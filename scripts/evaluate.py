#!/usr/bin/env python3
"""Model-agnostic evaluation CLI for CADICA lesion detection.

Accepts predictions and ground truth in JSON format and produces a full
evaluation report in outputs/eval/<run_name>/.

Input formats:
  --pred: Path to JSON file mapping image_id -> [[x,y,w,h,score,class_id], ...]
  --gt:   Path to JSON file mapping image_id -> [[x,y,w,h,class_id], ...]
  --split: If provided instead of --gt, loads GT from data_index/frames_index.csv
           and data_index/annotations.csv for the given split (train/val/test).
  --frame-patient: Path to JSON or CSV mapping frame_id -> patient_id.

Class scheme is determined by --scheme (7class, binary, severity3).

Example:
    python scripts/evaluate.py \\
        --pred outputs/predictions/val_preds.json \\
        --split val \\
        --scheme 7class \\
        --conf 0.5 \\
        --run-name val_7class \\
        --bootstrap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    compute_detection_metrics,
    compute_map_50_95,
    compute_confusion_matrix,
    compute_frame_metrics,
    compute_patient_metrics,
    bootstrap_confidence_intervals,
    find_best_conf_threshold,
    compute_f1_conf_curve,
)
from src.evaluation.report import generate_report
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("evaluate")


# ---------------------------------------------------------------------------
# Class scheme resolvers
# ---------------------------------------------------------------------------

BUILTIN_SCHEMES: dict[str, dict[str, int]] = {
    "7class": {
        "p0_20": 0, "p20_50": 1, "p50_70": 2,
        "p70_90": 3, "p90_98": 4, "p99": 5, "p100": 6,
    },
    "binary": {
        "nonlesion": 0, "p0_20": 1, "p20_50": 1, "p50_70": 1,
        "p70_90": 1, "p90_98": 1, "p99": 1, "p100": 1,
    },
    "severity3": {
        "p0_20": 0, "p20_50": 0, "p50_70": 1,
        "p70_90": 2, "p90_98": 2, "p99": 2, "p100": 2,
    },
}

SCHEME_NAMES: dict[str, dict[int, str]] = {
    "7class": {0: "p0_20", 1: "p20_50", 2: "p50_70", 3: "p70_90", 4: "p90_98", 5: "p99", 6: "p100"},
    "binary": {0: "normal", 1: "lesion"},
    "severity3": {0: "mild", 1: "moderate", 2: "severe"},
}

# Class IDs representing significant stenosis (>=70%)
SIGNIFICANT_CLASSES = {
    "7class": {3, 4, 5, 6},       # p70_90, p90_98, p99, p100
    "binary": {1},                 # lesion
    "severity3": {2},              # severe
}


# ---------------------------------------------------------------------------
# Ground truth loading from split
# ---------------------------------------------------------------------------

def load_gt_from_split(
    split: str,
    scheme: str,
    index_dir: Path,
    config: dict,
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """Load ground-truth boxes from frames_index.csv + annotations.csv for a split.

    Args:
        split: 'train', 'val', or 'test'.
        scheme: Class scheme name.
        index_dir: Path to data_index/ directory.
        config: Project configuration dict.

    Returns:
        (ground_truths, frame_to_patient): GT dict and patient mapping.
    """
    frames_path = index_dir / "frames_index.csv"
    ann_path = index_dir / "annotations.csv"
    splits_path = index_dir / "splits" / "split_v1.csv"

    if not frames_path.exists() or not ann_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Run build_index.py first.")

    frames_df = pd.read_csv(frames_path)
    ann_df = pd.read_csv(ann_path)

    if split != "all":
        if not splits_path.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_path}. Run make_splits.py first.")
        split_df = pd.read_csv(splits_path)
        split_patients = set(split_df[split_df["split"] == split]["patient_id"])
        frames_df = frames_df[frames_df["patient_id"].isin(split_patients)]
        ann_df = ann_df[ann_df["patient_id"].isin(split_patients)]

    class_mapping = _resolve_scheme(scheme, config)

    ground_truths: dict[str, list[list[float]]] = {}
    frame_to_patient: dict[str, str] = {}

    for _, row in frames_df.iterrows():
        fid = str(row["frame_id"])
        ground_truths[fid] = []
        frame_to_patient[fid] = str(row["patient_id"])

    for _, row in ann_df.iterrows():
        fid = str(row["frame_id"])
        category = str(row["category"])
        if category not in class_mapping:
            continue
        cls_id = class_mapping[category]
        # Use absolute pixel coords from index (image is 512x512)
        w_img = float(row.get("image_width", 512))
        h_img = float(row.get("image_height", 512))
        xc_n = float(row["x_center"])
        yc_n = float(row["y_center"])
        wn = float(row["w_norm"])
        hn = float(row["h_norm"])
        x = (xc_n - wn / 2) * w_img
        y = (yc_n - hn / 2) * h_img
        w = wn * w_img
        h = hn * h_img
        if fid in ground_truths:
            ground_truths[fid].append([x, y, w, h, cls_id])

    return ground_truths, frame_to_patient


def _resolve_scheme(scheme: str, config: dict) -> dict[str, int]:
    """Resolve class name -> class_id mapping for given scheme."""
    if scheme in BUILTIN_SCHEMES:
        return BUILTIN_SCHEMES[scheme]
    # Try config
    schemes = config.get("class_schemes", {}).get("schemes", {})
    if scheme in schemes:
        return {str(k): int(v) for k, v in schemes[scheme].items()}
    raise ValueError(f"Unknown class scheme: {scheme}. Choose from: {list(BUILTIN_SCHEMES)}")


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Model-agnostic evaluation for CADICA lesion detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pred", type=str, required=True,
        help="Path to JSON file: image_id -> [[x,y,w,h,score,class_id], ...]",
    )
    pred_gt = parser.add_mutually_exclusive_group(required=True)
    pred_gt.add_argument(
        "--gt", type=str,
        help="Path to JSON file: image_id -> [[x,y,w,h,class_id], ...]",
    )
    pred_gt.add_argument(
        "--split", type=str, choices=["train", "val", "test", "all"],
        help="Load GT from data_index for the given split.",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class",
        choices=["7class", "binary", "severity3"],
        help="Class scheme for evaluation.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold for object-level and frame-level metrics.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for positive matching.",
    )
    parser.add_argument(
        "--run-name", type=str, default="eval_run",
        help="Name of this evaluation run (used as output sub-directory).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval",
        help="Base output directory.",
    )
    parser.add_argument(
        "--frame-patient", type=str, default=None,
        help="JSON or CSV mapping frame_id -> patient_id for patient aggregation.",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Compute bootstrap 95%% CIs (adds runtime).",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=500,
        help="Number of bootstrap iterations.",
    )
    parser.add_argument(
        "--find-threshold", action="store_true",
        help="Sweep confidence thresholds and report best by macro F1.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--index-dir", type=str, default="data_index",
        help="Path to data_index/ directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main evaluation entry point."""
    args = parse_args(argv)

    # Load config
    config_path = Path(args.config)
    config = load_config(config_path) if config_path.exists() else {}

    # Resolve class scheme
    class_mapping = _resolve_scheme(args.scheme, config)
    class_names_map = SCHEME_NAMES.get(args.scheme, {int(v): k for k, v in class_mapping.items()})
    class_ids = sorted(set(class_mapping.values()))
    significant_cls = SIGNIFICANT_CLASSES.get(args.scheme, set())

    # Load predictions
    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    with open(pred_path, encoding="utf-8") as f:
        predictions: dict[str, list[list[float]]] = json.load(f)
    logger.info("Loaded predictions: %d images", len(predictions))

    # Load ground truths
    frame_to_patient: dict[str, str] = {}
    if args.gt:
        gt_path = Path(args.gt)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT file not found: {gt_path}")
        with open(gt_path, encoding="utf-8") as f:
            ground_truths: dict[str, list[list[float]]] = json.load(f)
        logger.info("Loaded ground truths: %d images", len(ground_truths))
    else:
        ground_truths, frame_to_patient = load_gt_from_split(
            args.split, args.scheme, Path(args.index_dir), config
        )
        logger.info("Loaded GT from split '%s': %d images", args.split, len(ground_truths))

    # Optionally override frame->patient mapping from file
    if args.frame_patient:
        fp_path = Path(args.frame_patient)
        if fp_path.suffix == ".json":
            with open(fp_path) as f:
                frame_to_patient = json.load(f)
        else:
            df = pd.read_csv(fp_path)
            frame_to_patient = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))

    # If frame_to_patient still empty, try to infer from frame IDs (pX_vY_ZZZZ)
    if not frame_to_patient:
        for fid in set(predictions) | set(ground_truths):
            parts = str(fid).split("_")
            frame_to_patient[fid] = parts[0] if parts else fid

    all_metrics: dict = {}

    # -----------------------------------------------------------------------
    # 1. Object detection metrics @ fixed threshold
    # -----------------------------------------------------------------------
    logger.info("Computing detection metrics (IoU=%.2f, Conf=%.2f)...", args.iou, args.conf)
    det_metrics = compute_detection_metrics(
        predictions, ground_truths, class_ids, class_names_map,
        iou_threshold=args.iou, conf_threshold=args.conf,
    )
    all_metrics["detection"] = det_metrics

    # -----------------------------------------------------------------------
    # 2. mAP@0.5 and mAP@0.5:0.95
    # -----------------------------------------------------------------------
    logger.info("Computing mAP@0.5 and mAP@0.5:0.95...")
    map_metrics = compute_map_50_95(predictions, ground_truths, class_ids)
    all_metrics["map"] = map_metrics
    logger.info("mAP@0.5=%.4f  mAP@0.5:0.95=%.4f",
                map_metrics["mAP_50"], map_metrics["mAP_50_95"])

    # -----------------------------------------------------------------------
    # 3. Confusion matrix
    # -----------------------------------------------------------------------
    logger.info("Computing confusion matrix...")
    cm = compute_confusion_matrix(
        predictions, ground_truths, class_ids,
        iou_threshold=args.iou, conf_threshold=args.conf,
    )
    all_metrics["confusion_matrix"] = cm.tolist()

    # -----------------------------------------------------------------------
    # 4. Frame-level classification
    # -----------------------------------------------------------------------
    logger.info("Computing frame-level metrics...")
    frame_metrics = compute_frame_metrics(predictions, ground_truths, conf_threshold=args.conf)
    all_metrics["frame"] = frame_metrics
    logger.info("Frame ROC-AUC=%.4f  PR-AUC=%.4f  F1=%.4f",
                frame_metrics["roc_auc"], frame_metrics["pr_auc"], frame_metrics["f1"])

    # -----------------------------------------------------------------------
    # 5. Patient-level aggregation
    # -----------------------------------------------------------------------
    if frame_to_patient:
        logger.info("Computing patient-level metrics...")
        patient_metrics = compute_patient_metrics(
            predictions, ground_truths,
            frame_to_patient=frame_to_patient,
            conf_threshold=args.conf,
            significant_class_ids=significant_cls,
        )
        all_metrics["patient"] = patient_metrics

    # -----------------------------------------------------------------------
    # 6. F1-vs-confidence curves
    # -----------------------------------------------------------------------
    logger.info("Computing F1-vs-confidence curves...")
    f1_curves = compute_f1_conf_curve(predictions, ground_truths, class_ids, iou_threshold=args.iou)
    all_metrics["f1_conf_curves"] = f1_curves

    # -----------------------------------------------------------------------
    # 7. Best confidence threshold
    # -----------------------------------------------------------------------
    if args.find_threshold:
        logger.info("Finding best confidence threshold by macro F1...")
        best_thr = find_best_conf_threshold(predictions, ground_truths, class_ids, args.iou)
        all_metrics["best_threshold"] = best_thr
        logger.info("Best conf threshold=%.4f (macro F1=%.4f)",
                    best_thr["best_conf_threshold"], best_thr["best_macro_f1"])

    # -----------------------------------------------------------------------
    # 8. Bootstrap CIs
    # -----------------------------------------------------------------------
    if args.bootstrap:
        logger.info("Computing bootstrap CIs (n=%d)...", args.n_bootstrap)
        boot = bootstrap_confidence_intervals(
            predictions, ground_truths, class_ids, frame_to_patient,
            class_names_map, iou_threshold=args.iou, conf_threshold=args.conf,
            n_bootstrap=args.n_bootstrap,
        )
        all_metrics["bootstrap"] = boot
        logger.info("Bootstrap macro F1: %.4f [%.4f, %.4f]",
                    boot["macro_f1"]["mean"], boot["macro_f1"]["lower"], boot["macro_f1"]["upper"])

    # -----------------------------------------------------------------------
    # 9. Generate full report
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir) / args.run_name
    logger.info("Generating report in %s...", output_dir)
    generate_report(
        all_metrics=all_metrics,
        class_ids=class_ids,
        class_names=class_names_map,
        run_name=args.run_name,
        output_dir=output_dir,
    )

    print(f"\n[OK] Evaluation complete: {output_dir}")
    print(f"   mAP@0.5         = {map_metrics['mAP_50']:.4f}")
    print(f"   mAP@0.5:0.95    = {map_metrics['mAP_50_95']:.4f}")
    print(f"   Micro F1        = {det_metrics['micro']['f1']:.4f}")
    print(f"   Frame ROC-AUC   = {frame_metrics['roc_auc']:.4f}")
    print(f"   Frame PR-AUC    = {frame_metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
