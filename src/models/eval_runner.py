"""Evaluation runner for calibrated YOLO model runs.

Calibrates confidence threshold on validation split via maximum macro F1,
freezes the threshold in threshold.json, evaluates both validation and test
splits using the model-agnostic evaluation framework, compares against
Ultralytics native metrics, and writes comprehensive clinical reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.data.eval_sets import load_eval_ground_truth
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

logger = get_logger("eval_runner")

SCHEME_NAMES = {
    "7class": {0: "p0_20", 1: "p20_50", 2: "p50_70", 3: "p70_90", 4: "p90_98", 5: "p99", 6: "p100"},
    "binary": {0: "normal", 1: "lesion"},
    "severity3": {0: "mild", 1: "moderate", 2: "severe"},
}

SIGNIFICANT_CLASSES = {
    "7class": {3, 4, 5, 6},       # p70_90, p90_98, p99, p100
    "binary": {1},                 # lesion
    "severity3": {2},              # severe
}


def read_ultralytics_map50_from_run(run_dir: Path | str) -> float | None:
    """Read the best mAP@0.5 achieved by Ultralytics from results.csv.

    Args:
        run_dir: Run directory path containing results.csv.

    Returns:
        Best mAP50 float, or None if unavailable.
    """
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        candidates = ["metrics/mAP50(B)", "mAP50(B)", "metrics/mAP_0.5"]
        col = next((c for c in candidates if c in df.columns), None)
        if col:
            best_map = float(df[col].max())
            return round(best_map, 4)
    except Exception as e:
        logger.warning("Could not read Ultralytics mAP@0.5 from %s: %s", csv_path, e)

    return None


def evaluate_split(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    frame_to_patient: dict[str, str],
    split: str,
    scheme: str,
    run_name: str,
    output_dir: Path | str = "outputs/eval",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    bootstrap: bool = True,
    n_bootstrap: int = 500,
    ultralytics_map50: float | None = None,
) -> dict[str, Any]:
    """Execute full Phase 2 evaluation report generation for a single split.

    Args:
        predictions: Dict frame_id -> [[x,y,w,h,score,class], ...].
        ground_truths: Dict frame_id -> [[x,y,w,h,class], ...].
        frame_to_patient: Dict frame_id -> patient_id.
        split: 'val' or 'test'.
        scheme: Class scheme ('7class', 'binary', 'severity3').
        run_name: Name of the run.
        output_dir: Base evaluation output directory.
        conf_threshold: Calibrated confidence threshold.
        iou_threshold: Match IoU threshold (default: 0.5).
        bootstrap: Whether to compute patient bootstrap CIs.
        n_bootstrap: Number of bootstrap iterations.
        ultralytics_map50: Optional Ultralytics mAP@0.5 for comparison.

    Returns:
        Dictionary of all computed metrics.
    """
    class_names_map = SCHEME_NAMES.get(scheme, {i: f"cls_{i}" for i in range(7)})
    class_ids = sorted(class_names_map.keys())
    sig_classes = SIGNIFICANT_CLASSES.get(scheme, set())

    eval_out_dir = Path(output_dir) / f"{run_name}_{split}"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, Any] = {
        "split": split,
        "scheme": scheme,
        "conf_threshold": round(conf_threshold, 4),
        "iou_threshold": round(iou_threshold, 4),
    }

    # 1. Detection metrics at confidence threshold
    logger.info("Computing detection metrics for %s split (IoU=%.2f, Conf=%.4f)...", split, iou_threshold, conf_threshold)
    det_metrics = compute_detection_metrics(
        predictions, ground_truths, class_ids, class_names_map,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    all_metrics["detection"] = det_metrics

    # 2. mAP@0.5 and mAP@0.5:0.95 (computed across full score range)
    logger.info("Computing mAP@0.5 and mAP@0.5:0.95 for %s split...", split)
    map_metrics = compute_map_50_95(predictions, ground_truths, class_ids)
    all_metrics["map"] = map_metrics

    # Cross-check against Ultralytics mAP@0.5
    if ultralytics_map50 is not None:
        our_map50 = float(map_metrics["mAP_50"])
        diff = abs(our_map50 - ultralytics_map50)
        comparison_info = {
            "custom_map50": round(our_map50, 4),
            "ultralytics_map50": round(ultralytics_map50, 4),
            "difference": round(diff, 4),
            "cross_check_passed": bool(diff <= 0.02),
        }
        all_metrics["ultralytics_comparison"] = comparison_info

        if diff > 0.02:
            warn_msg = (
                f"[WARNING] Difference between custom mAP@0.5 ({our_map50:.4f}) and "
                f"Ultralytics mAP@0.5 ({ultralytics_map50:.4f}) is {diff:.4f} (> 0.02)!"
            )
            logger.warning(warn_msg)
            warnings.warn(warn_msg, UserWarning, stacklevel=2)
        else:
            logger.info(
                "[PASS] mAP@0.5 cross-check passed: custom=%.4f, ultralytics=%.4f, diff=%.4f <= 0.02",
                our_map50,
                ultralytics_map50,
                diff,
            )

    # 3. Confusion matrix
    logger.info("Computing confusion matrix for %s split...", split)
    cm = compute_confusion_matrix(
        predictions, ground_truths, class_ids,
        iou_threshold=iou_threshold, conf_threshold=conf_threshold,
    )
    all_metrics["confusion_matrix"] = cm.tolist()

    # 4. Frame-level classification
    logger.info("Computing frame-level classification metrics for %s split...", split)
    frame_metrics = compute_frame_metrics(predictions, ground_truths, conf_threshold=conf_threshold)
    all_metrics["frame"] = frame_metrics

    # 5. Patient-level aggregation & significant stenosis
    logger.info("Computing patient-level metrics for %s split...", split)
    patient_metrics = compute_patient_metrics(
        predictions, ground_truths,
        frame_to_patient=frame_to_patient,
        conf_threshold=conf_threshold,
        significant_class_ids=sig_classes,
    )
    all_metrics["patient"] = patient_metrics

    # 6. F1-vs-confidence curves
    logger.info("Computing F1-vs-confidence curves for %s split...", split)
    f1_curves = compute_f1_conf_curve(predictions, ground_truths, class_ids, iou_threshold=iou_threshold)
    all_metrics["f1_conf_curves"] = f1_curves

    # 7. Patient bootstrap CIs
    if bootstrap:
        logger.info("Computing patient bootstrap 95%% CIs (n=%d) for %s split...", n_bootstrap, split)
        boot = bootstrap_confidence_intervals(
            predictions, ground_truths, class_ids, frame_to_patient,
            class_names_map, iou_threshold=iou_threshold, conf_threshold=conf_threshold,
            n_bootstrap=n_bootstrap,
        )
        all_metrics["bootstrap"] = boot

    # 8. Generate full report (Markdown, JSON, Figures)
    generate_report(
        all_metrics=all_metrics,
        class_ids=class_ids,
        class_names=class_names_map,
        run_name=f"{run_name}_{split}",
        output_dir=eval_out_dir,
    )

    logger.info("Evaluation report complete for %s: %s", split, eval_out_dir)
    return all_metrics


def run_pipeline_evaluation(
    run_dir: str | Path,
    scheme: str = "7class",
    val_preds_path: str | Path | None = None,
    test_preds_path: str | Path | None = None,
    output_dir: str | Path = "outputs/eval",
    yolo_dir: str | Path = "outputs/yolo_dataset",
    bootstrap: bool = True,
    n_bootstrap: int = 500,
    fold: int | None = None,
) -> dict[str, Any]:
    """Execute complete validation threshold freezing and test evaluation protocol.

    1. Load VAL predictions and ground truth.
    2. Sweep confidence thresholds on VAL only and identify optimal threshold by max macro F1.
    3. Freeze chosen threshold into threshold.json in run_dir.
    4. Generate full evaluation report for VAL split at frozen threshold.
    5. Compare against Ultralytics results.csv mAP@0.5 and warn if diff > 0.02.
    6. If TEST predictions are provided/present, apply frozen threshold to TEST split.

    Args:
        run_dir: Run directory (e.g., outputs/runs/<run_name>).
        scheme: Class scheme ('7class', 'binary', 'severity3').
        val_preds_path: Path to preds_val.json (defaults to <run_dir>/preds_val.json).
        test_preds_path: Path to preds_test.json (defaults to <run_dir>/preds_test.json).
        output_dir: Destination directory for evaluation reports.
        yolo_dir: YOLO dataset directory.
        bootstrap: Whether to compute bootstrap CIs.
        n_bootstrap: Number of bootstrap iterations.
        fold: Optional cross-validation fold index (0 to 4).

    Returns:
        Dictionary containing threshold details and evaluation summaries.
    """
    run_dir = Path(run_dir)
    run_name = run_dir.name

    # 1. Resolve predictions paths
    if val_preds_path is None:
        val_preds_path = run_dir / "preds_val.json"
    else:
        val_preds_path = Path(val_preds_path)

    if not val_preds_path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {val_preds_path}")

    with open(val_preds_path, encoding="utf-8") as f:
        val_preds = json.load(f)

    # 2. Load ground truths for val
    val_gts, val_f2p = load_eval_ground_truth(
        split="val",
        scheme=scheme,
        yolo_dir=yolo_dir,
        fold=fold,
    )

    class_names_map = SCHEME_NAMES.get(scheme, {i: f"cls_{i}" for i in range(7)})
    class_ids = sorted(class_names_map.keys())

    # 3. Find optimal threshold on VAL only
    logger.info("Searching for optimal confidence threshold on VAL split by max macro F1...")
    thr_search = find_best_conf_threshold(
        predictions=val_preds,
        ground_truths=val_gts,
        class_ids=class_ids,
        iou_threshold=0.5,
        n_thresholds=50,
    )
    best_conf = float(thr_search["best_conf_threshold"])
    best_macro_f1 = float(thr_search["best_macro_f1"])
    logger.info("Optimal VAL confidence threshold: %.4f (macro F1: %.4f)", best_conf, best_macro_f1)

    # 4. Freeze threshold into threshold.json
    threshold_info = {
        "run_name": run_name,
        "scheme": scheme,
        "best_conf_threshold": best_conf,
        "best_macro_f1": best_macro_f1,
        "sweep": thr_search.get("sweep", []),
    }
    threshold_json_path = run_dir / "threshold.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)
    logger.info("Frozen confidence threshold saved to %s", threshold_json_path)

    # 5. Read Ultralytics mAP@0.5 from run_dir
    ultralytics_map50 = read_ultralytics_map50_from_run(run_dir)

    # 6. Evaluate VAL split
    logger.info("Evaluating VAL split at frozen threshold %.4f...", best_conf)
    val_metrics = evaluate_split(
        predictions=val_preds,
        ground_truths=val_gts,
        frame_to_patient=val_f2p,
        split="val",
        scheme=scheme,
        run_name=run_name,
        output_dir=output_dir,
        conf_threshold=best_conf,
        iou_threshold=0.5,
        bootstrap=bootstrap,
        n_bootstrap=n_bootstrap,
        ultralytics_map50=ultralytics_map50,
    )

    # 7. Evaluate TEST split if present
    test_metrics = None
    if test_preds_path is None:
        candidate_test = run_dir / "preds_test.json"
        if candidate_test.exists():
            test_preds_path = candidate_test

    if test_preds_path and Path(test_preds_path).exists():
        test_preds_path = Path(test_preds_path)
        logger.info("Found TEST predictions at %s. Applying frozen threshold %.4f once to TEST...", test_preds_path, best_conf)
        with open(test_preds_path, encoding="utf-8") as f:
            test_preds = json.load(f)

        test_gts, test_f2p = load_eval_ground_truth(
            split="test",
            scheme=scheme,
            yolo_dir=yolo_dir,
            fold=fold,
        )

        test_metrics = evaluate_split(
            predictions=test_preds,
            ground_truths=test_gts,
            frame_to_patient=test_f2p,
            split="test",
            scheme=scheme,
            run_name=run_name,
            output_dir=output_dir,
            conf_threshold=best_conf,
            iou_threshold=0.5,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            ultralytics_map50=None,
        )

    return {
        "threshold": threshold_info,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
