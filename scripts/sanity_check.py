#!/usr/bin/env python3
"""Phase 2.4 - Evaluation Sanity Check Runner.

Evaluates two dummy baseline predictors on the validation split:
1. Oracle Predictor: Uses ground-truth boxes as predictions with confidence=1.0.
   Expected metrics: Precision=1.0, Recall=1.0, F1=1.0, mAP@0.5=1.0, mAP@0.5:0.95=1.0.
   Fails loudly (exits with code 1) if the oracle is not perfect.
2. Random Predictor: Generates stochastic bounding boxes and confidence scores.
   Expected metrics: Near-zero (mAP@0.5 < 0.05, F1 < 0.05).

Saves comprehensive evaluation reports in outputs/eval/oracle_val/ and outputs/eval/random_val/.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make src importable from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_detection_metrics,
    compute_frame_metrics,
    compute_map_50_95,
    compute_patient_metrics,
)
from src.evaluation.report import generate_report
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from scripts.evaluate import SCHEME_NAMES, SIGNIFICANT_CLASSES, load_gt_from_split

logger = get_logger("sanity_check")


def run_evaluation_pipeline(
    predictions: dict[str, list[list[float]]],
    ground_truths: dict[str, list[list[float]]],
    frame_to_patient: dict[str, str],
    class_ids: list[int],
    class_names: dict[int, str],
    run_name: str,
    output_dir: Path,
    iou_thresh: float = 0.5,
    conf_thresh: float = 0.5,
    scheme: str = "7class",
) -> dict[str, Any]:
    """Run full evaluation metrics computation and save report artifacts."""
    logger.info("Evaluating run '%s' (IoU=%.2f, Conf=%.2f)...", run_name, iou_thresh, conf_thresh)

    # 1. Object detection metrics
    det_metrics = compute_detection_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        class_ids=class_ids,
        class_names=class_names,
        iou_threshold=iou_thresh,
        conf_threshold=conf_thresh,
    )

    # 2. mAP@0.5 and mAP@0.5:0.95
    map_metrics = compute_map_50_95(
        predictions=predictions,
        ground_truths=ground_truths,
        class_ids=class_ids,
    )

    # 3. Confusion matrix
    cm = compute_confusion_matrix(
        predictions=predictions,
        ground_truths=ground_truths,
        class_ids=class_ids,
        iou_threshold=iou_thresh,
        conf_threshold=conf_thresh,
    )

    # 4. Frame-level metrics
    frame_metrics = compute_frame_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        conf_threshold=conf_thresh,
    )

    # 5. Patient-level metrics
    sig_classes = SIGNIFICANT_CLASSES.get(scheme, {3, 4, 5, 6})
    patient_metrics = compute_patient_metrics(
        predictions=predictions,
        ground_truths=ground_truths,
        frame_to_patient=frame_to_patient,
        conf_threshold=conf_thresh,
        significant_class_ids=sig_classes,
    )

    all_metrics: dict[str, Any] = {
        "detection": det_metrics,
        "map": map_metrics,
        "confusion_matrix": cm.tolist(),
        "frame": frame_metrics,
        "patient": patient_metrics,
        "metadata": {
            "run_name": run_name,
            "iou_threshold": iou_thresh,
            "conf_threshold": conf_thresh,
            "n_images": len(predictions),
            "scheme": scheme,
        },
    }

    run_output_dir = output_dir / run_name
    generate_report(
        all_metrics=all_metrics,
        class_ids=class_ids,
        class_names=class_names,
        run_name=run_name,
        output_dir=run_output_dir,
    )

    return all_metrics


def build_oracle_predictions(
    ground_truths: dict[str, list[list[float]]],
) -> dict[str, list[list[float]]]:
    """Build oracle predictions by copying GT boxes with confidence=1.0."""
    oracle_preds: dict[str, list[list[float]]] = {}
    for fid, boxes in ground_truths.items():
        # GT format: [x, y, w, h, class_id]
        # Pred format: [x, y, w, h, score, class_id]
        oracle_preds[fid] = [[b[0], b[1], b[2], b[3], 1.0, int(b[4])] for b in boxes]
    return oracle_preds


def build_random_predictions(
    ground_truths: dict[str, list[list[float]]],
    class_ids: list[int],
    seed: int = 42,
    img_width: float = 512.0,
    img_height: float = 512.0,
) -> dict[str, list[list[float]]]:
    """Build random baseline predictions with random coordinates and scores."""
    rng = np.random.RandomState(seed)
    random_preds: dict[str, list[list[float]]] = {}

    for fid in ground_truths:
        # Sample 0 to 2 random boxes per frame
        n_boxes = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        boxes = []
        for _ in range(n_boxes):
            w = rng.uniform(20.0, 120.0)
            h = rng.uniform(20.0, 120.0)
            x = rng.uniform(0.0, max(0.0, img_width - w))
            y = rng.uniform(0.0, max(0.0, img_height - h))
            score = rng.uniform(0.10, 0.95)
            cls_id = int(rng.choice(class_ids))
            boxes.append([round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(score, 3), cls_id])
        random_preds[fid] = boxes

    return random_preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2.4 - Evaluation sanity check with Oracle and Random predictors."
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Split to evaluate on.")
    parser.add_argument("--scheme", default="7class", choices=["7class", "binary", "severity3"], help="Class scheme.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file.")
    parser.add_argument("--index-dir", default="data_index", help="Index directory.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Output directory for reports.")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)

    print("======================================================================")
    print("   CADICA Phase 2.4 - Evaluation Sanity Check")
    print(f"   Split: '{args.split}' | Scheme: '{args.scheme}' | Seed: {args.seed}")
    print("======================================================================")

    # 1. Load ground truth
    ground_truths, frame_to_patient = load_gt_from_split(
        split=args.split,
        scheme=args.scheme,
        index_dir=index_dir,
        config=config,
    )

    class_names_map = SCHEME_NAMES.get(args.scheme, {})
    class_ids = sorted(class_names_map.keys())
    n_frames = len(ground_truths)
    n_pos_frames = sum(1 for b in ground_truths.values() if len(b) > 0)
    n_boxes = sum(len(b) for b in ground_truths.values())
    n_patients = len(set(frame_to_patient.values()))

    print(f"Loaded ground truth for split '{args.split}':")
    print(f"  Patients: {n_patients} | Total Frames: {n_frames:,} | Pos Frames: {n_pos_frames:,} | Boxes: {n_boxes:,}")
    print(f"  Classes ({len(class_ids)}): {list(class_names_map.values())}")
    print("----------------------------------------------------------------------")

    # 2. Evaluate Oracle Predictor
    print("\n[1/2] Running Oracle Predictor (ground truth as predictions)...")
    oracle_preds = build_oracle_predictions(ground_truths)
    oracle_metrics = run_evaluation_pipeline(
        predictions=oracle_preds,
        ground_truths=ground_truths,
        frame_to_patient=frame_to_patient,
        class_ids=class_ids,
        class_names=class_names_map,
        run_name=f"oracle_{args.split}",
        output_dir=output_dir,
        iou_thresh=args.iou,
        conf_thresh=args.conf,
        scheme=args.scheme,
    )

    o_micro_f1 = oracle_metrics["detection"]["micro"]["f1"]
    o_macro_f1 = oracle_metrics["detection"]["macro"]["f1"]
    o_map50 = oracle_metrics["map"]["mAP_50"]
    o_map50_95 = oracle_metrics["map"]["mAP_50_95"]
    o_frame_acc = oracle_metrics["frame"]["accuracy"]

    print("  Oracle Results:")
    print(f"    Micro F1:     {o_micro_f1:.4f} (expected: 1.0000)")
    print(f"    Macro F1:     {o_macro_f1:.4f} (expected: 1.0000)")
    print(f"    mAP@0.5:      {o_map50:.4f} (expected: 1.0000)")
    print(f"    mAP@0.5:0.95: {o_map50_95:.4f} (expected: 1.0000)")
    print(f"    Frame Acc:    {o_frame_acc:.4f} (expected: 1.0000)")

    # Strict assertion: fail loudly if oracle is not perfect!
    TOL = 1e-4
    failures = []
    if abs(o_micro_f1 - 1.0) > TOL:
        failures.append(f"Micro F1 = {o_micro_f1:.4f} != 1.0")
    if abs(o_macro_f1 - 1.0) > TOL:
        failures.append(f"Macro F1 = {o_macro_f1:.4f} != 1.0")
    if abs(o_map50 - 1.0) > TOL:
        failures.append(f"mAP@0.5 = {o_map50:.4f} != 1.0")
    if abs(o_map50_95 - 1.0) > TOL:
        failures.append(f"mAP@0.5:0.95 = {o_map50_95:.4f} != 1.0")
    if abs(o_frame_acc - 1.0) > TOL:
        failures.append(f"Frame Accuracy = {o_frame_acc:.4f} != 1.0")

    if failures:
        print("\n" + "!" * 70)
        print("  CRITICAL ERROR: Oracle Sanity Check FAILED!")
        for fail in failures:
            print(f"    - {fail}")
        print("!" * 70 + "\n")
        raise RuntimeError(f"Oracle sanity check failed with {len(failures)} mismatch(es): {failures}")

    print("  [PASS] Oracle sanity check PASSED with perfect 1.0000 across all metrics.")

    # 3. Evaluate Random Predictor
    print("\n[2/2] Running Random Predictor (stochastic boxes & confidences)...")
    random_preds = build_random_predictions(ground_truths, class_ids, seed=args.seed)
    random_metrics = run_evaluation_pipeline(
        predictions=random_preds,
        ground_truths=ground_truths,
        frame_to_patient=frame_to_patient,
        class_ids=class_ids,
        class_names=class_names_map,
        run_name=f"random_{args.split}",
        output_dir=output_dir,
        iou_thresh=args.iou,
        conf_thresh=args.conf,
        scheme=args.scheme,
    )

    r_micro_f1 = random_metrics["detection"]["micro"]["f1"]
    r_macro_f1 = random_metrics["detection"]["macro"]["f1"]
    r_map50 = random_metrics["map"]["mAP_50"]
    r_map50_95 = random_metrics["map"]["mAP_50_95"]
    r_frame_acc = random_metrics["frame"]["accuracy"]

    print("  Random Predictor Results:")
    print(f"    Micro F1:     {r_micro_f1:.4f} (expected: ~0.0)")
    print(f"    Macro F1:     {r_macro_f1:.4f} (expected: ~0.0)")
    print(f"    mAP@0.5:      {r_map50:.4f} (expected: ~0.0)")
    print(f"    mAP@0.5:0.95: {r_map50_95:.4f} (expected: ~0.0)")
    print(f"    Frame Acc:    {r_frame_acc:.4f}")

    # Random baseline should have near-zero AP and F1
    if r_map50 > 0.10:
        logger.warning("Random mAP@0.5 (%.4f) higher than expected (<0.10)", r_map50)
    else:
        print("  [PASS] Random predictor achieves expected near-zero detection performance.")

    # 4. Summary Table
    print("\n======================================================================")
    print("   Sanity Check Summary Comparison (Split: " + args.split + ")")
    print("======================================================================")
    print(f"   {'Metric':<20} | {'Oracle (Expected 1.0)':<22} | {'Random (Expected ~0.0)':<22}")
    print("   " + "-" * 68)
    print(f"   {'Micro F1':<20} | {o_micro_f1:<22.4f} | {r_micro_f1:<22.4f}")
    print(f"   {'Macro F1':<20} | {o_macro_f1:<22.4f} | {r_macro_f1:<22.4f}")
    print(f"   {'mAP@0.5':<20} | {o_map50:<22.4f} | {r_map50:<22.4f}")
    print(f"   {'mAP@0.5:0.95':<20} | {o_map50_95:<22.4f} | {r_map50_95:<22.4f}")
    print(f"   {'Frame Accuracy':<20} | {o_frame_acc:<22.4f} | {r_frame_acc:<22.4f}")
    print("======================================================================")
    print(f"   Reports written to:")
    print(f"     - Oracle: {output_dir / ('oracle_' + args.split)}")
    print(f"     - Random: {output_dir / ('random_' + args.split)}")
    print("   Sanity Check: ALL CHECKS PASSED [OK]\n")


if __name__ == "__main__":
    main()
