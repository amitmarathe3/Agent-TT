#!/usr/bin/env python3
"""CLI runner for evaluating a trained YOLO run.

Calibrates the optimal confidence threshold on the validation split (max macro F1),
freezes it in threshold.json, evaluates the validation and test splits with full
Phase 2 metrics, and compares custom mAP@0.5 against Ultralytics' logged mAP@0.5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.eval_runner import run_pipeline_evaluation
from src.utils.logger import get_logger

logger = get_logger("evaluate_run_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for YOLO run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO run: calibrate on val, freeze threshold, evaluate val & test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to training run directory (e.g. outputs/runs/yolo_7class).",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class",
        choices=["7class", "binary", "severity3"],
        help="Class scheme for evaluation.",
    )
    parser.add_argument(
        "--val-preds", type=str, default=None,
        help="Path to val predictions JSON (defaults to <run_dir>/preds_val.json).",
    )
    parser.add_argument(
        "--test-preds", type=str, default=None,
        help="Path to test predictions JSON (defaults to <run_dir>/preds_test.json if present).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval",
        help="Base output directory for evaluation reports.",
    )
    parser.add_argument(
        "--yolo-dir", type=str, default="outputs/yolo_dataset",
        help="Path to exported YOLO dataset directory.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip patient-level bootstrap CIs to speed up runtime.",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=500,
        help="Number of patient bootstrap iterations.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI execution entrypoint."""
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    logger.info("Starting evaluation for YOLO run '%s'...", run_dir.name)

    res = run_pipeline_evaluation(
        run_dir=run_dir,
        scheme=args.scheme,
        val_preds_path=args.val_preds,
        test_preds_path=args.test_preds,
        output_dir=args.output_dir,
        yolo_dir=args.yolo_dir,
        bootstrap=not args.no_bootstrap,
        n_bootstrap=args.n_bootstrap,
    )

    thr_info = res["threshold"]
    val_m = res["val_metrics"]
    test_m = res.get("test_metrics")

    print("\n" + "=" * 80)
    print(f"   [OK] YOLO Run Evaluation Complete: {run_dir.name}")
    print("=" * 80)
    print(f"   Frozen Threshold:    {thr_info['best_conf_threshold']:.4f} (best Val Macro F1: {thr_info['best_macro_f1']:.4f})")
    print(f"   Threshold File:      {run_dir}/threshold.json")
    print(f"   Validation Report:   {args.output_dir}/{run_dir.name}_val/metrics.md")
    print(f"   Val mAP@0.5:         {val_m['map']['mAP_50']:.4f}")
    print(f"   Val mAP@0.5:0.95:    {val_m['map']['mAP_50_95']:.4f}")
    print(f"   Val Micro F1:        {val_m['detection']['micro']['f1']:.4f}")
    print(f"   Val Frame ROC-AUC:   {val_m['frame']['roc_auc']:.4f}")

    if "ultralytics_comparison" in val_m:
        comp = val_m["ultralytics_comparison"]
        status_str = "[OK] Passed (<=0.02)" if comp["cross_check_passed"] else "[WARN] Exceeds 0.02"
        print(f"   Ultralytics Check:   Custom={comp['custom_map50']:.4f} vs Ultra={comp['ultralytics_map50']:.4f} (Diff: {comp['difference']:.4f}) -> {status_str}")

    if test_m:
        print(f"\n   Test Report:         {args.output_dir}/{run_dir.name}_test/metrics.md")
        print(f"   Test mAP@0.5:        {test_m['map']['mAP_50']:.4f}")
        print(f"   Test mAP@0.5:0.95:   {test_m['map']['mAP_50_95']:.4f}")
        print(f"   Test Micro F1:       {test_m['detection']['micro']['f1']:.4f}")
        print(f"   Test Frame ROC-AUC:  {test_m['frame']['roc_auc']:.4f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
