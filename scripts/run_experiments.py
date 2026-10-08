#!/usr/bin/env python3
"""CLI runner for CADICA YOLO baseline benchmark matrix and 5-fold cross-validation.

Orchestrates reproducible, resumable experiment suites and updates outputs/experiments/leaderboard.csv
and outputs/experiments/cv_summary.md.

Usage:
    # Run a single experiment
    python scripts/run_experiments.py --run-name exp_7class_yolo11s_oversample --scheme 7class --model yolo11s.pt --oversample-rare

    # Run complete baseline matrix (A, B, C, D)
    python scripts/run_experiments.py --matrix

    # Run 5-fold patient-wise cross-validation for best config
    python scripts/run_experiments.py --cv

    # Fast verification smoke test across matrix
    python scripts/run_experiments.py --matrix --smoke-test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.experiment_runner import (
    generate_cv_summary,
    load_leaderboard,
    run_single_experiment,
)
from src.utils.logger import get_logger

logger = get_logger("run_experiments_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for experiment runner."""
    parser = argparse.ArgumentParser(
        description="Run CADICA YOLO baseline benchmark matrix and 5-fold CV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Run full baseline experiment matrix (A: binary, B: severity3, C: 7class +/- oversample, D: yolo11n/m).",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold patient-wise cross validation for best config and write cv_summary.md.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Custom run name for single experiment execution.",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class",
        choices=["7class", "binary", "severity3"],
        help="Class scheme for single experiment run.",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11s.pt",
        help="Model architecture or pretrained weights for single run.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size in pixels.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--oversample-rare", action="store_true",
        help="Enable rare-class 3x oversampling.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Optional fold index (0 to 4).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast verification mode: train for 2 epochs on fraction=0.1 of data.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-execution even if experiment is already marked as completed.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing experiment results (alias for --force).",
    )
    parser.add_argument(
        "--best-run", type=str, default="yolo11s_7class_os",
        help="Base run name for 5-fold cross-validation.",
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Subset of experiment run names to execute.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Skip patient-level bootstrap CI computation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device ('0', 'cpu', etc.). Auto-detects by default.",
    )
    parser.add_argument(
        "--exp-dir", type=str, default="outputs/experiments",
        help="Base directory for experiment outputs and leaderboard.csv.",
    )
    parsed = parser.parse_args(argv)
    if parsed.overwrite:
        parsed.force = True
    return parsed


def run_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Execute standard baseline matrix (Runs A, B, C, D)."""
    matrix_configs = [
        # A. Binary
        {"run_name": "exp_binary_yolo11s", "scheme": "binary", "model": "yolo11s.pt", "oversample": False},
        # B. Severity3
        {"run_name": "exp_severity3_yolo11s", "scheme": "severity3", "model": "yolo11s.pt", "oversample": False},
        # C. 7class without & with oversample
        {"run_name": "exp_7class_yolo11s", "scheme": "7class", "model": "yolo11s.pt", "oversample": False},
        {"run_name": "exp_7class_yolo11s_oversample", "scheme": "7class", "model": "yolo11s.pt", "oversample": True},
        # D. Model variants (nano & medium)
        {"run_name": "exp_7class_yolo11n_oversample", "scheme": "7class", "model": "yolo11n.pt", "oversample": True},
        {"run_name": "exp_7class_yolo11m_oversample", "scheme": "7class", "model": "yolo11m.pt", "oversample": True},
    ]

    results = []
    for cfg in matrix_configs:
        logger.info("--- Starting Benchmark Matrix Run: %s ---", cfg["run_name"])
        res = run_single_experiment(
            run_name=cfg["run_name"],
            scheme=cfg["scheme"],
            model=cfg["model"],
            imgsz=args.imgsz,
            oversample=cfg["oversample"],
            fold=None,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            force=args.force,
            device=args.device,
            exp_dir=args.exp_dir,
        )
        results.append(res)

    return results


def run_5fold_cv(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Execute 5-fold cross-validation on best config and generate summary."""
    cv_results = []
    best_scheme = "7class"
    best_model = "yolo11s.pt"
    oversample = True

    for k in range(5):
        run_name = f"exp_cv_fold{k}"
        logger.info("--- Starting 5-Fold Cross-Validation: Fold %d (%s) ---", k, run_name)
        res = run_single_experiment(
            run_name=run_name,
            scheme=best_scheme,
            model=best_model,
            imgsz=args.imgsz,
            oversample=oversample,
            fold=k,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            force=args.force,
            device=args.device,
            exp_dir=args.exp_dir,
        )
        cv_results.append(res)

    # Generate cv_summary.md
    summary_path = Path(args.exp_dir) / "cv_summary.md"
    generate_cv_summary(cv_results, output_path=summary_path)
    return cv_results


def main(argv: list[str] | None = None) -> None:
    """CLI execution entrypoint."""
    args = parse_args(argv)

    if not args.matrix and not args.cv and not args.run_name:
        logger.info("No run target specified. Defaulting to running full benchmark matrix (--matrix)...")
        args.matrix = True

    if args.matrix:
        logger.info("Running complete Phase 3 baseline benchmark matrix...")
        run_matrix(args)

    if args.cv:
        logger.info("Running 5-fold patient-wise cross-validation...")
        run_5fold_cv(args)

    if args.run_name:
        logger.info("Running single experiment: %s", args.run_name)
        run_single_experiment(
            run_name=args.run_name,
            scheme=args.scheme,
            model=args.model,
            imgsz=args.imgsz,
            oversample=args.oversample_rare,
            fold=args.fold,
            epochs=args.epochs,
            smoke_test=args.smoke_test,
            force=args.force,
            device=args.device,
            exp_dir=args.exp_dir,
        )

    # Display Leaderboard
    df_lb = load_leaderboard(args.exp_dir)
    print("\n" + "=" * 90)
    print("   [OK] CADICA YOLO Benchmark Leaderboard")
    print("=" * 90)
    if not df_lb.empty:
        cols_to_show = ["run_name", "scheme", "model", "oversample", "val_mAP50", "test_mAP50", "test_f1", "test_frame_roc_auc", "train_minutes"]
        print(df_lb[cols_to_show].to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
