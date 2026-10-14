#!/usr/bin/env python3
"""Run baseline YOLO real experiments for CADICA coronary angiography.

Orchestrates real training, prediction, and evaluation for:
  Run A: binary scheme, yolo11s.pt, 100 epochs, patience 25
  Run B: severity3 scheme, yolo11s.pt, 100 epochs, patience 25
  Run C: 7class scheme + --oversample-rare, yolo11s.pt, 100 epochs, patience 25

Stops after Run C per Phase 3.2 (v0.3.2) requirements.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.experiment_runner import run_single_experiment, load_leaderboard
from src.utils.logger import get_logger


logger = get_logger("run_experiments")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_experiments."""
    parser = argparse.ArgumentParser(
        description="Run CADICA baseline experiments (Run A, B, C).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Run baseline experiment matrix.",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold cross-validation.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast verification mode (2 epochs, fraction=0.1).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing leaderboard entries.",
    )
    parser.add_argument(
        "--best-run", type=str, default="exp_7class_yolo11s_oversample",
        help="Run name for best configuration.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for single run.",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class", choices=["7class", "binary", "severity3"],
        help="Class scheme.",
    )
    parser.add_argument(
        "--oversample-rare", action="store_true",
        help="Oversample rare classes.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Disable bootstrap CI calculations.",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Filter specific runs to execute.",
    )
    return parser.parse_args(argv)


def run_experiment_and_print_summary(
    run_name: str,
    scheme: str,
    model: str = "yolo11s.pt",
    imgsz: int = 640,
    oversample: bool = False,
    epochs: int = 100,
    smoke_test: bool = False,
    force: bool = False,
    device: str | None = None,
) -> dict:
    """Execute single experiment run and print raw terminal summary details."""
    res = run_single_experiment(
        run_name=run_name,
        scheme=scheme,
        model=model,
        imgsz=imgsz,
        oversample=oversample,
        fold=None,
        epochs=epochs,
        smoke_test=smoke_test,
        force=force,
        device=device,
    )

    run_dir = Path("outputs/runs") / run_name
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "best.pt"

    # Read actual trained epochs from results.csv if available
    results_csv = run_dir / "results.csv"
    actual_epochs = epochs
    if results_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            actual_epochs = len(df)
        except Exception:
            pass

    size_mb = (best_pt.stat().st_size / (1024 * 1024)) if best_pt.exists() else 0.0

    print("\n" + "=" * 70)
    print(f"   [REAL RUN SUMMARY] Finished: {run_name}")
    print("=" * 70)
    print(f"   Scheme:               {scheme}")
    print(f"   Model:                {model}")
    print(f"   Oversample:           {oversample}")
    print(f"   Epochs Trained:       {actual_epochs} / {epochs}")
    print(f"   Train Minutes:        {res.get('train_minutes', 0.0):.2f} min")
    print(f"   Path of best.pt:      {best_pt.resolve()}")
    print(f"   Size of best.pt:      {size_mb:.2f} MB ({best_pt.stat().st_size if best_pt.exists() else 0} bytes)")
    print(f"   Val mAP@0.5:          {res.get('val_mAP50', 0.0):.4f} | Val F1: {res.get('val_f1', 0.0):.4f}")
    print(f"   Test mAP@0.5:         {res.get('test_mAP50', 0.0):.4f} | Test F1: {res.get('test_f1', 0.0):.4f}")
    print("=" * 70 + "\n")

    return res


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    runs_to_run = [
        {"run_name": "exp_binary_yolo11s", "scheme": "binary", "model": "yolo11s.pt", "oversample": False},
        {"run_name": "exp_severity3_yolo11s", "scheme": "severity3", "model": "yolo11s.pt", "oversample": False},
        {"run_name": "exp_7class_yolo11s_oversample", "scheme": "7class", "model": "yolo11s.pt", "oversample": True},
    ]

    if args.runs:
        runs_to_run = [r for r in runs_to_run if r["run_name"] in args.runs]

    for cfg in runs_to_run:
        print(f"\n>>> Launching Real Baseline Run: {cfg['run_name']} (scheme={cfg['scheme']})")
        run_experiment_and_print_summary(
            run_name=cfg["run_name"],
            scheme=cfg["scheme"],
            model=cfg["model"],
            imgsz=640,
            oversample=cfg["oversample"],
            epochs=100 if not args.smoke_test else 2,
            smoke_test=args.smoke_test,
            force=args.overwrite,
            device=args.device,
        )

    print("\n[OK] Baseline Runs A, B, and C Execution Completed.")


if __name__ == "__main__":
    main()
