#!/usr/bin/env python3
"""CLI runner for Ultralytics YOLO training on CADICA coronary angiography dataset.

Windows-safe with __main__ guard and auto-batch on GPU.
Hyperparameters tailored for medical angiography (no flips, mild scale/rotation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_trainer import train_yolo
from src.utils.logger import get_logger

logger = get_logger("train_yolo_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO model on CADICA angiography dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scheme", type=str, default="7class",
        choices=["7class", "binary", "severity3"],
        help="Class scheme for training.",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11s.pt",
        help="Model architecture or pretrained weights (e.g. yolo11s.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size (pixels) for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Optional 5-fold cross-validation fold index (0 to 4).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for run output directory in outputs/runs/.",
    )
    parser.add_argument(
        "--oversample-rare", action="store_true",
        help="Repeat train frames containing p99/p100/p90_98 up to 3x (train only).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast verification mode: train for 2 epochs on fraction=0.1 of data.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device ('0', 'cpu', etc.). Auto-detects GPU by default.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker count (Windows-safe default: 2).",
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (default: auto batch -1 on GPU, 16 on CPU).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="outputs/yolo_dataset",
        help="Directory containing exported YOLO data.yaml.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/runs",
        help="Base directory for training run outputs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI execution entrypoint."""
    args = parse_args(argv)

    logger.info("Initializing YOLO training pipeline...")
    logger.info("  Scheme:           %s", args.scheme)
    logger.info("  Model:            %s", args.model)
    logger.info("  Image Size:       %d", args.imgsz)
    logger.info("  Epochs:           %d%s", args.epochs, " (2 in smoke-test)" if args.smoke_test else "")
    logger.info("  Fold:             %s", args.fold)
    logger.info("  Seed:             %d", args.seed)
    logger.info("  Oversample Rare:  %s", args.oversample_rare)
    logger.info("  Smoke Test:       %s", args.smoke_test)
    logger.info("  Workers:          %d", args.workers)

    summary = train_yolo(
        scheme=args.scheme,
        model=args.model,
        imgsz=args.imgsz,
        epochs=args.epochs,
        fold=args.fold,
        seed=args.seed,
        run_name=args.run_name,
        oversample_rare=args.oversample_rare,
        smoke_test=args.smoke_test,
        device=args.device,
        workers=args.workers,
        batch=args.batch,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 70)
    print(f"   [OK] YOLO Training Complete: {summary['run_name']}")
    print("=" * 70)
    print(f"   Run Directory:   {args.output_dir}/{summary['run_name']}")
    print(f"   Device Used:     {summary['device']}")
    print(f"   Best Weights:    {summary['best_weights']}")
    print(f"   Elapsed Time:    {summary['elapsed_seconds']:.1f}s")
    print(f"   Config Saved:    {args.output_dir}/{summary['run_name']}/run_config.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
