#!/usr/bin/env python3
"""CLI runner for YOLO prediction on canonical evaluation splits.

Evaluates trained model weights (e.g. best.pt) on canonical evaluation frames
with low confidence (conf=0.001, iou=0.6, max_det=50) and saves detections in
model-agnostic format [x,y,w,h,score,class] to preds_<split>.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.yolo_infer import predict_canonical_split
from src.utils.logger import get_logger

logger = get_logger("predict_yolo_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for YOLO prediction."""
    parser = argparse.ArgumentParser(
        description="Run YOLO prediction on canonical CADICA evaluation sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--weights", type=str,
        help="Path to YOLO model weights file (.pt).",
    )
    group.add_argument(
        "--run-dir", type=str,
        help="Path to training run directory containing weights/best.pt.",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test", "train"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save output JSON. Defaults to <run_dir>/preds_<split>.json.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.001,
        help="Confidence threshold for prediction generation.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.6,
        help="NMS IoU threshold.",
    )
    parser.add_argument(
        "--max-det", type=int, default=50,
        help="Maximum detections per frame.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Inference compute device (default: auto).",
    )
    parser.add_argument(
        "--yolo-dir", type=str, default="outputs/yolo_dataset",
        help="Base directory of exported YOLO dataset.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI execution entrypoint."""
    args = parse_args(argv)

    # 1. Resolve weights path and default output path
    if args.run_dir:
        run_p = Path(args.run_dir)
        weights_p = run_p / "weights" / "best.pt"
        if not weights_p.exists():
            # Fallback to last.pt or direct file
            weights_p = run_p / "best.pt"
        if not weights_p.exists():
            raise FileNotFoundError(f"Could not find best.pt in run directory: {args.run_dir}")
        default_out = run_p / f"preds_{args.split}.json"
    else:
        weights_p = Path(args.weights)
        if not weights_p.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_p}")
        default_out = weights_p.parent.parent / f"preds_{args.split}.json"

    out_p = Path(args.output) if args.output else default_out

    logger.info("Starting canonical YOLO prediction...")
    logger.info("  Weights:    %s", weights_p)
    logger.info("  Split:      %s", args.split)
    logger.info("  Output:     %s", out_p)
    logger.info("  Conf:       %.4f", args.conf)
    logger.info("  IoU:        %.2f", args.iou)
    logger.info("  Max Det:    %d", args.max_det)

    predictions = predict_canonical_split(
        weights_path=weights_p,
        split=args.split,
        output_json_path=out_p,
        yolo_dir=args.yolo_dir,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
        device=args.device,
    )

    print("\n" + "=" * 70)
    print(f"   [OK] YOLO Prediction Complete for split '{args.split}'")
    print("=" * 70)
    print(f"   Evaluated Frames:   {len(predictions)}")
    print(f"   Predictions Saved:  {out_p}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
