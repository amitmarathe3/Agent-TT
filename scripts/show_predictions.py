#!/usr/bin/env python3
"""CLI runner for qualitative prediction visualization and error analysis.

Generates 3x4 image grids:
1. 12 sample validation images with GT (green) and predictions (red).
2. 12 worst False Negatives (missed lesions).
3. 12 worst False Positives (high-confidence false alarms).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src importable from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.eval_sets import load_eval_ground_truth
from src.models.eval_runner import SCHEME_NAMES
from src.models.visualization import generate_error_grids
from src.utils.logger import get_logger

logger = get_logger("show_predictions_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for prediction visualization."""
    parser = argparse.ArgumentParser(
        description="Generate qualitative prediction grids: samples, worst FNs, worst FPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="Path to training run directory (e.g. outputs/runs/yolo_7class).",
    )
    parser.add_argument(
        "--preds", type=str, default=None,
        help="Path to predictions JSON (defaults to <run_dir>/preds_val.json).",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class",
        choices=["7class", "binary", "severity3"],
        help="Class scheme for annotation labels.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Confidence threshold. If not provided, loads from <run_dir>/threshold.json (or defaults to 0.5).",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for matching predictions to GT.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for generated figures. Defaults to outputs/eval/<run_name>_<split>/visualizations.",
    )
    parser.add_argument(
        "--yolo-dir", type=str, default="outputs/yolo_dataset",
        help="Path to exported YOLO dataset directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI execution entrypoint."""
    args = parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # 1. Resolve predictions path
    preds_p = Path(args.preds) if args.preds else run_dir / f"preds_{args.split}.json"
    if not preds_p.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_p}")

    with open(preds_p, encoding="utf-8") as f:
        predictions = json.load(f)

    # 2. Resolve confidence threshold
    conf_thr = args.conf
    if conf_thr is None:
        thr_file = run_dir / "threshold.json"
        if thr_file.exists():
            with open(thr_file, encoding="utf-8") as f:
                thr_data = json.load(f)
            conf_thr = float(thr_data.get("best_conf_threshold", 0.5))
            logger.info("Loaded frozen confidence threshold %.4f from %s", conf_thr, thr_file)
        else:
            conf_thr = 0.5
            logger.info("Using default confidence threshold %.4f", conf_thr)

    # 3. Load ground truth
    ground_truths, _ = load_eval_ground_truth(
        split=args.split,
        scheme=args.scheme,
        yolo_dir=args.yolo_dir,
    )

    # 4. Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("outputs/eval") / f"{run_dir.name}_{args.split}" / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.yolo_dir) / "images" / args.split
    class_names = SCHEME_NAMES.get(args.scheme, {i: f"cls_{i}" for i in range(7)})

    logger.info("Generating prediction grids in %s...", out_dir)
    grid_paths = generate_error_grids(
        predictions=predictions,
        ground_truths=ground_truths,
        class_names=class_names,
        image_dir=img_dir,
        output_dir=out_dir,
        conf_threshold=conf_thr,
        iou_threshold=args.iou,
    )

    print("\n" + "=" * 75)
    print(f"   [OK] Prediction Visualizations Generated ({args.split} split)")
    print("=" * 75)
    print(f"   Output Directory:         {out_dir}")
    print(f"   Validation Sample Grid:   {grid_paths['val_grid']}")
    print(f"   Worst False Negatives:    {grid_paths['false_negatives_grid']}")
    print(f"   Worst False Positives:    {grid_paths['false_positives_grid']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
