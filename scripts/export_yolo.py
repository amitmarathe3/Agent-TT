"""CLI script to export CADICA dataset in Ultralytics YOLO format.

Outputs standard images/ and labels/ subdirectories for train/val/test splits along with data.yaml.
Supports class aggregation schemes, hard negative frame sampling, and K-fold CV export.

Usage:
    python scripts/export_yolo.py
    python scripts/export_yolo.py --scheme binary --neg-ratio 0.20
    python scripts/export_yolo.py --fold 0 --output-dir outputs/yolo_fold0
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.yolo_export import YOLOExporter


def main() -> None:
    """CLI entry point for exporting YOLO dataset."""
    parser = argparse.ArgumentParser(
        description="Agent-TT: Export CADICA dataset into standard YOLO format."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/yolo_dataset",
        help="Output directory for YOLO dataset (default: outputs/yolo_dataset)",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["7class", "binary", "severity3"],
        default="7class",
        help="Lesion classification scheme: 7class, binary, or severity3 (default: 7class)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=float,
        default=0.25,
        help="Proportion of negative (lesion-free) frames to sample relative to positives (default: 0.25)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional cross-validation fold index (0 to 4) to export (default: None, exports split_v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for negative sampling (default: 42)",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Force copying images instead of creating symlinks (default: False)",
    )

    args = parser.parse_args()

    exporter = YOLOExporter(
        config_path=args.config,
        output_dir=args.output_dir,
        scheme=args.scheme,
        neg_ratio=args.neg_ratio,
        fold=args.fold,
        seed=args.seed,
        use_symlinks=not args.copy_images,
    )

    print("============================================================")
    print("  Agent-TT — Exporting YOLO Format Dataset (Phase 2.2)      ")
    print("============================================================")
    print(f"  Scheme:     {args.scheme}")
    print(f"  Neg-Ratio:  {args.neg_ratio}")
    print(f"  Fold:       {args.fold if args.fold is not None else 'Default (split_v1)'}")
    print(f"  Output Dir: {args.output_dir}")
    print("------------------------------------------------------------")

    stats = exporter.export()

    print("\n------------------------------------------------------------")
    print("  YOLO Export Completed Successfully!")
    print("------------------------------------------------------------")
    for s_name in ["train", "val", "test"]:
        if s_name in stats:
            s = stats[s_name]
            print(
                f"  [{s_name.upper():5s}] Positives: {s['positives']:4d} | "
                f"Negatives: {s['negatives']:4d} | Total Images: {s['total_images']:4d} | "
                f"Boxes: {s['boxes']:4d}"
            )
    print(f"\n  Configuration: {stats.get('data_yaml')}")
    print(f"  Classes ({len(stats.get('class_names', []))}): {stats.get('class_names')}")


if __name__ == "__main__":
    main()
