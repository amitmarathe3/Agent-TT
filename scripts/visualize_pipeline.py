"""Visualization and dataset normalization statistics computation script for CADICA pipeline.

Generates:
1. reports/figures/augmentation_check.png: A 3x4 grid of 12 augmented training frames with bounding boxes.
2. data_index/norm_stats.json: Dataset pixel mean and standard deviation across training images.

Usage:
    python scripts/visualize_pipeline.py
    python scripts/visualize_pipeline.py --num-samples 12 --stat-samples 300
"""

import argparse
import json
from pathlib import Path
import random
import sys
import zipfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augment import get_train_transforms
from src.data.dataset import CADICADataset
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


def compute_normalization_stats(
    dataset: CADICADataset,
    num_samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute per-channel mean and standard deviation on sample of training images.

    Args:
        dataset: CADICADataset without data augmentation transforms.
        num_samples: Number of images to sample for calculation.
        seed: Random seed.

    Returns:
        Dictionary with mean and std values.
    """
    rng = random.Random(seed)
    total_imgs = len(dataset)
    sample_indices = rng.sample(range(total_imgs), min(num_samples, total_imgs))

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for idx in sample_indices:
        item = dataset[idx]
        img = item["image"].numpy()  # shape (3, H, W) normalized to [0, 1]
        _, h, w = img.shape
        num_px = h * w
        total_pixels += num_px

        channel_sum += img.sum(axis=(1, 2))
        channel_sq_sum += (img ** 2).sum(axis=(1, 2))

    mean = channel_sum / total_pixels
    variance = (channel_sq_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-7))

    return {
        "mean": [float(round(m, 4)) for m in mean],
        "std": [float(round(s, 4)) for s in std],
        "sample_size": len(sample_indices),
        "total_pixels_evaluated": int(total_pixels),
        "image_size": [512, 512],
    }


def draw_yolo_boxes(
    image: np.ndarray,
    boxes: list[list[float]] | np.ndarray,
    labels: list[int] | np.ndarray,
    class_names: list[str],
) -> np.ndarray:
    """Draw bounding boxes and class labels onto an RGB image."""
    img = image.copy()
    h, w, _ = img.shape

    colors = [
        (255, 69, 0),    # Red-Orange
        (30, 144, 255),  # Dodger Blue
        (50, 205, 50),   # Lime Green
        (255, 215, 0),   # Gold
        (148, 0, 211),   # Dark Violet
        (255, 20, 147),  # Deep Pink
        (0, 255, 255),   # Cyan
    ]

    for b, lbl in zip(boxes, labels):
        xc, yc, bw, bh = b[0], b[1], b[2], b[3]
        x1 = int(round((xc - bw / 2.0) * w))
        y1 = int(round((yc - bh / 2.0) * h))
        x2 = int(round((xc + bw / 2.0) * w))
        y2 = int(round((yc + bh / 2.0) * h))

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))

        c_idx = int(lbl) % len(colors)
        color = colors[c_idx]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label badge
        label_str = class_names[int(lbl)] if int(lbl) < len(class_names) else f"class_{lbl}"
        badge_txt = f"{label_str}"
        cv2.putText(
            img,
            badge_txt,
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


def main() -> None:
    """CLI runner to generate visualization grid and calculate normalization stats."""
    parser = argparse.ArgumentParser(description="Agent-TT: Augmentation pipeline visualization & statistics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--stat-samples", type=int, default=300, help="Number of frames for mean/std calculation")
    parser.add_argument("--output-fig", type=str, default="reports/figures/augmentation_check.png")
    parser.add_argument("--output-json", type=str, default="data_index/norm_stats.json")
    args = parser.parse_args()

    set_seed(42)
    logger = get_logger("VisualizePipeline")

    config = load_config(args.config)
    index_dir = Path(config.get("index_dir", "data_index"))
    dataset_root = Path(config.get("dataset_root", "CADICA.zip"))

    frames_df = pd.read_csv(index_dir / "frames_index.csv")
    ann_df = pd.read_csv(index_dir / "annotations.csv")
    split_df = pd.read_csv(index_dir / "splits" / "split_v1.csv")

    # Select annotated training frames
    merged_train = frames_df.merge(split_df, on="patient_id", how="inner")
    train_pos = merged_train[(merged_train["split"] == "train") & (merged_train["has_groundtruth"])].copy()

    # 1. Compute Dataset Normalization Stats (on unaugmented training frames)
    logger.info("Computing dataset normalization statistics on %d sample frames...", args.stat_samples)
    base_dataset = CADICADataset(
        frames_df=train_pos,
        annotations_df=ann_df,
        dataset_root=dataset_root,
        transforms=None,
        normalize=True,
    )
    norm_stats = compute_normalization_stats(base_dataset, num_samples=args.stat_samples)

    out_json_path = Path(args.output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info("Saved normalization stats to %s: mean=%s, std=%s", out_json_path, norm_stats["mean"], norm_stats["std"])

    # 2. Generate 12 Augmented Samples Grid
    logger.info("Generating 3x4 augmentation verification grid...")
    aug_pipeline = get_train_transforms(image_size=512, bbox_format="yolo", horizontal_flip=True)

    aug_dataset = CADICADataset(
        frames_df=train_pos,
        annotations_df=ann_df,
        dataset_root=dataset_root,
        transforms=aug_pipeline,
        normalize=False,  # Keep uint8 [0, 255] for visual plotting
    )

    class_names = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

    # Select 12 frames with annotations
    sample_indices = random.Random(42).sample(range(len(aug_dataset)), 12)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sample_indices):
        item = aug_dataset[idx]
        img_np = item["image"].permute(1, 2, 0).numpy().astype(np.uint8)  # (H, W, 3)
        boxes = item["boxes"].numpy()
        labels = item["labels"].numpy()

        rendered = draw_yolo_boxes(img_np, boxes, labels, class_names)

        axes[i].imshow(rendered)
        axes[i].set_title(f"{item['frame_id']} ({len(boxes)} boxes)", fontsize=10, fontweight="bold")
        axes[i].axis("off")

    plt.suptitle("Agent-TT: CADICA Medical Image Augmentation & Bounding Box Check", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_fig_path = Path(args.output_fig)
    out_fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
    plt.close()

    logger.info("Saved augmentation check figure to %s", out_fig_path)
    print("============================================================")
    print("  Pipeline Visualization & Normalization Stats Complete!    ")
    print("============================================================")
    print(f"  Normalization stats: {out_json_path}")
    print(f"    - Mean: {norm_stats['mean']}")
    print(f"    - Std:  {norm_stats['std']}")
    print(f"  Augmentation grid:   {out_fig_path}")


if __name__ == "__main__":
    main()
