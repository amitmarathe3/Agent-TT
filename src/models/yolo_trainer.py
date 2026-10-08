"""YOLO training pipeline logic for CADICA coronary angiography.

Handles model instantiation (with fallback), rare-class oversampling,
Windows-safe execution, medical augmentation hyperparameters, run tracking,
and curve plotting.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.utils.logger import get_logger

logger = get_logger("yolo_trainer")


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash, or 'unknown' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_safe_device(requested_device: str | None = None) -> tuple[str, str]:
    """Determine safe compute device and return device identifier and description.

    Args:
        requested_device: Optional user override (e.g. '0', 'cpu', 'cuda').

    Returns:
        (device_id, device_name)
    """
    if requested_device is not None:
        dev = str(requested_device).strip()
        if dev.lower() == "cpu":
            return "cpu", "CPU"
        name = f"Custom ({dev})"
        if torch.cuda.is_available():
            try:
                idx = int(dev) if dev.isdigit() else 0
                name = torch.cuda.get_device_name(idx)
            except Exception:
                pass
        return dev, name

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA detected: %s (Device 0)", gpu_name)
        return "0", f"CUDA:0 ({gpu_name})"

    logger.info("CUDA not available. Using CPU.")
    return "cpu", "CPU"


def load_yolo_model(model_name: str = "yolo11s.pt"):
    """Load Ultralytics YOLO model with graceful fallback to yolov8s.pt.

    Args:
        model_name: Model weights identifier (default: yolo11s.pt).

    Returns:
        (model_instance, loaded_model_name)
    """
    from ultralytics import YOLO

    try:
        logger.info("Attempting to load model '%s'...", model_name)
        model = YOLO(model_name)
        logger.info("Successfully loaded model '%s'", model_name)
        return model, model_name
    except Exception as e:
        logger.warning(
            "Failed to load '%s' (%s). Falling back to 'yolov8s.pt'...",
            model_name,
            e,
        )
        fallback_name = "yolov8s.pt"
        model = YOLO(fallback_name)
        logger.info("Successfully loaded fallback model '%s'", fallback_name)
        return model, fallback_name


def prepare_oversampled_dataset(
    data_yaml_path: Path | str,
    output_dir: Path | str,
    rare_categories: tuple[str, ...] = ("p99", "p100", "p90_98"),
    multiplier: int = 3,
    annotations_csv: Path | str = "data_index/annotations.csv",
) -> Path:
    """Create an oversampled training manifest repeating rare-class images up to multiplier x.

    Ultralytics natively supports text files with image paths for the 'train' key.
    This repeats lines for images containing rare stenosis categories (p99, p100, p90_98)
    without duplicating image files on disk.

    Args:
        data_yaml_path: Original data.yaml path.
        output_dir: Directory to save oversampled manifest and yaml.
        rare_categories: Categories to oversample.
        multiplier: Total number of times rare images appear (default: 3).
        annotations_csv: Path to annotations.csv to identify images with rare categories.

    Returns:
        Path to new oversampled data.yaml.
    """
    data_yaml_path = Path(data_yaml_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_yaml_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_dataset_path = Path(data_cfg.get("path", data_yaml_path.parent))
    train_entry = data_cfg.get("train", "images/train")

    if Path(train_entry).is_absolute():
        train_img_dir = Path(train_entry)
    else:
        train_img_dir = base_dataset_path / train_entry

    if not train_img_dir.exists():
        raise FileNotFoundError(f"Training image directory not found: {train_img_dir}")

    # Identify images containing rare categories
    rare_frame_ids: set[str] = set()
    ann_path = Path(annotations_csv)
    if ann_path.exists():
        df_ann = pd.read_csv(ann_path)
        rare_df = df_ann[df_ann["category"].astype(str).isin(rare_categories)]
        rare_frame_ids = set(rare_df["frame_id"].astype(str))
        logger.info(
            "Identified %d frames containing rare categories %s from %s",
            len(rare_frame_ids),
            rare_categories,
            ann_path,
        )
    else:
        # Fallback: scan labels/train txt files
        labels_dir = base_dataset_path / "labels" / "train"
        if labels_dir.exists():
            for lbl_file in labels_dir.glob("*.txt"):
                rare_frame_ids.add(lbl_file.stem)

    # Gather all train images
    train_images = sorted(
        [p for p in train_img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    oversampled_lines: list[str] = []
    rare_count = 0
    standard_count = 0

    for img_path in train_images:
        resolved_str = str(img_path.resolve()).replace("\\", "/")
        stem = img_path.stem
        if stem in rare_frame_ids:
            # Repeat up to multiplier times
            for _ in range(max(1, multiplier)):
                oversampled_lines.append(resolved_str)
            rare_count += 1
        else:
            oversampled_lines.append(resolved_str)
            standard_count += 1

    oversampled_txt = output_dir / "train_oversampled.txt"
    oversampled_txt.write_text("\n".join(oversampled_lines) + "\n", encoding="utf-8")
    logger.info(
        "Oversampled manifest written: %d rare frames repeated %dx (%d entries), "
        "%d standard frames (total: %d images)",
        rare_count,
        multiplier,
        rare_count * multiplier,
        standard_count,
        len(oversampled_lines),
    )

    # Create new data.yaml pointing to oversampled manifest
    new_yaml_cfg = dict(data_cfg)
    new_yaml_cfg["train"] = str(oversampled_txt.resolve()).replace("\\", "/")
    oversampled_yaml = output_dir / "data_oversampled.yaml"
    with open(oversampled_yaml, "w", encoding="utf-8") as f:
        yaml.dump(new_yaml_cfg, f, sort_keys=False)

    return oversampled_yaml


def plot_training_curves(results_csv_path: Path | str, output_png_path: Path | str) -> None:
    """Plot high-resolution training loss, precision, recall, and mAP curves from results.csv.

    Args:
        results_csv_path: Path to Ultralytics results.csv.
        output_png_path: Destination path for curves.png.
    """
    results_csv_path = Path(results_csv_path)
    output_png_path = Path(output_png_path)

    if not results_csv_path.exists():
        logger.warning("results.csv not found at %s. Skipping custom curve plot.", results_csv_path)
        return

    try:
        df = pd.read_csv(results_csv_path)
        # Strip whitespace from column names
        df.columns = [c.strip() for c in df.columns]

        epochs = df["epoch"] if "epoch" in df else np.arange(len(df))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss curves
        ax = axes[0, 0]
        for col in ["train/box_loss", "train/cls_loss", "train/dfl_loss"]:
            if col in df:
                ax.plot(epochs, df[col], label=col.replace("train/", ""))
        ax.set_title("Training Losses", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Validation Losses
        ax = axes[0, 1]
        for col in ["val/box_loss", "val/cls_loss", "val/dfl_loss"]:
            if col in df:
                ax.plot(epochs, df[col], label=col.replace("val/", ""))
        ax.set_title("Validation Losses", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Precision & Recall
        ax = axes[1, 0]
        if "metrics/precision(B)" in df:
            ax.plot(epochs, df["metrics/precision(B)"], label="Precision", color="dodgerblue")
        if "metrics/recall(B)" in df:
            ax.plot(epochs, df["metrics/recall(B)"], label="Recall", color="forestgreen")
        ax.set_title("Metrics: Precision & Recall", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. mAP Curves
        ax = axes[1, 1]
        if "metrics/mAP50(B)" in df:
            ax.plot(epochs, df["metrics/mAP50(B)"], label="mAP@0.5", color="crimson", lw=2)
        if "metrics/mAP50-95(B)" in df:
            ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="purple", lw=2)
        ax.set_title("Mean Average Precision (mAP)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.set_ylim(-0.02, 1.02)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle("YOLO Training Progress & Convergence", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        output_png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_png_path, dpi=300)
        plt.close(fig)
        logger.info("Saved custom training curves to %s", output_png_path)
    except Exception as e:
        logger.warning("Failed to plot training curves from %s: %s", results_csv_path, e)


def train_yolo(
    scheme: str = "7class",
    model: str = "yolo11s.pt",
    imgsz: int = 640,
    epochs: int = 100,
    fold: int | None = None,
    seed: int = 42,
    run_name: str | None = None,
    oversample_rare: bool = False,
    smoke_test: bool = False,
    device: str | None = None,
    workers: int = 2,
    batch: int | None = None,
    data_dir: str | Path = "outputs/yolo_dataset",
    output_dir: str | Path = "outputs/runs",
) -> dict[str, Any]:
    """Execute YOLO training pipeline with medical angiography configuration.

    Args:
        scheme: Class scheme ('7class', 'binary', 'severity3').
        model: Model weights string.
        imgsz: Image size for training (default: 640).
        epochs: Number of epochs (default: 100, overridden to 2 in smoke test).
        fold: Optional cross-validation fold index.
        seed: Random seed for reproducibility.
        run_name: Custom name for the run directory.
        oversample_rare: Whether to oversample p99/p100/p90_98 images 3x.
        smoke_test: Fast 2-epoch, fraction=0.1 verification mode.
        device: Device override (default: auto-detected).
        workers: DataLoader workers (default 2 for Windows safety).
        batch: Batch size override (default: -1 auto-batch on GPU, 16 on CPU).
        data_dir: Base directory containing data.yaml.
        output_dir: Base directory for run outputs.

    Returns:
        Summary dict containing run paths, config, and best weights.
    """
    start_time = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 1. Determine run name
    if not run_name:
        tag = f"yolo_{scheme}"
        if fold is not None:
            tag += f"_fold{fold}"
        if smoke_test:
            tag += "_smoke"
        run_name = tag

    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 2. Resolve safe device
    device_id, device_name = get_safe_device(device)
    logger.info("Run '%s' initialized on device: %s (id: %s)", run_name, device_name, device_id)

    # 3. Batch size selection
    if batch is None:
        if device_id != "cpu":
            # Auto-batch on GPU
            train_batch = -1
        else:
            train_batch = 16
    else:
        train_batch = batch

    # 4. Resolve data configuration
    data_yaml_path = data_dir / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found in {data_dir}. Run export_yolo.py first."
        )

    active_data_yaml = data_yaml_path
    if oversample_rare:
        logger.info("Applying rare-class oversampling (p99/p100/p90_98 x3)...")
        active_data_yaml = prepare_oversampled_dataset(
            data_yaml_path=data_yaml_path,
            output_dir=run_dir,
            rare_categories=("p99", "p100", "p90_98"),
            multiplier=3,
        )

    # 5. Handle smoke-test
    actual_epochs = 2 if smoke_test else epochs
    actual_fraction = 0.1 if smoke_test else 1.0

    if smoke_test:
        logger.info("[SMOKE TEST] Training configured for 2 epochs, fraction=0.1")

    # 6. Load Model
    yolo_model, resolved_model_name = load_yolo_model(model)

    # 7. Execute Ultralytics training with medical augmentation settings
    logger.info("Starting Ultralytics training for %d epochs...", actual_epochs)
    train_results = yolo_model.train(
        data=str(active_data_yaml.resolve()),
        imgsz=imgsz,
        epochs=actual_epochs,
        batch=train_batch,
        device=device_id,
        workers=workers,
        seed=seed,
        project=str(output_dir.resolve()),
        name=run_name,
        exist_ok=True,
        # Medical angiography augmentation settings
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.2,
        flipud=0.0,
        fliplr=0.0,
        degrees=10.0,
        scale=0.1,
        translate=0.05,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        patience=25,
        cos_lr=True,
        fraction=actual_fraction,
        verbose=True,
    )

    elapsed_sec = time.time() - start_time
    logger.info("Training completed in %.2f seconds.", elapsed_sec)

    # 8. Check and save curves.png
    curves_dest = run_dir / "curves.png"
    results_csv = run_dir / "results.csv"
    ultralytics_results_png = run_dir / "results.png"

    if results_csv.exists():
        plot_training_curves(results_csv, curves_dest)
    elif ultralytics_results_png.exists():
        shutil.copy2(ultralytics_results_png, curves_dest)
        logger.info("Copied Ultralytics results.png to %s", curves_dest)

    # 9. Save run_config.json
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    run_config = {
        "run_name": run_name,
        "scheme": scheme,
        "requested_model": model,
        "resolved_model": resolved_model_name,
        "imgsz": imgsz,
        "epochs": actual_epochs,
        "batch": train_batch,
        "fold": fold,
        "seed": seed,
        "oversample_rare": oversample_rare,
        "smoke_test": smoke_test,
        "device": device_name,
        "device_id": device_id,
        "workers": workers,
        "git_hash": get_git_commit_hash(),
        "versions": {
            "torch": torch.__version__,
            "ultralytics": sys.modules.get("ultralytics", None).__version__
            if "ultralytics" in sys.modules
            else "unknown",
            "python": sys.version,
        },
        "elapsed_seconds": round(elapsed_sec, 2),
        "best_weights": str(best_pt.resolve()) if best_pt.exists() else None,
        "last_weights": str(last_pt.resolve()) if last_pt.exists() else None,
    }

    config_dest = run_dir / "run_config.json"
    with open(config_dest, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    logger.info("Saved run configuration to %s", config_dest)

    return run_config
