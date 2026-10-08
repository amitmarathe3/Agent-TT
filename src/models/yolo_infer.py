"""YOLO inference engine for canonical evaluation sets.

Runs trained weights on canonical validation and test splits with
low-threshold detection (conf=0.001, iou=0.6, max_det=50) for fine-grained
confidence-threshold calibration and PR-curve generation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from src.data.eval_sets import get_canonical_eval_frame_ids
from src.utils.logger import get_logger

logger = get_logger("yolo_infer")


def predict_canonical_split(
    weights_path: str | Path,
    split: str = "val",
    output_json_path: str | Path | None = None,
    yolo_dir: str | Path = "outputs/yolo_dataset",
    conf: float = 0.001,
    iou: float = 0.6,
    max_det: int = 50,
    imgsz: int = 640,
    device: str | None = None,
    batch_size: int = 16,
) -> dict[str, list[list[float]]]:
    """Run model inference on canonical evaluation split and format predictions.

    Args:
        weights_path: Path to model weights (e.g., best.pt).
        split: 'val' or 'test'.
        output_json_path: Optional path to save preds_<split>.json.
        yolo_dir: Path to exported YOLO dataset directory.
        conf: Prediction confidence threshold (default: 0.001).
        iou: NMS IoU threshold (default: 0.6).
        max_det: Maximum detections per image (default: 50).
        imgsz: Inference image size (default: 640).
        device: Device string (e.g. '0', 'cpu', None for auto).
        batch_size: Batch size for batched inference.

    Returns:
        Dict mapping frame_id -> list of [x, y, w, h, score, class_id] in absolute pixels.
    """
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    yolo_path = Path(yolo_dir)
    images_dir = yolo_path / "images" / split
    if not images_dir.exists():
        raise FileNotFoundError(f"Canonical image directory not found: {images_dir}")

    # 1. Retrieve canonical frame IDs
    frame_ids = get_canonical_eval_frame_ids(split, yolo_dir=yolo_path)
    logger.info(
        "Loaded %d canonical eval frames for split '%s' from %s",
        len(frame_ids),
        split,
        images_dir,
    )

    # 2. Collect image paths
    image_paths: list[Path] = []
    missing_frames: list[str] = []
    for fid in frame_ids:
        # Check png first, then jpg
        p = images_dir / f"{fid}.png"
        if not p.exists():
            p = images_dir / f"{fid}.jpg"
        if p.exists():
            image_paths.append(p)
        else:
            missing_frames.append(fid)

    if missing_frames:
        logger.warning(
            "Found %d canonical frames missing on disk in %s",
            len(missing_frames),
            images_dir,
        )

    # 3. Load YOLO model
    logger.info("Loading YOLO model from %s...", weights_path)
    model = YOLO(str(weights_path))

    # Determine device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Running prediction on %d images (conf=%.4f, iou=%.2f, max_det=%d, imgsz=%d, device=%s)...",
        len(image_paths),
        conf,
        iou,
        max_det,
        imgsz,
        device,
    )

    predictions: dict[str, list[list[float]]] = {}
    # Initialize all canonical frames with empty lists
    for fid in frame_ids:
        predictions[fid] = []

    # Run inference in batches
    path_strs = [str(p.resolve()) for p in image_paths]
    results = model.predict(
        source=path_strs,
        conf=conf,
        iou=iou,
        max_det=max_det,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        verbose=False,
    )

    total_detections = 0
    for r, img_p in zip(results, image_paths):
        fid = img_p.stem
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # Extract [x1, y1, x2, y2], scores, and classes
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()

        frame_preds: list[list[float]] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            score = float(scores[i])
            cls_id = int(classes[i])

            # Convert to [x, y, w, h] in absolute pixels
            x = float(x1)
            y = float(y1)
            w = float(max(0.0, x2 - x1))
            h = float(max(0.0, y2 - y1))

            frame_preds.append([round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(score, 6), cls_id])

        predictions[fid] = frame_preds
        total_detections += len(frame_preds)

    logger.info(
        "Inference complete: %d total detections across %d frames (average: %.2f boxes/frame)",
        total_detections,
        len(predictions),
        total_detections / max(1, len(predictions)),
    )

    # 4. Save to JSON if requested
    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)
        logger.info("Saved formatted predictions to %s", out_p)

    return predictions
