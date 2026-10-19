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
    preprocess: str = "512",
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
        preprocess: '512' (direct feed) or '544_padded' (letterbox 512 into 544 with 114 gray padding).

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
        "Running prediction on %d images (conf=%.4f, iou=%.2f, max_det=%d, imgsz=%d, preprocess=%s, device=%s)...",
        len(image_paths),
        conf,
        iou,
        max_det,
        imgsz,
        preprocess,
        device,
    )

    predictions: dict[str, list[list[float]]] = {}
    for fid in frame_ids:
        predictions[fid] = []

    total_detections = 0
    if preprocess == "544_padded":
        import cv2
        from ultralytics.data.augment import LetterBox

        lb = LetterBox(new_shape=(544, 544), auto=False, scaleup=False, center=True)
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i : i + batch_size]
            chunk_imgs = [lb(image=cv2.imread(str(p))) for p in chunk_paths]
            chunk_results = model.predict(
                source=chunk_imgs,
                conf=conf,
                iou=iou,
                max_det=max_det,
                imgsz=544,
                device=device,
                verbose=False,
            )
            for r, img_p in zip(chunk_results, chunk_paths):
                fid = img_p.stem
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                frame_preds: list[list[float]] = []
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    score = float(scores[j])
                    cls_id = int(classes[j])

                    # Map coordinates back from 544 (16px centered padding) to 512
                    x1_512 = float(np.clip(x1 - 16.0, 0.0, 512.0))
                    y1_512 = float(np.clip(y1 - 16.0, 0.0, 512.0))
                    x2_512 = float(np.clip(x2 - 16.0, 0.0, 512.0))
                    y2_512 = float(np.clip(y2 - 16.0, 0.0, 512.0))

                    w = float(max(0.0, x2_512 - x1_512))
                    h = float(max(0.0, y2_512 - y1_512))

                    frame_preds.append([round(x1_512, 2), round(y1_512, 2), round(w, 2), round(h, 2), round(score, 6), cls_id])

                predictions[fid] = frame_preds
                total_detections += len(frame_preds)
    else:
        # Standard unpadded prediction (chunked to avoid GPU OOM)
        effective_imgsz = 512 if imgsz == 640 else imgsz
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i : i + batch_size]
            path_strs = [str(p.resolve()) for p in chunk_paths]
            chunk_results = model.predict(
                source=path_strs,
                conf=conf,
                iou=iou,
                max_det=max_det,
                imgsz=effective_imgsz,
                device=device,
                verbose=False,
            )
            for r, img_p in zip(chunk_results, chunk_paths):
                fid = img_p.stem
                boxes = r.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                frame_preds: list[list[float]] = []
                for j in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[j]
                    score = float(scores[j])
                    cls_id = int(classes[j])

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


def predict_single_image(
    model: YOLO,
    image: np.ndarray,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 100,
    device: str | None = None,
    preprocess: str = "544_padded",
) -> list[list[float]]:
    """Predict on a single RGB/BGR image array.

    Args:
        model: Loaded YOLO model.
        image: Numpy array of shape (H, W, 3).
        conf: Confidence threshold (default: 0.001).
        iou: NMS IoU threshold (default: 0.7).
        max_det: Max detections (default: 100).
        device: '0', 'cpu', or None.
        preprocess: '544_padded' or '512'.

    Returns:
        List of [x, y, w, h, score, class_id] in pixels of the input image.
    """
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    orig_h, orig_w = image.shape[:2]
    if orig_h == 0 or orig_w == 0:
        return []

    # Ensure 512x512 base resolution
    if (orig_w, orig_h) != (512, 512):
        import cv2

        im_512 = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
    else:
        im_512 = image

    scale_x = orig_w / 512.0
    scale_y = orig_h / 512.0

    if preprocess == "544_padded":
        from ultralytics.data.augment import LetterBox

        lb = LetterBox(new_shape=(544, 544), auto=False, scaleup=False, center=True)
        img_input = lb(image=im_512)
        results = model.predict(
            source=img_input,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=544,
            device=device,
            verbose=False,
        )
        r = results[0] if results else None
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        dets: list[list[float]] = []
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            x1_512 = float(np.clip(x1 - 16.0, 0.0, 512.0))
            y1_512 = float(np.clip(y1 - 16.0, 0.0, 512.0))
            x2_512 = float(np.clip(x2 - 16.0, 0.0, 512.0))
            y2_512 = float(np.clip(y2 - 16.0, 0.0, 512.0))

            x = x1_512 * scale_x
            y = y1_512 * scale_y
            w = max(0.0, x2_512 - x1_512) * scale_x
            h = max(0.0, y2_512 - y1_512) * scale_y
            dets.append([round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(float(scores[j]), 6), int(classes[j])])
        return dets
    else:
        results = model.predict(
            source=im_512,
            conf=conf,
            iou=iou,
            max_det=max_det,
            imgsz=512,
            device=device,
            verbose=False,
        )
        r = results[0] if results else None
        if r is None or r.boxes is None or len(r.boxes) == 0:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        dets = []
        for j in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[j]
            x = float(x1) * scale_x
            y = float(y1) * scale_y
            w = float(max(0.0, x2 - x1)) * scale_x
            h = float(max(0.0, y2 - y1)) * scale_y
            dets.append([round(x, 2), round(y, 2), round(w, 2), round(h, 2), round(float(scores[j]), 6), int(classes[j])])
        return dets

