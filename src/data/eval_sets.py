"""Canonical evaluation datasets for CADICA coronary angiography.

Defines standard evaluation sets:
- Canonical evaluation set (default): Exactly the images present in
  outputs/yolo_dataset/images/<split>/, which includes all annotated positive
  frames for the split plus clinically vetted hard negatives sampled at
  the designated negative ratio (e.g., 722 frames for val: 578 positives + 144 negatives).
- Full frames mode (--full-frames): Evaluates all frames from all videos of the
  split patients. Emits an explicit warning because unannotated video frames
  far from keyframes may contain unlabeled lesions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("eval_sets")


def get_canonical_eval_frame_ids(
    split: str,
    yolo_dir: Path | str = "outputs/yolo_dataset",
) -> list[str]:
    """Retrieve the sorted list of canonical image frame IDs for a split.

    Args:
        split: Split name ('train', 'val', or 'test').
        yolo_dir: Base directory of the exported YOLO dataset.

    Returns:
        List of frame IDs (e.g. ['p6_v1_00010', ...]).
    """
    yolo_path = Path(yolo_dir)
    img_dir = yolo_path / "images" / split

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Canonical YOLO image directory not found: {img_dir}. "
            f"Please run 'python scripts/export_yolo.py --scheme 7class' first to build the dataset."
        )

    frame_ids = sorted(p.stem for p in img_dir.glob("*.png"))
    if not frame_ids:
        # Fallback to jpg or generic files if applicable
        frame_ids = sorted(p.stem for p in img_dir.iterdir() if p.is_file())

    return frame_ids


def load_eval_ground_truth(
    split: str,
    scheme: str = "7class",
    yolo_dir: Path | str = "outputs/yolo_dataset",
    index_dir: Path | str = "data_index",
    full_frames: bool = False,
    config: dict[str, Any] | None = None,
    fold: int | None = None,
) -> tuple[dict[str, list[list[float]]], dict[str, str]]:
    """Load evaluation ground truth for a given split.

    By default (full_frames=False), loads only canonical evaluation frames
    (annotated frames + sampled hard negatives, exactly 722 for val).
    If full_frames=True, loads all video frames for the split patients and
    warns that unannotated frames may contain unlabeled lesions.

    Args:
        split: 'val', 'test', 'train', or 'all'.
        scheme: Class scheme ('7class', 'binary', 'severity3').
        yolo_dir: Path to exported YOLO dataset directory.
        index_dir: Path to data_index directory.
        full_frames: If True, evaluate across all sequence frames.
        config: Project configuration dict (optional).
        fold: Optional cross-validation fold index (0 to 4).

    Returns:
        (ground_truths, frame_to_patient):
            ground_truths: Dict frame_id -> [[x, y, w, h, class_id], ...]
            frame_to_patient: Dict frame_id -> patient_id
    """
    from scripts.evaluate import BUILTIN_SCHEMES, _resolve_scheme

    index_path = Path(index_dir)
    frames_path = index_path / "frames_index.csv"
    ann_path = index_path / "annotations.csv"
    splits_path = index_path / "splits" / "split_v1.csv"
    folds_path = index_path / "splits" / "folds_5.csv"

    if not frames_path.exists() or not ann_path.exists():
        raise FileNotFoundError(f"Index files not found in {index_dir}. Run build_index.py first.")

    frames_df = pd.read_csv(frames_path)
    ann_df = pd.read_csv(ann_path)

    # Filter patients by split or fold
    if split != "all":
        if fold is not None:
            if not folds_path.exists():
                raise FileNotFoundError(f"Folds file not found: {folds_path}. Run make_splits.py first.")
            folds_df = pd.read_csv(folds_path)
            val_fold = (fold + 1) % 5
            if split == "test":
                split_patients = set(folds_df[folds_df["fold"] == fold]["patient_id"])
            elif split == "val":
                split_patients = set(folds_df[folds_df["fold"] == val_fold]["patient_id"])
            else:  # train
                split_patients = set(folds_df[~folds_df["fold"].isin([fold, val_fold])]["patient_id"])
            frames_df = frames_df[frames_df["patient_id"].isin(split_patients)]
            ann_df = ann_df[ann_df["patient_id"].isin(split_patients)]
        else:
            if not splits_path.exists():
                raise FileNotFoundError(f"Splits file not found: {splits_path}. Run make_splits.py first.")
            split_df = pd.read_csv(splits_path)
            split_patients = set(split_df[split_df["split"] == split]["patient_id"])
            frames_df = frames_df[frames_df["patient_id"].isin(split_patients)]
            ann_df = ann_df[ann_df["patient_id"].isin(split_patients)]

    class_mapping = _resolve_scheme(scheme, config or {})

    # Determine frame universe
    if full_frames:
        warning_msg = (
            f"[WARNING] --full-frames flag is active for split '{split}'. "
            f"Evaluating against all {len(frames_df):,} raw video frames. "
            f"Note: unannotated sequence frames from lesion videos are treated as negatives "
            f"but may contain unlabeled coronary lesions!"
        )
        logger.warning(warning_msg)
        warnings.warn(warning_msg, UserWarning, stacklevel=2)
        target_frame_ids = set(str(fid) for fid in frames_df["frame_id"])
    else:
        canonical_ids = get_canonical_eval_frame_ids(split, yolo_dir=yolo_dir)
        target_frame_ids = set(canonical_ids)
        logger.info(
            "Using canonical evaluation set for split '%s': %d frames (positives + hard negatives)",
            split, len(target_frame_ids),
        )

    # Build patient lookup and initialize empty GT for all eval frames
    frame_to_patient: dict[str, str] = {}
    ground_truths: dict[str, list[list[float]]] = {}

    for _, row in frames_df.iterrows():
        fid = str(row["frame_id"])
        if fid in target_frame_ids:
            ground_truths[fid] = []
            frame_to_patient[fid] = str(row["patient_id"])

    # Populate annotations for positive frames
    for _, row in ann_df.iterrows():
        fid = str(row["frame_id"])
        if fid not in ground_truths:
            continue

        category = str(row["category"])
        if category not in class_mapping:
            continue
        cls_id = class_mapping[category]

        w_img = float(row.get("image_width", 512))
        h_img = float(row.get("image_height", 512))
        if "x" in row and pd.notna(row["x"]):
            x = float(row["x"])
            y = float(row["y"])
            w = float(row["w"])
            h = float(row["h"])
        else:
            xc_n = float(row["x_center"])
            yc_n = float(row["y_center"])
            wn = float(row["w_norm"])
            hn = float(row["h_norm"])
            x = (xc_n - wn / 2) * w_img
            y = (yc_n - hn / 2) * h_img
            w = wn * w_img
            h = hn * h_img

        ground_truths[fid].append([x, y, w, h, cls_id])

    return ground_truths, frame_to_patient
