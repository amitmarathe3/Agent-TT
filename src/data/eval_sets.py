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

# Hard invariants for the canonical evaluation sets (patient-wise 6/6 split of
# the 12 eval patients). These are the ONLY frames the Phase 3 evaluation and
# bootstrap are allowed to see: the bootstrap universe must never be built from
# frames_index.csv (31,499 frames / 42 patients) because that admits 36 patients
# with no eval frames and inflates frame AUC from ~0.65 to ~0.99.
EXPECTED_EVAL_UNIVERSE: dict[str, tuple[int, int]] = {
    "val": (722, 6),
    "test": (852, 6),
}


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
    """Load evaluation ground truth directly from YOLO dataset label files.

    Reads bounding boxes from outputs/yolo_dataset/labels/<split>/*.txt,
    converts normalized YOLO coordinates [cls_id, xc, yc, w, h] to absolute
    pixel coordinates scaled to 512x512, applies class scheme mapping, and asserts
    that loaded n_gt equals the total label-file box count.

    Args:
        split: 'val', 'test', 'train', or 'all'.
        scheme: Class scheme ('7class', 'binary', 'severity3').
        yolo_dir: Path to exported YOLO dataset directory.
        index_dir: Path to data_index directory.
        full_frames: Reserved for compatibility.
        config: Project configuration dict (optional).
        fold: Optional cross-validation fold index (0 to 4).

    Returns:
        (ground_truths, frame_to_patient):
            ground_truths: Dict frame_id -> [[x, y, w, h, class_id], ...]
            frame_to_patient: Dict frame_id -> patient_id
    """
    yolo_path = Path(yolo_dir)
    lbl_dir = yolo_path / "labels" / split
    if not lbl_dir.exists():
        lbl_dir = yolo_path / split / "labels"

    if not lbl_dir.exists():
        raise FileNotFoundError(f"YOLO label directory not found at {lbl_dir}")

    # Build patient lookup from frames_index.csv
    frames_path = Path(index_dir) / "frames_index.csv"
    frame_to_patient: dict[str, str] = {}
    if frames_path.exists():
        frames_df = pd.read_csv(frames_path)
        for _, r in frames_df.iterrows():
            frame_to_patient[str(r["frame_id"])] = str(r["patient_id"])

    ground_truths: dict[str, list[list[float]]] = {}
    total_label_box_count = 0
    per_class_counts: dict[int, int] = {}

    if full_frames:
        warnings.warn(
            "--full-frames selected: evaluating against all raw frames may treat "
            "unannotated frames from lesion videos as false negatives; prefer the "
            "canonical evaluation set by default.",
            UserWarning,
            stacklevel=2,
        )

    txt_files = sorted(lbl_dir.glob("*.txt"))
    for txt_file in txt_files:
        fid = txt_file.stem
        ground_truths[fid] = []
        if fid not in frame_to_patient:
            pid = fid.split("_")[0]
            frame_to_patient[fid] = pid

        lines = [l.strip() for l in txt_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            orig_cls = int(parts[0])
            xc_n, yc_n, wn, hn = map(float, parts[1:])

            total_label_box_count += 1

            # Map scheme
            if scheme == "7class":
                cls_id = orig_cls
            elif scheme == "binary":
                cls_id = 1
            elif scheme == "severity3":
                if orig_cls in (0, 1):
                    cls_id = 0
                elif orig_cls in (2, 3):
                    cls_id = 1
                else:
                    cls_id = 2
            else:
                cls_id = orig_cls

            # Scale normalized YOLO xywh (512x512) to absolute x, y, w, h
            w = wn * 512.0
            h = hn * 512.0
            x = (xc_n - wn / 2.0) * 512.0
            y = (yc_n - hn / 2.0) * 512.0

            ground_truths[fid].append([x, y, w, h, cls_id])
            per_class_counts[cls_id] = per_class_counts.get(cls_id, 0) + 1

    n_gt = sum(len(boxes) for boxes in ground_truths.values())

    # Hard assertion: n_gt loaded must strictly match label file box count
    assert n_gt == total_label_box_count, f"GT count mismatch! loaded n_gt={n_gt} vs file count={total_label_box_count}"

    logger.info(
        "Loaded ground truth for split '%s' (%s scheme): %d images, %d total boxes across classes %s",
        split, scheme, len(ground_truths), n_gt, dict(sorted(per_class_counts.items())),
    )

    return ground_truths, frame_to_patient

