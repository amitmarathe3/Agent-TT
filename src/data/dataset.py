"""PyTorch Dataset module for CADICA coronary angiography dataset.

Supports dynamic image loading from zip archives or folders, bounding-box annotations,
negative frames, Albumentations transforms, multiple class schemes, and clinical metadata integration.
"""

from collections.abc import Callable
import io
from pathlib import Path
from typing import Any
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augment import clip_yolo_boxes
from src.utils.config import load_config


class CADICADataset(Dataset):
    """PyTorch Dataset for CADICA angiography frames and lesion bounding boxes."""

    def __init__(
        self,
        frames_df: pd.DataFrame,
        annotations_df: pd.DataFrame | None = None,
        dataset_root: str | Path = "CADICA.zip",
        transforms: Callable | None = None,
        class_scheme: str = "7class",
        config_path: str | Path = "configs/config.yaml",
        metadata_df: pd.DataFrame | None = None,
        box_format: str = "yolo",
        image_size: int = 512,
        normalize: bool = True,
    ) -> None:
        """Initialize CADICADataset.

        Args:
            frames_df: DataFrame containing frame records with columns:
                ['patient_id', 'video_id', 'frame_id', 'frame_path'].
            annotations_df: Optional DataFrame with bounding box records with columns:
                ['frame_id', 'category', 'x_center', 'y_center', 'w_norm', 'h_norm'].
            dataset_root: Path to CADICA.zip archive or extracted directory.
            transforms: Optional Albumentations transform pipeline.
            class_scheme: Category mapping scheme: '7class', 'binary', or 'severity3'.
            config_path: Path to YAML config for loading scheme mappings if needed.
            metadata_df: Optional DataFrame containing cleaned patient clinical metadata.
            box_format: Bounding box format ('yolo' = [xc, yc, w, h] normalized).
            image_size: Target image square dimension.
            normalize: Whether to normalize image pixels to [0.0, 1.0].
        """
        self.frames_df = frames_df.reset_index(drop=True)
        self.dataset_root = Path(dataset_root)
        self.transforms = transforms
        self.class_scheme = class_scheme
        self.box_format = box_format
        self.image_size = image_size
        self.normalize = normalize

        # Load class mappings
        self.class_mapping = self._load_class_mapping(config_path, class_scheme)

        # Build fast annotations lookup by frame_id
        self.ann_lookup: dict[str, list[dict[str, Any]]] = {}
        if annotations_df is not None and len(annotations_df) > 0:
            for _, row in annotations_df.iterrows():
                fid = str(row["frame_id"])
                if fid not in self.ann_lookup:
                    self.ann_lookup[fid] = []
                self.ann_lookup[fid].append({
                    "category": str(row["category"]),
                    "x_center": float(row["x_center"]),
                    "y_center": float(row["y_center"]),
                    "w_norm": float(row["w_norm"]),
                    "h_norm": float(row["h_norm"]),
                })

        # Build fast patient clinical metadata lookup
        self.metadata_lookup: dict[str, np.ndarray] = {}
        if metadata_df is not None and len(metadata_df) > 0:
            # Select numeric clinical columns for metadata tensor
            meta = metadata_df.copy()
            numeric_cols = [
                col for col in meta.select_dtypes(include=[np.number]).columns
                if col not in ["patient_id"]
            ]
            for _, row in meta.iterrows():
                pid = str(row["patient_id"])
                vals = row[numeric_cols].fillna(0.0).values.astype(np.float32)
                self.metadata_lookup[pid] = vals

        # Zipfile handle management (lazy initialization per worker process)
        self._zip_file: zipfile.ZipFile | None = None
        self.is_zip = self.dataset_root.is_file() and self.dataset_root.suffix.lower() == ".zip"

    def _load_class_mapping(self, config_path: str | Path, scheme: str) -> dict[str, int]:
        """Load class scheme mapping from config file or standard defaults."""
        cfg_file = Path(config_path)
        if cfg_file.exists():
            cfg = load_config(cfg_file)
            schemes = cfg.get("class_schemes", {}).get("schemes", {})
            if scheme in schemes:
                return {str(k): int(v) for k, v in schemes[scheme].items()}

        # Standard built-in fallbacks
        if scheme == "binary":
            return {
                "nonlesion": 0, "p0_20": 1, "p20_50": 1, "p50_70": 1,
                "p70_90": 1, "p90_98": 1, "p99": 1, "p100": 1,
            }
        elif scheme == "severity3":
            return {
                "p0_20": 0, "p20_50": 0,
                "p50_70": 1,
                "p70_90": 2, "p90_98": 2, "p99": 2, "p100": 2,
            }
        else:  # 7class
            return {
                "p0_20": 0, "p20_50": 1, "p50_70": 2,
                "p70_90": 3, "p90_98": 4, "p99": 5, "p100": 6,
            }

    @property
    def zip_handle(self) -> zipfile.ZipFile:
        """Lazy-open zipfile handle (thread/process safe)."""
        if self._zip_file is None:
            self._zip_file = zipfile.ZipFile(self.dataset_root, "r")
        return self._zip_file

    def __len__(self) -> int:
        """Return total number of frames in dataset."""
        return len(self.frames_df)

    def _read_image(self, frame_path_str: str) -> np.ndarray:
        """Load image as uint8 RGB numpy array from zip or filesystem."""
        if self.is_zip:
            # Normalize path inside zip (forward slashes)
            norm_path = frame_path_str.replace("\\", "/").lstrip("/")
            try:
                img_bytes = self.zip_handle.read(norm_path)
            except KeyError:
                # Try matching by suffix or filename if full path varies
                filename = Path(norm_path).name
                candidates = [n for n in self.zip_handle.namelist() if n.endswith(filename)]
                if candidates:
                    img_bytes = self.zip_handle.read(candidates[0])
                else:
                    raise FileNotFoundError(f"Frame image not found in zip: {norm_path}")

            img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Failed to decode image from zip bytes: {norm_path}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            full_path = self.dataset_root / frame_path_str
            if not full_path.exists():
                full_path = Path(frame_path_str)
            if not full_path.exists():
                raise FileNotFoundError(f"Frame image file not found: {full_path}")
            img_bgr = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError(f"Failed to read image from disk: {full_path}")
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get dataset item by integer index.

        Returns dictionary with keys:
            - 'image': torch.FloatTensor of shape (3, H, W) normalized to [0, 1]
            - 'boxes': torch.FloatTensor of shape (N, 4) in YOLO format [xc, yc, w, h]
            - 'labels': torch.LongTensor of shape (N,)
            - 'patient_id': str
            - 'frame_id': str
            - 'metadata': torch.FloatTensor or None
        """
        row = self.frames_df.iloc[idx]
        patient_id = str(row["patient_id"])
        frame_id = str(row["frame_id"])
        frame_path = str(row["frame_path"])

        # 1. Load image (RGB)
        image = self._read_image(frame_path)

        # 2. Gather bboxes & labels
        boxes_list: list[list[float]] = []
        labels_list: list[int] = []

        if frame_id in self.ann_lookup:
            for ann in self.ann_lookup[frame_id]:
                cat = ann["category"]
                if cat in self.class_mapping:
                    class_id = self.class_mapping[cat]
                    boxes_list.append([
                        ann["x_center"],
                        ann["y_center"],
                        ann["w_norm"],
                        ann["h_norm"],
                    ])
                    labels_list.append(class_id)

        # Clip boxes to valid [0, 1] range
        if boxes_list:
            boxes_list = clip_yolo_boxes(boxes_list)
            # Ensure labels match clipped count
            labels_list = labels_list[: len(boxes_list)]

        # 3. Apply Albumentations augmentations
        if self.transforms is not None:
            try:
                augmented = self.transforms(image=image, bboxes=boxes_list, labels=labels_list)
                image = augmented["image"]
                boxes_list = clip_yolo_boxes(augmented.get("bboxes", []))
                labels_list = augmented.get("labels", [])[: len(boxes_list)]
            except Exception:
                # If transform drops all boxes or errors, fallback to unaugmented
                pass

        # 4. Convert image to PyTorch FloatTensor (C, H, W)
        if image.dtype != np.float32:
            img_tensor = torch.from_numpy(image).float()
        else:
            img_tensor = torch.from_numpy(image)

        # Normalize pixel values
        if self.normalize:
            img_tensor = img_tensor / 255.0

        # Change dimension order from (H, W, C) to (C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).contiguous()

        # 5. Convert boxes & labels to Tensors
        if len(boxes_list) > 0:
            boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_list, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        # 6. Clinical metadata features
        metadata_tensor = None
        if patient_id in self.metadata_lookup:
            metadata_tensor = torch.from_numpy(self.metadata_lookup[patient_id])

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "patient_id": patient_id,
            "frame_id": frame_id,
            "metadata": metadata_tensor,
        }

    def __del__(self) -> None:
        """Close zip file handle on object destruction."""
        if self._zip_file is not None:
            try:
                self._zip_file.close()
            except Exception:
                pass

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Custom collate function for PyTorch DataLoader handling variable number of bboxes."""
        images = torch.stack([item["image"] for item in batch], dim=0)
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]
        patient_ids = [item["patient_id"] for item in batch]
        frame_ids = [item["frame_id"] for item in batch]

        metadata_items = [item["metadata"] for item in batch if item["metadata"] is not None]
        if len(metadata_items) == len(batch):
            metadata = torch.stack(metadata_items, dim=0)
        else:
            metadata = None

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "patient_ids": patient_ids,
            "frame_ids": frame_ids,
            "metadata": metadata,
        }
