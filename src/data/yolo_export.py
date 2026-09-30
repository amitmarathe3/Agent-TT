"""YOLO dataset exporter module for CADICA coronary angiography dataset.

Exports structured datasets in Ultralytics YOLO format (images/ and labels/ for train/val/test,
plus data.yaml) with configurable class schemes, negative frame sampling, and K-fold support.
"""

from collections.abc import Sequence
import logging
import os
from pathlib import Path
import shutil
from typing import Any
import zipfile

import numpy as np
import pandas as pd
import yaml

from src.utils.config import load_config
from src.utils.logger import get_logger


class YOLOExporter:
    """Exports CADICA frames and annotations to standard Ultralytics YOLO format."""

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        output_dir: str | Path = "outputs/yolo_dataset",
        scheme: str = "7class",
        neg_ratio: float = 0.25,
        fold: int | None = None,
        seed: int = 42,
        use_symlinks: bool = True,
        min_neg_distance: int = 15,
    ) -> None:
        """Initialize YOLOExporter.

        Args:
            config_path: Path to YAML configuration file.
            output_dir: Directory where YOLO dataset will be exported.
            scheme: Class scheme ('7class', 'binary', 'severity3').
            neg_ratio: Ratio of negative (lesion-free) frames to positive frames.
            fold: Optional fold index (0 to 4) to export a specific CV fold.
            seed: Random seed for deterministic negative sampling.
            use_symlinks: Whether to try creating symlinks for images before fallback to copy.
            min_neg_distance: Minimum frame distance from annotated keyframes for lesion-video negatives.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path) if self.config_path.exists() else {}

        logging_cfg = self.config.get("logging", {}) if isinstance(self.config.get("logging"), dict) else {}
        self.logger = get_logger(
            name="YOLOExporter",
            log_file=logging_cfg.get("log_file", "outputs/logs/run.log"),
            level=logging_cfg.get("level", "INFO"),
        )

        self.dataset_root = Path(self.config.get("dataset_root", "CADICA.zip"))
        self.index_dir = Path(self.config.get("index_dir", "data_index"))
        self.output_dir = Path(output_dir)
        self.scheme = scheme
        self.neg_ratio = max(0.0, float(neg_ratio))
        self.fold = int(fold) if fold is not None else None
        self.seed = int(seed)
        self.use_symlinks = use_symlinks
        self.min_neg_distance = int(min_neg_distance)

        # Resolve class names and mappings
        self.class_mapping, self.class_names = self._resolve_class_scheme(self.scheme)

    def _resolve_class_scheme(self, scheme: str) -> tuple[dict[str, int], list[str]]:
        """Resolve mapping and ordered class names for YOLO data.yaml."""
        schemes = self.config.get("class_schemes", {}).get("schemes", {})
        if scheme in schemes:
            raw_map = schemes[scheme]
            if scheme == "binary":
                # Binary names: 0: normal, 1: lesion
                mapping = {str(k): int(v) for k, v in raw_map.items()}
                names = ["normal", "lesion"]
                return mapping, names
            elif scheme == "severity3":
                # Tri-tier: 0: mild, 1: moderate, 2: severe
                mapping = {str(k): int(v) for k, v in raw_map.items()}
                names = ["mild", "moderate", "severe"]
                return mapping, names
            else:
                mapping = {str(k): int(v) for k, v in raw_map.items()}
                # Invert mapping to find names in order 0..6
                inv = {int(v): str(k) for k, v in mapping.items()}
                names = [inv[i] for i in sorted(inv.keys())]
                return mapping, names

        # Built-in fallbacks
        if scheme == "binary":
            mapping = {
                "nonlesion": 0, "p0_20": 1, "p20_50": 1, "p50_70": 1,
                "p70_90": 1, "p90_98": 1, "p99": 1, "p100": 1,
            }
            return mapping, ["normal", "lesion"]
        elif scheme == "severity3":
            mapping = {
                "p0_20": 0, "p20_50": 0, "p50_70": 1,
                "p70_90": 2, "p90_98": 2, "p99": 2, "p100": 2,
            }
            return mapping, ["mild", "moderate", "severe"]
        else:  # 7class
            mapping = {
                "p0_20": 0, "p20_50": 1, "p50_70": 2,
                "p70_90": 3, "p90_98": 4, "p99": 5, "p100": 6,
            }
            return mapping, ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]

    def load_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load frames, annotations, and split mappings from data_index."""
        frames_path = self.index_dir / "frames_index.csv"
        ann_path = self.index_dir / "annotations.csv"

        if not frames_path.exists() or not ann_path.exists():
            raise FileNotFoundError(
                f"Missing index files in {self.index_dir}. Run scripts/build_index.py first."
            )

        frames_df = pd.read_csv(frames_path)
        ann_df = pd.read_csv(ann_path)

        # Load split definition
        if self.fold is not None:
            folds_path = self.index_dir / "splits" / "folds_5.csv"
            if not folds_path.exists():
                raise FileNotFoundError(
                    f"Folds file not found: {folds_path}. Run scripts/make_splits.py first."
                )
            folds_df = pd.read_csv(folds_path)
            # Map fold assignment to train/val/test:
            # test = fold N, val = (N + 1) % 5, train = remaining
            k = 5
            val_fold = (self.fold + 1) % k
            split_rows = []
            for _, r in folds_df.iterrows():
                f_val = int(r["fold"])
                if f_val == self.fold:
                    s = "test"
                elif f_val == val_fold:
                    s = "val"
                else:
                    s = "train"
                split_rows.append({"patient_id": r["patient_id"], "split": s})
            split_df = pd.DataFrame(split_rows)
            self.logger.info(
                "Exporting Fold %d (Test: Fold %d, Val: Fold %d, Train: other folds)",
                self.fold,
                self.fold,
                val_fold,
            )
        else:
            split_path = self.index_dir / "splits" / "split_v1.csv"
            if not split_path.exists():
                raise FileNotFoundError(
                    f"Splits file not found: {split_path}. Run scripts/make_splits.py first."
                )
            split_df = pd.read_csv(split_path)

        return frames_df, ann_df, split_df

    def _sample_negatives(
        self,
        frames_df: pd.DataFrame,
        ann_df: pd.DataFrame,
        split_patients: set[str],
        n_negatives: int,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Sample clean negative frames for a split from nonlesion videos and far unannotated frames."""
        if n_negatives <= 0:
            return pd.DataFrame()

        # Build fast video -> annotated frame numbers lookup
        annotated_fids = set(ann_df["frame_id"])
        ann_by_video: dict[str, set[int]] = {}
        for fid in annotated_fids:
            parts = str(fid).split("_")
            if len(parts) >= 3:
                vid_key = f"{parts[0]}_{parts[1]}"
                try:
                    f_num = int(parts[2])
                    ann_by_video.setdefault(vid_key, set()).add(f_num)
                except ValueError:
                    pass

        # Filter unannotated frames belonging to patients in this split
        split_frames = frames_df[
            frames_df["patient_id"].isin(split_patients) & (~frames_df["has_groundtruth"])
        ]

        candidate_indices = []
        for idx, row in split_frames.iterrows():
            fid = str(row["frame_id"])
            has_lesion = bool(row["has_lesion_video"])

            if not has_lesion:
                # 1. Frames from nonlesion videos are always clean negatives
                candidate_indices.append(idx)
            else:
                # 2. Frames from lesion videos must be far from keyframes
                parts = fid.split("_")
                if len(parts) >= 3:
                    vid_key = f"{parts[0]}_{parts[1]}"
                    try:
                        f_num = int(parts[2])
                        ann_nums = ann_by_video.get(vid_key, set())
                        if not ann_nums or min(abs(f_num - a) for a in ann_nums) >= self.min_neg_distance:
                            candidate_indices.append(idx)
                    except ValueError:
                        pass

        if not candidate_indices:
            return pd.DataFrame()

        selected_count = min(n_negatives, len(candidate_indices))
        sampled_idxs = rng.choice(candidate_indices, size=selected_count, replace=False)
        return frames_df.loc[sampled_idxs].copy()

    def export(self) -> dict[str, Any]:
        """Execute full YOLO dataset export workflow.

        Returns:
            Dictionary with counts and paths of exported data.
        """
        frames_df, ann_df, split_df = self.load_data()
        rng = np.random.RandomState(self.seed)

        # Merge split assignment into frames and annotations
        merged_frames = frames_df.merge(split_df, on="patient_id", how="inner")
        merged_ann = ann_df.merge(split_df, on="patient_id", how="inner")

        # Create output directories
        for split_name in ["train", "val", "test"]:
            (self.output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        # Build annotations lookup: frame_id -> list of yolo lines
        ann_lines_by_frame: dict[str, list[str]] = {}
        for _, row in merged_ann.iterrows():
            fid = str(row["frame_id"])
            cat = str(row["category"])
            if cat in self.class_mapping:
                cid = self.class_mapping[cat]
                xc = float(row["x_center"])
                yc = float(row["y_center"])
                wn = float(row["w_norm"])
                hn = float(row["h_norm"])
                # Clamp coordinates to [0, 1]
                xc = max(0.0, min(1.0, xc))
                yc = max(0.0, min(1.0, yc))
                wn = max(0.001, min(1.0, wn))
                hn = max(0.001, min(1.0, hn))
                line = f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
                ann_lines_by_frame.setdefault(fid, []).append(line)

        # Open zipfile if dataset root is zip
        is_zip = self.dataset_root.is_file() and self.dataset_root.suffix.lower() == ".zip"
        zip_handle = zipfile.ZipFile(self.dataset_root, "r") if is_zip else None

        zip_namelist_set = set(zip_handle.namelist()) if is_zip and zip_handle else set()

        stats: dict[str, Any] = {"train": {}, "val": {}, "test": {}}

        try:
            for s_name in ["train", "val", "test"]:
                s_patients = set(split_df[split_df["split"] == s_name]["patient_id"])

                # Positive frames: annotated frames in this split
                s_pos_frames = merged_frames[
                    (merged_frames["split"] == s_name) & (merged_frames["has_groundtruth"])
                ].drop_duplicates(subset=["frame_id"])

                n_pos = len(s_pos_frames)

                # Sample negatives at neg_ratio
                n_neg_target = int(round(n_pos * self.neg_ratio))
                s_neg_frames = self._sample_negatives(
                    frames_df, ann_df, s_patients, n_neg_target, rng
                )
                n_neg = len(s_neg_frames)

                self.logger.info(
                    "Exporting %s: %d positive frames, %d negative frames (ratio: %.2f)",
                    s_name,
                    n_pos,
                    n_neg,
                    self.neg_ratio,
                )

                img_dest_dir = self.output_dir / "images" / s_name
                lbl_dest_dir = self.output_dir / "labels" / s_name

                # 1. Export Positives
                for _, row in s_pos_frames.iterrows():
                    fid = str(row["frame_id"])
                    fpath = str(row["frame_path"])
                    self._export_single_frame(
                        fid, fpath, is_zip, zip_handle, zip_namelist_set, img_dest_dir, lbl_dest_dir, ann_lines_by_frame.get(fid, [])
                    )

                # 2. Export Negatives (with empty label files)
                for _, row in s_neg_frames.iterrows():
                    fid = str(row["frame_id"])
                    fpath = str(row["frame_path"])
                    self._export_single_frame(
                        fid, fpath, is_zip, zip_handle, zip_namelist_set, img_dest_dir, lbl_dest_dir, []
                    )

                stats[s_name] = {
                    "positives": n_pos,
                    "negatives": n_neg,
                    "total_images": n_pos + n_neg,
                    "boxes": sum(len(ann_lines_by_frame.get(str(fid), [])) for fid in s_pos_frames["frame_id"]),
                }

            # 3. Generate data.yaml
            yaml_content = {
                "path": str(self.output_dir.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(self.class_names),
                "names": self.class_names,
            }
            yaml_path = self.output_dir / "data.yaml"
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yaml_content, f, sort_keys=False)

            self.logger.info("Wrote YOLO data configuration to %s", yaml_path)
            stats["data_yaml"] = str(yaml_path)
            stats["class_names"] = self.class_names

        finally:
            if zip_handle is not None:
                zip_handle.close()

        return stats

    def _export_single_frame(
        self,
        frame_id: str,
        frame_path_str: str,
        is_zip: bool,
        zip_handle: zipfile.ZipFile | None,
        zip_namelist_set: set[str],
        img_dest_dir: Path,
        lbl_dest_dir: Path,
        label_lines: list[str],
    ) -> None:
        """Export image file and label file for one frame."""
        target_img = img_dest_dir / f"{frame_id}.png"
        target_lbl = lbl_dest_dir / f"{frame_id}.txt"

        # 1. Write labels
        target_lbl.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

        # 2. Link or copy image
        if not target_img.exists():
            if is_zip and zip_handle is not None:
                norm_path = frame_path_str.replace("\\", "/").lstrip("/")
                if norm_path not in zip_namelist_set:
                    # Suffix match
                    filename = Path(norm_path).name
                    candidates = [n for n in zip_namelist_set if n.endswith(filename)]
                    if candidates:
                        norm_path = candidates[0]
                    else:
                        raise FileNotFoundError(f"Frame {norm_path} not found in zip")
                # Extract image bytes
                img_bytes = zip_handle.read(norm_path)
                target_img.write_bytes(img_bytes)
            else:
                src_path = self.dataset_root / frame_path_str
                if not src_path.exists():
                    src_path = Path(frame_path_str)
                if not src_path.exists():
                    raise FileNotFoundError(f"Source image not found: {src_path}")

                if self.use_symlinks:
                    try:
                        os.symlink(str(src_path.resolve()), str(target_img.resolve()))
                    except OSError:
                        # Fallback to copy if symlink fails (e.g. non-admin Windows)
                        shutil.copy2(src_path, target_img)
                else:
                    shutil.copy2(src_path, target_img)
