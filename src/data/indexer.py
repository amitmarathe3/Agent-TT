"""Dataset indexer module for CADICA coronary angiography dataset.

Scans the dataset (zip or directory) and builds four structured CSV index
tables saved in data_index/:

- frames_index.csv   – one row per image frame
- annotations.csv    – one row per bounding-box annotation
- videos_index.csv   – one row per video sequence
- patients_index.csv – one row per patient

Also discovers all unique lesion categories and saves data_index/classes.json.

Groundtruth txt format (auto-detected):
  <x> <y> <w> <h> [<category>]
  Separator: space or comma.  One box per line.
  Category: stenosis-percentage string e.g. p0_20, p20_50, … p100, p99.
"""

import io
import json
import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from PIL import Image

from src.utils.config import load_config
from src.utils.logger import get_logger


# ────────────────────────────── helpers ──────────────────────────────


def _parse_gt_line(line: str) -> tuple[int, int, int, int, str] | None:
    """Parse a single groundtruth bounding-box line.

    Supports space- or comma-separated ``x y w h [category]`` format.

    Args:
        line: Raw text line from a groundtruth file.

    Returns:
        Tuple ``(x, y, w, h, category)`` or ``None`` if line is malformed.
    """
    line = line.strip()
    if not line:
        return None
    # normalise comma → space
    parts = re.split(r"[,\s]+", line.replace(",", " "))
    parts = [p for p in parts if p]
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    except (IndexError, ValueError):
        return None
    category = parts[4] if len(parts) >= 5 else "__no_category__"
    return x, y, w, h, category


def _pil_image_size(raw_bytes: bytes) -> tuple[int, int]:
    """Read image width and height from bytes without decoding pixels.

    Args:
        raw_bytes: Raw file content bytes.

    Returns:
        ``(width, height)`` tuple, or ``(0, 0)`` on error.
    """
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return img.size  # (width, height)
    except Exception:
        return 0, 0


def _iter(iterable: list, desc: str) -> Any:
    """Wrap iterable with tqdm if available, else return as-is."""
    if TQDM_AVAILABLE:
        return tqdm(iterable, desc=desc, unit="item", leave=True)
    return iterable


# ────────────────────────────── main class ──────────────────────────────


class DatasetIndexer:
    """Scans the CADICA dataset and builds structured CSV index tables."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Initialise the indexer from central configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        log_cfg = self.config.get("logging", {}) if isinstance(self.config.get("logging"), dict) else {}
        self.logger = get_logger(
            name="DatasetIndexer",
            log_file=log_cfg.get("log_file", "outputs/logs/run.log"),
            level=log_cfg.get("level", "INFO"),
        )

        self.dataset_root = Path(self.config.get("dataset_root", "CADICA.zip"))
        self.meta_file = self.config.get("metadata_file", "metadata.xlsx")
        folders = self.config.get("folders", {})
        self.sel_folder = folders.get("selected", "selectedVideos")
        self.nonsel_folder = folders.get("nonselected", "nonselectedVideos")
        self.index_dir = Path(self.config.get("index_dir", "data_index"))
        self.image_ext = self.config.get("image_extension", ".png")

        # Auto-detect whether we are working on a zip or a directory
        self._is_zip: bool = (
            self.dataset_root.is_file() and self.dataset_root.suffix.lower() == ".zip"
        )

    # ── public API ───────────────────────────────────────────────────────

    def build(self, limit_patients: int | None = None) -> dict[str, pd.DataFrame]:
        """Build all index tables and save them to ``data_index/``.

        Args:
            limit_patients: If set, process at most this many patients
                (useful for quick integration tests).

        Returns:
            Dict mapping table name → DataFrame for the four tables.
        """
        self.index_dir.mkdir(parents=True, exist_ok=True)

        if self._is_zip:
            results = self._build_from_zip(limit_patients=limit_patients)
        else:
            results = self._build_from_dir(limit_patients=limit_patients)

        # Save each table
        for name, df in results.items():
            out = self.index_dir / f"{name}.csv"
            df.to_csv(out, index=False)
            self.logger.info("Saved %s (%d rows) → %s", name, len(df), out)

        # Save classes.json
        self._save_classes(results.get("annotations", pd.DataFrame()))

        return results

    # ── zip-based indexer ────────────────────────────────────────────────

    def _build_from_zip(self, limit_patients: int | None) -> dict[str, pd.DataFrame]:
        """Build index tables by reading directly from the zip archive."""
        self.logger.info("Building index from zip: %s", self.dataset_root)

        with zipfile.ZipFile(self.dataset_root, "r") as zf:
            namelist = zf.namelist()

            # ── Determine zip prefix (e.g. "CADICA/") ──────────────────
            prefix = _detect_zip_prefix(namelist, self.sel_folder, self.nonsel_folder)
            sel_root = f"{prefix}{self.sel_folder}/"
            nonsel_root = f"{prefix}{self.nonsel_folder}/"

            # ── Build lookup structures ─────────────────────────────────
            members: set[str] = set(namelist)

            # lesionVideos per patient
            lesion_map: dict[str, set[str]] = defaultdict(set)
            for name in namelist:
                if name.startswith(sel_root) and Path(name).name == "lesionVideos.txt":
                    parts = Path(name).parts
                    # parts: [prefix?, selectedVideos, pX, lesionVideos.txt]
                    try:
                        p_idx = parts.index(self.sel_folder)
                        patient_id = parts[p_idx + 1]
                        lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
                        for ln in lines:
                            v = ln.strip()
                            if v:
                                lesion_map[patient_id].add(v)
                    except Exception as exc:
                        self.logger.warning("Could not read lesionVideos.txt %s: %s", name, exc)

            # selectedFrames per (patient, video)
            sf_map: dict[tuple[str, str], set[str]] = {}
            for name in namelist:
                if name.endswith("_selectedFrames.txt"):
                    parts = Path(name).parts
                    try:
                        p_idx = _find_patient_idx(parts)
                        patient_id = parts[p_idx]
                        video_id = parts[p_idx + 1]
                        lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
                        sf_map[(patient_id, video_id)] = {ln.strip() for ln in lines if ln.strip()}
                    except Exception as exc:
                        self.logger.warning("Could not read selectedFrames.txt %s: %s", name, exc)

            # ── Collect patient IDs ─────────────────────────────────────
            patient_ids: set[str] = set()
            for name in namelist:
                parts = Path(name).parts
                try:
                    for folder in (self.sel_folder, self.nonsel_folder):
                        if folder in parts:
                            idx = parts.index(folder)
                            if idx + 1 < len(parts) and re.match(r"p\d+", parts[idx + 1]):
                                patient_ids.add(parts[idx + 1])
                except Exception:
                    pass

            sorted_patients = sorted(patient_ids, key=lambda p: int(p[1:]))
            if limit_patients is not None:
                sorted_patients = sorted_patients[:limit_patients]

            self.logger.info(
                "Processing %d patient(s) from zip (limit=%s)", len(sorted_patients), limit_patients
            )

            # ── Build rows ─────────────────────────────────────────────
            frame_rows: list[dict] = []
            ann_rows: list[dict] = []
            video_rows: list[dict] = []
            patient_rows: list[dict] = []

            gt_format_logged = False

            for patient_id in _iter(sorted_patients, "Patients"):
                p_videos_count = p_nonsel_count = 0
                p_frames_total = p_lesion_videos = p_boxes_total = 0

                for split_type, root_pfx in [
                    ("selected", sel_root),
                    ("nonselected", nonsel_root),
                ]:
                    # gather videos for this patient
                    vid_prefix = f"{root_pfx}{patient_id}/"
                    video_ids: set[str] = set()
                    for name in namelist:
                        if name.startswith(vid_prefix):
                            rest = name[len(vid_prefix):]
                            parts = rest.split("/")
                            if parts[0] and re.match(r"v\d+", parts[0]):
                                video_ids.add(parts[0])

                    for video_id in sorted(video_ids, key=lambda v: int(v[1:])):
                        v_pfx = f"{vid_prefix}{video_id}/"
                        is_lesion = video_id in lesion_map.get(patient_id, set())
                        selected_frames = sf_map.get((patient_id, video_id), set())

                        # input frames
                        input_pfx = f"{v_pfx}input/"
                        frame_names = [
                            Path(n).name
                            for n in namelist
                            if n.startswith(input_pfx) and n.endswith(self.image_ext)
                        ]

                        # groundtruth files
                        gt_pfx = f"{v_pfx}groundtruth/"
                        gt_files_list = [
                            n
                            for n in namelist
                            if n.startswith(gt_pfx) and n.endswith(".txt")
                        ]
                        gt_stems: set[str] = {Path(n).stem for n in gt_files_list}

                        v_ann_count = 0
                        v_annotated_frames = 0

                        for frame_name in frame_names:
                            frame_id = Path(frame_name).stem
                            frame_path = f"{input_pfx}{frame_name}"
                            is_key = frame_id in selected_frames or frame_name in selected_frames
                            has_gt = frame_id in gt_stems

                            # Image size – read lazily from zip
                            w, h = 0, 0
                            try:
                                raw = zf.read(frame_path)
                                w, h = _pil_image_size(raw)
                            except Exception:
                                pass

                            frame_rows.append(
                                {
                                    "split_type": split_type,
                                    "patient_id": patient_id,
                                    "video_id": video_id,
                                    "frame_id": frame_id,
                                    "frame_path": frame_path,
                                    "is_selected_keyframe": is_key,
                                    "has_lesion_video": is_lesion,
                                    "has_groundtruth": has_gt,
                                    "image_width": w,
                                    "image_height": h,
                                }
                            )

                            # Annotations for this frame
                            if has_gt:
                                gt_name = f"{gt_pfx}{frame_id}.txt"
                                try:
                                    gt_text = zf.read(gt_name).decode("utf-8", errors="replace")
                                except Exception as exc:
                                    self.logger.warning("Cannot read %s: %s", gt_name, exc)
                                    continue

                                boxes = self._parse_gt_text(
                                    gt_text, gt_name, w, h, not gt_format_logged
                                )
                                gt_format_logged = True
                                frame_ann_count = 0
                                for x, y, bw, bh, cat in boxes:
                                    xc, yc, wn, hn = self._normalise_box(x, y, bw, bh, w, h)
                                    ann_rows.append(
                                        {
                                            "patient_id": patient_id,
                                            "video_id": video_id,
                                            "frame_id": frame_id,
                                            "frame_path": frame_path,
                                            "x": x,
                                            "y": y,
                                            "w": bw,
                                            "h": bh,
                                            "category": cat,
                                            "x_center": xc,
                                            "y_center": yc,
                                            "w_norm": wn,
                                            "h_norm": hn,
                                        }
                                    )
                                    frame_ann_count += 1

                                if frame_ann_count > 0:
                                    v_ann_count += frame_ann_count
                                    v_annotated_frames += 1

                        n_frames = len(frame_names)
                        n_sel = sum(
                            1
                            for fn in frame_names
                            if Path(fn).stem in selected_frames or fn in selected_frames
                        )
                        video_rows.append(
                            {
                                "patient_id": patient_id,
                                "video_id": video_id,
                                "split_type": split_type,
                                "num_frames": n_frames,
                                "num_selected_frames": n_sel,
                                "is_lesion_video": is_lesion,
                                "num_annotated_frames": v_annotated_frames,
                                "num_boxes": v_ann_count,
                            }
                        )

                        p_videos_count += 1
                        if split_type == "nonselected":
                            p_nonsel_count += 1
                        if is_lesion:
                            p_lesion_videos += 1
                        p_frames_total += n_frames
                        p_boxes_total += v_ann_count

                sel_vids = p_videos_count - p_nonsel_count
                patient_rows.append(
                    {
                        "patient_id": patient_id,
                        "num_videos": p_videos_count,
                        "num_selected_videos": sel_vids,
                        "num_nonselected_videos": p_nonsel_count,
                        "num_frames": p_frames_total,
                        "num_lesion_videos": p_lesion_videos,
                        "num_boxes": p_boxes_total,
                    }
                )

        return {
            "frames_index": pd.DataFrame(frame_rows),
            "annotations": pd.DataFrame(ann_rows),
            "videos_index": pd.DataFrame(video_rows),
            "patients_index": pd.DataFrame(patient_rows),
        }

    # ── directory-based indexer ──────────────────────────────────────────

    def _build_from_dir(self, limit_patients: int | None) -> dict[str, pd.DataFrame]:
        """Build index tables from an extracted directory structure."""
        root = self.dataset_root
        # Support nested CADICA/ subfolder
        if not (root / self.sel_folder).exists() and (root / "CADICA" / self.sel_folder).exists():
            root = root / "CADICA"

        self.logger.info("Building index from directory: %s", root)

        frame_rows: list[dict] = []
        ann_rows: list[dict] = []
        video_rows: list[dict] = []
        patient_rows: list[dict] = []

        patient_ids: set[str] = set()
        for folder in (root / self.sel_folder, root / self.nonsel_folder):
            if folder.exists():
                for p in folder.iterdir():
                    if p.is_dir() and re.match(r"p\d+", p.name):
                        patient_ids.add(p.name)

        sorted_patients = sorted(patient_ids, key=lambda p: int(p[1:]))
        if limit_patients is not None:
            sorted_patients = sorted_patients[:limit_patients]

        gt_format_logged = False

        for patient_id in _iter(sorted_patients, "Patients"):
            p_videos_count = p_nonsel_count = 0
            p_frames_total = p_lesion_videos = p_boxes_total = 0

            for split_type, top_dir in [
                ("selected", root / self.sel_folder),
                ("nonselected", root / self.nonsel_folder),
            ]:
                p_dir = top_dir / patient_id
                if not p_dir.exists():
                    continue

                # lesionVideos.txt
                lesion_videos: set[str] = set()
                lv_file = p_dir / "lesionVideos.txt"
                if lv_file.exists():
                    try:
                        lesion_videos = {
                            ln.strip()
                            for ln in lv_file.read_text("utf-8").splitlines()
                            if ln.strip()
                        }
                    except Exception as exc:
                        self.logger.warning("Cannot read %s: %s", lv_file, exc)

                for v_dir in sorted(p_dir.glob("v*"), key=lambda d: int(d.name[1:])):
                    if not v_dir.is_dir():
                        continue
                    video_id = v_dir.name
                    is_lesion = video_id in lesion_videos

                    # selectedFrames
                    sf_file = v_dir / f"{patient_id}_{video_id}_selectedFrames.txt"
                    selected_frames: set[str] = set()
                    if sf_file.exists():
                        try:
                            selected_frames = {
                                ln.strip()
                                for ln in sf_file.read_text("utf-8").splitlines()
                                if ln.strip()
                            }
                        except Exception as exc:
                            self.logger.warning("Cannot read %s: %s", sf_file, exc)

                    input_dir = v_dir / "input"
                    gt_dir = v_dir / "groundtruth"
                    gt_stems: set[str] = set()
                    if gt_dir.exists():
                        gt_stems = {f.stem for f in gt_dir.glob("*.txt")}

                    frame_files = sorted(input_dir.glob(f"*{self.image_ext}")) if input_dir.exists() else []
                    v_ann_count = 0
                    v_annotated_frames = 0

                    for frame_path in frame_files:
                        frame_id = frame_path.stem
                        rel_path = str(frame_path)
                        is_key = frame_id in selected_frames or frame_path.name in selected_frames
                        has_gt = frame_id in gt_stems

                        w, h = 0, 0
                        try:
                            img = Image.open(frame_path)
                            w, h = img.size
                        except Exception:
                            pass

                        frame_rows.append(
                            {
                                "split_type": split_type,
                                "patient_id": patient_id,
                                "video_id": video_id,
                                "frame_id": frame_id,
                                "frame_path": rel_path,
                                "is_selected_keyframe": is_key,
                                "has_lesion_video": is_lesion,
                                "has_groundtruth": has_gt,
                                "image_width": w,
                                "image_height": h,
                            }
                        )

                        if has_gt:
                            gt_file = gt_dir / f"{frame_id}.txt"
                            try:
                                gt_text = gt_file.read_text("utf-8", errors="replace")
                            except Exception as exc:
                                self.logger.warning("Cannot read %s: %s", gt_file, exc)
                                continue

                            boxes = self._parse_gt_text(
                                gt_text, str(gt_file), w, h, not gt_format_logged
                            )
                            gt_format_logged = True
                            frame_ann_count = 0
                            for x, y, bw, bh, cat in boxes:
                                xc, yc, wn, hn = self._normalise_box(x, y, bw, bh, w, h)
                                ann_rows.append(
                                    {
                                        "patient_id": patient_id,
                                        "video_id": video_id,
                                        "frame_id": frame_id,
                                        "frame_path": rel_path,
                                        "x": x,
                                        "y": y,
                                        "w": bw,
                                        "h": bh,
                                        "category": cat,
                                        "x_center": xc,
                                        "y_center": yc,
                                        "w_norm": wn,
                                        "h_norm": hn,
                                    }
                                )
                                frame_ann_count += 1

                            if frame_ann_count > 0:
                                v_ann_count += frame_ann_count
                                v_annotated_frames += 1

                    n_frames = len(frame_files)
                    n_sel = sum(
                        1
                        for fp in frame_files
                        if fp.stem in selected_frames or fp.name in selected_frames
                    )
                    video_rows.append(
                        {
                            "patient_id": patient_id,
                            "video_id": video_id,
                            "split_type": split_type,
                            "num_frames": n_frames,
                            "num_selected_frames": n_sel,
                            "is_lesion_video": is_lesion,
                            "num_annotated_frames": v_annotated_frames,
                            "num_boxes": v_ann_count,
                        }
                    )

                    p_videos_count += 1
                    if split_type == "nonselected":
                        p_nonsel_count += 1
                    if is_lesion:
                        p_lesion_videos += 1
                    p_frames_total += n_frames
                    p_boxes_total += v_ann_count

            sel_vids = p_videos_count - p_nonsel_count
            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "num_videos": p_videos_count,
                    "num_selected_videos": sel_vids,
                    "num_nonselected_videos": p_nonsel_count,
                    "num_frames": p_frames_total,
                    "num_lesion_videos": p_lesion_videos,
                    "num_boxes": p_boxes_total,
                }
            )

        return {
            "frames_index": pd.DataFrame(frame_rows),
            "annotations": pd.DataFrame(ann_rows),
            "videos_index": pd.DataFrame(video_rows),
            "patients_index": pd.DataFrame(patient_rows),
        }

    # ── annotation parsing ───────────────────────────────────────────────

    def _parse_gt_text(
        self,
        text: str,
        source_name: str,
        img_w: int,
        img_h: int,
        log_format: bool,
    ) -> list[tuple[int, int, int, int, str]]:
        """Parse groundtruth text returning list of ``(x, y, w, h, category)``.

        Args:
            text: Raw file content.
            source_name: Path string used in log messages.
            img_w: Image width (for warnings).
            img_h: Image height (for warnings).
            log_format: If True, emit an INFO message about detected format.

        Returns:
            List of parsed bounding box tuples.
        """
        boxes: list[tuple[int, int, int, int, str]] = []
        lines = text.strip().splitlines()
        if not lines:
            return boxes

        if log_format:
            sample = lines[0].strip()
            sep = "comma+space" if "," in sample else "space"
            n_fields = len(re.split(r"[,\s]+", sample))
            has_cat = n_fields >= 5
            self.logger.info(
                "Groundtruth format detected from %s: separator=%s, fields=%d, has_category=%s, sample=%r",
                source_name, sep, n_fields, has_cat, sample
            )

        for i, line in enumerate(lines):
            parsed = _parse_gt_line(line)
            if parsed is None:
                if line.strip():
                    self.logger.warning(
                        "Malformed groundtruth line %d in %s: %r", i + 1, source_name, line
                    )
                continue
            boxes.append(parsed)

        return boxes

    # ── YOLO normalisation ───────────────────────────────────────────────

    @staticmethod
    def _normalise_box(
        x: int, y: int, w: int, h: int, img_w: int, img_h: int
    ) -> tuple[float, float, float, float]:
        """Convert absolute ``[x, y, w, h]`` box to YOLO-normalised values.

        Args:
            x: Top-left x pixel coordinate.
            y: Top-left y pixel coordinate.
            w: Box width in pixels.
            h: Box height in pixels.
            img_w: Image width.
            img_h: Image height.

        Returns:
            ``(x_center, y_center, w_norm, h_norm)`` normalised by image size.
            Returns ``(0.0, 0.0, 0.0, 0.0)`` if image dimensions are zero.
        """
        if img_w == 0 or img_h == 0:
            return 0.0, 0.0, 0.0, 0.0
        x_center = round((x + w / 2) / img_w, 6)
        y_center = round((y + h / 2) / img_h, 6)
        w_norm = round(w / img_w, 6)
        h_norm = round(h / img_h, 6)
        return x_center, y_center, w_norm, h_norm

    # ── classes.json ─────────────────────────────────────────────────────

    def _save_classes(self, ann_df: pd.DataFrame) -> None:
        """Discover unique annotation categories and save to classes.json.

        Args:
            ann_df: Annotations DataFrame with a ``category`` column.
        """
        out = self.index_dir / "classes.json"
        if ann_df.empty or "category" not in ann_df.columns:
            classes = {}
        else:
            unique_cats = sorted(ann_df["category"].dropna().unique().tolist())
            classes = {cat: idx for idx, cat in enumerate(unique_cats)}

        out.write_text(json.dumps(classes, indent=2), encoding="utf-8")
        self.logger.info("Saved classes.json with %d categories: %s", len(classes), list(classes.keys()))


# ── module-level helpers ────────────────────────────────────────────────


def _detect_zip_prefix(namelist: list[str], sel_folder: str, nonsel_folder: str) -> str:
    """Detect top-level prefix inside a zip archive (e.g. 'CADICA/').

    Args:
        namelist: Full zip namelist.
        sel_folder: Expected selected-videos folder name.
        nonsel_folder: Expected non-selected-videos folder name.

    Returns:
        Prefix string (may be empty if folders are at root).
    """
    for name in namelist:
        parts = Path(name).parts
        for i, part in enumerate(parts):
            if part in (sel_folder, nonsel_folder):
                return "/".join(parts[:i]) + "/" if i > 0 else ""
    return ""


def _find_patient_idx(parts: tuple[str, ...]) -> int:
    """Return the index of the patient part (pXX) in a path parts tuple.

    Args:
        parts: Tuple of path components.

    Returns:
        Index of patient component.

    Raises:
        ValueError: If no patient folder found.
    """
    for i, part in enumerate(parts):
        if re.match(r"^p\d+$", part):
            return i
    raise ValueError(f"No patient folder found in path parts: {parts}")
