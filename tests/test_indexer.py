"""Unit tests for DatasetIndexer using a small synthetic CADICA directory fixture."""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.data.indexer import DatasetIndexer, _parse_gt_line, _detect_zip_prefix


# ── helpers ─────────────────────────────────────────────────────────────────


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return minimal bytes for a tiny PNG image."""
    buf = BytesIO()
    Image.new("L", (width, height), color=128).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def base_config(tmp_path: Path) -> tuple[Path, Path]:
    """Return (config_path, dataset_root) pointing at a synthetic directory dataset."""
    dataset_root = tmp_path / "CADICA_synth"
    dataset_root.mkdir()

    # ── structure: selectedVideos/p1/v1 with lesion, p1/v2 without ──────────
    sel = dataset_root / "selectedVideos" / "p1"
    sel.mkdir(parents=True)
    (sel / "lesionVideos.txt").write_text("v1\n", encoding="utf-8")
    (sel / "nonlesionVideos.txt").write_text("v2\n", encoding="utf-8")

    # v1 – lesion video with groundtruth
    v1_input = sel / "v1" / "input"
    v1_gt = sel / "v1" / "groundtruth"
    v1_input.mkdir(parents=True)
    v1_gt.mkdir(parents=True)
    (v1_input / "p1_v1_00001.png").write_bytes(_make_png_bytes(128, 128))
    (v1_input / "p1_v1_00002.png").write_bytes(_make_png_bytes(128, 128))
    (v1_gt / "p1_v1_00001.txt").write_text("10 20 30 40 p0_20\n", encoding="utf-8")
    (sel / "v1" / "p1_v1_selectedFrames.txt").write_text("p1_v1_00001\np1_v1_00002\n", encoding="utf-8")

    # v2 – non-lesion, no groundtruth
    v2_input = sel / "v2" / "input"
    v2_input.mkdir(parents=True)
    (v2_input / "p1_v2_00001.png").write_bytes(_make_png_bytes(128, 128))
    (sel / "v2" / "p1_v2_selectedFrames.txt").write_text("p1_v2_00001\n", encoding="utf-8")

    # ── nonselectedVideos/p1/v3 ─────────────────────────────────────────────
    nonsel = dataset_root / "nonselectedVideos" / "p1"
    nonsel.mkdir(parents=True)
    v3_input = nonsel / "v3" / "input"
    v3_input.mkdir(parents=True)
    (v3_input / "p1_v3_00001.png").write_bytes(_make_png_bytes(64, 64))

    # ── config ──────────────────────────────────────────────────────────────
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
dataset_root: "{dataset_root.as_posix()}"
metadata_file: "metadata.xlsx"
folders:
  selected: "selectedVideos"
  nonselected: "nonselectedVideos"
index_dir: "{(tmp_path / 'data_index').as_posix()}"
reports_dir: "{(tmp_path / 'reports').as_posix()}"
outputs_dir: "{(tmp_path / 'outputs').as_posix()}"
random_seed: 42
image_extension: ".png"
logging:
  level: "WARNING"
  log_file: "{(tmp_path / 'outputs/logs/run.log').as_posix()}"
""",
        encoding="utf-8",
    )
    return cfg, dataset_root


@pytest.fixture
def zip_config(tmp_path: Path) -> tuple[Path, Path]:
    """Return (config_path, zip_path) pointing at a synthetic zip-based dataset."""
    zip_path = tmp_path / "CADICA_synth.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # lesion control files
        zf.writestr("CADICA/selectedVideos/p1/lesionVideos.txt", "v1\n")
        zf.writestr("CADICA/selectedVideos/p1/nonlesionVideos.txt", "v2\n")
        # selectedFrames
        zf.writestr("CADICA/selectedVideos/p1/v1/p1_v1_selectedFrames.txt", "p1_v1_00001\n")
        # input frames
        zf.writestr("CADICA/selectedVideos/p1/v1/input/p1_v1_00001.png", _make_png_bytes(64, 64))
        zf.writestr("CADICA/selectedVideos/p1/v1/input/p1_v1_00002.png", _make_png_bytes(64, 64))
        # groundtruth
        zf.writestr("CADICA/selectedVideos/p1/v1/groundtruth/p1_v1_00001.txt", "5 10 15 20 p50_70\n")
        # nonselected
        zf.writestr("CADICA/nonselectedVideos/p1/v2/input/p1_v2_00001.png", _make_png_bytes(64, 64))

    cfg = tmp_path / "config_zip.yaml"
    cfg.write_text(
        f"""
dataset_root: "{zip_path.as_posix()}"
metadata_file: "metadata.xlsx"
folders:
  selected: "selectedVideos"
  nonselected: "nonselectedVideos"
index_dir: "{(tmp_path / 'data_index').as_posix()}"
reports_dir: "reports"
outputs_dir: "outputs"
random_seed: 42
image_extension: ".png"
logging:
  level: "WARNING"
  log_file: "{(tmp_path / 'run.log').as_posix()}"
""",
        encoding="utf-8",
    )
    return cfg, zip_path


# ── unit tests: _parse_gt_line ───────────────────────────────────────────────


def test_parse_gt_line_space_separated() -> None:
    """Parse standard space-separated annotation line."""
    result = _parse_gt_line("10 20 30 40 p0_20")
    assert result == (10, 20, 30, 40, "p0_20")


def test_parse_gt_line_comma_separated() -> None:
    """Parse comma-separated annotation line."""
    result = _parse_gt_line("10,20,30,40,p50_70")
    assert result == (10, 20, 30, 40, "p50_70")


def test_parse_gt_line_no_category() -> None:
    """Parse annotation line without a category field."""
    result = _parse_gt_line("5 6 7 8")
    assert result == (5, 6, 7, 8, "__no_category__")


def test_parse_gt_line_malformed() -> None:
    """Malformed line returns None."""
    assert _parse_gt_line("abc def") is None
    assert _parse_gt_line("") is None


# ── unit tests: directory-based indexer ─────────────────────────────────────


def test_build_from_dir_frame_count(base_config: tuple[Path, Path]) -> None:
    """frames_index should have one row per PNG frame across all splits."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    frames = tables["frames_index"]
    # 2 sel v1 + 1 sel v2 + 1 nonsel v3 = 4
    assert len(frames) == 4


def test_build_from_dir_annotations(base_config: tuple[Path, Path]) -> None:
    """annotations.csv should have exactly one box from the single groundtruth file."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    ann = tables["annotations"]
    assert len(ann) == 1
    assert ann.iloc[0]["category"] == "p0_20"
    assert ann.iloc[0]["x"] == 10
    # YOLO normalisation with 128×128 image
    assert abs(ann.iloc[0]["w_norm"] - 30 / 128) < 1e-4


def test_build_from_dir_videos_index(base_config: tuple[Path, Path]) -> None:
    """videos_index should have one row per video (3 total)."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    videos = tables["videos_index"]
    assert len(videos) == 3
    v1 = videos[videos["video_id"] == "v1"].iloc[0]
    assert v1["is_lesion_video"] is True or v1["is_lesion_video"] == True
    assert v1["num_boxes"] == 1


def test_build_from_dir_patients_index(base_config: tuple[Path, Path]) -> None:
    """patients_index should have one row (p1) with correct aggregated counts."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    patients = tables["patients_index"]
    assert len(patients) == 1
    p1 = patients.iloc[0]
    assert p1["patient_id"] == "p1"
    assert p1["num_videos"] == 3
    assert p1["num_frames"] == 4
    assert p1["num_boxes"] == 1


def test_build_from_dir_classes_json(base_config: tuple[Path, Path], tmp_path: Path) -> None:
    """classes.json must be saved and map category strings to integer IDs."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    indexer.build()
    classes_path = indexer.index_dir / "classes.json"
    assert classes_path.exists()
    classes = json.loads(classes_path.read_text())
    assert "p0_20" in classes
    assert isinstance(classes["p0_20"], int)


def test_build_from_dir_csv_files_saved(base_config: tuple[Path, Path]) -> None:
    """All four CSV index files must be written to data_index/."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    indexer.build()
    for table in ("frames_index", "annotations", "videos_index", "patients_index"):
        assert (indexer.index_dir / f"{table}.csv").exists()


def test_build_limit_patients(base_config: tuple[Path, Path]) -> None:
    """--limit-patients=0 should produce empty tables (no patients processed)."""
    cfg, _ = base_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build(limit_patients=0)
    assert len(tables["frames_index"]) == 0
    assert len(tables["patients_index"]) == 0


# ── unit tests: zip-based indexer ───────────────────────────────────────────


def test_build_from_zip_frames(zip_config: tuple[Path, Path]) -> None:
    """Zip-based indexer produces correct frame count."""
    cfg, _ = zip_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    assert len(tables["frames_index"]) == 3  # 2 sel + 1 nonsel


def test_build_from_zip_annotations(zip_config: tuple[Path, Path]) -> None:
    """Zip-based indexer parses bounding box and normalises correctly."""
    cfg, _ = zip_config
    indexer = DatasetIndexer(config_path=cfg)
    tables = indexer.build()
    ann = tables["annotations"]
    assert len(ann) == 1
    assert ann.iloc[0]["category"] == "p50_70"
    assert ann.iloc[0]["x"] == 5


def test_detect_zip_prefix() -> None:
    """_detect_zip_prefix should correctly identify the zip root prefix."""
    names = [
        "CADICA/selectedVideos/p1/v1/input/frame.png",
        "CADICA/nonselectedVideos/p2/v3/input/frame.png",
    ]
    prefix = _detect_zip_prefix(names, "selectedVideos", "nonselectedVideos")
    assert prefix == "CADICA/"
