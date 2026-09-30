"""Unit tests for dataset indexer module using synthetic datasets."""

import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pytest

from src.data.indexer import DatasetIndexer


@pytest.fixture
def synthetic_dataset_for_indexer(tmp_path: Path) -> Path:
    """Fixture creating a synthetic CADICA dataset structure for index testing."""
    root = tmp_path / "synthetic_indexer_dataset"
    root.mkdir()

    # Metadata
    (root / "metadata.xlsx").write_bytes(b"dummy metadata")

    # Selected Videos - Patient p1, Video v1
    v1_input = root / "selectedVideos" / "p1" / "v1" / "input"
    v1_input.mkdir(parents=True)
    v1_gt = root / "selectedVideos" / "p1" / "v1" / "groundtruth"
    v1_gt.mkdir(parents=True)

    # Create dummy 100x100 PNG image
    img1 = Image.new("RGB", (100, 100), color="blue")
    img1.save(v1_input / "p1_v1_00001.png")

    img2 = Image.new("RGB", (200, 100), color="red")
    img2.save(v1_input / "p1_v1_00002.png")

    # Bounding box groundtruth for frame 00001 (x=10, y=20, w=30, h=40, category=p50_70)
    (v1_gt / "p1_v1_00001.txt").write_text("10 20 30 40 p50_70\n", encoding="utf-8")

    # Control files
    p1_dir = root / "selectedVideos" / "p1"
    (p1_dir / "lesionVideos.txt").write_text("v1\n", encoding="utf-8")
    (p1_dir / "nonlesionVideos.txt").write_text("", encoding="utf-8")
    (root / "selectedVideos" / "p1" / "v1" / "p1_v1_selectedFrames.txt").write_text("p1_v1_00001\n", encoding="utf-8")

    # Nonselected Videos - Patient p2, Video v1
    v2_input = root / "nonselectedVideos" / "p2" / "v1" / "input"
    v2_input.mkdir(parents=True)
    img3 = Image.new("RGB", (100, 100), color="green")
    img3.save(v2_input / "p2_v1_00001.png")

    return root


@pytest.fixture
def indexer_config(tmp_path: Path, synthetic_dataset_for_indexer: Path) -> Path:
    """Fixture creating configuration pointing to synthetic dataset."""
    config_file = tmp_path / "config.yaml"
    content = f"""
dataset_root: "{synthetic_dataset_for_indexer.as_posix()}"
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
  level: "DEBUG"
  log_file: "{(tmp_path / 'outputs' / 'logs' / 'run.log').as_posix()}"
"""
    config_file.write_text(content, encoding="utf-8")
    return config_file


def test_build_index_tables(indexer_config: Path, tmp_path: Path) -> None:
    """Test building full index creates all CSV files and classes.json with correct schema."""
    indexer = DatasetIndexer(config_path=indexer_config)
    paths = indexer.build_index()

    assert paths["classes"].exists()
    assert paths["frames_index"].exists()
    assert paths["annotations"].exists()
    assert paths["videos_index"].exists()
    assert paths["patients_index"].exists()

    # Verify classes.json
    classes_mapping = json.loads(paths["classes"].read_text(encoding="utf-8"))
    assert "p50_70" in classes_mapping

    # Verify frames_index.csv
    df_frames = pd.read_csv(paths["frames_index"])
    assert len(df_frames) == 3
    assert set(df_frames.columns) == {
        "split_type", "patient_id", "video_id", "frame_id", "frame_path",
        "is_selected_keyframe", "has_lesion_video", "has_groundtruth",
        "image_width", "image_height"
    }

    # Check keyframe flag for p1_v1_00001
    f1_row = df_frames[df_frames["frame_id"] == "p1_v1_00001"].iloc[0]
    assert f1_row["is_selected_keyframe"] == True
    assert f1_row["has_groundtruth"] == True
    assert f1_row["image_width"] == 100
    assert f1_row["image_height"] == 100

    # Verify annotations.csv
    df_ann = pd.read_csv(paths["annotations"])
    assert len(df_ann) == 1
    ann_row = df_ann.iloc[0]
    assert ann_row["patient_id"] == "p1"
    assert ann_row["video_id"] == "v1"
    assert ann_row["x"] == 10
    assert ann_row["y"] == 20
    assert ann_row["w"] == 30
    assert ann_row["h"] == 40
    assert ann_row["category"] == "p50_70"

    # Check YOLO normalization: x_center = (10 + 30/2)/100 = 0.25, y_center = (20 + 40/2)/100 = 0.40
    assert pytest.approx(ann_row["x_center"], 0.001) == 0.25
    assert pytest.approx(ann_row["y_center"], 0.001) == 0.40
    assert pytest.approx(ann_row["w_norm"], 0.001) == 0.30
    assert pytest.approx(ann_row["h_norm"], 0.001) == 0.40

    # Verify patients_index.csv
    df_patients = pd.read_csv(paths["patients_index"])
    assert len(df_patients) == 2


def test_build_index_limit_patients(indexer_config: Path) -> None:
    """Test limiting patient processing via limit_patients parameter."""
    indexer = DatasetIndexer(config_path=indexer_config)
    paths = indexer.build_index(limit_patients=1)

    df_patients = pd.read_csv(paths["patients_index"])
    assert len(df_patients) == 1
    assert df_patients.iloc[0]["patient_id"] == "p1"


def test_parse_malformed_groundtruth(indexer_config: Path) -> None:
    """Test parsing empty and malformed groundtruth files gracefully."""
    indexer = DatasetIndexer(config_path=indexer_config)

    # Empty content
    boxes_empty = indexer.parse_groundtruth_file("")
    assert len(boxes_empty) == 0

    # Malformed text (insufficient tokens or non-numeric)
    malformed_content = "invalid_line_tokens\n10 20 abc 40 p0_20\n15 25 35 45 p70_90\n"
    boxes = indexer.parse_groundtruth_file(malformed_content)
    assert len(boxes) == 1
    assert boxes[0] == (15.0, 25.0, 35.0, 45.0, "p70_90")
