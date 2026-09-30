"""Tests for Exploratory Data Analysis (EDA) module."""

import io
import zipfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from PIL import Image
import pytest
import yaml

from src.data.eda import DatasetEDA, _natural_patient_sort_key


@pytest.fixture
def synthetic_eda_env(tmp_path: Path) -> Path:
    """Create a self-contained temporary environment for EDA testing."""
    index_dir = tmp_path / "data_index"
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    outputs_dir = tmp_path / "outputs"
    index_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. frames_index.csv
    frames_data = {
        "split_type": ["selected", "selected", "selected", "nonselected"],
        "patient_id": ["p1", "p1", "p2", "p2"],
        "video_id": ["v1", "v2", "v1", "v2"],
        "frame_id": ["p1_v1_00001", "p1_v2_00001", "p2_v1_00001", "p2_v2_00001"],
        "frame_path": [
            "CADICA/selectedVideos/p1/v1/input/p1_v1_00001.png",
            "CADICA/selectedVideos/p1/v2/input/p1_v2_00001.png",
            "CADICA/selectedVideos/p2/v1/input/p2_v1_00001.png",
            "CADICA/nonselectedVideos/p2/v2/input/p2_v2_00001.png",
        ],
        "is_selected_keyframe": [True, True, True, False],
        "has_lesion_video": [True, True, True, False],
        "has_groundtruth": [True, True, True, False],
        "image_width": [512, 512, 512, 512],
        "image_height": [512, 512, 512, 512],
    }
    pd.DataFrame(frames_data).to_csv(index_dir / "frames_index.csv", index=False)

    # 2. annotations.csv (with an intentional imbalance: p0_20 has 10, p99 has 1)
    ann_data = {
        "patient_id": ["p1"] * 6 + ["p2"] * 5,
        "video_id": ["v1"] * 3 + ["v2"] * 3 + ["v1"] * 5,
        "frame_id": ["p1_v1_00001"] * 3 + ["p1_v2_00001"] * 3 + ["p2_v1_00001"] * 5,
        "frame_path": [
            "CADICA/selectedVideos/p1/v1/input/p1_v1_00001.png"
        ] * 3 + [
            "CADICA/selectedVideos/p1/v2/input/p1_v2_00001.png"
        ] * 3 + [
            "CADICA/selectedVideos/p2/v1/input/p2_v1_00001.png"
        ] * 5,
        "x": [100, 150, 200, 120, 180, 220, 50, 80, 110, 140, 300],
        "y": [100, 150, 200, 120, 180, 220, 50, 80, 110, 140, 300],
        "w": [50, 60, 40, 55, 65, 45, 10, 12, 35, 40, 15],  # some < 400 area
        "h": [50, 60, 40, 55, 65, 45, 10, 12, 35, 40, 15],
        "category": [
            "p0_20", "p0_20", "p0_20", "p0_20", "p0_20",
            "p0_20", "p0_20", "p0_20", "p0_20", "p0_20",
            "p99",  # 10 vs 1 => ratio 10.0 > 5.0
        ],
        "x_center": [0.24, 0.35, 0.43, 0.29, 0.41, 0.47, 0.11, 0.17, 0.25, 0.31, 0.60],
        "y_center": [0.24, 0.35, 0.43, 0.29, 0.41, 0.47, 0.11, 0.17, 0.25, 0.31, 0.60],
        "w_norm": [0.1, 0.12, 0.08, 0.11, 0.13, 0.09, 0.02, 0.02, 0.07, 0.08, 0.03],
        "h_norm": [0.1, 0.12, 0.08, 0.11, 0.13, 0.09, 0.02, 0.02, 0.07, 0.08, 0.03],
    }
    pd.DataFrame(ann_data).to_csv(index_dir / "annotations.csv", index=False)

    # 3. videos_index.csv
    videos_data = {
        "patient_id": ["p1", "p1", "p2", "p2"],
        "video_id": ["v1", "v2", "v1", "v2"],
        "split_type": ["selected", "selected", "selected", "nonselected"],
        "num_frames": [1, 1, 1, 1],
        "num_selected_frames": [1, 1, 1, 0],
        "is_lesion_video": [True, True, True, False],
        "num_annotated_frames": [1, 1, 1, 0],
        "num_boxes": [3, 3, 5, 0],
    }
    pd.DataFrame(videos_data).to_csv(index_dir / "videos_index.csv", index=False)

    # 4. patients_index.csv
    patients_data = {
        "patient_id": ["p1", "p2", "p3"],  # p3 has 0 boxes to test 0-lesion detection
        "num_videos": [2, 2, 1],
        "num_selected_videos": [2, 1, 1],
        "num_nonselected_videos": [0, 1, 0],
        "num_frames": [2, 2, 1],
        "num_lesion_videos": [2, 1, 0],
        "num_boxes": [6, 5, 0],
    }
    pd.DataFrame(patients_data).to_csv(index_dir / "patients_index.csv", index=False)

    # 5. metadata_clean.csv
    meta_data = {
        "patient_id": ["p1", "p2", "p3"],
        "Age": [55, 65, 45],
        "BMI": [24.5, 29.1, 22.0],
        "Smoker": [1, 0, 1],
        "Diabetes": [0, 1, 0],
    }
    pd.DataFrame(meta_data).to_csv(index_dir / "metadata_clean.csv", index=False)

    # 6. Create synthetic zip dataset with real 512x512 PNG images
    zip_path = tmp_path / "CADICA.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for pth in [
            "CADICA/selectedVideos/p1/v1/input/p1_v1_00001.png",
            "CADICA/selectedVideos/p1/v2/input/p1_v2_00001.png",
            "CADICA/selectedVideos/p2/v1/input/p2_v1_00001.png",
            "CADICA/nonselectedVideos/p2/v2/input/p2_v2_00001.png",
        ]:
            arr = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            zf.writestr(pth, buf.getvalue())

    # 7. config.yaml
    cfg = {
        "dataset_root": str(zip_path),
        "metadata_file": "metadata.xlsx",
        "folders": {"selected": "selectedVideos", "nonselected": "nonselectedVideos"},
        "index_dir": str(index_dir),
        "reports_dir": str(reports_dir),
        "outputs_dir": str(outputs_dir),
        "random_seed": 42,
        "image_extension": ".png",
        "logging": {"level": "DEBUG"},
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    return config_file


def test_natural_sort_key():
    """Verify natural patient sort key helper."""
    assert _natural_patient_sort_key("p1") == 1
    assert _natural_patient_sort_key("p10") == 10
    assert _natural_patient_sort_key("p2") == 2
    assert sorted(["p10", "p2", "p1"], key=_natural_patient_sort_key) == ["p1", "p2", "p10"]
    assert _natural_patient_sort_key("unknown") == 999999


def test_eda_initialization(synthetic_eda_env: Path):
    """Verify DatasetEDA properly loads configs and paths."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    assert eda.index_dir.is_dir()
    assert eda.figures_dir.is_dir()
    assert eda.random_seed == 42


def test_eda_lazy_loading(synthetic_eda_env: Path):
    """Verify DataFrame properties load from disk upon first access."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    assert len(eda.frames) == 4
    assert len(eda.annotations) == 11
    assert len(eda.videos) == 4
    assert len(eda.patients) == 3
    assert eda.metadata is not None
    assert len(eda.metadata) == 3


def test_plot_per_patient_stats(synthetic_eda_env: Path):
    """Verify per-patient bar chart creation."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_per_patient_stats()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_class_distribution(synthetic_eda_env: Path):
    """Verify class distribution bar chart, stats table, and imbalance warning."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path, summary_df, warnings = eda.plot_class_distribution()

    assert out_path.is_file()
    assert len(summary_df) == 2
    assert "p0_20" in summary_df["category"].values
    assert "p99" in summary_df["category"].values
    # Check that imbalance warning was triggered (10 vs 1 = ratio 10.0 > 5.0)
    assert len(warnings) == 1
    assert "Class imbalance detected" in warnings[0]


def test_plot_bbox_statistics(synthetic_eda_env: Path):
    """Verify bbox geometry plots generation."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_bbox_statistics()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_resolution_distribution(synthetic_eda_env: Path):
    """Verify resolution distribution bar chart."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_resolution_distribution()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_lesion_selected_splits(synthetic_eda_env: Path):
    """Verify dataset split pie/bar charts."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_lesion_selected_splits()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_boxes_per_frame(synthetic_eda_env: Path):
    """Verify boxes per frame histogram generation."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_boxes_per_frame()
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_sample_annotations(synthetic_eda_env: Path):
    """Verify sample annotation grid draws boxes on image frames."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_sample_annotations(n_samples=3)
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_metadata_correlations(synthetic_eda_env: Path):
    """Verify correlation heatmap generation."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    out_path = eda.plot_metadata_correlations()
    assert out_path is not None
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_plot_metadata_correlations_missing(synthetic_eda_env: Path):
    """Verify graceful handling when metadata is missing."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    # Remove metadata file
    meta_file = eda.index_dir / "metadata_clean.csv"
    if meta_file.exists():
        meta_file.unlink()
    eda._metadata = None

    out_path = eda.plot_metadata_correlations()
    assert out_path is None


def test_full_eda_run_pipeline(synthetic_eda_env: Path):
    """Verify end-to-end EDA execution generating all figures and report."""
    eda = DatasetEDA(config_path=synthetic_eda_env)
    results = eda.run()

    # Check figures dictionary
    assert "per_patient" in results["figure_paths"]
    assert "class_dist" in results["figure_paths"]
    assert "bbox_stats" in results["figure_paths"]
    assert "resolution" in results["figure_paths"]
    assert "splits" in results["figure_paths"]
    assert "boxes_per_frame" in results["figure_paths"]
    assert "samples" in results["figure_paths"]
    assert "correlations" in results["figure_paths"]

    # Verify report file exists and contains essential sections
    report_path = results["report_path"]
    assert report_path.is_file()
    content = report_path.read_text(encoding="utf-8")
    assert "# CADICA Dataset — Exploratory Data Analysis Report" in content
    assert "## Key Findings & Potential Issues" in content
    assert "Class imbalance detected" in content
    assert "Tiny boxes" in content
    assert "Patients with zero bounding boxes" in content
    assert "p3" in content
    assert "## 1. Per-Patient Statistics" in content
    assert "## 2. Lesion Category Distribution" in content
    assert "## 3. Bounding Box Geometry Statistics" in content
    assert "## 7. Sample Annotated Frames" in content
