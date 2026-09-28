"""Unit tests for dataset scanning functionality."""

import json
from pathlib import Path
import zipfile
import pytest
from src.data.dataset_scanner import DatasetScanner


@pytest.fixture
def mock_zip_dataset(tmp_path: Path) -> Path:
    """Fixture creating a temporary zip archive with dummy image and annotation files."""
    zip_path = tmp_path / "mock_dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("patient_01/image_01.png", b"fake_image_bytes")
        zf.writestr("patient_01/image_02.dcm", b"fake_dicom_bytes")
        zf.writestr("patient_01/annotation.json", json.dumps({"lesion": "CAD"}).encode("utf-8"))
        zf.writestr("patient_02/notes.txt", b"some notes")
    return zip_path


@pytest.fixture
def mock_config(tmp_path: Path, mock_zip_dataset: Path) -> Path:
    """Fixture creating a temporary configuration file pointing to mock zip dataset."""
    config_path = tmp_path / "config.yaml"
    content = f"""
dataset:
  zip_path: "{mock_zip_dataset.as_posix()}"
  scan_output_path: "{(tmp_path / 'summary.json').as_posix()}"
logging:
  level: "DEBUG"
"""
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_scan_zip(mock_config: Path, mock_zip_dataset: Path, tmp_path: Path) -> None:
    """Test zip archive scanner structure discovery."""
    scanner = DatasetScanner(config_path=mock_config)
    summary = scanner.scan_zip(mock_zip_dataset)

    assert summary["source_type"] == "zip"
    assert summary["total_files"] == 4
    assert summary["image_count"] == 2
    assert summary["annotation_count"] == 2
    assert ".png" in summary["extension_counts"]
    assert ".dcm" in summary["extension_counts"]


def test_scan_directory(mock_config: Path, tmp_path: Path) -> None:
    """Test directory scanner structure discovery."""
    dir_path = tmp_path / "data_dir"
    dir_path.mkdir()
    (dir_path / "img.png").write_bytes(b"data")
    (dir_path / "label.csv").write_text("id,label\n1,cad", encoding="utf-8")

    scanner = DatasetScanner(config_path=mock_config)
    summary = scanner.scan_directory(dir_path)

    assert summary["source_type"] == "directory"
    assert summary["total_files"] == 2
    assert summary["image_count"] == 1
    assert summary["annotation_count"] == 1


def test_save_summary(mock_config: Path, mock_zip_dataset: Path, tmp_path: Path) -> None:
    """Test saving summary JSON output."""
    scanner = DatasetScanner(config_path=mock_config)
    summary = scanner.scan_zip(mock_zip_dataset)

    out_file = tmp_path / "output_summary.json"
    written_path = scanner.save_summary(summary, output_path=out_file)

    assert written_path.exists()
    data = json.loads(written_path.read_text(encoding="utf-8"))
    assert data["total_files"] == 4
