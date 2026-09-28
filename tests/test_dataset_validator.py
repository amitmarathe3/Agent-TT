"""Unit tests for dataset validator module using synthetic dataset fixtures."""

from pathlib import Path
import pytest

from src.data.validator import DatasetValidator


@pytest.fixture
def synthetic_cadica_dir(tmp_path: Path) -> Path:
    """Fixture creating a valid synthetic CADICA dataset structure."""
    dataset_root = tmp_path / "synthetic_CADICA"
    dataset_root.mkdir()

    # 1. metadata.xlsx
    (dataset_root / "metadata.xlsx").write_bytes(b"dummy metadata excel content")

    # 2. selectedVideos
    sel_dir = dataset_root / "selectedVideos" / "p1" / "v1"
    sel_input = sel_dir / "input"
    sel_input.mkdir(parents=True)
    sel_gt = sel_dir / "groundtruth"
    sel_gt.mkdir(parents=True)

    # Frame images
    (sel_input / "p1_v1_00001.png").write_bytes(b"fake image data 1")
    (sel_input / "p1_v1_00002.png").write_bytes(b"fake image data 2")

    # Bounding box groundtruth annotation for frame 00001
    (sel_gt / "p1_v1_00001.txt").write_text("10 20 30 40\n", encoding="utf-8")

    # Control files
    p1_dir = dataset_root / "selectedVideos" / "p1"
    (p1_dir / "lesionVideos.txt").write_text("v1\n", encoding="utf-8")
    (p1_dir / "nonlesionVideos.txt").write_text("", encoding="utf-8")
    (sel_dir / "p1_v1_selectedFrames.txt").write_text("p1_v1_00001\np1_v1_00002\n", encoding="utf-8")

    # 3. nonselectedVideos
    non_dir = dataset_root / "nonselectedVideos" / "p1" / "v2" / "input"
    non_dir.mkdir(parents=True)
    (non_dir / "p1_v2_00001.png").write_bytes(b"fake nonselected frame")

    return dataset_root


@pytest.fixture
def validator_config(tmp_path: Path, synthetic_cadica_dir: Path) -> Path:
    """Fixture creating a configuration YAML file pointing to synthetic dataset."""
    config_file = tmp_path / "config.yaml"
    content = f"""
dataset_root: "{synthetic_cadica_dir.as_posix()}"
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


def test_validate_synthetic_pass(validator_config: Path) -> None:
    """Test validating a valid synthetic dataset returns PASS status."""
    validator = DatasetValidator(config_path=validator_config)
    results = validator.validate()

    assert results["status"] == "PASS"
    assert results["totals"]["patients"] == 1
    assert results["totals"]["videos"] == 2
    assert results["totals"]["frames"] == 3
    assert results["totals"]["groundtruth"] == 1
    assert len(results["inconsistencies"]) == 0


def test_validate_missing_metadata(validator_config: Path, synthetic_cadica_dir: Path) -> None:
    """Test validator detects missing metadata.xlsx file."""
    (synthetic_cadica_dir / "metadata.xlsx").unlink()

    validator = DatasetValidator(config_path=validator_config)
    results = validator.validate()

    assert results["status"] == "FAIL"
    assert any("Missing metadata file" in inc for inc in results["inconsistencies"])


def test_validate_orphaned_groundtruth(validator_config: Path, synthetic_cadica_dir: Path) -> None:
    """Test validator detects groundtruth text file with no matching input frame image."""
    orphaned_gt = synthetic_cadica_dir / "selectedVideos" / "p1" / "v1" / "groundtruth" / "p1_v1_99999.txt"
    orphaned_gt.write_text("5 5 10 10\n", encoding="utf-8")

    validator = DatasetValidator(config_path=validator_config)
    results = validator.validate()

    assert results["status"] == "FAIL"
    assert any("p1_v1_99999.txt has no matching frame" in inc for inc in results["inconsistencies"])


def test_generate_validation_report(validator_config: Path, tmp_path: Path) -> None:
    """Test report generation writes formatted text to output path."""
    validator = DatasetValidator(config_path=validator_config)
    results = validator.validate()

    report_path = tmp_path / "test_report.txt"
    report_text = validator.generate_report(results, output_path=report_path)

    assert report_path.exists()
    assert "AGENT-TT DATASET VALIDATION REPORT" in report_text
    assert "Overall Status: PASS" in report_text
