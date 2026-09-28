"""Unit tests for configuration loader, logger, and reproducibility utilities."""

import logging
from pathlib import Path
import random
import numpy as np
import pytest

from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed


@pytest.fixture
def valid_config_yaml(tmp_path: Path) -> Path:
    """Fixture providing a temporary valid YAML config file."""
    config_file = tmp_path / "config.yaml"
    content = """
dataset_root: "PATH/TO/CADICA"
metadata_file: "metadata.xlsx"
folders:
  selected: "selectedVideos"
  nonselected: "nonselectedVideos"
index_dir: "data_index"
reports_dir: "reports"
outputs_dir: "outputs"
random_seed: 42
image_extension: ".png"
"""
    config_file.write_text(content, encoding="utf-8")
    return config_file


def test_load_valid_config(valid_config_yaml: Path) -> None:
    """Test loading a valid YAML config file with required keys."""
    cfg = load_config(valid_config_yaml)
    assert cfg["dataset_root"] == "PATH/TO/CADICA"
    assert cfg["metadata_file"] == "metadata.xlsx"
    assert cfg["folders"]["selected"] == "selectedVideos"
    assert cfg["folders"]["nonselected"] == "nonselectedVideos"
    assert cfg["random_seed"] == 42
    assert cfg["image_extension"] == ".png"


def test_load_missing_required_key(tmp_path: Path) -> None:
    """Test loading config missing a required key raises ValueError."""
    config_file = tmp_path / "invalid_config.yaml"
    config_file.write_text("dataset_root: 'test'\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required configuration key"):
        load_config(config_file)


def test_load_missing_folder_keys(tmp_path: Path) -> None:
    """Test loading config missing nested folder keys raises ValueError."""
    config_file = tmp_path / "invalid_folders.yaml"
    content = """
dataset_root: "test"
metadata_file: "metadata.xlsx"
folders:
  selected: "selectedVideos"
index_dir: "data_index"
reports_dir: "reports"
outputs_dir: "outputs"
random_seed: 42
image_extension: ".png"
"""
    config_file.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required key"):
        load_config(config_file)


def test_get_logger(tmp_path: Path) -> None:
    """Test get_logger helper logs to console and to file."""
    log_file = tmp_path / "outputs" / "logs" / "run.log"
    logger = get_logger(name="TestLogger", log_file=log_file, level="DEBUG")

    assert logger.name == "TestLogger"
    logger.info("Logger initialization test")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Logger initialization test" in content


def test_set_seed() -> None:
    """Test set_seed produces deterministic pseudo-random sequences."""
    set_seed(123)
    val_py1 = random.randint(0, 10000)
    val_np1 = np.random.rand()

    set_seed(123)
    val_py2 = random.randint(0, 10000)
    val_np2 = np.random.rand()

    assert val_py1 == val_py2
    assert val_np1 == val_np2

    try:
        import torch
        set_seed(456)
        t1 = torch.randn(5)
        set_seed(456)
        t2 = torch.randn(5)
        assert torch.equal(t1, t2)
    except ImportError:
        pass
