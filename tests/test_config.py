"""Unit tests for configuration loader and logging utilities."""

from pathlib import Path
import pytest
from src.utils.config import load_config, setup_logging


def test_load_valid_config(tmp_path: Path) -> None:
    """Test loading a valid YAML config file."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("dataset:\n  zip_path: 'test.zip'\nlogging:\n  level: 'DEBUG'\n", encoding="utf-8")

    cfg = load_config(config_file)
    assert "dataset" in cfg
    assert cfg["dataset"]["zip_path"] == "test.zip"
    assert cfg["logging"]["level"] == "DEBUG"


def test_load_nonexistent_config() -> None:
    """Test loading a non-existent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_file_path_xyz.yaml")


def test_setup_logging(tmp_path: Path) -> None:
    """Test logger initialization and file handler creation."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", log_file=log_file)

    assert logger.name == "Agent-TT"
    logger.debug("Test debug message")

    assert log_file.exists()
    assert "Test debug message" in log_file.read_text(encoding="utf-8")
