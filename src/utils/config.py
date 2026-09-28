"""Configuration and logging utilities for Agent-TT."""

import logging
from pathlib import Path
from typing import Any
import yaml


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration dictionary from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        ValueError: If YAML parsing fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration file at {path}: {exc}") from exc

    return config


def setup_logging(
    level: str = "INFO",
    log_format: str | None = None,
    log_file: str | Path | None = None
) -> logging.Logger:
    """Configure logger with standard formatting.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Optional logging format string.
        log_file: Optional file path to log output to.

    Returns:
        Configured logging.Logger instance.
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("Agent-TT")
    logger.setLevel(numeric_level)

    # Clear existing handlers to prevent duplicated logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
