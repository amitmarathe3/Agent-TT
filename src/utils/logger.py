"""Centralized logger helper utility for Agent-TT."""

import logging
from pathlib import Path


def get_logger(
    name: str = "Agent-TT",
    log_file: str | Path = "outputs/logs/run.log",
    level: str = "INFO"
) -> logging.Logger:
    """Create and configure a logger instance that outputs to console and a log file.

    Args:
        name: Name of the logger instance.
        log_file: File path for output logs (defaults to outputs/logs/run.log).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger instance.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if get_logger is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    file_path = Path(log_file)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
