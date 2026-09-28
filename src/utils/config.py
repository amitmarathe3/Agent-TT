"""Configuration loader and validation module for Agent-TT."""

from pathlib import Path
from typing import Any
import yaml

REQUIRED_KEYS = [
    "dataset_root",
    "metadata_file",
    "folders",
    "index_dir",
    "reports_dir",
    "outputs_dir",
    "random_seed",
    "image_extension",
]

REQUIRED_FOLDER_KEYS = ["selected", "nonselected"]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load configuration dictionary from a YAML file and validate required keys.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing validated configuration parameters.

    Raises:
        FileNotFoundError: If the config_path does not exist.
        ValueError: If YAML parsing fails or required keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    try:
        with open(path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML configuration file at {path}: {exc}") from exc

    # Validate top-level required keys
    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration key(s): {', '.join(missing_keys)}")

    # Validate nested folders structure
    folders = config.get("folders")
    if not isinstance(folders, dict):
        raise ValueError("Configuration key 'folders' must be a dictionary.")

    missing_folders = [fk for fk in REQUIRED_FOLDER_KEYS if fk not in folders]
    if missing_folders:
        raise ValueError(f"Missing required key(s) in 'folders': {', '.join(missing_folders)}")

    return config
