"""CLI script to inspect, clean, and cross-reference CADICA clinical metadata."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.metadata import MetadataInspector


def main() -> None:
    """CLI entry point for inspecting metadata."""
    parser = argparse.ArgumentParser(
        description="Agent-TT: Inspect CADICA clinical metadata, normalize patient IDs, and audit alignment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=None,
        help="Path to metadata.xlsx (standalone file, directory, or zip archive)",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset root path or zip archive from config",
    )
    parser.add_argument(
        "--output-summary",
        type=str,
        default=None,
        help="Override output path for metadata summary report text file",
    )
    parser.add_argument(
        "--output-clean",
        type=str,
        default=None,
        help="Override output path for cleaned metadata CSV file",
    )

    args = parser.parse_args()

    inspector = MetadataInspector(config_path=args.config)
    results = inspector.inspect(
        metadata_path_override=args.metadata_path,
        dataset_root_override=args.dataset_root,
        output_summary_path=args.output_summary,
        output_clean_path=args.output_clean,
    )

    # Print summary report to console
    print(results["summary_report"])


if __name__ == "__main__":
    main()
