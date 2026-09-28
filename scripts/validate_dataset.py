"""CLI script to run dataset validation for CADICA angiography dataset."""

import argparse
from pathlib import Path
from src.data.validator import DatasetValidator


def main() -> None:
    """Run CLI dataset validator."""
    parser = argparse.ArgumentParser(description="Agent-TT CADICA Dataset Validator CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset root path or zip archive")
    parser.add_argument("--output", type=str, default=None, help="Override path to write validation report file")

    args = parser.parse_args()

    validator = DatasetValidator(config_path=args.config)
    results = validator.validate(dataset_path_override=args.dataset_root)
    report = validator.generate_report(results, output_path=args.output)

    # Print report to console
    print(report)


if __name__ == "__main__":
    main()
