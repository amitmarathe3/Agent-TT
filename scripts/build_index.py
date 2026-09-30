"""CLI script to scan CADICA dataset and build CSV index tables in data_index/."""

import argparse
from pathlib import Path

from src.data.indexer import DatasetIndexer


def main() -> None:
    """Run CLI dataset indexer."""
    parser = argparse.ArgumentParser(description="Agent-TT CADICA Dataset Indexer CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file")
    parser.add_argument("--dataset_root", type=str, default=None, help="Override dataset root path or zip archive")
    parser.add_argument("--limit-patients", "--limit_patients", type=int, default=None, help="Limit number of patient folders for quick indexing")

    args = parser.parse_args()

    indexer = DatasetIndexer(config_path=args.config)
    output_paths = indexer.build_index(
        dataset_path_override=args.dataset_root,
        limit_patients=args.limit_patients
    )

    print("==================================================")
    print("        AGENT-TT DATASET INDEXING COMPLETE        ")
    print("==================================================")
    for index_name, path in output_paths.items():
        print(f"  - {index_name:15s}: {path.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
