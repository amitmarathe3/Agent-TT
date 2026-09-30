"""CLI script to build the CADICA dataset index tables in data_index/."""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.indexer import DatasetIndexer


def main() -> None:
    """Entrypoint for build_index CLI."""
    parser = argparse.ArgumentParser(
        description="Agent-TT: Build CADICA dataset index tables (CSV + classes.json)."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--dataset_root", type=str, default=None,
        help="Override dataset root path from config (zip or directory)"
    )
    parser.add_argument(
        "--limit-patients", type=int, default=None, metavar="N",
        help="Process only the first N patients (for quick testing)"
    )
    args = parser.parse_args()

    indexer = DatasetIndexer(config_path=args.config)

    # Optionally override dataset root
    if args.dataset_root:
        indexer.dataset_root = Path(args.dataset_root)
        indexer._is_zip = (
            indexer.dataset_root.is_file()
            and indexer.dataset_root.suffix.lower() == ".zip"
        )

    tables = indexer.build(limit_patients=args.limit_patients)

    print("\n--- Index Build Complete ---")
    for name, df in tables.items():
        print(f"  {name}.csv : {len(df):,} rows")
    print(f"  classes.json written to {indexer.index_dir / 'classes.json'}")
    print()


if __name__ == "__main__":
    main()
