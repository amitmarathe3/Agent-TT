"""CLI script to run Exploratory Data Analysis on the CADICA dataset.

Reads pre-built CSV index tables from ``data_index/`` and generates
PNG figures in ``reports/figures/`` plus an integrated Markdown report
at ``reports/eda_summary.md``.

Usage examples:
    # Run full EDA with default config
    python -m scripts.run_eda

    # Use a custom config file
    python -m scripts.run_eda --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.eda import DatasetEDA


def main() -> None:
    """CLI entry point for running CADICA dataset EDA."""
    parser = argparse.ArgumentParser(
        description="Agent-TT: Run Exploratory Data Analysis on CADICA dataset indexes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_eda
  python -m scripts.run_eda --config configs/config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Agent-TT — CADICA Exploratory Data Analysis")
    print("=" * 60)
    print()

    eda = DatasetEDA(config_path=args.config)
    results = eda.run()

    print()
    print("-" * 60)
    print("  EDA Complete!")
    print("-" * 60)
    print()

    # Print generated figures
    print("Generated Figures:")
    for name, path in results["figure_paths"].items():
        status = f"  {path}" if path else "  (skipped)"
        print(f"  [{name}] {status}")
    print()

    # Print key findings
    print("Key Findings:")
    for f in results["findings"]:
        # Strip markdown bold markers for console
        clean = f.replace("**", "").replace("`", "")
        print(f"  - {clean}")
    print()

    print(f"Summary report: {results['report_path']}")
    print()


if __name__ == "__main__":
    main()
