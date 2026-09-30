"""CLI script to generate patient-wise train/val/test and 5-fold cross-validation splits.

Ensures zero patient overlap (no patient appears in multiple splits), balanced class distributions
(specifically for rare classes p100 and p99), and saves split tables and summary report in data_index/splits/.

Usage:
    python scripts/make_splits.py
    python -m scripts.make_splits --seed 42 --n-folds 5
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splits import PatientSplitter


def main() -> None:
    """CLI entry point for generating patient-wise splits."""
    parser = argparse.ArgumentParser(
        description="Agent-TT: Generate stratified patient-wise train/val/test and K-fold CV splits."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Override path to directory containing index CSVs (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override path to output splits directory (default: data_index/splits)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed for deterministic split search (default: from config)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target proportion for training split (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target proportion for validation split (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Target proportion for test split (default: 0.15)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of patient-wise cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10000,
        help="Maximum search iterations for optimal split (default: 10000)",
    )

    args = parser.parse_args()

    splitter = PatientSplitter(
        config_path=args.config,
        index_dir=args.index_dir,
        random_seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_folds=args.n_folds,
        max_search_iter=args.max_iter,
    )

    print("============================================================")
    print("  Agent-TT — Generating Patient-Wise Splits (Phase 2.1)     ")
    print("============================================================")

    split_path, folds_path, report_path = splitter.run_all(
        output_dir=args.output_dir,
        seed=args.seed,
    )

    print("\n------------------------------------------------------------")
    print("  Splits Generated Successfully!")
    print("------------------------------------------------------------")
    print(f"  [Train/Val/Test] -> {split_path}")
    print(f"  [{args.n_folds}-Fold CV]     -> {folds_path}")
    print(f"  [Report]         -> {report_path}")
    print("\nCheck the Markdown report for per-split class distributions & validation.")


if __name__ == "__main__":
    main()
