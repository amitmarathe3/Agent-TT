#!/usr/bin/env python3
"""Run baseline YOLO experiments and cross‑validation.

This script orchestrates:
1. Baseline runs:
   - A. binary scheme, yolo11s
   - B. severity3 scheme, yolo11s
   - C. 7class scheme, yolo11s (with and without rare‑class oversampling)
   - D. best scheme (determined after baselines) with yolo11n and yolo11m
2. 5‑fold patient‑wise cross‑validation for the best configuration.

Each experiment is executed via :pyfunc:`src.models.experiment_runner.run_single_experiment`
which handles training, prediction, evaluation and leaderboard persistence.
The script is resumable – already‑completed runs are skipped automatically.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.models.experiment_runner import (
    run_single_experiment,
    load_leaderboard,
    generate_cv_summary,
)

logger = logging.getLogger("run_experiments")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_experiments."""
    parser = argparse.ArgumentParser(
        description="Run CADICA baseline experiments and 5-fold cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--matrix", action="store_true",
        help="Run baseline experiment matrix.",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold cross-validation.",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Fast verification mode.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing leaderboard entries.",
    )
    parser.add_argument(
        "--best-run", type=str, default="yolo11s_7class_os",
        help="Run name for best configuration.",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for single run.",
    )
    parser.add_argument(
        "--scheme", type=str, default="7class", choices=["7class", "binary", "severity3"],
        help="Class scheme.",
    )
    parser.add_argument(
        "--oversample-rare", action="store_true",
        help="Oversample rare classes.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true",
        help="Disable bootstrap CI calculations.",
    )
    parser.add_argument(
        "--runs", nargs="*", default=None,
        help="Filter specific runs to execute.",
    )
    return parser.parse_args(argv)




def baseline_runs() -> List[Dict[str, Any]]:
    """Execute the baseline experiment matrix.

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries from each run (as returned by ``run_single_experiment``).
    """
    runs_cfg = []

    # A. binary scheme, yolo11s
    runs_cfg.append({"scheme": "binary", "model": "yolo11s.pt", "oversample": False})
    # B. severity3 scheme, yolo11s
    runs_cfg.append({"scheme": "severity3", "model": "yolo11s.pt", "oversample": False})
    # C. 7class scheme, yolo11s – without oversampling
    runs_cfg.append({"scheme": "7class", "model": "yolo11s.pt", "oversample": False})
    # C. 7class scheme, yolo11s – with oversampling
    runs_cfg.append({"scheme": "7class", "model": "yolo11s.pt", "oversample": True})

    results = []
    for cfg in runs_cfg:
        run_name = f"{cfg['scheme']}_{cfg['model'].replace('.pt', '')}"
        if cfg["oversample"]:
            run_name += "_oversample"
        logger.info("Starting baseline run: %s", run_name)
        res = run_single_experiment(
            run_name=run_name,
            scheme=cfg["scheme"],
            model=cfg["model"],
            imgsz=640,
            oversample=cfg["oversample"],
            fold=None,
            epochs=100,
            smoke_test=False,
            force=False,
        )
        results.append(res)
    return results


def select_best_config(leaderboard_path: Path) -> Dict[str, Any]:
    """Pick the best configuration based on validation F1.

    Parameters
    ----------
    leaderboard_path: Path
        Path to ``leaderboard.csv``.

    Returns
    -------
    Dict[str, Any]
        Mapping containing ``scheme``, ``model`` and ``oversample`` of the best run.
    """
    df = load_leaderboard(leaderboard_path.parent)
    if df.empty:
        raise RuntimeError("Leaderboard is empty – cannot select best config.")
    # Prefer higher validation F1; tie‑breaker: higher mAP50.
    best_row = df.sort_values(["val_f1", "val_mAP50"], ascending=False).iloc[0]
    return {
        "scheme": best_row["scheme"],
        "model": best_row["model"],
        "oversample": bool(best_row["oversample"]),
    }


def cross_validation(best_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run 5‑fold CV for the supplied configuration.

    Returns
    -------
    List[Dict[str, Any]]
        Result dicts for each fold.
    """
    cv_results = []
    for fold in range(5):
        run_name = f"{best_cfg['scheme']}_{best_cfg['model'].replace('.pt', '')}_cv_fold{fold}"
        if best_cfg["oversample"]:
            run_name += "_oversample"
        logger.info("CV fold %d – run: %s", fold, run_name)
        res = run_single_experiment(
            run_name=run_name,
            scheme=best_cfg["scheme"],
            model=best_cfg["model"],
            imgsz=640,
            oversample=best_cfg["oversample"],
            fold=fold,
            epochs=100,
            smoke_test=False,
            force=False,
        )
        cv_results.append(res)
    return cv_results


def main():
    # 1️⃣ Baseline matrix
    baseline_runs()

    # 2️⃣ Determine best config from leaderboard
    leaderboard_path = Path("outputs/experiments/leaderboard.csv")
    best_cfg = select_best_config(leaderboard_path)
    logger.info("Best configuration selected: %s", best_cfg)

    # 3️⃣ 5‑fold CV for best config
    cv_results = cross_validation(best_cfg)

    # 4️⃣ Generate CV summary markdown
    cv_md_path = Path("outputs/experiments/cv_summary.md")
    generate_cv_summary(cv_results, output_path=cv_md_path)
    logger.info("Cross‑validation summary written to %s", cv_md_path)


if __name__ == "__main__":
    main()
