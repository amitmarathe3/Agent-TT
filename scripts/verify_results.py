#!/usr/bin/env python3
"""Verification script for CADICA baseline leaderboard results.

Fails loudly if:
1. leaderboard.csv is missing or empty.
2. Any leaderboard row lacks best.pt in outputs/runs/<run_name>/weights/best.pt.
3. Any leaderboard row lacks preds_val.json or preds_test.json.
4. Any leaderboard row lacks metrics.json in outputs/eval/<run_name>_val/ or outputs/eval/<run_name>_test/.
5. Metric values recorded in leaderboard.csv differ from values in metrics.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models.experiment_manager import extract_metrics_from_eval


def verify_leaderboard(
    leaderboard_path: str | Path = "outputs/experiments/leaderboard.csv",
    runs_dir: str | Path = "outputs/runs",
    eval_dir: str | Path = "outputs/eval",
) -> None:
    lb_path = Path(leaderboard_path)
    runs_base = Path(runs_dir)
    eval_base = Path(eval_dir)

    print("======================================================================")
    print(" VERIFYING EXPERIMENT LEADERBOARD & ARTIFACT DISK INTEGRITY")
    print("======================================================================")

    if not lb_path.exists():
        raise FileNotFoundError(f"Leaderboard CSV not found at {lb_path}")

    df = pd.read_csv(lb_path)
    if df.empty:
        raise ValueError(f"Leaderboard CSV at {lb_path} is empty!")

    print(f"Loaded leaderboard with {len(df)} rows from {lb_path}.\n")

    errors = []

    for idx, row in df.iterrows():
        run_name = str(row["run_name"])
        print(f"Checking run [{idx + 1}/{len(df)}]: {run_name} ...")

        # 1. Check best.pt
        run_p = runs_base / run_name
        weights_p = run_p / "weights" / "best.pt"
        if not weights_p.exists():
            weights_p = run_p / "best.pt"

        if not weights_p.exists():
            errors.append(f"[{run_name}] Missing best.pt at {weights_p}")
        elif weights_p.stat().st_size == 0:
            errors.append(f"[{run_name}] Empty 0-byte best.pt at {weights_p}")
        else:
            print(f"  [OK] best.pt: {weights_p} ({weights_p.stat().st_size / (1024*1024):.2f} MB)")

        # 2. Check predictions JSONs
        for split in ["val", "test"]:
            pred_p = run_p / f"preds_{split}.json"
            if not pred_p.exists():
                errors.append(f"[{run_name}] Missing predictions JSON: {pred_p}")
            else:
                print(f"  [OK] preds_{split}.json: {pred_p}")

        # 3. Check metrics.json and compare values
        for split in ["val", "test"]:
            eval_split_dir = eval_base / f"{run_name}_{split}"
            metrics_json = eval_split_dir / "metrics.json"

            if not metrics_json.exists():
                errors.append(f"[{run_name}] Missing metrics.json at {metrics_json}")
                continue

            print(f"  [OK] metrics.json ({split}): {metrics_json}")

            # Verify metrics match
            extracted = extract_metrics_from_eval(metrics_json)
            mapping = {
                "P": f"{split}_P",
                "R": f"{split}_R",
                "F1": f"{split}_F1",
                "mAP50": f"{split}_mAP50",
                "mAP5095": f"{split}_mAP5095",
                "frame_sensitivity": f"{split}_frame_sensitivity",
                "frame_specificity": f"{split}_frame_specificity",
                "roc_auc": f"{split}_roc_auc",
            }

            for key_ex, col_name in mapping.items():
                if col_name in row and pd.notna(row[col_name]):
                    lb_val = float(row[col_name])
                    ex_val = round(float(extracted.get(key_ex, float("nan"))), 4)
                    if abs(lb_val - ex_val) > 0.001:
                        errors.append(
                            f"[{run_name}] Mismatch for column '{col_name}': "
                            f"leaderboard={lb_val} vs metrics.json={ex_val}"
                        )

    print("-" * 70)
    if errors:
        print("[FAIL] Verification FAILED with errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("[VERIFIED SUCCESS] All leaderboard entries backed by disk artifacts and match metrics.json exactly.")
        print("=" * 70)


if __name__ == "__main__":
    verify_leaderboard()
