"""Experiment leaderboard management and cross-validation summary utilities.

Handles reading/writing leaderboard.csv, extracting metrics from evaluation
reports, and computing mean ± std CV summaries with class-aware p100 handling.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("experiment_manager")

# Canonical leaderboard columns
LEADERBOARD_COLUMNS = [
    "run_name",
    "scheme",
    "model",
    "imgsz",
    "oversample",
    "fold",
    "train_minutes",
    # Val metrics
    "val_P",
    "val_R",
    "val_F1",
    "val_mAP50",
    "val_mAP5095",
    "val_frame_sensitivity",
    "val_frame_specificity",
    "val_roc_auc",
    # Test metrics
    "test_P",
    "test_R",
    "test_F1",
    "test_mAP50",
    "test_mAP5095",
    "test_frame_sensitivity",
    "test_frame_specificity",
    "test_roc_auc",
]


def load_leaderboard(leaderboard_path):
    """Load existing leaderboard CSV or create empty DataFrame."""
    path = Path(leaderboard_path)
    if path.exists():
        df = pd.read_csv(path, dtype={"fold": "object"})
        for col in LEADERBOARD_COLUMNS:
            if col not in df.columns:
                df[col] = float("nan")
        return df[LEADERBOARD_COLUMNS]
    return pd.DataFrame(columns=LEADERBOARD_COLUMNS)


def save_leaderboard(df, leaderboard_path):
    """Save leaderboard DataFrame to CSV."""
    path = Path(leaderboard_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Leaderboard saved to %s (%d rows)", path, len(df))


def row_exists(df, run_name):
    """Check whether a run_name is already in the leaderboard."""
    return run_name in df["run_name"].values


def extract_metrics_from_eval(metrics_json_path):
    """Extract key scalar metrics from a metrics.json produced by evaluate_split."""
    path = Path(metrics_json_path)
    if not path.exists():
        logger.warning("metrics.json not found: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return {}

    out = {}
    det = m.get("detection", {})
    micro = det.get("micro", {})
    out["P"] = float(micro.get("precision", float("nan")))
    out["R"] = float(micro.get("recall", float("nan")))
    out["F1"] = float(micro.get("f1", float("nan")))
    map_m = m.get("map", {})
    out["mAP50"] = float(map_m.get("mAP_50", float("nan")))
    out["mAP5095"] = float(map_m.get("mAP_50_95", float("nan")))
    frame = m.get("frame", {})
    out["frame_sensitivity"] = float(frame.get("sensitivity", float("nan")))
    out["frame_specificity"] = float(frame.get("specificity", float("nan")))
    out["roc_auc"] = float(frame.get("roc_auc", float("nan")))
    return out


def append_run_to_leaderboard(
    leaderboard_path,
    run_name,
    scheme,
    model,
    imgsz,
    oversample,
    fold,
    train_minutes,
    val_metrics_json,
    test_metrics_json=None,
    overwrite_existing=False,
):
    """Read leaderboard, append or update run row, save, and return updated DataFrame."""
    df = load_leaderboard(leaderboard_path)
    if row_exists(df, run_name):
        if overwrite_existing:
            df = df[df["run_name"] != run_name].reset_index(drop=True)
            logger.info("Overwriting existing leaderboard entry for '%s'", run_name)
        else:
            logger.info("Run '%s' already in leaderboard. Skipping.", run_name)
            return df

    val_m = extract_metrics_from_eval(val_metrics_json)
    test_m = extract_metrics_from_eval(test_metrics_json) if test_metrics_json else {}

    row = {
        "run_name": run_name,
        "scheme": scheme,
        "model": model,
        "imgsz": imgsz,
        "oversample": oversample,
        "fold": "" if fold is None else str(fold),
        "train_minutes": round(train_minutes, 2),
        "val_P": round(val_m.get("P", float("nan")), 4),
        "val_R": round(val_m.get("R", float("nan")), 4),
        "val_F1": round(val_m.get("F1", float("nan")), 4),
        "val_mAP50": round(val_m.get("mAP50", float("nan")), 4),
        "val_mAP5095": round(val_m.get("mAP5095", float("nan")), 4),
        "val_frame_sensitivity": round(val_m.get("frame_sensitivity", float("nan")), 4),
        "val_frame_specificity": round(val_m.get("frame_specificity", float("nan")), 4),
        "val_roc_auc": round(val_m.get("roc_auc", float("nan")), 4),
        "test_P": round(test_m.get("P", float("nan")), 4),
        "test_R": round(test_m.get("R", float("nan")), 4),
        "test_F1": round(test_m.get("F1", float("nan")), 4),
        "test_mAP50": round(test_m.get("mAP50", float("nan")), 4),
        "test_mAP5095": round(test_m.get("mAP5095", float("nan")), 4),
        "test_frame_sensitivity": round(test_m.get("frame_sensitivity", float("nan")), 4),
        "test_frame_specificity": round(test_m.get("frame_specificity", float("nan")), 4),
        "test_roc_auc": round(test_m.get("roc_auc", float("nan")), 4),
    }

    new_row_df = pd.DataFrame([row])[LEADERBOARD_COLUMNS]
    df = pd.concat([df, new_row_df], ignore_index=True)
    save_leaderboard(df, leaderboard_path)
    logger.info("Appended run '%s' to leaderboard.", run_name)
    return df


def compute_cv_summary(
    leaderboard_path,
    base_run_name,
    n_folds=5,
    folds_with_p100=None,
    output_md=None,
):
    """Compute mean +/- std over CV folds from leaderboard, noting p100-absent folds."""
    if folds_with_p100 is None:
        folds_with_p100 = [0, 1, 2]  # Folds 3 and 4 have no p100

    df = load_leaderboard(leaderboard_path)
    fold_rows = df[df["run_name"].str.startswith(base_run_name) & (df["fold"] != "")].copy()
    fold_rows["fold_int"] = pd.to_numeric(fold_rows["fold"], errors="coerce").astype("Int64")

    if len(fold_rows) == 0:
        logger.warning("No CV fold rows found for base_run_name='%s'", base_run_name)
        return {}

    metric_cols = [c for c in LEADERBOARD_COLUMNS if c.startswith("val_") or c.startswith("test_")]
    summary = {"base_run_name": base_run_name, "n_folds_found": len(fold_rows)}

    for col in metric_cols:
        vals = pd.to_numeric(fold_rows[col], errors="coerce").dropna().values
        if len(vals) == 0:
            summary[f"{col}_mean"] = float("nan")
            summary[f"{col}_std"] = float("nan")
        else:
            summary[f"{col}_mean"] = float(round(float(np.mean(vals)), 4))
            summary[f"{col}_std"] = float(round(float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0), 4))

    if output_md:
        _write_cv_summary_md(
            summary=summary,
            fold_rows=fold_rows,
            metric_cols=metric_cols,
            folds_with_p100=folds_with_p100,
            output_path=Path(output_md),
            base_run_name=base_run_name,
        )

    return summary


def _write_cv_summary_md(
    summary,
    fold_rows,
    metric_cols,
    folds_with_p100,
    output_path,
    base_run_name,
):
    """Write CV summary to a Markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_folds = len(fold_rows)
    folds_missing_p100 = [i for i in range(5) if i not in folds_with_p100]

    lines = [
        "# CADICA YOLO — 5-Fold Patient-Wise Cross-Validation Summary",
        "",
        f"**Base Run**: {base_run_name}  ",
        f"**Folds Found**: {n_folds} / 5  ",
        f"**Note on p100 class**: Folds {folds_missing_p100} contain **no p100 boxes** (patients with "
        f"total occlusion are only in folds {folds_with_p100}). Class-specific p100 metrics are averaged "
        "only over folds that contain it and this is noted explicitly below.",
        "",
        "---",
        "",
        "## Per-Fold Leaderboard Rows",
        "",
    ]

    display_cols = ["run_name", "fold"] + metric_cols[:8]
    avail = [c for c in display_cols if c in fold_rows.columns]
    lines.append(fold_rows[avail].to_markdown(index=False))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Mean +/- Std across All Folds")
    lines.append("")

    split_groups = [
        ("Val", [c for c in metric_cols if c.startswith("val_")]),
        ("Test", [c for c in metric_cols if c.startswith("test_")]),
    ]
    for split_label, cols in split_groups:
        lines.append(f"### {split_label} Metrics")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        for col in cols:
            label = col.replace("val_", "").replace("test_", "")
            mean_val = summary.get(f"{col}_mean", float("nan"))
            std_val = summary.get(f"{col}_std", float("nan"))
            mean_str = f"{mean_val:.4f}" if not math.isnan(mean_val) else "N/A"
            std_str = f"{std_val:.4f}" if not math.isnan(std_val) else "N/A"
            lines.append(f"| {label} | {mean_str} | {std_str} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        f"> **Note**: Folds {folds_missing_p100} (old_3 and old_4) have zero p100 bounding boxes.",
        "> Per-class p100 AP values for these folds will be 0.0 or NaN by construction, so any",
        f"> p100-specific reporting should average only over folds {folds_with_p100}.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("CV summary written to %s", output_path)
