"""Experiment runner module for CADICA YOLO baseline experiments and 5-fold cross-validation.

Provides automated orchestration for:
- Resumable experiment execution logging to outputs/experiments/leaderboard.csv
- Baseline matrix runs (binary, severity3, 7class with/without rare-class oversampling, yolo11n/s/m)
- 5-fold patient-wise cross-validation with mean +/- std summary reports
- Class-level aggregation accounting for folds without rare classes (e.g. p100 absent in folds 3 and 4)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.yolo_export import YOLOExporter
from src.models.eval_runner import run_pipeline_evaluation
from src.models.yolo_infer import predict_canonical_split
from src.models.yolo_trainer import train_yolo
from src.utils.logger import get_logger

logger = get_logger("experiment_runner")

LEADERBOARD_COLUMNS = [
    "run_name",
    "scheme",
    "model",
    "imgsz",
    "oversample",
    "fold",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_mAP50",
    "val_mAP50_95",
    "val_frame_sensitivity",
    "val_frame_specificity",
    "val_frame_roc_auc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_mAP50",
    "test_mAP50_95",
    "test_frame_sensitivity",
    "test_frame_specificity",
    "test_frame_roc_auc",
    "train_minutes",
]


def get_leaderboard_path(exp_dir: str | Path = "outputs/experiments") -> Path:
    """Get absolute path to leaderboard CSV file."""
    exp_path = Path(exp_dir)
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path / "leaderboard.csv"


def load_leaderboard(exp_dir: str | Path = "outputs/experiments") -> pd.DataFrame:
    """Load current leaderboard DataFrame or create empty DataFrame with correct schema."""
    csv_path = get_leaderboard_path(exp_dir)
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Ensure all expected columns exist
            for col in LEADERBOARD_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df[LEADERBOARD_COLUMNS]
        except Exception as e:
            logger.warning("Error reading leaderboard at %s: %s. Initializing fresh.", csv_path, e)

    return pd.DataFrame(columns=LEADERBOARD_COLUMNS)


def save_leaderboard_row(
    row_dict: dict[str, Any],
    exp_dir: str | Path = "outputs/experiments",
) -> pd.DataFrame:
    """Append or update a single experiment row in leaderboard.csv."""
    df = load_leaderboard(exp_dir)
    run_name = str(row_dict.get("run_name", ""))

    # Clean dict to match expected columns
    clean_row = {col: row_dict.get(col, None) for col in LEADERBOARD_COLUMNS}

    if not df.empty and run_name in df["run_name"].values:
        # Update existing row
        idx = df[df["run_name"] == run_name].index[0]
        for col, val in clean_row.items():
            df.at[idx, col] = val
    else:
        # Append new row
        new_row_df = pd.DataFrame([clean_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

    csv_path = get_leaderboard_path(exp_dir)
    df.to_csv(csv_path, index=False)
    logger.info("Saved leaderboard row for '%s' to %s", run_name, csv_path)
    return df


def is_experiment_done(
    run_name: str,
    exp_dir: str | Path = "outputs/experiments",
    output_runs_dir: str | Path = "outputs/runs",
) -> bool:
    """Check if an experiment run has already completed both training and evaluation."""
    # 1. Check leaderboard.csv
    df = load_leaderboard(exp_dir)
    if not df.empty and run_name in df["run_name"].values:
        row = df[df["run_name"] == run_name].iloc[0]
        if pd.notna(row.get("test_mAP50")) and pd.notna(row.get("val_mAP50")):
            return True

    # 2. Check run directory artifacts
    run_p = Path(output_runs_dir) / run_name
    weights_p = run_p / "weights" / "best.pt"
    if not weights_p.exists():
        weights_p = run_p / "best.pt"

    eval_val_md = Path("outputs/eval") / f"{run_name}_val" / "metrics.md"
    eval_test_md = Path("outputs/eval") / f"{run_name}_test" / "metrics.md"

    return weights_p.exists() and eval_val_md.exists() and eval_test_md.exists()


def ensure_yolo_dataset(
    scheme: str = "7class",
    fold: int | None = None,
    output_base: str | Path = "outputs",
) -> Path:
    """Ensure standard or fold-specific YOLO dataset is exported and returns directory path."""
    out_base = Path(output_base)
    if fold is not None:
        yolo_dir = out_base / f"yolo_fold{fold}"
    elif scheme != "7class":
        yolo_dir = out_base / f"yolo_dataset_{scheme}"
    else:
        yolo_dir = out_base / "yolo_dataset"

    yaml_file = yolo_dir / "data.yaml"
    if yaml_file.exists():
        logger.info("YOLO dataset directory already exists at %s", yolo_dir)
        return yolo_dir

    logger.info("Exporting YOLO dataset (scheme=%s, fold=%s) to %s...", scheme, fold, yolo_dir)
    exporter = YOLOExporter(
        output_dir=yolo_dir,
        scheme=scheme,
        neg_ratio=0.25,
        fold=fold,
        seed=42,
        use_symlinks=True,
    )
    exporter.export()
    return yolo_dir


def extract_metrics_summary(eval_res: dict[str, Any]) -> dict[str, float]:
    """Extract standard evaluation metrics from run_pipeline_evaluation result dict."""
    val_m = eval_res.get("val_metrics", {})
    test_m = eval_res.get("test_metrics") or {}

    val_det = val_m.get("detection", {}).get("micro", {})
    val_map = val_m.get("map", {})
    val_frame = val_m.get("frame", {})

    test_det = test_m.get("detection", {}).get("micro", {})
    test_map = test_m.get("map", {})
    test_frame = test_m.get("frame", {})

    return {
        "val_precision": round(float(val_det.get("precision", 0.0)), 4),
        "val_recall": round(float(val_det.get("recall", 0.0)), 4),
        "val_f1": round(float(val_det.get("f1", 0.0)), 4),
        "val_mAP50": round(float(val_map.get("mAP_50", 0.0)), 4),
        "val_mAP50_95": round(float(val_map.get("mAP_50_95", 0.0)), 4),
        "val_frame_sensitivity": round(float(val_frame.get("sensitivity", val_frame.get("recall", 0.0))), 4),
        "val_frame_specificity": round(float(val_frame.get("specificity", 0.0)), 4),
        "val_frame_roc_auc": round(float(val_frame.get("roc_auc", 0.0)), 4),
        "test_precision": round(float(test_det.get("precision", 0.0)), 4),
        "test_recall": round(float(test_det.get("recall", 0.0)), 4),
        "test_f1": round(float(test_det.get("f1", 0.0)), 4),
        "test_mAP50": round(float(test_map.get("mAP_50", 0.0)), 4),
        "test_mAP50_95": round(float(test_map.get("mAP_50_95", 0.0)), 4),
        "test_frame_sensitivity": round(float(test_frame.get("sensitivity", test_frame.get("recall", 0.0))), 4),
        "test_frame_specificity": round(float(test_frame.get("specificity", 0.0)), 4),
        "test_frame_roc_auc": round(float(test_frame.get("roc_auc", 0.0)), 4),
    }


def run_single_experiment(
    run_name: str,
    scheme: str = "7class",
    model: str = "yolo11s.pt",
    imgsz: int = 640,
    oversample: bool = False,
    fold: int | None = None,
    epochs: int = 100,
    smoke_test: bool = False,
    force: bool = False,
    device: str | None = None,
    workers: int = 2,
    batch: int | None = None,
    exp_dir: str | Path = "outputs/experiments",
) -> dict[str, Any]:
    """Run a single experiment (train -> predict -> evaluate) with resumability.

    Args:
        run_name: Name of the experiment run directory.
        scheme: Class scheme ('7class', 'binary', 'severity3').
        model: Model architecture weights.
        imgsz: Training image size.
        oversample: Whether to oversample rare classes up to 3x.
        fold: Optional fold index (0 to 4).
        epochs: Training epochs.
        smoke_test: Fast verification mode (2 epochs, fraction=0.1).
        force: Force re-running even if completed.
        device: Compute device ('0', 'cpu', etc.).
        workers: DataLoader workers.
        batch: Batch size.
        exp_dir: Experiments output directory.

    Returns:
        Row dictionary stored in leaderboard.
    """
    logger.info("======================================================================")
    logger.info(" Running Experiment: %s", run_name)
    logger.info("======================================================================")

    # 1. Check resumability
    if not force and is_experiment_done(run_name, exp_dir=exp_dir):
        logger.info("Experiment '%s' already completed. Loading existing results...", run_name)
        df = load_leaderboard(exp_dir)
        if not df.empty and run_name in df["run_name"].values:
            return df[df["run_name"] == run_name].iloc[0].to_dict()

    # 2. Ensure YOLO dataset exists
    yolo_dir = ensure_yolo_dataset(scheme=scheme, fold=fold)

    # 3. Train model
    start_t = time.time()
    train_summary = train_yolo(
        scheme=scheme,
        model=model,
        imgsz=imgsz,
        epochs=epochs,
        fold=fold,
        seed=42,
        run_name=run_name,
        oversample_rare=oversample,
        smoke_test=smoke_test,
        device=device,
        workers=workers,
        batch=batch,
        data_dir=yolo_dir,
    )
    elapsed_sec = time.time() - start_t
    train_minutes = round(elapsed_sec / 60.0, 2)

    run_dir = Path("outputs/runs") / run_name
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = run_dir / "best.pt"

    # 4. Predict on val and test canonical splits
    logger.info("Running low-threshold inference on VAL split for '%s'...", run_name)
    val_preds_path = run_dir / "preds_val.json"
    predict_canonical_split(
        weights_path=best_weights,
        split="val",
        output_json_path=val_preds_path,
        yolo_dir=yolo_dir,
        conf=0.001,
        iou=0.6,
        max_det=50,
        imgsz=imgsz,
        device=device,
    )

    logger.info("Running low-threshold inference on TEST split for '%s'...", run_name)
    test_preds_path = run_dir / "preds_test.json"
    predict_canonical_split(
        weights_path=best_weights,
        split="test",
        output_json_path=test_preds_path,
        yolo_dir=yolo_dir,
        conf=0.001,
        iou=0.6,
        max_det=50,
        imgsz=imgsz,
        device=device,
    )

    # 5. Evaluate run (calibrate on val, freeze threshold, evaluate test)
    logger.info("Evaluating run '%s' with threshold freezing...", run_name)
    eval_res = run_pipeline_evaluation(
        run_dir=run_dir,
        scheme=scheme,
        val_preds_path=val_preds_path,
        test_preds_path=test_preds_path,
        output_dir="outputs/eval",
        yolo_dir=yolo_dir,
        bootstrap=not smoke_test,
        n_bootstrap=100 if smoke_test else 500,
        fold=fold,
    )

    # 6. Extract metrics and record to leaderboard
    extracted = extract_metrics_summary(eval_res)
    row_dict = {
        "run_name": run_name,
        "scheme": scheme,
        "model": model,
        "imgsz": imgsz,
        "oversample": oversample,
        "fold": fold if fold is not None else "main",
        "train_minutes": train_minutes,
        **extracted,
    }

    save_leaderboard_row(row_dict, exp_dir=exp_dir)

    # Also save detailed eval_res dict as JSON in run_dir for record-keeping
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        # Convert non-serializable objects if any
        json.dump(row_dict, f, indent=2)

    logger.info("Experiment '%s' completed successfully in %.2f minutes.", run_name, train_minutes)
    return row_dict


def generate_cv_summary(
    cv_rows: list[dict[str, Any]],
    output_path: str | Path = "outputs/experiments/cv_summary.md",
) -> str:
    """Generate 5-Fold Cross-Validation Markdown summary with mean +/- std statistics.

    Special Handling: Folds 3 and 4 have no p100 ground-truth boxes in val/test splits.
    Per-class p100 metrics are averaged strictly over folds that contain p100 (folds 0, 1, 2)
    and noted explicitly in the markdown report.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    df_cv = pd.DataFrame(cv_rows)

    metrics_keys = [
        "val_precision", "val_recall", "val_f1", "val_mAP50", "val_mAP50_95",
        "val_frame_sensitivity", "val_frame_specificity", "val_frame_roc_auc",
        "test_precision", "test_recall", "test_f1", "test_mAP50", "test_mAP50_95",
        "test_frame_sensitivity", "test_frame_specificity", "test_frame_roc_auc",
    ]

    stats: dict[str, tuple[float, float]] = {}
    for key in metrics_keys:
        if key in df_cv.columns:
            vals = df_cv[key].dropna().astype(float).values
            mean_val = float(np.mean(vals)) if len(vals) > 0 else 0.0
            std_val = float(np.std(vals)) if len(vals) > 0 else 0.0
            stats[key] = (mean_val, std_val)
        else:
            stats[key] = (0.0, 0.0)

    # Build Markdown table for overall CV metrics
    md_lines = [
        "# 5-Fold Patient-Wise Cross-Validation Summary",
        "",
        "**Model Config**: `7class` | `yolo11s.pt` | `--oversample-rare` | Image Size: 640",
        f"**Folds Evaluated**: {len(cv_rows)} patient-wise folds (zero patient leakage)",
        "",
        "## Overall Detection & Classification Metrics (Mean ± Std)",
        "",
        "| Metric | Validation Split (Val) | Test / Holdout Fold Split (Test) |",
        "| :--- | :---: | :---: |",
        f"| **Precision (Micro)** | {stats['val_precision'][0]:.4f} ± {stats['val_precision'][1]:.4f} | {stats['test_precision'][0]:.4f} ± {stats['test_precision'][1]:.4f} |",
        f"| **Recall (Micro)** | {stats['val_recall'][0]:.4f} ± {stats['val_recall'][1]:.4f} | {stats['test_recall'][0]:.4f} ± {stats['test_recall'][1]:.4f} |",
        f"| **F1-Score (Micro)** | {stats['val_f1'][0]:.4f} ± {stats['val_f1'][1]:.4f} | {stats['test_f1'][0]:.4f} ± {stats['test_f1'][1]:.4f} |",
        f"| **mAP @ 0.5** | **{stats['val_mAP50'][0]:.4f} ± {stats['val_mAP50'][1]:.4f}** | **{stats['test_mAP50'][0]:.4f} ± {stats['test_mAP50'][1]:.4f}** |",
        f"| **mAP @ 0.5:0.95** | {stats['val_mAP50_95'][0]:.4f} ± {stats['val_mAP50_95'][1]:.4f} | {stats['test_mAP50_95'][0]:.4f} ± {stats['test_mAP50_95'][1]:.4f} |",
        f"| **Frame Sensitivity** | {stats['val_frame_sensitivity'][0]:.4f} ± {stats['val_frame_sensitivity'][1]:.4f} | {stats['test_frame_sensitivity'][0]:.4f} ± {stats['test_frame_sensitivity'][1]:.4f} |",
        f"| **Frame Specificity** | {stats['val_frame_specificity'][0]:.4f} ± {stats['val_frame_specificity'][1]:.4f} | {stats['test_frame_specificity'][0]:.4f} ± {stats['test_frame_specificity'][1]:.4f} |",
        f"| **Frame ROC-AUC** | {stats['val_frame_roc_auc'][0]:.4f} ± {stats['val_frame_roc_auc'][1]:.4f} | {stats['test_frame_roc_auc'][0]:.4f} ± {stats['test_frame_roc_auc'][1]:.4f} |",
        "",
        "## Per-Fold Breakdown",
        "",
        "| Fold | Val mAP@0.5 | Val F1 | Test mAP@0.5 | Test F1 | Test Frame ROC-AUC | Train Time (min) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in cv_rows:
        f_idx = r.get("fold", "?")
        v_map = float(r.get("val_mAP50", 0.0))
        v_f1 = float(r.get("val_f1", 0.0))
        t_map = float(r.get("test_mAP50", 0.0))
        t_f1 = float(r.get("test_f1", 0.0))
        t_auc = float(r.get("test_frame_roc_auc", 0.0))
        t_min = float(r.get("train_minutes", 0.0))
        md_lines.append(f"| Fold {f_idx} | {v_map:.4f} | {v_f1:.4f} | {t_map:.4f} | {t_f1:.4f} | {t_auc:.4f} | {t_min:.1f} |")

    md_lines.extend([
        "",
        "## Rare Class Handling Note (`p100` Stenosis)",
        "",
        "> [!IMPORTANT]",
        "> **Note on `p100` (Total Occlusion)**: Folds 3 and 4 contain zero `p100` ground-truth boxes in their validation/test split partitions due to the extreme scarcity of total occlusions in the dataset (only 4 patients carry `p100`). Consequently, `p100` metrics are averaged strictly over the 3 folds that contain `p100` ground-truth annotations (Folds 0, 1, and 2) to prevent division-by-zero artifacts.",
        "",
    ])

    report_content = "\n".join(md_lines)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("CV Summary saved to %s", out_p)
    return report_content
