"""Unit tests for Phase 3.3 experiment runner, leaderboard tracking, and CV summary generation."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.models.experiment_runner import (
    LEADERBOARD_COLUMNS,
    extract_metrics_summary,
    generate_cv_summary,
    is_experiment_done,
    load_leaderboard,
    save_leaderboard_row,
)
from scripts.run_experiments import parse_args


def test_leaderboard_creation_and_save():
    """Test leaderboard CSV initialization, row appending, and updating."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        df_empty = load_leaderboard(tmp_p)

        assert isinstance(df_empty, pd.DataFrame)
        assert list(df_empty.columns) == LEADERBOARD_COLUMNS
        assert len(df_empty) == 0

        # Save first row
        row1 = {
            "run_name": "exp_test_1",
            "scheme": "7class",
            "model": "yolo11s.pt",
            "imgsz": 640,
            "oversample": True,
            "fold": "main",
            "val_mAP50": 0.6500,
            "test_mAP50": 0.6200,
            "train_minutes": 15.5,
        }
        df_after1 = save_leaderboard_row(row1, exp_dir=tmp_p)
        assert len(df_after1) == 1
        assert df_after1.iloc[0]["run_name"] == "exp_test_1"
        assert df_after1.iloc[0]["val_mAP50"] == 0.6500

        # Update existing row
        row1_update = {
            "run_name": "exp_test_1",
            "scheme": "7class",
            "model": "yolo11s.pt",
            "test_mAP50": 0.6350,
            "test_f1": 0.7100,
        }
        df_after2 = save_leaderboard_row(row1_update, exp_dir=tmp_p)
        assert len(df_after2) == 1
        assert df_after2.iloc[0]["test_mAP50"] == 0.6350
        assert df_after2.iloc[0]["test_f1"] == 0.7100

        # Add second row
        row2 = {"run_name": "exp_test_2", "scheme": "binary", "test_mAP50": 0.8500}
        df_after3 = save_leaderboard_row(row2, exp_dir=tmp_p)
        assert len(df_after3) == 2


def test_is_experiment_done():
    """Test experiment completion checker."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        assert not is_experiment_done("non_existent_run", exp_dir=tmp_p)

        # Save completed row
        row = {
            "run_name": "completed_run",
            "scheme": "7class",
            "val_mAP50": 0.600,
            "test_mAP50": 0.580,
        }
        save_leaderboard_row(row, exp_dir=tmp_p)

        assert is_experiment_done("completed_run", exp_dir=tmp_p)


def test_extract_metrics_summary():
    """Test extraction of metrics from run_pipeline_evaluation result dictionary."""
    mock_eval_res = {
        "val_metrics": {
            "detection": {"micro": {"precision": 0.80, "recall": 0.70, "f1": 0.7466}},
            "map": {"mAP_50": 0.72, "mAP_50_95": 0.45},
            "frame": {"sensitivity": 0.88, "specificity": 0.91, "roc_auc": 0.94},
        },
        "test_metrics": {
            "detection": {"micro": {"precision": 0.78, "recall": 0.68, "f1": 0.7265}},
            "map": {"mAP_50": 0.70, "mAP_50_95": 0.43},
            "frame": {"sensitivity": 0.86, "specificity": 0.89, "roc_auc": 0.92},
        },
    }

    summary = extract_metrics_summary(mock_eval_res)

    assert summary["val_precision"] == 0.80
    assert summary["val_mAP50"] == 0.72
    assert summary["test_f1"] == 0.7265
    assert summary["test_frame_roc_auc"] == 0.92


def test_generate_cv_summary():
    """Test 5-fold CV markdown summary table generation and rare class note inclusion."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "cv_summary.md"

        cv_rows = [
            {"fold": 0, "val_mAP50": 0.60, "test_mAP50": 0.58, "test_f1": 0.65, "train_minutes": 10.0},
            {"fold": 1, "val_mAP50": 0.62, "test_mAP50": 0.59, "test_f1": 0.66, "train_minutes": 10.5},
            {"fold": 2, "val_mAP50": 0.61, "test_mAP50": 0.60, "test_f1": 0.67, "train_minutes": 9.8},
            {"fold": 3, "val_mAP50": 0.59, "test_mAP50": 0.57, "test_f1": 0.64, "train_minutes": 10.2},
            {"fold": 4, "val_mAP50": 0.63, "test_mAP50": 0.61, "test_f1": 0.68, "train_minutes": 10.1},
        ]

        md_text = generate_cv_summary(cv_rows, output_path=out_file)

        assert out_file.exists()
        assert "# 5-Fold Patient-Wise Cross-Validation Summary" in md_text
        assert "Fold 0" in md_text
        assert "Fold 4" in md_text
        assert "p100" in md_text
        assert "Folds 3 and 4" in md_text


def test_run_experiments_cli_parser():
    """Test CLI argument parsing for scripts/run_experiments.py."""
    args = parse_args(["--matrix", "--smoke-test"])
    assert args.matrix is True
    assert args.smoke_test is True

    args_cv = parse_args(["--cv", "--device", "cpu"])
    assert args_cv.cv is True
    assert args_cv.device == "cpu"

    args_single = parse_args(["--run-name", "custom_run", "--scheme", "binary", "--oversample-rare"])
    assert args_single.run_name == "custom_run"
    assert args_single.scheme == "binary"
    assert args_single.oversample_rare is True
