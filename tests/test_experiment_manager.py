#!/usr/bin/env python3
"""Unit tests for the Phase 3.3 experiment manager and run_experiments pipeline.

Tests:
  - Leaderboard load/save/append (empty, existing, overwrite)
  - row_exists detection
  - extract_metrics_from_eval (valid JSON, missing file, empty)
  - compute_cv_summary mean/std correctness
  - CV summary Markdown output generation
  - run_experiments.py argument parsing
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.models.experiment_manager import (
    LEADERBOARD_COLUMNS,
    append_run_to_leaderboard,
    compute_cv_summary,
    extract_metrics_from_eval,
    load_leaderboard,
    row_exists,
    save_leaderboard,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_metrics_json(tmp_path, split="val", P=0.8, R=0.7, F1=0.75,
                      mAP50=0.65, mAP5095=0.40,
                      sensitivity=0.72, specificity=0.85, roc_auc=0.90):
    """Create a minimal metrics.json compatible with extract_metrics_from_eval."""
    d = {
        "detection": {
            "micro": {
                "precision": P,
                "recall": R,
                "f1": F1,
            }
        },
        "map": {
            "mAP_50": mAP50,
            "mAP_50_95": mAP5095,
        },
        "frame": {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "roc_auc": roc_auc,
        },
    }
    p = tmp_path / f"metrics_{split}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Test: leaderboard I/O
# ---------------------------------------------------------------------------

class TestLeaderboardIO:
    def test_load_empty_leaderboard(self, tmp_path):
        path = tmp_path / "leaderboard.csv"
        df = load_leaderboard(path)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == LEADERBOARD_COLUMNS
        assert len(df) == 0

    def test_save_and_reload(self, tmp_path):
        path = tmp_path / "lb.csv"
        df = pd.DataFrame(columns=LEADERBOARD_COLUMNS)
        save_leaderboard(df, path)
        assert path.exists()
        df2 = load_leaderboard(path)
        assert len(df2) == 0
        assert list(df2.columns) == LEADERBOARD_COLUMNS

    def test_row_exists_false_on_empty(self, tmp_path):
        path = tmp_path / "lb.csv"
        df = load_leaderboard(path)
        assert not row_exists(df, "nonexistent_run")

    def test_row_exists_true_after_append(self, tmp_path):
        lb = tmp_path / "lb.csv"
        val_json = make_metrics_json(tmp_path, split="val")
        test_json = make_metrics_json(tmp_path, split="test")
        df = append_run_to_leaderboard(
            leaderboard_path=lb,
            run_name="run_alpha",
            scheme="7class",
            model="yolo11s.pt",
            imgsz=640,
            oversample=False,
            fold=None,
            train_minutes=12.5,
            val_metrics_json=val_json,
            test_metrics_json=test_json,
        )
        assert row_exists(df, "run_alpha")
        assert len(df) == 1

    def test_append_skips_existing_by_default(self, tmp_path):
        lb = tmp_path / "lb.csv"
        val_json = make_metrics_json(tmp_path)
        for _ in range(2):  # call twice
            df = append_run_to_leaderboard(
                leaderboard_path=lb,
                run_name="run_beta",
                scheme="binary",
                model="yolo11s.pt",
                imgsz=640,
                oversample=False,
                fold=None,
                train_minutes=5.0,
                val_metrics_json=val_json,
            )
        assert len(df) == 1  # should not duplicate

    def test_append_overwrites_when_flag_set(self, tmp_path):
        lb = tmp_path / "lb.csv"
        val_json = make_metrics_json(tmp_path, P=0.5)
        append_run_to_leaderboard(
            leaderboard_path=lb,
            run_name="run_gamma",
            scheme="binary",
            model="yolo11n.pt",
            imgsz=640,
            oversample=False,
            fold=None,
            train_minutes=3.0,
            val_metrics_json=val_json,
        )
        val_json2 = make_metrics_json(tmp_path / "v2", P=0.9)
        df = append_run_to_leaderboard(
            leaderboard_path=lb,
            run_name="run_gamma",
            scheme="binary",
            model="yolo11n.pt",
            imgsz=640,
            oversample=False,
            fold=None,
            train_minutes=3.5,
            val_metrics_json=val_json2,
            overwrite_existing=True,
        )
        assert len(df) == 1
        assert float(df.iloc[0]["val_P"]) == pytest.approx(0.9, abs=1e-3)


# ---------------------------------------------------------------------------
# Test: metrics extraction
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_extract_valid_json(self, tmp_path):
        p = make_metrics_json(tmp_path, P=0.81, R=0.73, F1=0.77,
                              mAP50=0.66, mAP5095=0.42,
                              sensitivity=0.74, specificity=0.88, roc_auc=0.92)
        m = extract_metrics_from_eval(p)
        assert m["P"] == pytest.approx(0.81, abs=1e-4)
        assert m["R"] == pytest.approx(0.73, abs=1e-4)
        assert m["F1"] == pytest.approx(0.77, abs=1e-4)
        assert m["mAP50"] == pytest.approx(0.66, abs=1e-4)
        assert m["mAP5095"] == pytest.approx(0.42, abs=1e-4)
        assert m["frame_sensitivity"] == pytest.approx(0.74, abs=1e-4)
        assert m["roc_auc"] == pytest.approx(0.92, abs=1e-4)

    def test_extract_missing_file(self, tmp_path):
        m = extract_metrics_from_eval(tmp_path / "nonexistent.json")
        assert m == {}

    def test_extract_empty_json(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("{}", encoding="utf-8")
        m = extract_metrics_from_eval(p)
        assert math.isnan(m.get("P", float("nan")))
        assert math.isnan(m.get("mAP50", float("nan")))


# ---------------------------------------------------------------------------
# Test: CV summary computation
# ---------------------------------------------------------------------------

class TestCVSummary:
    def _setup_folds(self, tmp_path, base_name, n_folds=3):
        lb = tmp_path / "lb.csv"
        for i in range(n_folds):
            val_json = make_metrics_json(
                tmp_path / f"fold{i}",
                split="val",
                P=0.7 + i * 0.05,
                R=0.65 + i * 0.04,
                F1=0.67 + i * 0.04,
                mAP50=0.55 + i * 0.05,
                mAP5095=0.30 + i * 0.05,
            )
            append_run_to_leaderboard(
                leaderboard_path=lb,
                run_name=f"{base_name}_fold{i}",
                scheme="7class",
                model="yolo11s.pt",
                imgsz=640,
                oversample=False,
                fold=i,
                train_minutes=15.0,
                val_metrics_json=val_json,
            )
        return lb

    def test_cv_summary_mean_std(self, tmp_path):
        lb = self._setup_folds(tmp_path, "test_run_cv", n_folds=3)
        summary = compute_cv_summary(
            leaderboard_path=lb,
            base_run_name="test_run_cv",
            n_folds=3,
            folds_with_p100=[0, 1],
        )
        # mean of P values: 0.70, 0.75, 0.80 -> 0.75
        assert summary["val_P_mean"] == pytest.approx(0.75, abs=1e-3)
        assert summary["val_P_std"] > 0.0  # non-trivial std
        assert summary["n_folds_found"] == 3

    def test_cv_summary_writes_markdown(self, tmp_path):
        lb = self._setup_folds(tmp_path, "cv_run", n_folds=2)
        out_md = tmp_path / "cv_summary.md"
        compute_cv_summary(
            leaderboard_path=lb,
            base_run_name="cv_run",
            n_folds=2,
            folds_with_p100=[0],
            output_md=out_md,
        )
        assert out_md.exists()
        text = out_md.read_text(encoding="utf-8")
        assert "5-Fold" in text or "Fold" in text
        assert "p100" in text
        assert "Mean" in text

    def test_cv_summary_empty_leaderboard(self, tmp_path):
        lb = tmp_path / "empty.csv"
        summary = compute_cv_summary(
            leaderboard_path=lb,
            base_run_name="missing_run",
        )
        assert summary == {}


# ---------------------------------------------------------------------------
# Test: run_experiments argument parsing
# ---------------------------------------------------------------------------

class TestRunExperimentsArgParsing:
    def test_default_args(self):
        from scripts.run_experiments import parse_args
        args = parse_args([])
        assert args.smoke_test is False
        assert args.overwrite is False
        assert args.best_run == "yolo11s_7class_os"
        assert args.seed == 42
        assert args.workers == 2
        assert args.no_bootstrap is False

    def test_smoke_test_flag(self):
        from scripts.run_experiments import parse_args
        args = parse_args(["--smoke-test", "--best-run", "yolo11s_binary"])
        assert args.smoke_test is True
        assert args.best_run == "yolo11s_binary"

    def test_runs_filter(self):
        from scripts.run_experiments import parse_args
        args = parse_args(["--runs", "yolo11s_binary", "yolo11s_7class"])
        assert "yolo11s_binary" in args.runs
        assert "yolo11s_7class" in args.runs

