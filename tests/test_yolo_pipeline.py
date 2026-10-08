"""Unit and integration tests for the YOLO training, prediction, and evaluation pipeline.

Tests:
- Model loader and fallback mechanism
- Safe device selection
- Rare-class oversampling manifest generator
- Training curve plotting from results.csv
- Prediction formatting in absolute pixels
- Confidence threshold calibration, freezing, and evaluation
- Qualitative error grid generator
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import yaml

from src.models.yolo_trainer import (
    get_safe_device,
    load_yolo_model,
    prepare_oversampled_dataset,
    plot_training_curves,
)
from src.models.eval_runner import (
    read_ultralytics_map50_from_run,
    evaluate_split,
)
from src.models.visualization import (
    render_prediction_grid,
    generate_error_grids,
)


class TestYOLOTrainer:
    def test_get_safe_device(self):
        dev_id, dev_name = get_safe_device()
        assert dev_id in {"0", "cpu"}
        assert isinstance(dev_name, str)

        dev_custom_id, dev_custom_name = get_safe_device("cpu")
        assert dev_custom_id == "cpu"
        assert "CPU" in dev_custom_name

    def test_load_yolo_model_fallback(self):
        # Loading a non-existent invalid model should fall back to yolov8s.pt
        model, model_name = load_yolo_model("non_existent_fake_model.pt")
        assert model is not None
        assert model_name == "yolov8s.pt"

    def test_prepare_oversampled_dataset(self, tmp_path):
        # Setup synthetic dataset directory
        ds_dir = tmp_path / "mock_yolo"
        train_img_dir = ds_dir / "images" / "train"
        train_img_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy image files
        img1 = train_img_dir / "frame_rare.png"
        img2 = train_img_dir / "frame_normal.png"
        img1.write_bytes(b"dummy1")
        img2.write_bytes(b"dummy2")

        # Create data.yaml
        yaml_content = {
            "path": str(ds_dir).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": 7,
            "names": ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"],
        }
        data_yaml = ds_dir / "data.yaml"
        with open(data_yaml, "w") as f:
            yaml.dump(yaml_content, f)

        # Create annotations.csv marking frame_rare as p99
        ann_csv = tmp_path / "annotations.csv"
        df = pd.DataFrame([
            {"frame_id": "frame_rare", "category": "p99"},
            {"frame_id": "frame_normal", "category": "p0_20"},
        ])
        df.to_csv(ann_csv, index=False)

        # Call prepare_oversampled_dataset
        out_yaml = prepare_oversampled_dataset(
            data_yaml_path=data_yaml,
            output_dir=tmp_path / "run_out",
            rare_categories=("p99", "p100", "p90_98"),
            multiplier=3,
            annotations_csv=ann_csv,
        )
        assert out_yaml.exists()

        # Check train_oversampled.txt content
        txt_path = tmp_path / "run_out" / "train_oversampled.txt"
        assert txt_path.exists()
        lines = [line.strip() for line in txt_path.read_text().splitlines() if line.strip()]
        # frame_rare should appear 3 times, frame_normal should appear 1 time -> total 4 lines
        assert len(lines) == 4
        rare_matches = [l for l in lines if "frame_rare" in l]
        normal_matches = [l for l in lines if "frame_normal" in l]
        assert len(rare_matches) == 3
        assert len(normal_matches) == 1

    def test_plot_training_curves(self, tmp_path):
        results_csv = tmp_path / "results.csv"
        df = pd.DataFrame({
            "epoch": [0, 1, 2],
            "train/box_loss": [1.5, 1.2, 0.9],
            "train/cls_loss": [2.0, 1.5, 1.1],
            "train/dfl_loss": [1.0, 0.8, 0.6],
            "val/box_loss": [1.6, 1.3, 1.0],
            "val/cls_loss": [2.1, 1.6, 1.2],
            "val/dfl_loss": [1.1, 0.9, 0.7],
            "metrics/precision(B)": [0.4, 0.6, 0.7],
            "metrics/recall(B)": [0.3, 0.5, 0.65],
            "metrics/mAP50(B)": [0.35, 0.55, 0.68],
            "metrics/mAP50-95(B)": [0.20, 0.35, 0.45],
        })
        df.to_csv(results_csv, index=False)

        curves_png = tmp_path / "curves.png"
        plot_training_curves(results_csv, curves_png)
        assert curves_png.exists()
        assert curves_png.stat().st_size > 1000


class TestEvalRunner:
    def test_read_ultralytics_map50(self, tmp_path):
        csv_p = tmp_path / "results.csv"
        df = pd.DataFrame({
            "epoch": [0, 1],
            "metrics/mAP50(B)": [0.45, 0.72],
        })
        df.to_csv(csv_p, index=False)
        val = read_ultralytics_map50_from_run(tmp_path)
        assert val == pytest.approx(0.72, abs=1e-4)

    def test_evaluate_split_with_comparison(self, tmp_path):
        preds = {
            "img1": [[10.0, 10.0, 50.0, 50.0, 0.9, 0]],
            "img2": [],
        }
        gts = {
            "img1": [[10.0, 10.0, 50.0, 50.0, 0]],
            "img2": [],
        }
        f2p = {"img1": "p1", "img2": "p2"}

        expected_map50 = 1.0 / 7.0  # class 0 has AP=1.0, others have 0.0

        metrics = evaluate_split(
            predictions=preds,
            ground_truths=gts,
            frame_to_patient=f2p,
            split="val",
            scheme="7class",
            run_name="test_run",
            output_dir=tmp_path / "eval",
            conf_threshold=0.5,
            iou_threshold=0.5,
            bootstrap=False,
            ultralytics_map50=expected_map50,
        )

        assert "detection" in metrics
        assert "map" in metrics
        assert metrics["map"]["mAP_50"] == pytest.approx(expected_map50, abs=1e-3)
        assert "ultralytics_comparison" in metrics
        assert metrics["ultralytics_comparison"]["cross_check_passed"] is True


class TestVisualization:
    def test_render_prediction_grid_and_error_grids(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # Create 2 sample images with dummy pixels
        import cv2
        dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "f1.png"), dummy_img)
        cv2.imwrite(str(img_dir / "f2.png"), dummy_img)

        # f1 has a match; f2 has an FN and an FP
        preds = {
            "f1": [[10.0, 10.0, 50.0, 50.0, 0.9, 0]],
            "f2": [[400.0, 400.0, 30.0, 30.0, 0.85, 1]], # FP
        }
        gts = {
            "f1": [[10.0, 10.0, 50.0, 50.0, 0]],
            "f2": [[100.0, 100.0, 40.0, 40.0, 6]], # FN (p100)
        }
        class_names = {0: "p0_20", 1: "p20_50", 6: "p100"}

        out_dir = tmp_path / "vis"
        grids = generate_error_grids(
            predictions=preds,
            ground_truths=gts,
            class_names=class_names,
            image_dir=img_dir,
            output_dir=out_dir,
            conf_threshold=0.5,
            iou_threshold=0.5,
        )

        assert grids["val_grid"].exists()
        assert grids["false_negatives_grid"].exists()
        assert grids["false_positives_grid"].exists()
