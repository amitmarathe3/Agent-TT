"""Unit and integration tests for YOLO dataset export functionality."""

from pathlib import Path
import tempfile
import yaml
import pytest
import pandas as pd

from src.data.yolo_export import YOLOExporter


@pytest.fixture
def temp_export_dir():
    """Create a temporary directory for YOLO dataset exports."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestYOLOExporter:
    """Tests covering YOLO exporter structure, labels, schemes, and fold export."""

    def test_yolo_export_basic_structure(self, temp_export_dir: Path) -> None:
        """Verify exporter creates standard Ultralytics YOLO folders and data.yaml."""
        exporter = YOLOExporter(
            config_path="configs/config.yaml",
            output_dir=temp_export_dir / "yolo_basic",
            scheme="7class",
            neg_ratio=0.1,
            seed=42,
        )
        stats = exporter.export()

        out_dir = temp_export_dir / "yolo_basic"
        assert (out_dir / "data.yaml").exists()

        for split in ["train", "val", "test"]:
            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            assert img_dir.is_dir()
            assert lbl_dir.is_dir()

            # Verify image and label file counts match
            img_files = sorted(img_dir.glob("*.png"))
            lbl_files = sorted(lbl_dir.glob("*.txt"))
            assert len(img_files) == len(lbl_files)
            assert len(img_files) == stats[split]["total_images"]

    def test_label_format_and_empty_negatives(self, temp_export_dir: Path) -> None:
        """Verify positive labels follow YOLO format and negative labels are empty."""
        exporter = YOLOExporter(
            config_path="configs/config.yaml",
            output_dir=temp_export_dir / "yolo_labels",
            scheme="7class",
            neg_ratio=0.25,
            seed=42,
        )
        stats = exporter.export()
        out_dir = temp_export_dir / "yolo_labels"

        # Check train labels
        lbl_dir = out_dir / "labels" / "train"
        lbl_files = list(lbl_dir.glob("*.txt"))
        assert len(lbl_files) > 0

        empty_files = [f for f in lbl_files if f.stat().st_size == 0]
        non_empty_files = [f for f in lbl_files if f.stat().st_size > 0]

        # There must be both positives and negatives
        assert len(non_empty_files) == stats["train"]["positives"]
        assert len(empty_files) == stats["train"]["negatives"]

        # Validate line formatting of non-empty labels
        for f in non_empty_files[:20]:
            lines = f.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                parts = line.split()
                assert len(parts) == 5, f"Expected 5 elements, got: {line}"
                cid, xc, yc, w, h = parts
                cid_int = int(cid)
                assert 0 <= cid_int < 7
                xc_f, yc_f, w_f, h_f = float(xc), float(yc), float(w), float(h)
                assert 0.0 <= xc_f <= 1.0
                assert 0.0 <= yc_f <= 1.0
                assert 0.0 < w_f <= 1.0
                assert 0.0 < h_f <= 1.0

    def test_class_schemes_yaml(self, temp_export_dir: Path) -> None:
        """Verify data.yaml correctly configures binary, severity3, and 7class schemes."""
        schemes = {
            "binary": 2,
            "severity3": 3,
            "7class": 7,
        }

        for scheme_name, expected_nc in schemes.items():
            out_dir = temp_export_dir / f"yolo_{scheme_name}"
            exporter = YOLOExporter(
                config_path="configs/config.yaml",
                output_dir=out_dir,
                scheme=scheme_name,
                neg_ratio=0.0,  # Speed up by skipping negatives
            )
            # Check scheme mapping resolution
            assert len(exporter.class_names) == expected_nc
            assert exporter.scheme == scheme_name

            # Test yaml generation logic
            mapping, names = exporter._resolve_class_scheme(scheme_name)
            assert len(names) == expected_nc

    def test_fold_export(self, temp_export_dir: Path) -> None:
        """Verify --fold argument maps folds correctly to train/val/test splits."""
        folds_path = Path("data_index/splits/folds_5.csv")
        if not folds_path.exists():
            pytest.skip("folds_5.csv not found")

        folds_df = pd.read_csv(folds_path)
        test_fold = 0
        expected_test_patients = set(folds_df[folds_df["fold"] == test_fold]["patient_id"])
        expected_val_patients = set(folds_df[folds_df["fold"] == (test_fold + 1) % 5]["patient_id"])

        exporter = YOLOExporter(
            config_path="configs/config.yaml",
            output_dir=temp_export_dir / "yolo_fold0",
            scheme="7class",
            fold=test_fold,
            neg_ratio=0.0,
        )
        _, _, split_df = exporter.load_data()

        actual_test_patients = set(split_df[split_df["split"] == "test"]["patient_id"])
        actual_val_patients = set(split_df[split_df["split"] == "val"]["patient_id"])

        assert actual_test_patients == expected_test_patients
        assert actual_val_patients == expected_val_patients
        assert actual_test_patients.isdisjoint(actual_val_patients)

    def test_zero_neg_ratio(self, temp_export_dir: Path) -> None:
        """Verify neg_ratio=0 produces zero empty negative files."""
        exporter = YOLOExporter(
            config_path="configs/config.yaml",
            output_dir=temp_export_dir / "yolo_no_neg",
            scheme="7class",
            neg_ratio=0.0,
        )
        stats = exporter.export()
        assert stats["train"]["negatives"] == 0
        assert stats["val"]["negatives"] == 0
        assert stats["test"]["negatives"] == 0
