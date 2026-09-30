"""Tests for patient-wise splits module and CLI."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.data.splits import PatientSplitter, natural_patient_sort_key


@pytest.fixture
def splitter_instance() -> PatientSplitter:
    """Fixture providing a PatientSplitter instance initialized with configs/config.yaml."""
    return PatientSplitter(config_path="configs/config.yaml")


@pytest.fixture
def loaded_splits(splitter_instance: PatientSplitter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fixture providing train/val/test split and folds alongside annotations and patients."""
    ann_df, pts_df, frames_df = splitter_instance.load_index_data()
    split_df = splitter_instance.create_train_val_test_split(ann_df, pts_df, seed=42)
    folds_df = splitter_instance.create_kfold_splits(ann_df, pts_df, n_folds=5, seed=42)
    return split_df, folds_df, ann_df, pts_df


class TestZeroPatientOverlap:
    """Tests asserting strictly zero patient overlap across splits and cross-validation folds."""

    def test_no_overlap_train_val(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert zero patient overlap between train and validation sets."""
        split_df, _, _, _ = loaded_splits
        train_pts = set(split_df[split_df["split"] == "train"]["patient_id"])
        val_pts = set(split_df[split_df["split"] == "val"]["patient_id"])
        overlap = train_pts.intersection(val_pts)
        assert len(overlap) == 0, f"Found patient overlap between train and val: {overlap}"

    def test_no_overlap_train_test(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert zero patient overlap between train and test sets."""
        split_df, _, _, _ = loaded_splits
        train_pts = set(split_df[split_df["split"] == "train"]["patient_id"])
        test_pts = set(split_df[split_df["split"] == "test"]["patient_id"])
        overlap = train_pts.intersection(test_pts)
        assert len(overlap) == 0, f"Found patient overlap between train and test: {overlap}"

    def test_no_overlap_val_test(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert zero patient overlap between validation and test sets."""
        split_df, _, _, _ = loaded_splits
        val_pts = set(split_df[split_df["split"] == "val"]["patient_id"])
        test_pts = set(split_df[split_df["split"] == "test"]["patient_id"])
        overlap = val_pts.intersection(test_pts)
        assert len(overlap) == 0, f"Found patient overlap between val and test: {overlap}"

    def test_all_patients_covered(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert union of train, val, test contains every patient exactly once."""
        split_df, _, _, pts_df = loaded_splits
        all_pts = set(pts_df["patient_id"])
        assigned_pts = set(split_df["patient_id"])
        assert assigned_pts == all_pts
        assert len(split_df) == len(all_pts)

    def test_no_overlap_across_folds(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert zero patient overlap across all K-fold cross-validation partitions."""
        _, folds_df, _, pts_df = loaded_splits
        all_folds = sorted(folds_df["fold"].unique())
        assert len(all_folds) == 5

        fold_patient_sets = [
            set(folds_df[folds_df["fold"] == f]["patient_id"]) for f in all_folds
        ]

        for i in range(len(all_folds)):
            for j in range(i + 1, len(all_folds)):
                overlap = fold_patient_sets[i].intersection(fold_patient_sets[j])
                assert (
                    len(overlap) == 0
                ), f"Found patient overlap between Fold {all_folds[i]} and Fold {all_folds[j]}: {overlap}"

    def test_all_patients_in_folds(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert union of all fold partitions covers 100% of patients."""
        _, folds_df, _, pts_df = loaded_splits
        all_pts = set(pts_df["patient_id"])
        assigned_pts = set(folds_df["patient_id"])
        assert assigned_pts == all_pts
        assert len(folds_df) == len(all_pts)

    def test_correct_n_folds(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert exactly 5 distinct folds exist."""
        _, folds_df, _, _ = loaded_splits
        assert set(folds_df["fold"]) == {0, 1, 2, 3, 4}


class TestClassCoverage:
    """Tests asserting all stenosis classes are represented in validation and test splits."""

    def test_all_classes_in_val(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert no class has zero bounding boxes in validation."""
        split_df, _, ann_df, _ = loaded_splits
        val_pts = set(split_df[split_df["split"] == "val"]["patient_id"])
        val_ann = ann_df[ann_df["patient_id"].isin(val_pts)]
        val_classes = set(val_ann["category"])
        all_classes = set(ann_df["category"])
        assert val_classes == all_classes, f"Missing classes in val: {all_classes - val_classes}"

    def test_all_classes_in_test(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert no class has zero bounding boxes in test."""
        split_df, _, ann_df, _ = loaded_splits
        test_pts = set(split_df[split_df["split"] == "test"]["patient_id"])
        test_ann = ann_df[ann_df["patient_id"].isin(test_pts)]
        test_classes = set(test_ann["category"])
        all_classes = set(ann_df["category"])
        assert test_classes == all_classes, f"Missing classes in test: {all_classes - test_classes}"

    def test_rare_classes_distributed(self, loaded_splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
        """Assert rare class p100 and p99 have bounding boxes in all splits."""
        split_df, _, ann_df, _ = loaded_splits
        for s in ["train", "val", "test"]:
            s_pts = set(split_df[split_df["split"] == s]["patient_id"])
            s_ann = ann_df[ann_df["patient_id"].isin(s_pts)]
            p100_count = (s_ann["category"] == "p100").sum()
            p99_count = (s_ann["category"] == "p99").sum()
            assert p100_count > 0, f"Split {s} has 0 boxes for rare class p100!"
            assert p99_count > 0, f"Split {s} has 0 boxes for rare class p99!"


class TestOutputFiles:
    """Tests verifying generated split CSV files, reports, and consistency."""

    def test_split_files_generated_and_valid(self, splitter_instance: PatientSplitter, tmp_path: Path) -> None:
        """Run run_all and assert output files exist and are valid CSVs."""
        split_path, folds_path, report_path = splitter_instance.run_all(output_dir=tmp_path, seed=42)

        assert split_path.exists()
        assert folds_path.exists()
        assert report_path.exists()

        split_df = pd.read_csv(split_path)
        assert list(split_df.columns) == ["patient_id", "split"]
        assert len(split_df) == 42
        assert set(split_df["split"]) == {"train", "val", "test"}

        folds_df = pd.read_csv(folds_path)
        assert list(folds_df.columns) == ["patient_id", "fold"]
        assert len(folds_df) == 42
        assert set(folds_df["fold"]) == {0, 1, 2, 3, 4}

        report_txt = report_path.read_text(encoding="utf-8")
        assert "CADICA Dataset — Patient-Wise Splits Report" in report_txt
        assert "Leakage Verification" in report_txt
        assert "Zero patient overlap" in report_txt

    def test_no_duplicate_patient_rows(self, splitter_instance: PatientSplitter, tmp_path: Path) -> None:
        """Assert no duplicate patient rows in generated files."""
        split_path, folds_path, _ = splitter_instance.run_all(output_dir=tmp_path, seed=42)
        split_df = pd.read_csv(split_path)
        folds_df = pd.read_csv(folds_path)

        assert not split_df["patient_id"].duplicated().any()
        assert not folds_df["patient_id"].duplicated().any()


class TestConfigAndClassSchemes:
    """Tests validating ratio validation, class schemes mapping, and edge case failures."""

    def test_bad_ratios_raise_error(self) -> None:
        """Assert that split ratios not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            PatientSplitter(
                config_path="configs/config.yaml",
                train_ratio=0.80,
                val_ratio=0.20,
                test_ratio=0.20,
            )

    def test_class_schemes_mappings(self, splitter_instance: PatientSplitter) -> None:
        """Test mapping categories across 7class, binary, and severity3 schemes."""
        # 7class
        assert splitter_instance.map_category("p0_20", "7class") == 0
        assert splitter_instance.map_category("p100", "7class") == 6

        # binary
        assert splitter_instance.map_category("nonlesion", "binary") == 0
        assert splitter_instance.map_category("p0_20", "binary") == 1
        assert splitter_instance.map_category("p100", "binary") == 1

        # severity3
        assert splitter_instance.map_category("p0_20", "severity3") == 0
        assert splitter_instance.map_category("p20_50", "severity3") == 0
        assert splitter_instance.map_category("p50_70", "severity3") == 1
        assert splitter_instance.map_category("p70_90", "severity3") == 2
        assert splitter_instance.map_category("p90_98", "severity3") == 2
        assert splitter_instance.map_category("p99", "severity3") == 2
        assert splitter_instance.map_category("p100", "severity3") == 2

    def test_unknown_scheme_raises(self, splitter_instance: PatientSplitter) -> None:
        """Test that unknown scheme raises KeyError."""
        with pytest.raises(KeyError, match="Unknown class scheme"):
            splitter_instance.map_category("p0_20", "invalid_scheme")

    def test_natural_patient_sort_key(self) -> None:
        """Test natural sorting order of patient IDs."""
        pts = ["p10", "p2", "p1", "p20", "p3"]
        sorted_pts = sorted(pts, key=natural_patient_sort_key)
        assert sorted_pts == ["p1", "p2", "p3", "p10", "p20"]
