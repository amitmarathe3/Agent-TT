"""Tests for clinical metadata inspection, normalization, and auditing module."""

from pathlib import Path
import pandas as pd
import pytest

from src.data.metadata import (
    MetadataInspector,
    natural_patient_key,
    normalize_patient_id,
)


@pytest.fixture
def synthetic_metadata_file(tmp_path: Path) -> Path:
    """Create a temporary synthetic Excel file with diverse clinical columns."""
    excel_path = tmp_path / "metadata.xlsx"

    data = {
        "Patient ID": [1, 2, 3, 4, 5],
        "Age (years)": [55, 62, 48, 71, 59],
        "Sex": ["M", "F", "M", "M", "F"],
        "Height (m)": [1.75, None, 1.62, 1.80, None],
        "Weight (kg)": [78.0, 65.0, 82.5, 90.0, 70.0],
        "Smoker": [1, 0, 1, 0, 0],
        "Diabetes mellitus": [0, 1, 0, 1, 0],
        "Comorbidity": ["None", "Hypertension", "COPD", "None", "None"],
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)
    return excel_path


@pytest.fixture
def synthetic_px_metadata_file(tmp_path: Path) -> Path:
    """Create a synthetic Excel file with string 'pX' patient IDs and arbitrary column name."""
    excel_path = tmp_path / "study_meta.xlsx"

    data = {
        "Subject_Code": ["p01", "P2", "p3", "p004"],
        "Age": [50, 60, 70, 80],
        "Diagnosis": ["CAD", "CAD", "Normal", "CAD"],
    }
    df = pd.DataFrame(data)
    df.to_excel(excel_path, index=False)
    return excel_path


class TestPatientIDNormalization:
    """Tests for patient ID normalization logic."""

    def test_normalize_integer(self) -> None:
        """Integer IDs should normalize to 'pX'."""
        assert normalize_patient_id(1) == "p1"
        assert normalize_patient_id(42) == "p42"
        assert normalize_patient_id(100) == "p100"

    def test_normalize_float(self) -> None:
        """Float IDs should normalize to 'pX' without decimal points."""
        assert normalize_patient_id(5.0) == "p5"

    def test_normalize_string_variations(self) -> None:
        """Various string representations should normalize to 'pX'."""
        assert normalize_patient_id("1") == "p1"
        assert normalize_patient_id("p1") == "p1"
        assert normalize_patient_id("P1") == "p1"
        assert normalize_patient_id("p01") == "p1"
        assert normalize_patient_id("p0042") == "p42"
        assert normalize_patient_id("patient_7") == "p7"

    def test_normalize_empty_or_none(self) -> None:
        """Null or empty inputs should return empty string."""
        assert normalize_patient_id(None) == ""
        assert normalize_patient_id("") == ""
        assert normalize_patient_id("   ") == ""

    def test_natural_sort_order(self) -> None:
        """Patient IDs should sort naturally by integer part."""
        raw_list = ["p10", "p2", "p1", "p20", "p3"]
        sorted_list = sorted(raw_list, key=natural_patient_key)
        assert sorted_list == ["p1", "p2", "p3", "p10", "p20"]


class TestPatientIDAutoDetection:
    """Tests for automated patient ID column detection."""

    def test_detect_integer_column(self, tmp_path: Path) -> None:
        """Detect column named 'Patient ID' containing integers."""
        df = pd.DataFrame({
            "Patient ID": [1, 2, 3],
            "Age": [60, 65, 70],
            "Smoker": [0, 1, 0],
        })
        inspector = MetadataInspector()
        detected_col, score = inspector.detect_patient_id_column(df)
        assert detected_col == "Patient ID"
        assert score > 0

    def test_detect_nonstandard_column_name(self) -> None:
        """Detect patient ID even when column name is unusual (e.g. 'Subject_Code')."""
        df = pd.DataFrame({
            "Subject_Code": ["p1", "p2", "p3", "p4"],
            "Heart failure": [0, 0, 1, 0],
            "Weight": [70.0, 75.0, 80.0, 85.0],
        })
        inspector = MetadataInspector()
        detected_col, score = inspector.detect_patient_id_column(df)
        assert detected_col == "Subject_Code"
        assert score > 0

    def test_detect_pure_integer_unnamed_col(self) -> None:
        """Detect integer sequence when column name is generic like 'code'."""
        df = pd.DataFrame({
            "code": [101, 102, 103, 104],
            "diabetes": [0, 1, 0, 0],
        })
        inspector = MetadataInspector()
        detected_col, score = inspector.detect_patient_id_column(df)
        assert detected_col == "code"
        assert score > 0


class TestMetadataCleaning:
    """Tests for cleaning and standardizing metadata DataFrames."""

    def test_clean_metadata_reorders_and_normalizes(self) -> None:
        """Clean metadata should have 'patient_id' first and normalized values."""
        df = pd.DataFrame({
            "Age": [55, 60],
            "Patient ID": [1, 2],
            "Sex": ["M  ", "F\xa0"],
        })
        inspector = MetadataInspector()
        clean_df, detected_col = inspector.clean_metadata(df)

        assert detected_col == "Patient ID"
        assert list(clean_df.columns)[0] == "patient_id"
        assert clean_df["patient_id"].tolist() == ["p1", "p2"]
        # Whitespace and non-breaking space stripped
        assert clean_df["Sex"].tolist() == ["M", "F"]


class TestPatientCrossReferencing:
    """Tests for comparing metadata patients against image folders."""

    def test_compare_patients_perfect_match(self) -> None:
        """Perfect alignment when sets match exactly."""
        inspector = MetadataInspector()
        res = inspector.compare_patients(
            metadata_patients=["p1", "p2", "p3"],
            image_patients=["p3", "p1", "p2"],
        )
        assert res["status"] == "PERFECT_MATCH"
        assert res["is_perfect_match"] is True
        assert res["missing_from_folders"] == []
        assert res["missing_from_metadata"] == []

    def test_compare_patients_with_discrepancies(self) -> None:
        """Detect missing patients in both directions."""
        inspector = MetadataInspector()
        res = inspector.compare_patients(
            metadata_patients=["p1", "p2", "p3", "p5"],
            image_patients=["p2", "p3", "p4"],
        )
        assert res["status"] == "DISCREPANCY_DETECTED"
        assert res["is_perfect_match"] is False
        assert res["missing_from_folders"] == ["p1", "p5"]
        assert res["missing_from_metadata"] == ["p4"]


class TestSummaryReportGeneration:
    """Tests for generating comprehensive text summary report."""

    def test_summary_report_contains_all_sections(self, synthetic_metadata_file: Path) -> None:
        """Generated report must include shape, columns, missing, categoricals, describe, and alignment."""
        inspector = MetadataInspector()
        raw_df = pd.read_excel(synthetic_metadata_file)
        clean_df, detected_col = inspector.clean_metadata(raw_df)
        alignment = inspector.compare_patients(
            metadata_patients=clean_df["patient_id"].tolist(),
            image_patients=["p1", "p2", "p3", "p4", "p5"],
        )

        report = inspector.generate_summary(
            raw_df=raw_df,
            clean_df=clean_df,
            patient_col=detected_col,
            source_desc=str(synthetic_metadata_file),
            alignment=alignment,
        )

        assert "CADICA CLINICAL METADATA AUDIT & SUMMARY REPORT" in report
        assert "5 rows x 8 columns" in report
        assert "COLUMN SCHEMA & DATA TYPES" in report
        assert "MISSING VALUES DETAIL PER COLUMN" in report
        assert "CATEGORICAL COLUMNS: UNIQUE VALUES & VALUE COUNTS" in report
        assert "NUMERIC COLUMNS SUMMARY: describe()" in report
        assert "PATIENT ALIGNMENT AUDIT" in report
        assert "PERFECT_MATCH" in report


class TestEndToEndPipeline:
    """End-to-end integration tests using synthetic files and real dataset."""

    def test_inspect_with_synthetic_file(
        self, synthetic_metadata_file: Path, tmp_path: Path
    ) -> None:
        """Run full inspect pipeline with synthetic excel file."""
        inspector = MetadataInspector()
        summary_out = tmp_path / "summary.txt"
        clean_out = tmp_path / "clean.csv"

        results = inspector.inspect(
            metadata_path_override=synthetic_metadata_file,
            output_summary_path=summary_out,
            output_clean_path=clean_out,
        )

        assert summary_out.is_file()
        assert clean_out.is_file()
        assert results["clean_df"].shape[0] == 5
        assert results["patient_col"] == "Patient ID"
        assert results["clean_df"]["patient_id"].tolist() == ["p1", "p2", "p3", "p4", "p5"]

    def test_inspect_with_cadica_zip_if_available(self) -> None:
        """If CADICA.zip is present, test full metadata inspection against real data."""
        zip_path = Path("CADICA.zip")
        if not zip_path.is_file():
            pytest.skip("CADICA.zip not present in test environment.")

        inspector = MetadataInspector()
        results = inspector.inspect(dataset_root_override=zip_path)

        assert results["clean_df"].shape[0] == 42
        assert results["patient_col"] == "Patient ID"
        assert len(results["alignment"]["metadata_patients"]) == 42
        assert len(results["alignment"]["image_patients"]) == 42
        assert results["alignment"]["is_perfect_match"] is True
        assert results["alignment"]["status"] == "PERFECT_MATCH"
