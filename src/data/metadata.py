"""Clinical metadata inspector and cleaner for the CADICA coronary angiography dataset.

Loads metadata.xlsx using pandas and performs:
- Automated patient ID column detection without hardcoded column names
- Patient ID normalization to the standard "pX" folder format
- Comprehensive summary generation: shape, column names, dtypes, missing values per column,
  unique values for categorical columns, and describe() for numeric columns
- Cross-validation against image folder patient IDs (identifying missing patients in both directions)
- Export of clean metadata to data_index/metadata_clean.csv
- Generation of detailed audit report saved to reports/metadata_summary.txt
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import get_logger


def natural_patient_key(pid: str) -> tuple[int, str]:
    """Sort key for natural ordering of patient IDs (e.g. p1, p2, ..., p10).

    Args:
        pid: Patient identifier string.

    Returns:
        Tuple of (integer_value, original_string).
    """
    m = re.search(r"(\d+)", str(pid))
    if m:
        return (int(m.group(1)), str(pid))
    return (999999, str(pid))


def normalize_patient_id(val: Any) -> str:
    """Normalize any patient identifier value into standard 'pX' format.

    Handles integers, floats, numeric strings, and strings with 'p' or 'P' prefixes.

    Examples:
        >>> normalize_patient_id(1)
        'p1'
        >>> normalize_patient_id("p01")
        'p1'
        >>> normalize_patient_id("42")
        'p42'

    Args:
        val: Raw value from metadata dataframe.

    Returns:
        Normalized patient string 'pX' (where X is an integer without leading zeros),
        or empty string if value is null or invalid.
    """
    if val is None or pd.isna(val):
        return ""

    if isinstance(val, (int, float)):
        try:
            int_val = int(round(float(val)))
            return f"p{int_val}"
        except (ValueError, OverflowError):
            pass

    s = str(val).strip()
    if not s:
        return ""

    # Matches p1, P1, p01, P001, p 1
    m = re.match(r"^[pP]?\s*0*([0-9]+)$", s)
    if m:
        return f"p{int(m.group(1))}"

    # Matches patient_1, sub-02, case 3
    m_end = re.search(r"(\d+)$", s)
    if m_end:
        return f"p{int(m_end.group(1))}"

    return s


class MetadataInspector:
    """Inspects, validates, cleans, and cross-references CADICA clinical metadata."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Initialize the metadata inspector using project configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        logging_cfg = self.config.get("logging", {}) if isinstance(self.config.get("logging"), dict) else {}
        self.logger = get_logger(
            name="MetadataInspector",
            log_file=logging_cfg.get("log_file", "outputs/logs/run.log"),
            level=logging_cfg.get("level", "INFO"),
        )

        self.dataset_root = Path(self.config.get("dataset_root", "CADICA.zip"))
        self.metadata_file = self.config.get("metadata_file", "metadata.xlsx")
        folders_cfg = self.config.get("folders", {})
        self.selected_folder_name = folders_cfg.get("selected", "selectedVideos")
        self.nonselected_folder_name = folders_cfg.get("nonselected", "nonselectedVideos")
        self.index_dir = Path(self.config.get("index_dir", "data_index"))
        self.reports_dir = Path(self.config.get("reports_dir", "reports"))

    def resolve_metadata_source(
        self,
        metadata_path_override: str | Path | None = None,
        dataset_root_override: str | Path | None = None,
    ) -> tuple[Path | None, bytes | None, str]:
        """Locate metadata.xlsx from override, dataset_root (dir or zip), or default paths.

        Args:
            metadata_path_override: Explicit path to metadata file, folder, or zip archive.
            dataset_root_override: Explicit path to dataset root directory or zip.

        Returns:
            Tuple of (file_path, raw_bytes, description_source). Exactly one of
            file_path or raw_bytes will be non-None on success.

        Raises:
            FileNotFoundError: If metadata file cannot be located.
        """
        # 1. Direct metadata_path_override provided
        if metadata_path_override:
            meta_path = Path(metadata_path_override)
            if meta_path.is_file():
                if meta_path.suffix.lower() == ".zip":
                    return self._find_metadata_in_zip(meta_path)
                return meta_path, None, str(meta_path.resolve())
            if meta_path.is_dir():
                found = self._find_metadata_in_dir(meta_path)
                if found:
                    return found, None, str(found.resolve())

        # 2. Check dataset_root (override or config)
        ds_root = Path(dataset_root_override) if dataset_root_override else self.dataset_root
        if ds_root.exists():
            if ds_root.is_file() and ds_root.suffix.lower() == ".zip":
                return self._find_metadata_in_zip(ds_root)
            if ds_root.is_dir():
                found = self._find_metadata_in_dir(ds_root)
                if found:
                    return found, None, str(found.resolve())

        # 3. Check standalone path in current directory or data/
        for candidate in [
            Path(self.metadata_file),
            Path("data") / self.metadata_file,
            Path("CADICA") / self.metadata_file,
        ]:
            if candidate.is_file():
                return candidate, None, str(candidate.resolve())

        raise FileNotFoundError(
            f"Could not locate metadata file '{self.metadata_file}' in dataset root "
            f"'{ds_root}' or default paths."
        )

    def _find_metadata_in_dir(self, directory: Path) -> Path | None:
        """Search directory for metadata.xlsx."""
        candidates = [
            directory / self.metadata_file,
            directory / "CADICA" / self.metadata_file,
        ]
        for c in candidates:
            if c.is_file():
                return c

        for f in directory.rglob(f"*{self.metadata_file}"):
            if f.is_file():
                return f
        return None

    def _find_metadata_in_zip(self, zip_path: Path) -> tuple[None, bytes, str]:
        """Extract metadata.xlsx bytes from a zip archive."""
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            target_entry = None
            for name in namelist:
                if name.endswith(self.metadata_file) and not name.startswith("__MACOSX"):
                    target_entry = name
                    break

            if target_entry is None:
                raise FileNotFoundError(
                    f"Metadata file '{self.metadata_file}' not found inside zip archive: {zip_path}"
                )

            self.logger.info("Reading '%s' directly from zip '%s'", target_entry, zip_path)
            raw_bytes = zf.read(target_entry)
            return None, raw_bytes, f"{zip_path} :: {target_entry}"

    def load_raw_metadata(
        self,
        metadata_path_override: str | Path | None = None,
        dataset_root_override: str | Path | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Load raw metadata dataframe from Excel file or zip stream.

        Args:
            metadata_path_override: Optional path override.
            dataset_root_override: Optional dataset root override.

        Returns:
            Tuple of (DataFrame, source_description).
        """
        file_path, raw_bytes, source_desc = self.resolve_metadata_source(
            metadata_path_override=metadata_path_override,
            dataset_root_override=dataset_root_override,
        )

        if raw_bytes is not None:
            df = pd.read_excel(io.BytesIO(raw_bytes), engine="openpyxl")
        else:
            df = pd.read_excel(file_path, engine="openpyxl")

        self.logger.info("Loaded metadata from '%s' with shape %s", source_desc, df.shape)
        return df, source_desc

    def detect_patient_id_column(self, df: pd.DataFrame) -> tuple[str, float]:
        """Automatically detect the patient identifier column without assuming column names.

        Uses a hybrid heuristic combining name tokens and value distribution:
        - High uniqueness ratio (>0.8)
        - Patient pattern matching ('p1', integers, etc.)
        - Penalization for binary flags or clinical measurement columns

        Args:
            df: Raw metadata DataFrame.

        Returns:
            Tuple of (detected_column_name, confidence_score).

        Raises:
            ValueError: If no suitable patient ID column could be identified.
        """
        best_col: str | None = None
        best_score = -1000.0

        patient_name_tokens = {"patient", "pat", "subject", "sub", "case", "id", "pid", "paciente"}
        clinical_penalty_tokens = {
            "age", "height", "weight", "bmi", "pressure", "vessel", "failure",
            "smoker", "diabetes", "evolution", "cholesterol", "dyslipidemia",
            "event", "stenosis", "comorbidity", "ejection", "indication",
        }

        total_rows = len(df)
        if total_rows == 0:
            raise ValueError("Metadata DataFrame is empty; cannot detect patient ID column.")

        for col in df.columns:
            score = 0.0
            col_str = str(col).lower()

            # 1. Column name scoring
            name_words = set(re.split(r"[\s_\-(),.]+", col_str))
            if any(token in col_str for token in patient_name_tokens):
                score += 40.0
            if "patient id" in col_str or "patient_id" in col_str or "patientid" in col_str:
                score += 30.0
            if any(pen in col_str for pen in clinical_penalty_tokens):
                score -= 60.0

            # 2. Value distribution scoring
            series = df[col].dropna()
            if len(series) == 0:
                continue

            nunique = series.nunique()
            uniqueness_ratio = nunique / total_rows

            if uniqueness_ratio == 1.0:
                score += 50.0
            elif uniqueness_ratio >= 0.8:
                score += 25.0
            elif uniqueness_ratio < 0.3:
                score -= 50.0

            # Check for boolean or binary flags {0, 1}
            unique_set = set(series.unique())
            if unique_set.issubset({0, 1, 0.0, 1.0, "0", "1"}):
                score -= 80.0

            # 3. Check value patterns
            p_pattern_matches = 0
            integer_matches = 0

            for val in series:
                s_val = str(val).strip()
                if re.match(r"^[pP]\s*0*([1-9]\d*)$", s_val):
                    p_pattern_matches += 1
                try:
                    num = float(val)
                    if num.is_integer() and 1 <= int(num) <= 100000:
                        integer_matches += 1
                except (ValueError, TypeError):
                    pass

            p_ratio = p_pattern_matches / len(series)
            int_ratio = integer_matches / len(series)

            if p_ratio >= 0.8:
                score += 60.0
            elif int_ratio >= 0.8 and uniqueness_ratio >= 0.8:
                score += 40.0

            self.logger.debug(
                "Candidate column '%s': score=%.1f (uniqueness=%.2f, p_ratio=%.2f, int_ratio=%.2f)",
                col, score, uniqueness_ratio, p_ratio, int_ratio,
            )

            if score > best_score:
                best_score = score
                best_col = str(col)

        if best_col is None or best_score <= 0:
            raise ValueError(
                f"Failed to automatically detect patient ID column. Columns: {list(df.columns)}"
            )

        self.logger.info("Detected patient ID column: '%s' (score=%.1f)", best_col, best_score)
        return best_col, best_score

    def clean_metadata(
        self,
        df: pd.DataFrame,
        patient_col: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Normalize patient IDs and produce cleaned DataFrame.

        Normalizes patient IDs to 'pX' format, standardizes column name to 'patient_id',
        and strips surrounding whitespace from text fields.

        Args:
            df: Raw metadata DataFrame.
            patient_col: Optional column name. If None, auto-detected.

        Returns:
            Tuple of (cleaned_df, detected_patient_col).
        """
        if patient_col is None:
            patient_col, _ = self.detect_patient_id_column(df)

        clean_df = df.copy()

        # Normalize patient IDs
        normalized_ids = clean_df[patient_col].apply(normalize_patient_id)
        clean_df[patient_col] = normalized_ids

        # Rename to standardized 'patient_id'
        if patient_col != "patient_id":
            clean_df = clean_df.rename(columns={patient_col: "patient_id"})

        # Clean string columns: strip trailing whitespace and non-breaking spaces
        for col in clean_df.columns:
            if clean_df[col].dtype == "object":
                clean_df[col] = clean_df[col].apply(
                    lambda x: x.replace("\xa0", " ").strip() if isinstance(x, str) else x
                )

        # Place 'patient_id' as the first column
        cols = ["patient_id"] + [c for c in clean_df.columns if c != "patient_id"]
        clean_df = clean_df[cols]

        return clean_df, patient_col

    def discover_image_patients(
        self,
        dataset_root_override: str | Path | None = None,
    ) -> list[str]:
        """Discover all patient IDs present in the CADICA image folders.

        Checks data_index/patients_index.csv if available, or scans dataset_root
        (zip or directory).

        Args:
            dataset_root_override: Optional path to dataset directory or zip archive.

        Returns:
            Sorted list of normalized patient IDs (e.g. ['p1', 'p2', ...]).
        """
        # 1. Check data_index/patients_index.csv first if it exists
        patients_csv = self.index_dir / "patients_index.csv"
        if patients_csv.is_file():
            try:
                p_df = pd.read_csv(patients_csv)
                if "patient_id" in p_df.columns:
                    p_list = [normalize_patient_id(p) for p in p_df["patient_id"].dropna()]
                    p_list = sorted(list(set(p_list)), key=natural_patient_key)
                    self.logger.info("Discovered %d patients from %s", len(p_list), patients_csv)
                    return p_list
            except Exception as e:
                self.logger.warning("Could not read patients_index.csv: %s", e)

        # 2. Scan dataset_root (override or config)
        ds_root = Path(dataset_root_override) if dataset_root_override else self.dataset_root
        if not ds_root.exists():
            self.logger.warning("Dataset root does not exist: %s", ds_root)
            return []

        patient_set: set[str] = set()

        if ds_root.is_file() and ds_root.suffix.lower() == ".zip":
            with zipfile.ZipFile(ds_root, "r") as zf:
                for name in zf.namelist():
                    for part in Path(name).parts:
                        if re.match(r"^p\d+$", part, re.IGNORECASE):
                            patient_set.add(normalize_patient_id(part))
        elif ds_root.is_dir():
            for folder_name in [self.selected_folder_name, self.nonselected_folder_name]:
                folder_dir = ds_root / folder_name
                if not folder_dir.exists() and (ds_root / "CADICA" / folder_name).exists():
                    folder_dir = ds_root / "CADICA" / folder_name

                if folder_dir.exists():
                    for p_dir in folder_dir.glob("p*"):
                        if p_dir.is_dir() and re.match(r"^p\d+$", p_dir.name, re.IGNORECASE):
                            patient_set.add(normalize_patient_id(p_dir.name))

        sorted_patients = sorted(list(patient_set), key=natural_patient_key)
        self.logger.info(
            "Discovered %d patients from image folders in %s", len(sorted_patients), ds_root
        )
        return sorted_patients

    def compare_patients(
        self,
        metadata_patients: list[str],
        image_patients: list[str],
    ) -> dict[str, Any]:
        """Cross-reference metadata patient IDs against image folder patient IDs.

        Args:
            metadata_patients: List of patient IDs from metadata.
            image_patients: List of patient IDs from image folders.

        Returns:
            Dict containing comparison counts, missing lists, and match status.
        """
        meta_set = set(normalize_patient_id(p) for p in metadata_patients if p)
        image_set = set(normalize_patient_id(p) for p in image_patients if p)

        missing_from_folders = sorted(list(meta_set - image_set), key=natural_patient_key)
        missing_from_metadata = sorted(list(image_set - meta_set), key=natural_patient_key)
        common = sorted(list(meta_set & image_set), key=natural_patient_key)

        is_perfect = len(missing_from_folders) == 0 and len(missing_from_metadata) == 0

        status = "PERFECT_MATCH" if is_perfect else "DISCREPANCY_DETECTED"

        return {
            "status": status,
            "is_perfect_match": is_perfect,
            "metadata_patients_count": len(meta_set),
            "image_patients_count": len(image_set),
            "common_count": len(common),
            "missing_from_folders": missing_from_folders,
            "missing_from_metadata": missing_from_metadata,
            "metadata_patients": sorted(list(meta_set), key=natural_patient_key),
            "image_patients": sorted(list(image_set), key=natural_patient_key),
        }

    def generate_summary(
        self,
        raw_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        patient_col: str,
        source_desc: str,
        alignment: dict[str, Any],
    ) -> str:
        """Format an extensive textual summary report for metadata audit.

        Covers shape, column names, dtypes, missing values per column,
        unique values for categorical columns, describe() for numeric columns,
        and patient alignment.

        Args:
            raw_df: Original metadata DataFrame.
            clean_df: Cleaned metadata DataFrame with normalized 'patient_id'.
            patient_col: Original detected patient ID column name.
            source_desc: Path or description of metadata source.
            alignment: Result dictionary from compare_patients().

        Returns:
            Structured report string.
        """
        lines: list[str] = []
        sep_major = "=" * 80
        sep_minor = "-" * 80

        lines.append(sep_major)
        lines.append("CADICA CLINICAL METADATA AUDIT & SUMMARY REPORT")
        lines.append(sep_major)
        lines.append(f"Source Location       : {source_desc}")
        lines.append(f"Raw Metadata Shape    : {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")
        lines.append(f"Clean Metadata Shape  : {clean_df.shape[0]} rows x {clean_df.shape[1]} columns")
        lines.append(f"Detected Patient Col  : '{patient_col}' -> normalized to 'patient_id' (pX)")
        lines.append("")

        # ── 1. Column Names and Data Types ────────────────────────────────────
        lines.append(sep_minor)
        lines.append("1. COLUMN SCHEMA & DATA TYPES")
        lines.append(sep_minor)
        lines.append(f"{'Idx':<4} {'Column Name':<50} {'Dtype':<12} {'Missing':<10} {'Missing %':<10}")
        lines.append(f"{'---':<4} {'-'*48:<50} {'-'*10:<12} {'-'*8:<10} {'-'*8:<10}")

        total_cells = raw_df.shape[0] * raw_df.shape[1]
        total_missing = int(raw_df.isnull().sum().sum())

        for idx, col in enumerate(raw_df.columns, start=1):
            dtype_str = str(raw_df[col].dtype)
            missing_count = int(raw_df[col].isnull().sum())
            missing_pct = (missing_count / len(raw_df)) * 100.0 if len(raw_df) > 0 else 0.0
            lines.append(f"{idx:<4} {str(col):<50} {dtype_str:<12} {missing_count:<10} {missing_pct:>6.2f}%")

        lines.append("")
        lines.append(f"Total Dataset Cells   : {total_cells:,}")
        lines.append(f"Total Missing Values  : {total_missing:,} ({total_missing / total_cells * 100.0:.2f}% of all cells)")
        lines.append("")

        # ── 2. Missing Values Detail ──────────────────────────────────────────
        lines.append(sep_minor)
        lines.append("2. MISSING VALUES DETAIL PER COLUMN")
        lines.append(sep_minor)
        missing_series = raw_df.isnull().sum()
        cols_with_missing = missing_series[missing_series > 0].sort_values(ascending=False)

        if cols_with_missing.empty:
            lines.append("No missing values found across any columns (100% complete).")
        else:
            lines.append(f"Columns with missing values ({len(cols_with_missing)} total):")
            for col_name, cnt in cols_with_missing.items():
                pct = (cnt / len(raw_df)) * 100.0
                lines.append(f"  - {col_name:<48} : {cnt:>3} missing ({pct:>6.2f}%)")
        lines.append("")

        # ── 3. Categorical Columns: Unique Values ─────────────────────────────
        lines.append(sep_minor)
        lines.append("3. CATEGORICAL COLUMNS: UNIQUE VALUES & VALUE COUNTS")
        lines.append(sep_minor)

        # Identify categorical columns: object, category, bool, or discrete low-cardinality
        cat_cols = []
        for c in raw_df.columns:
            if c == patient_col:
                continue
            is_obj = raw_df[c].dtype in ("object", "category", "bool")
            # Low cardinality numeric discrete flags (e.g. binary flags <= 5 unique values)
            is_low_cardinality = (
                pd.api.types.is_numeric_dtype(raw_df[c])
                and raw_df[c].nunique(dropna=True) <= 3
            )
            if is_obj or is_low_cardinality:
                cat_cols.append(c)

        if not cat_cols:
            lines.append("No categorical or discrete columns found.")
        else:
            for c in cat_cols:
                n_uniq = raw_df[c].nunique(dropna=False)
                lines.append(f"Column: '{c}' (Type: {raw_df[c].dtype}, Unique values: {n_uniq})")
                val_counts = raw_df[c].value_counts(dropna=False)
                for val, count in val_counts.items():
                    val_repr = "<NA / Missing>" if pd.isna(val) else str(val)
                    pct = (count / len(raw_df)) * 100.0
                    lines.append(f"    - {val_repr:<45} : {count:>3} ({pct:>5.1f}%)")
                lines.append("")

        # ── 4. Numeric Columns: describe() ────────────────────────────────────
        lines.append(sep_minor)
        lines.append("4. NUMERIC COLUMNS SUMMARY: describe()")
        lines.append(sep_minor)

        # Exclude patient_id from numeric summary
        num_cols = [
            c for c in clean_df.select_dtypes(include=["number"]).columns
            if c != "patient_id"
        ]
        if num_cols:
            desc_table = clean_df[num_cols].describe().to_string()
            lines.append(desc_table)
        else:
            lines.append("No numeric columns present.")
        lines.append("")

        # ── 5. Patient Alignment Report ───────────────────────────────────────
        lines.append(sep_minor)
        lines.append("5. PATIENT ALIGNMENT AUDIT (Metadata vs Image Folders)")
        lines.append(sep_minor)
        lines.append(f"Patients in Metadata       : {alignment['metadata_patients_count']}")
        lines.append(f"Patients in Image Folders  : {alignment['image_patients_count']}")
        lines.append(f"Common Patients Aligned    : {alignment['common_count']}")
        lines.append(f"Alignment Status           : {alignment['status']}")
        lines.append("")

        lines.append("Patients in Metadata MISSING from Image Folders:")
        if alignment["missing_from_folders"]:
            lines.append(f"  Count: {len(alignment['missing_from_folders'])}")
            for p in alignment["missing_from_folders"]:
                lines.append(f"    - {p}")
        else:
            lines.append("  None (all metadata patients exist in image folders).")

        lines.append("")
        lines.append("Patients in Image Folders MISSING from Metadata:")
        if alignment["missing_from_metadata"]:
            lines.append(f"  Count: {len(alignment['missing_from_metadata'])}")
            for p in alignment["missing_from_metadata"]:
                lines.append(f"    - {p}")
        else:
            lines.append("  None (all folder patients exist in clinical metadata).")

        lines.append("")
        lines.append(sep_major)
        lines.append("END OF METADATA SUMMARY REPORT")
        lines.append(sep_major)

        return "\n".join(lines)

    def inspect(
        self,
        metadata_path_override: str | Path | None = None,
        dataset_root_override: str | Path | None = None,
        output_summary_path: str | Path | None = None,
        output_clean_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Execute the full metadata inspection, normalization, and audit pipeline.

        1. Loads raw metadata from Excel or zip stream.
        2. Detects patient ID column and normalizes to 'pX'.
        3. Saves cleaned CSV to data_index/metadata_clean.csv.
        4. Cross-references patient IDs with image folders.
        5. Generates and saves audit report to reports/metadata_summary.txt.

        Args:
            metadata_path_override: Optional path to metadata file or archive.
            dataset_root_override: Optional dataset root directory or zip.
            output_summary_path: Optional output path for summary report text.
            output_clean_path: Optional output path for clean CSV.

        Returns:
            Dict containing raw_df, clean_df, patient_col, summary_report,
            summary_path, clean_path, and alignment details.
        """
        # 1. Load raw metadata
        raw_df, source_desc = self.load_raw_metadata(
            metadata_path_override=metadata_path_override,
            dataset_root_override=dataset_root_override,
        )

        # 2. Detect & clean
        clean_df, patient_col = self.clean_metadata(raw_df)

        # 3. Discover image folder patients
        image_patients = self.discover_image_patients(
            dataset_root_override=dataset_root_override
        )
        meta_patients = clean_df["patient_id"].tolist()

        # 4. Compare
        alignment = self.compare_patients(
            metadata_patients=meta_patients,
            image_patients=image_patients,
        )

        # 5. Generate summary report text
        summary_text = self.generate_summary(
            raw_df=raw_df,
            clean_df=clean_df,
            patient_col=patient_col,
            source_desc=source_desc,
            alignment=alignment,
        )

        # 6. Save clean CSV
        clean_path = Path(output_clean_path) if output_clean_path else self.index_dir / "metadata_clean.csv"
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(clean_path, index=False, encoding="utf-8")
        self.logger.info("Saved cleaned metadata (%d rows) -> %s", len(clean_df), clean_path)

        # 7. Save summary report text
        summary_path = (
            Path(output_summary_path)
            if output_summary_path
            else self.reports_dir / "metadata_summary.txt"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text, encoding="utf-8")
        self.logger.info("Saved metadata summary report -> %s", summary_path)

        return {
            "raw_df": raw_df,
            "clean_df": clean_df,
            "patient_col": patient_col,
            "summary_report": summary_text,
            "summary_path": summary_path,
            "clean_path": clean_path,
            "alignment": alignment,
        }
