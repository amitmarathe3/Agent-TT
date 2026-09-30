"""Dataset validator module for CADICA coronary angiography dataset."""

import argparse
from pathlib import Path
from typing import Any
import zipfile

from src.utils.config import load_config
from src.utils.logger import get_logger


class DatasetValidator:
    """Validates the structural integrity, completeness, and consistency of the CADICA dataset."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Initialize DatasetValidator using central configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        logging_cfg = self.config.get("logging", {}) if isinstance(self.config.get("logging"), dict) else {}
        self.logger = get_logger(
            name="DatasetValidator",
            log_file=logging_cfg.get("log_file", "outputs/logs/run.log"),
            level=logging_cfg.get("level", "INFO")
        )

        self.dataset_root = Path(self.config.get("dataset_root", "CADICA.zip"))
        self.metadata_file = self.config.get("metadata_file", "metadata.xlsx")
        folders_cfg = self.config.get("folders", {})
        self.selected_folder_name = folders_cfg.get("selected", "selectedVideos")
        self.nonselected_folder_name = folders_cfg.get("nonselected", "nonselectedVideos")
        self.reports_dir = Path(self.config.get("reports_dir", "reports"))
        self.image_extension = self.config.get("image_extension", ".png")

    def validate(self, dataset_path_override: str | Path | None = None) -> dict[str, Any]:
        """Perform comprehensive dataset validation.

        Args:
            dataset_path_override: Optional override for dataset root path.

        Returns:
            Dict containing validation results, statistics, inconsistencies, and status.
        """
        root_path = Path(dataset_path_override) if dataset_path_override else self.dataset_root

        if not root_path.exists():
            # Check if root_path exists relative to workspace or as absolute
            self.logger.error("Dataset root path not found: %s", root_path.resolve())
            return {
                "status": "FAIL",
                "summary": "Dataset root path does not exist.",
                "inconsistencies": [f"Path not found: {root_path.resolve()}"],
                "totals": {"patients": 0, "videos": 0, "frames": 0, "groundtruth": 0}
            }

        if root_path.is_file() and root_path.suffix.lower() == ".zip":
            return self._validate_zip(root_path)
        else:
            return self._validate_directory(root_path)

    def _validate_directory(self, base_dir: Path) -> dict[str, Any]:
        """Validate an uncompressed directory structure."""
        inconsistencies: list[str] = []
        warnings: list[str] = []

        # Handle nested top-level folder e.g. base_dir/CADICA
        root = base_dir
        if not (root / self.selected_folder_name).exists() and (root / "CADICA" / self.selected_folder_name).exists():
            root = root / "CADICA"

        self.logger.info("Validating dataset directory: %s", root.resolve())

        # 1. Check metadata.xlsx
        meta_path = root / self.metadata_file
        if not meta_path.exists():
            inconsistencies.append(f"Missing metadata file: {self.metadata_file} in {root}")

        # 2. Check top-level selected / nonselected folders
        selected_dir = root / self.selected_folder_name
        nonselected_dir = root / self.nonselected_folder_name

        if not selected_dir.exists():
            inconsistencies.append(f"Missing folder: {self.selected_folder_name}")
        if not nonselected_dir.exists():
            inconsistencies.append(f"Missing folder: {self.nonselected_folder_name}")

        patients_set = set()
        total_videos = 0
        total_frames = 0
        total_groundtruth = 0

        # 3. Validate selectedVideos
        if selected_dir.exists():
            for p_dir in sorted(selected_dir.glob("p*")):
                if not p_dir.is_dir():
                    continue
                patient_id = p_dir.name
                patients_set.add(patient_id)

                # Check lesionVideos.txt & nonlesionVideos.txt
                for control_file in ["lesionVideos.txt", "nonlesionVideos.txt"]:
                    if not (p_dir / control_file).exists():
                        warnings.append(f"Patient {patient_id}: missing control file {control_file}")

                for v_dir in sorted(p_dir.glob("v*")):
                    if not v_dir.is_dir():
                        continue
                    total_videos += 1
                    video_id = v_dir.name

                    # Validate input folder & frames
                    input_dir = v_dir / "input"
                    if not input_dir.exists() or not input_dir.is_dir():
                        inconsistencies.append(f"Patient {patient_id}, Video {video_id}: missing 'input' directory")
                        input_frames = set()
                    else:
                        frame_files = list(input_dir.glob(f"*{self.image_extension}"))
                        input_frames = {f.stem for f in frame_files}
                        total_frames += len(frame_files)

                    # Validate selectedFrames.txt
                    sf_file = v_dir / f"{patient_id}_{video_id}_selectedFrames.txt"
                    selected_frames_list = set()
                    if not sf_file.exists():
                        warnings.append(f"Patient {patient_id}, Video {video_id}: missing selectedFrames.txt file ({sf_file.name})")
                    else:
                        try:
                            lines = sf_file.read_text(encoding="utf-8").splitlines()
                            selected_frames_list = {line.strip() for line in lines if line.strip()}
                        except Exception as exc:
                            warnings.append(f"Patient {patient_id}, Video {video_id}: failed to read {sf_file.name} ({exc})")

                    # Validate groundtruth folder & matching frames
                    gt_dir = v_dir / "groundtruth"
                    if gt_dir.exists() and gt_dir.is_dir():
                        gt_files = list(gt_dir.glob("*.txt"))
                        total_groundtruth += len(gt_files)

                        for gt_file in gt_files:
                            frame_name = gt_file.stem
                            # Check matching frame in input/
                            if frame_name not in input_frames:
                                inconsistencies.append(
                                    f"Patient {patient_id}, Video {video_id}: Groundtruth file {gt_file.name} has no matching frame in input/"
                                )

                            # Check entry in selectedFrames.txt if selectedFrames.txt exists
                            if selected_frames_list and frame_name not in selected_frames_list and f"{frame_name}{self.image_extension}" not in selected_frames_list:
                                warnings.append(
                                    f"Patient {patient_id}, Video {video_id}: Groundtruth frame {frame_name} is not listed in selectedFrames.txt"
                                )

        # 4. Validate nonselectedVideos
        if nonselected_dir.exists():
            for p_dir in sorted(nonselected_dir.glob("p*")):
                if not p_dir.is_dir():
                    continue
                patient_id = p_dir.name
                patients_set.add(patient_id)

                for v_dir in sorted(p_dir.glob("v*")):
                    if not v_dir.is_dir():
                        continue
                    total_videos += 1
                    video_id = v_dir.name

                    input_dir = v_dir / "input"
                    if not input_dir.exists() or not input_dir.is_dir():
                        inconsistencies.append(f"Nonselected Patient {patient_id}, Video {video_id}: missing 'input' directory")
                    else:
                        frame_files = list(input_dir.glob(f"*{self.image_extension}"))
                        total_frames += len(frame_files)

        status = "FAIL" if inconsistencies else ("WARN" if warnings else "PASS")
        all_issues = inconsistencies + [f"[WARN] {w}" for w in warnings]

        return {
            "status": status,
            "totals": {
                "patients": len(patients_set),
                "videos": total_videos,
                "frames": total_frames,
                "groundtruth": total_groundtruth
            },
            "inconsistencies": all_issues,
            "errors_count": len(inconsistencies),
            "warnings_count": len(warnings)
        }

    def _validate_zip(self, zip_path: Path) -> dict[str, Any]:
        """Validate zip file contents by analyzing member paths."""
        self.logger.info("Validating dataset zip archive: %s", zip_path.resolve())

        inconsistencies: list[str] = []
        warnings: list[str] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = set(zf.namelist())

            # Find root prefix inside zip if applicable
            has_meta = any(n.endswith(self.metadata_file) for n in namelist)
            if not has_meta:
                inconsistencies.append(f"Missing metadata file: {self.metadata_file} inside zip")

            has_selected = any(f"/{self.selected_folder_name}/" in n or n.startswith(f"{self.selected_folder_name}/") for n in namelist)
            has_nonselected = any(f"/{self.nonselected_folder_name}/" in n or n.startswith(f"{self.nonselected_folder_name}/") for n in namelist)

            if not has_selected:
                inconsistencies.append(f"Missing folder: {self.selected_folder_name} inside zip")
            if not has_nonselected:
                inconsistencies.append(f"Missing folder: {self.nonselected_folder_name} inside zip")

            patients_set = set()
            videos_set = set()
            input_frames = set()
            gt_files = set()
            selected_frames_entries = set()

            total_frames = 0
            total_gt = 0

            for name in namelist:
                parts = Path(name).parts
                if not parts:
                    continue

                # Locate patient (pX) and video (vY) in path parts
                curr_p = None
                curr_v = None
                for i, p in enumerate(parts):
                    if p.startswith("p") and p[1:].isdigit():
                        patients_set.add(p)
                        curr_p = p
                        if i + 1 < len(parts) and parts[i+1].startswith("v") and parts[i+1][1:].isdigit():
                            curr_v = parts[i+1]
                            video_key = f"{p}_{parts[i+1]}"
                            videos_set.add(video_key)

                if name.endswith(self.image_extension) and "/input/" in name:
                    total_frames += 1
                    stem = Path(name).stem
                    input_frames.add(stem)
                    # Check for misplaced frame filenames
                    if curr_p and curr_v and not stem.startswith(f"{curr_p}_{curr_v}_"):
                        warnings.append(
                            f"Misplaced/anomalous frame filename '{stem}' found in {curr_p}/{curr_v}/input/"
                        )
                elif name.endswith(".txt") and "/groundtruth/" in name:
                    total_gt += 1
                    gt_files.add(Path(name).stem)

            # Check groundtruth matching frame
            for gt_stem in gt_files:
                if gt_stem not in input_frames:
                    inconsistencies.append(f"Groundtruth frame {gt_stem}.txt has no matching frame image in input/")

        status = "FAIL" if inconsistencies else ("WARN" if warnings else "PASS")
        all_issues = inconsistencies + [f"[WARN] {w}" for w in warnings]

        return {
            "status": status,
            "totals": {
                "patients": len(patients_set),
                "videos": len(videos_set),
                "frames": total_frames,
                "groundtruth": total_gt
            },
            "inconsistencies": all_issues,
            "errors_count": len(inconsistencies),
            "warnings_count": len(warnings)
        }

    def generate_report(self, results: dict[str, Any], output_path: str | Path | None = None) -> str:
        """Format validation results into a human-readable text report and save to disk.

        Args:
            results: Validation result dictionary.
            output_path: File path to write text report to.

        Returns:
            Formatted text report string.
        """
        save_path = Path(output_path) if output_path else self.reports_dir / "dataset_validation.txt"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "==================================================",
            "        AGENT-TT DATASET VALIDATION REPORT        ",
            "==================================================",
            f"Overall Status: {results['status']}",
            "--------------------------------------------------",
            "Dataset Statistics:",
            f"  - Total Patients Detected:  {results['totals']['patients']}",
            f"  - Total Video Sequences:     {results['totals']['videos']}",
            f"  - Total Image Frames:        {results['totals']['frames']}",
            f"  - Total Groundtruth Labels:  {results['totals']['groundtruth']}",
            "--------------------------------------------------",
            f"Inconsistencies & Issues ({len(results['inconsistencies'])} item(s)):",
        ]

        if not results['inconsistencies']:
            lines.append("  [NONE] All dataset checks passed cleanly.")
        else:
            for issue in results['inconsistencies']:
                lines.append(f"  - {issue}")

        lines.extend([
            "==================================================",
            ""
        ])

        report_str = "\n".join(lines)
        save_path.write_text(report_str, encoding="utf-8")
        self.logger.info("Validation report written to: %s", save_path.resolve())

        return report_str
