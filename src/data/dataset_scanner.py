"""Scanner module to inspect dataset archives and directory structures without assuming layout."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import zipfile

from src.utils.config import load_config
from src.utils.logger import get_logger


class DatasetScanner:
    """Discovers file structure, image formats, annotations, and metadata from dataset paths or zip archives."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".tif", ".tiff", ".bmp"}
    ANNOTATION_EXTENSIONS = {".csv", ".json", ".xml", ".txt", ".mat", ".tsv"}

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        """Initialize DatasetScanner using project configuration.

        Args:
            config_path: Path to YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)

        logging_cfg = self.config.get("logging", {}) if isinstance(self.config.get("logging"), dict) else {}
        self.logger = get_logger(
            name="DatasetScanner",
            log_file=logging_cfg.get("log_file", "outputs/logs/run.log"),
            level=logging_cfg.get("level", "INFO")
        )

        dataset_cfg = self.config.get("dataset", {})
        self.zip_path = Path(dataset_cfg.get("zip_path", "CADICA.zip"))
        self.extracted_dir = Path(dataset_cfg.get("extracted_dir", "data/extracted"))
        self.scan_output_path = Path(dataset_cfg.get("scan_output_path", "data/dataset_summary.json"))

    def scan_zip(self, zip_filepath: str | Path | None = None) -> dict[str, Any]:
        """Scan a zip archive without extracting all contents into memory or repo.

        Args:
            zip_filepath: Optional path override for zip file.

        Returns:
            Dict containing structural summary of the archive.
        """
        target_zip = Path(zip_filepath) if zip_filepath else self.zip_path

        if not target_zip.exists():
            raise FileNotFoundError(f"Zip dataset file not found at: {target_zip.resolve()}")

        self.logger.info("Scanning zip file: %s (Size: %.2f MB)", target_zip, target_zip.stat().st_size / (1024 * 1024))

        extension_counts: dict[str, int] = {}
        dir_depths: dict[int, int] = {}
        sample_files: list[str] = []
        image_files: list[str] = []
        annotation_files: list[str] = []
        total_files = 0
        total_uncompressed_bytes = 0

        with zipfile.ZipFile(target_zip, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                total_files += 1
                total_uncompressed_bytes += info.file_size
                path_obj = Path(info.filename)
                ext = path_obj.suffix.lower() if path_obj.suffix else "[no_extension]"

                extension_counts[ext] = extension_counts.get(ext, 0) + 1
                depth = len(path_obj.parts) - 1
                dir_depths[depth] = dir_depths.get(depth, 0) + 1

                if ext in self.IMAGE_EXTENSIONS:
                    image_files.append(info.filename)
                elif ext in self.ANNOTATION_EXTENSIONS:
                    annotation_files.append(info.filename)

                if len(sample_files) < 20:
                    sample_files.append(info.filename)

        summary = {
            "source_type": "zip",
            "source_path": str(target_zip.resolve()),
            "total_files": total_files,
            "total_uncompressed_size_mb": round(total_uncompressed_bytes / (1024 * 1024), 2),
            "extension_counts": extension_counts,
            "directory_depth_distribution": dir_depths,
            "image_count": len(image_files),
            "annotation_count": len(annotation_files),
            "sample_files": sample_files,
            "sample_images": image_files[:10],
            "sample_annotations": annotation_files[:10],
        }

        self.logger.info(
            "Zip Scan Complete. Found %d files (%d images, %d annotations).",
            total_files, len(image_files), len(annotation_files)
        )
        return summary

    def scan_directory(self, dir_path: str | Path) -> dict[str, Any]:
        """Scan an extracted or raw directory.

        Args:
            dir_path: Path to target directory.

        Returns:
            Dict containing structural summary.
        """
        target_dir = Path(dir_path)
        if not target_dir.exists() or not target_dir.is_dir():
            raise FileNotFoundError(f"Target dataset directory not found at: {target_dir.resolve()}")

        self.logger.info("Scanning directory: %s", target_dir)

        extension_counts: dict[str, int] = {}
        image_files: list[str] = []
        annotation_files: list[str] = []
        sample_files: list[str] = []
        total_files = 0
        total_bytes = 0

        for file_path in target_dir.rglob("*"):
            if file_path.is_file():
                total_files += 1
                file_size = file_path.stat().st_size
                total_bytes += file_size

                rel_path = str(file_path.relative_to(target_dir))
                ext = file_path.suffix.lower() if file_path.suffix else "[no_extension]"
                extension_counts[ext] = extension_counts.get(ext, 0) + 1

                if ext in self.IMAGE_EXTENSIONS:
                    image_files.append(rel_path)
                elif ext in self.ANNOTATION_EXTENSIONS:
                    annotation_files.append(rel_path)

                if len(sample_files) < 20:
                    sample_files.append(rel_path)

        summary = {
            "source_type": "directory",
            "source_path": str(target_dir.resolve()),
            "total_files": total_files,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "extension_counts": extension_counts,
            "image_count": len(image_files),
            "annotation_count": len(annotation_files),
            "sample_files": sample_files,
            "sample_images": image_files[:10],
            "sample_annotations": annotation_files[:10],
        }

        self.logger.info(
            "Directory Scan Complete. Found %d files (%d images, %d annotations).",
            total_files, len(image_files), len(annotation_files)
        )
        return summary

    def save_summary(self, summary: dict[str, Any], output_path: str | Path | None = None) -> Path:
        """Save dataset summary to a JSON file.

        Args:
            summary: Summary dictionary to save.
            output_path: Optional custom path to save JSON.

        Returns:
            Path object where file was written.
        """
        save_path = Path(output_path) if output_path else self.scan_output_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Saved dataset summary to: %s", save_path.resolve())
        return save_path


def main() -> None:
    """CLI entrypoint for dataset scanning."""
    parser = argparse.ArgumentParser(description="Agent-TT Dataset Scanner CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--zip_path", type=str, default=None, help="Override path to dataset zip archive")
    parser.add_argument("--dir_path", type=str, default=None, help="Path to scan an uncompressed directory")
    parser.add_argument("--output", type=str, default=None, help="Output path for JSON summary report")

    args = parser.parse_args()

    scanner = DatasetScanner(config_path=args.config)

    if args.dir_path:
        summary = scanner.scan_directory(args.dir_path)
    else:
        summary = scanner.scan_zip(args.zip_path)

    scanner.save_summary(summary, output_path=args.output)


if __name__ == "__main__":
    main()
