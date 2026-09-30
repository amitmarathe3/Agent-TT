# Changelog

All notable changes to the Agent-TT (CADICA-Agent) project will be documented in this file.

## [0.5.0] - 2026-09-30

### Added
- Dataset Indexer module [`src/data/indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/indexer.py) generating CSV index manifests and class mapping JSON in [`data_index/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index):
  - [`data_index/frames_index.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/frames_index.csv): 31,500 frame records with split type, patient/video/frame IDs, keyframe status, groundtruth flags, and image dimensions.
  - [`data_index/annotations.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/annotations.csv): 6,161 bounding box records with absolute `[x, y, w, h]` and YOLO-normalized `[x_center, y_center, w_norm, h_norm]` coordinates.
  - [`data_index/videos_index.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/videos_index.csv): 668 video sequence summary records.
  - [`data_index/patients_index.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/patients_index.csv): 42 patient summary records.
  - [`data_index/classes.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/classes.json): Category string to integer mapping (`p0_20`, `p20_50`, `p50_70`, `p70_90`, `p90_98`, `p90_99`, `p100`).
- CLI script [`scripts/build_index.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/build_index.py) with `--limit-patients` option for quick subset indexing and `--dataset_root` path override.
- Unit test suite [`tests/test_dataset_indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_dataset_indexer.py) with synthetic dataset fixture and malformed text parsing tests (15 total tests passing clean).

## [0.4.0] - 2026-09-29

### Added
- Dataset Validator module [`src/data/validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/validator.py) checking `metadata.xlsx`, `selectedVideos`/`nonselectedVideos` directories, patient folders (`pX`), video folders (`vY`), `input/` frame PNG files, `selectedFrames.txt`, control files (`lesionVideos.txt`, `nonlesionVideos.txt`), and groundtruth annotation matches.
- Command-line interface script [`scripts/validate_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/validate_dataset.py) for running validation and rendering clear status reports (`PASS`/`WARN`/`FAIL`).
- Generated validation report at [`reports/dataset_validation.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/dataset_validation.txt) (verified 42 patients, 668 video sequences, 31,499 image frames, 3,996 groundtruth labels with PASS status).
- Unit test suite in [`tests/test_dataset_validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_dataset_validator.py) using a synthetic dataset fixture created in a temporary pytest environment (12 total tests passing clean).

## [0.3.0] - 2026-09-29

### Added
- Expanded [`requirements.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/requirements.txt) with full computer vision, deep learning, and data science dependencies (`albumentations`, `ultralytics`, `seaborn`, `scipy`, `jupyter`, etc.).
- Central configuration validation in [`src/utils/config.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/utils/config.py) validating required top-level keys (`dataset_root`, `metadata_file`, `folders`, `index_dir`, `reports_dir`, `outputs_dir`, `random_seed`, `image_extension`) and folder structure (`selected`, `nonselected`).
- Logging helper utility [`src/utils/logger.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/utils/logger.py) (`get_logger`) writing to console and `outputs/logs/run.log`.
- Reproducibility helper module [`src/utils/seed.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/utils/seed.py) (`set_seed`) covering Python `random`, `NumPy`, and `PyTorch` (including CUDA/cuDNN deterministic flags).
- Virtual environment setup guide in [`README.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/README.md) for Windows (PowerShell & CMD) and Linux/macOS.
- Unit tests in [`tests/test_config.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_config.py) verifying config validation, logging file output, and seed determinism (8 tests passing clean).

## [0.2.0] - 2026-09-29

### Phase 1.1 - Project scaffold created
- Created complete project scaffold directory structure (`src/`, `data_index/`, `scripts/`, `tests/`, `notebooks/`, `reports/figures/`, `outputs/`).
- Added central YAML configuration system ([`configs/config.yaml`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/configs/config.yaml)) and loader ([`src/utils/config.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/utils/config.py)).
- Developed `DatasetScanner` module ([`src/data/dataset_scanner.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/dataset_scanner.py)) to dynamically discover file structures, images, and annotations from zip archives and folders.
- Implemented unit test suite ([`tests/test_config.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_config.py), [`tests/test_dataset_scanner.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_dataset_scanner.py)).
- Created comprehensive [`.gitignore`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/.gitignore) excluding raw datasets, zip archives, model checkpoints, environment folders, and bytecode caches.
- Scanned `CADICA.zip` (36,233 files discovered: 31,500 PNG images, 4,732 annotation files) and generated [`data/dataset_summary.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data/dataset_summary.json).
- Added [`requirements.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/requirements.txt) with core computer vision and deep learning dependencies.

## [0.1.0] - 2026-09-29
- Project repository initialization and baseline guidelines acknowledgement.
