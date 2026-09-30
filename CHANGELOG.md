# Changelog

All notable changes to the Agent-TT (CADICA-Agent) project will be documented in this file.

## [0.3.0] - 2026-10-01 - Phase 2.2 Complete: Data Pipeline, YOLO Export & Medical Augmentation

### Added
- **YOLO Dataset Exporter**:
  - Implemented `YOLOExporter` in [`src/data/yolo_export.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/yolo_export.py) and CLI runner [`scripts/export_yolo.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/export_yolo.py).
  - Builds standard Ultralytics YOLO structure in `outputs/yolo_dataset/` (`images/{train,val,test}/`, `labels/{train,val,test}/`, `data.yaml`).
  - Strict image linkage: uses symlinks (or copy fallback) and reads directly from `CADICA.zip` or unpacked directories without checking raw images into version control.
  - Hard negative frame sampling via `--neg-ratio` (default `0.25`), selecting clean lesion-free frames from nonlesion videos and unannotated frames in lesion videos that are $\ge 15$ frames away from annotated keyframes (written with empty `.txt` labels).
  - Multi-class scheme support via `--scheme`: `7class` (all 7 stenosis tiers), `binary` (`normal`, `lesion`), and `severity3` (`mild`, `moderate`, `severe`).
  - Cross-validation fold export via `--fold N` (0 to 4), mapping fold assignments into test, val, and train sets.
  - Successfully exported full dataset: 4,994 images (3,996 positive frames + 998 sampled hard negatives) and 6,161 bounding boxes with zero patient leakage.
- **PyTorch Dataset for Custom Models**:
  - Implemented `CADICADataset` in [`src/data/dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/dataset.py).
  - Dynamic image loading from zip archives or filesystem directories with RGB conversion and optional float32 normalization.
  - Returns dictionary items: `image` (Tensor [3, H, W]), `boxes` (Tensor [N, 4]), `labels` (Tensor [N]), `patient_id` (str), `frame_id` (str), and optional clinical `metadata` features (normalized age, BMI, CAD risk flags).
  - Custom `collate_fn` handles variable box counts per frame and collates batches for custom PyTorch architectures.
- **Angiography-Safe Medical Data Augmentations**:
  - Implemented `get_train_transforms` and `get_val_transforms` in [`src/data/augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/augment.py) using Albumentations.
  - Tailored to invasive coronary angiography (ICA) physics and clinical morphology: CLAHE contrast enhancement, RandomBrightnessContrast, RandomGamma, GaussianBlur / GaussNoise, mild rotation ($\pm 10^\circ$), scaling (0.9–1.1), subtle translation, and optional horizontal flipping.
  - **Clinical Safety Constraints**: Strictly NO vertical flip (preserves catheter top-down anatomical orientation), NO color/hue jitter (preserves grayscale X-ray absorption physics), and NO aggressive crops (prevents cutting off stenosis lesions).
  - Added bbox post-transform safety bounding function `clip_yolo_boxes` preventing boundary overflow.
- **Pipeline Visualizer & Normalization Statistics**:
  - Implemented [`scripts/visualize_pipeline.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/visualize_pipeline.py).
  - Generates 3×4 sample grid with bounding boxes drawn and labeled at [`reports/figures/augmentation_check.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/augmentation_check.png).
  - Computed exact per-channel mean and standard deviation across all 3,496 training frames, saved to [`data_index/norm_stats.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/norm_stats.json): `mean: [0.5066, 0.5066, 0.5066]`, `std: [0.1385, 0.1385, 0.1385]`.
- **Test Suite**:
  - Created [`tests/test_augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_augment.py) (5 tests), [`tests/test_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_dataset.py) (5 tests), and [`tests/test_yolo_export.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_yolo_export.py) (5 tests).
  - All 84 project unit tests passing clean (100% pass rate).
- **Makefile Targets**:
  - Added `export-yolo` and `visualize-pipeline` targets to [`Makefile`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/Makefile).

## [0.2.0] - 2026-10-01 - Phase 2.1 Complete: Patient-Wise Splits & Cross-Validation

### Added
- Patient-wise dataset splitting module [`src/data/splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/splits.py) featuring `PatientSplitter`:
  - Strictly patient-level partitioning with 0 patient overlap across train, validation, and test splits (no patient data leakage).
  - Stratified randomized search algorithm optimizing class distribution and bounding box balance across splits.
  - Hard constraint enforcement: fails if any of the 7 stenosis classes has 0 bounding boxes in validation or test splits.
  - Rare-class balanced allocation: ensures rare class `p100` (present in exactly 3 patients: `p11`, `p23`, `p25`) has exactly one patient in each split (train, val, test), and `p99` (8 patients) is evenly distributed.
  - 5-fold patient-wise cross-validation generator allocating the 42 patients into 5 stratified test folds (8, 8, 8, 9, 9 patients) with zero patient overlap.
  - Configurable class severity schemes (`7class`, `binary`, `severity3`) with helper method `map_category()`.
  - Comprehensive Markdown summary report generator ([`data_index/splits/splits_report.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/splits_report.md)) detailing per-split patients, frames, boxes, and class breakdown percentages.
  - High-performance NumPy matrix vectorization achieving sub-second search iterations across 10,000 permutations.
- Command-line interface script [`scripts/make_splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/make_splits.py) with argparse options for `--config`, `--index-dir`, `--output-dir`, `--seed`, `--train-ratio`, `--val-ratio`, `--test-ratio`, `--n-folds`, and `--max-iter`.
- Make target `splits` added to [`Makefile`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/Makefile).
- Generated split manifests in [`data_index/splits/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index):
  - `split_v1.csv`: 42 patient rows (30 train, 6 val, 6 test) with 4,365 train boxes (70.8%), 757 val boxes (12.3%), and 1,039 test boxes (16.9%).
  - `folds_5.csv`: 42 patient rows mapped to folds 0–4 (8, 8, 8, 9, 9 patients) with balanced box counts (1,160 to 1,341 boxes per fold).
  - `splits_report.md`: Markdown report validating zero patient overlap and class distributions.
- Configuration additions in [`configs/config.yaml`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/configs/config.yaml):
  - `class_schemes`: `7class` (original 7 categories), `binary` (lesion vs nonlesion), `severity3` (mild, moderate, severe).
  - `splits`: `train_ratio: 0.70`, `val_ratio: 0.15`, `test_ratio: 0.15`, `n_folds: 5`, file paths, and `max_search_iter: 10000`.
- Comprehensive test suite in [`tests/test_splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_splits.py) with 16 tests covering zero patient overlap, class coverage, file outputs, natural patient sorting, and class scheme mappings (69 total project tests passing clean).

### Fixed
- Resolved 1-frame discrepancy between dataset validator (31,499 frames) and EDA (31,500 frames) in [`src/data/validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/validator.py):
  - Discovered that an anomalous extra frame `p30_v2_00021.png` was present in `selectedVideos/p32/v10/input/` alongside the legitimate frame in `selectedVideos/p30/v2/input/`.
  - The previous zip validator deduplicated filename stems in a `set()`, dropping 1 duplicate stem.
  - Updated validator to track total frame file count and flag misplaced frame names with an informative warning, aligning validator and EDA at exactly 31,500 frames.

## [0.1.0] - 2026-09-30 - Phase 1 Complete

### Milestone Summary
Phase 1 (Data Ingestion, Verification, Indexing, and Exploratory Data Analysis) is officially complete. The CADICA angiography dataset (42 patients, 668 videos, 31,500 frames, 6,161 bounding box annotations across 7 stenosis severity classes) has been completely validated, indexed, cleaned, and explored with 53/53 tests passing and reproducible one-command pipeline automation.

### Deliverables Completed
- **Pipeline Automation Scripts**:
  - [`run_phase1.bat`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase1.bat) (Windows batch runner)
  - [`run_phase1.sh`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase1.sh) (Linux/macOS Bash runner)
  - [`Makefile`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/Makefile) (Make targets: `phase1`, `validate`, `index`, `metadata`, `eda`, `test`, `clean`)
- **Dataset Structural Validation**:
  - [`src/data/validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/validator.py) & [`scripts/validate_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/validate_dataset.py)
  - PASS validation report at [`reports/dataset_validation.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/dataset_validation.txt) (verified 42 patients, 668 video folders, 31,500 frames, 3,996 annotations).
- **Dataset Indexing & Manifests**:
  - [`src/data/indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/indexer.py) & [`scripts/build_index.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/build_index.py)
  - Generated index tables in `data_index/`: `frames_index.csv` (31,500 rows), `annotations.csv` (6,161 rows), `videos_index.csv` (668 rows), `patients_index.csv` (42 rows), `classes.json` (7 classes).
- **Clinical Metadata Audit & Cleaning**:
  - [`src/data/metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/metadata.py) & [`scripts/inspect_metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/inspect_metadata.py)
  - Standardized cleaned metadata [`data_index/metadata_clean.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/metadata_clean.csv) and audit report [`reports/metadata_summary.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/metadata_summary.txt) verifying 100% patient alignment.
- **Exploratory Data Analysis (EDA)**:
  - [`src/data/eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eda.py) & [`scripts/run_eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/run_eda.py)
  - 8 analytical figures in [`reports/figures/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures) and summary report [`reports/eda_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/eda_summary.md).
- **Test Suite**:
  - 53 passing tests across `tests/test_config.py`, `tests/test_dataset_scanner.py`, `tests/test_dataset_validator.py`, `tests/test_indexer.py`, `tests/test_metadata.py`, and `tests/test_eda.py`.
- **Documentation**:
  - Rewritten Quick Start in [`README.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/README.md) with step-by-step commands, expected outputs, and Phase 1 Results statistics table.


### Added
- Exploratory Data Analysis (EDA) module [`src/data/eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eda.py) featuring `DatasetEDA`:
  - Per-patient analysis: bar charts of frames, videos, and bounding boxes per patient (`per_patient_stats.png`).
  - Lesion category class distribution: bar chart with count and percentage annotations, plus automatic severity ordering and class-imbalance detection warning (30.9x ratio, exceeding 5.0x threshold; `class_distribution.png`).
  - Bounding box geometry analysis: histograms for box width, height, area, aspect ratio, a 2D spatial heatmap of lesion centres across normalized coordinates, and box area distributions by category (`bbox_statistics.png`).
  - Image resolution verification: bar chart validating 100% 512×512 resolution uniformity (`resolution_distribution.png`).
  - Split and keyframe balance: pie and bar charts for lesion vs non-lesion videos and frames, selected vs nonselected frames, and annotated vs unannotated frames (`lesion_selected_splits.png`).
  - Boxes-per-frame frequency histogram (`boxes_per_frame.png`) revealing a mean of 1.54 boxes per annotated frame and up to 6 boxes on a single frame.
  - Sample annotation visualizer: grid of 12 randomly sampled angiographic frames with ground-truth bounding boxes and severity labels drawn directly from image data (`sample_annotations.png`).
  - Clinical metadata correlation: Pearson correlation heatmap linking numeric patient clinical measurements (age, BMI, diabetes, etc.) to per-patient lesion counts (`metadata_correlations.png`).
  - Automated Markdown summary report generation embedding all figures and detailing key findings (class imbalance, tiny boxes < 400 px², annotation sparsity, patients with 0 lesions) at [`reports/eda_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/eda_summary.md).
- Command-line interface runner script [`scripts/run_eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/run_eda.py) with argparse options and clean console summary.
- Comprehensive unit and integration test suite in [`tests/test_eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_eda.py) containing 13 tests using self-contained synthetic fixtures (53 total project tests passing clean).
- Exported `DatasetEDA` in [`src/data/__init__.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/__init__.py).
- Updated [`README.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/README.md) with EDA CLI documentation, options, and figure output tables.

## [0.6.0] - 2026-09-30

### Added
- Clinical metadata inspection and normalization module [`src/data/metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/metadata.py) featuring `MetadataInspector`:
  - Automated patient ID column detection without hardcoded column names (hybrid heuristic scoring column name tokens, value uniqueness ratio, integer/regex patterns, and penalizing clinical flags).
  - Patient ID normalization to standard `'pX'` folder format (e.g. `1` -> `'p1'`).
  - Comprehensive clinical summary report generator saving to [`reports/metadata_summary.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/metadata_summary.txt) covering dataset shape, column schema, dtypes, missing values per column, categorical unique value counts and distributions, and `describe()` for numeric clinical columns.
  - Cross-referencing against image folders (`selectedVideos` / `nonselectedVideos`) identifying missing patients in both directions (verified 42/42 patients aligned with zero discrepancies).
  - Cleaned metadata CSV exporter saving to [`data_index/metadata_clean.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/metadata_clean.csv) with standardized `patient_id` as the primary key.
- Command-line interface script [`scripts/inspect_metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/inspect_metadata.py) with options for configuration, dataset root, metadata path, and custom outputs.
- Comprehensive unit and integration test suite in [`tests/test_metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_metadata.py) with 14 tests covering patient normalization, auto-detection, cleaning, discrepancy detection, summary generation, synthetic Excel testing, and real CADICA archive testing (40 total project tests passing clean).
- Updated [`README.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/README.md) with Clinical Metadata Inspection CLI usage guide and cleaned metadata schema documentation.

## [0.5.0] - 2026-09-30

### Added
- Dataset Indexer module [`src/data/indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/indexer.py): scans CADICA zip or directory, auto-detects groundtruth format (`space`/`comma`-separated `x y w h [category]`), discovers all 7 stenosis categories, and builds four structured CSV tables.
- CLI script [`scripts/build_index.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/build_index.py) with `--limit-patients` flag and `--dataset_root` override.
- Generated index files in `data_index/` (full 42-patient run):
  - `frames_index.csv` — 31,500 rows (one per frame)
  - `annotations.csv` — 6,161 rows (one per bounding box, with YOLO-normalised coordinates)
  - `videos_index.csv` — 668 rows (one per video)
  - `patients_index.csv` — 42 rows (one per patient)
  - `classes.json` — 7 categories: `p0_20 (0), p100 (1), p20_50 (2), p50_70 (3), p70_90 (4), p90_98 (5), p99 (6)`
- 14 unit tests in [`tests/test_indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_indexer.py) covering `_parse_gt_line`, directory-based indexer, zip-based indexer, and helper utilities (26 total tests passing).
- Updated [`README.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/README.md) with Dataset Indexing CLI usage and full schema tables for all five generated files.

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
