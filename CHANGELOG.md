# Changelog

All notable changes to the Agent-TT (CADICA-Agent) project will be documented in this file.

## [0.2.1] - 2026-10-01 - Phase 3.1: Evaluation Framework Bugfixes & Canonical Evaluation Sets

### Milestone Summary
Phase 3.1 resolves critical evaluation metric bugs discovered prior to model training, defines canonical evaluation datasets matching exported YOLO validation and test splits, and introduces a calibrated Jittered Oracle baseline to certify that `mAP@0.5` and `mAP@0.5:0.95` respond correctly to localization displacement. All 125 project tests pass with zero warnings, and both `val` (722 frames) and `test` (852 frames) pass all sanity check assertions.

### Fixed
- **AP Precision Envelope & Zero-TP Bug**:
  - Previously, `build_pr_curve` artificially prepended `(0.0, 1.0)` to precision and recall arrays, causing the 101-threshold interpolation to sample precision 1.0 at threshold $r=0.0$, leading to non-zero average precision ($\approx 1/101 = 0.0099$) even for predictors with zero True Positives.
  - Rewrote 101-point COCO-style interpolation in [`src/evaluation/metrics.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/metrics.py) with an exact precision envelope where precision at recall threshold $r$ is $\max(\text{precisions}[\text{recalls} \ge r])$ or $0.0$ if no recall satisfies the threshold. Predictors with zero TPs now evaluate to AP exactly $0.0000$.
- **Frame-Level PR-AUC Bug**:
  - Previously, `_compute_pr` prepended baseline prevalence $P = \frac{n_{\text{pos}}}{n_{\text{pos}} + n_{\text{neg}}}$ and applied a trapezoid integral that bridged from $(0.0, 1.0)$ down to baseline, yielding PR-AUC of 0.6004 for a perfect Oracle predictor.
  - Replaced trapezoidal integration with a step-integral formulation matching `sklearn.metrics.average_precision_score` ($\sum (R_k - R_{k-1}) P_k$). Perfect Oracle predictions now achieve frame PR-AUC = 1.0000.

### Added
- **Canonical Evaluation Datasets ([`src/data/eval_sets.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eval_sets.py))**:
  - Defined canonical evaluation frames matching `outputs/yolo_dataset/images/<split>/` (positives + vetted hard negatives): exactly 722 frames for `val` (578 positives, 144 negatives) and 852 frames for `test` (682 positives, 170 negatives).
  - Wired [`scripts/evaluate.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/evaluate.py) and [`scripts/sanity_check.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/sanity_check.py) to load canonical frames by default.
  - Added `--full-frames` flag with a prominent clinical notice warning that evaluating against all raw frames (e.g. 2,879 in val) treats unannotated sequence frames from lesion videos as false negatives despite potentially harboring unlabeled lesions.
- **Jittered Oracle Baseline Predictor**:
  - Added `build_jittered_oracle_predictions` in [`scripts/sanity_check.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/sanity_check.py) that shifts ground-truth boxes by $0.22 \times \text{width}$, resulting in mean IoU $\approx 0.64$.
  - Sanity check and unit tests verify that Jittered Oracle achieves perfect $\text{mAP}@0.5 = 1.0000$ while $\text{mAP}@0.5:0.95 = 0.3000 < \text{mAP}@0.5$.
- **Test Suite Expansion**:
  - Added unit tests in [`tests/test_evaluation.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_evaluation.py) for zero-TP AP=0.0, Jittered Oracle mAP response, Oracle frame PR-AUC=1.0, canonical frame counts, and `--full-frames` warning emission. Total passing tests: 125 (100% pass rate).

## [0.2.0] - 2026-10-01 - Phase 2 Complete: Data Pipeline, Augmentation, Splits & Evaluation

### Milestone Summary
Phase 2 (Patient-Wise Splitting, Data Pipeline, Medical Data Augmentation, Model-Agnostic Evaluation Framework, and Baseline Sanity Checks) is officially complete. The CADICA angiography pipeline now provides patient-level stratified cross-validation with zero patient data leakage, hard negative sampling, angiography-safe medical augmentations, comprehensive multi-level detection/classification metrics, reproducible pipeline automation (`run_phase2.bat`, `run_phase2.sh`, `make phase2`), and 120/120 automated pytest tests passing clean with zero warnings.

### Added
- **Phase 2.4 - Evaluation Sanity Check & Automation**:
  - Implemented [`scripts/sanity_check.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/sanity_check.py) running dual baseline predictors on validation split:
    - **Oracle Predictor**: Ground-truth boxes as predictions with confidence=1.0; verified perfect 1.0000 across Micro/Macro F1, mAP@0.5, mAP@0.5:0.95, and Frame Accuracy (fails loudly if any metric deviates from 1.0).
    - **Random Predictor**: Stochastic boxes and scores; verified near-zero performance (Micro F1=0.0011, Macro F1=0.0007, mAP@0.5=0.0099).
  - Created unified Phase 2 pipeline scripts:
    - [`run_phase2.bat`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase2.bat) (Windows batch runner)
    - [`run_phase2.sh`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase2.sh) (Linux/macOS Bash runner)
    - Updated [`Makefile`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/Makefile) with `sanity-check` and `phase2` targets.
- **Phase 2.3 - Model-Agnostic Evaluation Package (`src/evaluation/`)**:
  - Module [`src/evaluation/matching.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/matching.py): Vectorized pairwise IoU and greedy score-ordered prediction-to-ground-truth matching (class-aware and class-agnostic).
  - Module [`src/evaluation/metrics.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/metrics.py): Object detection metrics (TP, FP, FN, Precision, Recall, F1, mean IoU), 101-point COCO-style mAP@0.5 and mAP@0.5:0.95, confusion matrices with background class, frame-level AUROC/sensitivity/specificity, patient-level maximum score aggregation, validation macro F1 threshold optimization, and patient bootstrap 95% CIs (1,000 iterations).
  - Module [`src/evaluation/report.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/report.py): JSON serialization (`metrics.json`), Markdown report generation (`metrics.md`), and high-resolution figures: confusion matrices, PR curves, F1 vs confidence curves, ROC and PR curves.
  - CLI [`scripts/evaluate.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/evaluate.py): Model-agnostic evaluator accepting `--pred`, `--gt` or `--split`, `--scheme`, `--conf`, `--iou`, and `--bootstrap`.
- **Phase 2.2 - Data Pipeline, YOLO Export & Medical Augmentation**:
  - Exporter [`src/data/yolo_export.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/yolo_export.py) and CLI [`scripts/export_yolo.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/export_yolo.py): Standard Ultralytics YOLO dataset in `outputs/yolo_dataset/` with hard negatives at ratio 0.25 (4,994 total images: 3,996 positives + 998 negatives; 6,161 bounding boxes).
  - PyTorch Dataset [`src/data/dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/dataset.py): `CADICADataset` returning images, boxes, labels, patient_id, frame_id, and clinical metadata with custom collate function.
  - Augmentation pipeline [`src/data/augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/augment.py): Angiography-safe Albumentations transforms (CLAHE, brightness/contrast, gamma, noise/blur, rotation $\pm 10^\circ$; strictly NO vertical flip, NO color jitter, NO aggressive crops).
  - Visual check grid [`reports/figures/augmentation_check.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/augmentation_check.png) and normalization stats [`data_index/norm_stats.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/norm_stats.json).
- **Phase 2.1 - Patient-Wise Splits & Stratified Cross-Validation**:
  - Splitter [`src/data/splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/splits.py) & CLI [`scripts/make_splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/make_splits.py): Partitioned 42 patients into `split_v1.csv` (30 train, 6 val, 6 test) and `folds_5.csv` with zero patient leakage, guaranteed per-class representation, rare-class allocation (`p100` and `p99`), and summary report [`data_index/splits/splits_report.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/splits_report.md).
- **Test Suite**:
  - 120 total automated pytest tests passing clean with zero warnings (100% pass rate).

### Fixed
- Resolved 1-frame discrepancy between dataset validator (31,499 frames) and EDA (31,500 frames) caused by misplaced frame duplicate stem `p30_v2_00021.png` in `selectedVideos/p32/v10/input/`. The validator was updated to count total frame files and flag misplaced frame names with an informative warning, aligning validator and EDA at exactly 31,500 frames.
- Fixed JSON serialization of NumPy types and private plotting arrays in evaluation report generator.
- Added ASCII-safe console status prefix `[OK]` replacing unicode character to prevent Windows CP1252 UnicodeEncodeError.

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
