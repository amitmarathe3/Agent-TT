# Agent-TT: Coronary Artery Disease Lesion Detection from ICA Images

Agent-TT (CADICA-Agent) is an end-to-end medical imaging machine learning pipeline for coronary artery disease (CAD) lesion detection using Invasive Coronary Angiography (ICA) images from the CADICA dataset.

---

## 1. Project Overview
Agent-TT is designed to assist cardiologists by automatically detecting and localizing coronary artery lesions from ICA video sequences and frame extractions. Built with modularity, strict configuration control, dynamic dataset discovery, and comprehensive evaluation metrics, Agent-TT provides a reproducible AI pipeline for CAD detection.

---

## 2. Dataset Description
The project utilizes the **CADICA** dataset:
- **Patients**: 42 patients (`p1` through `p42`).
- **Modality**: Invasive Coronary Angiography (ICA) frame sequences and static images.
- **Metadata**: [`metadata.xlsx`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data/dataset_summary.json) contains clinical parameters, patient metadata, and video classifications.
- **Directory Layout**:
  - `selectedVideos/`: Videos selected for detailed annotation and evaluation.
  - `nonselectedVideos/`: Unselected angiographic sequences.
- **Folder Convention**:
  - `pX/vY/`: Patient `pX`, Video `vY` folder hierarchy (e.g., `p10/v1/`).
  - `input/`: Contains frame image extractions (e.g., `p10_v1_00016.png`).
  - `groundtruth/`: Contains lesion bounding box annotations (e.g., `p10_v1_00016.txt`).
- **Control Files**:
  - `selectedFrames.txt`: List of frames designated for training/testing.
  - `lesionVideos.txt`: Videos containing confirmed CAD lesions.
  - `nonlesionVideos.txt`: Control videos without lesions.
- **Bounding Box Format**: `[x, y, w, h]` (x-coordinate, y-coordinate, width, height in pixels).

---

## 3. Project Structure
```text
CADICA-Agent/
├── README.md                  # Main documentation file
├── CHANGELOG.md               # Version history and task update log
├── requirements.txt           # Project dependencies
├── .gitignore                 # Exclusion rules for data, models, environments
├── configs/
│   └── config.yaml            # Central configuration file
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_scanner.py # Dataset discovery & zip scanning module
│   │   ├── validator.py       # Dataset integrity & consistency validator
│   │   ├── indexer.py         # Groundtruth parser & index table builder
│   │   ├── metadata.py        # Clinical metadata inspector & cleaner
│   │   └── eda.py             # Exploratory Data Analysis & figure generator
│   ├── models/
│   │   └── __init__.py        # Lesion detection model definitions
│   ├── evaluation/
│   │   └── __init__.py        # Evaluation metrics & loss functions
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration loader
│       ├── logger.py          # Dual console/file logger
│       └── seed.py            # Reproducibility seeding
├── scripts/
│   ├── validate_dataset.py    # Dataset integrity verification CLI
│   ├── build_index.py         # Manifest and index generation CLI
│   ├── inspect_metadata.py    # Clinical metadata audit & cleaning CLI
│   └── run_eda.py             # Exploratory Data Analysis runner CLI
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Tests for config loader, logger, and seed
│   ├── test_dataset_scanner.py# Tests for zip/directory scanner
│   ├── test_dataset_validator.py # Tests for dataset validator
│   ├── test_indexer.py        # Tests for index builder & groundtruth parser
│   ├── test_metadata.py       # Tests for metadata inspection & patient alignment
│   └── test_eda.py            # Tests for EDA plotting & report generation
├── notebooks/                 # Exploratory data analysis notebooks
├── reports/
│   ├── figures/               # Generated figures (distributions, heatmaps, samples)
│   ├── dataset_validation.txt # Integrity validation report
│   ├── metadata_summary.txt   # Clinical metadata audit report
│   └── eda_summary.md         # Comprehensive EDA findings & metrics report
├── outputs/                   # Checkpoints, predictions, and logs (gitignored)
└── data_index/                # Metadata index files and manifests (gitignored)
```

---

## 4. Virtual Environment Setup Instructions

### Prerequisites
- Python 3.10+

### Windows Setup (PowerShell / Command Prompt)
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Or activate virtual environment (Command Prompt)
.\.venv\Scripts\activate.bat

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS Setup (Bash / Zsh)
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Phase 1 Quick Start & Pipeline Usage

The Phase 1 pipeline automates dataset validation, index generation, metadata inspection, and exploratory data analysis.

### Option A: One-Command Execution

Run the complete Phase 1 pipeline in order (`validate_dataset` → `build_index` → `inspect_metadata` → `run_eda`):

```bash
# Windows (Command Prompt or PowerShell)
.\run_phase1.bat

# Linux / macOS / Bash
chmod +x run_phase1.sh
./run_phase1.sh

# Using Make
make phase1
```

---

### Option B: Step-by-Step Execution

#### Step 1: Validate Dataset Integrity
Verifies directory structure, presence of `metadata.xlsx`, patient folders (`p1`–`p42`), video subfolders, frame PNGs, control files (`selectedFrames.txt`, `lesionVideos.txt`, `nonlesionVideos.txt`), and ground-truth annotation matches.

```bash
python scripts/validate_dataset.py
# Or: make validate
```

- **Expected Console Output**:
  - `Dataset Validation Status: PASS`
  - Total Patients: `42`
  - Total Video Folders: `668` (196 selected, 472 non-selected)
  - Total PNG Frames: `31,500`
  - Total Groundtruth Annotation Files: `3,996`
  - Missing/Inconsistent Items: `0`
- **Expected Artifact Generated**:
  - [`reports/dataset_validation.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/dataset_validation.txt)

#### Step 2: Build Dataset Index & Annotation Manifests
Parses groundtruth bounding boxes with adaptive delimiter handling (`space` and `comma`), computes normalized YOLO coordinates `[x_center, y_center, w_norm, h_norm]`, extracts frame dimensions, and generates structured CSV tables and class dictionary.

```bash
python scripts/build_index.py
# Or: make index
```

- **Expected Console Output**:
  - `Discovered categories: ['p0_20', 'p100', 'p20_50', 'p50_70', 'p70_90', 'p90_98', 'p99']`
  - `Generated frames_index.csv: 31,500 rows`
  - `Generated annotations.csv: 6,161 rows`
  - `Generated videos_index.csv: 668 rows`
  - `Generated patients_index.csv: 42 rows`
- **Expected Artifacts Generated** in [`data_index/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index):
  - `frames_index.csv` — Frame-level index table (31,500 rows)
  - `annotations.csv` — Bounding-box annotations with normalized coordinates (6,161 rows)
  - `videos_index.csv` — Video sequence metadata and frame counts (668 rows)
  - `patients_index.csv` — Patient-level aggregation table (42 rows)
  - `classes.json` — Stenosis category to integer ID mapping

#### Step 3: Inspect & Clean Clinical Metadata
Audits `metadata.xlsx`, detects the patient identifier column dynamically, normalizes patient IDs to `pX` format, computes missing values and clinical distribution statistics, and verifies 100% patient alignment between metadata and image folders.

```bash
python scripts/inspect_metadata.py
# Or: make metadata
```

- **Expected Console Output**:
  - `Auto-detected patient column: 'PACIENTES'`
  - `Normalized patient range: p1 to p42`
  - `Metadata Alignment Status: PERFECT_MATCH (42 / 42 patients)`
  - `Discrepancies: 0`
- **Expected Artifacts Generated**:
  - [`reports/metadata_summary.txt`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/metadata_summary.txt) — Comprehensive clinical metadata audit report
  - [`data_index/metadata_clean.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/metadata_clean.csv) — Cleaned metadata with standardized `patient_id`

#### Step 4: Run Exploratory Data Analysis (EDA)
Generates 8 publication-grade visualization figures and an integrated Markdown analysis report detailing class imbalance, bounding box geometry, image resolutions, split balances, and clinical correlations.

```bash
python scripts/run_eda.py
# Or: make eda
```

- **Expected Console Output**:
  - `Generated per-patient statistics`
  - `Generated class distribution (WARNING: 30.9x class imbalance detected)`
  - `Generated bounding box geometry statistics`
  - `Generated resolution distribution (100% 512x512 uniform)`
  - `Generated split and keyframe distributions`
  - `Generated boxes per frame histogram (mean: 1.54 boxes/frame)`
  - `Generated 12-sample annotated visual grid`
  - `Generated clinical metadata correlation matrix`
  - `Summary report saved -> reports/eda_summary.md`
- **Expected Artifacts Generated**:
  - [`reports/eda_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/eda_summary.md) — Comprehensive EDA report
  - [`reports/figures/per_patient_stats.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/per_patient_stats.png)
  - [`reports/figures/class_distribution.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/class_distribution.png)
  - [`reports/figures/bbox_statistics.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/bbox_statistics.png)
  - [`reports/figures/resolution_distribution.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/resolution_distribution.png)
  - [`reports/figures/lesion_selected_splits.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/lesion_selected_splits.png)
  - [`reports/figures/boxes_per_frame.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/boxes_per_frame.png)
  - [`reports/figures/sample_annotations.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/sample_annotations.png)
  - [`reports/figures/metadata_correlations.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/metadata_correlations.png)

#### Step 5: Run Verification Test Suite
Executes the comprehensive automated pytest test suite:

```bash
pytest -v
# Or: make test
```

- **Expected Console Output**: `53 passed in ~14s`

---

### Python API & Utility Usage

```python
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.validator import DatasetValidator
from src.data.indexer import DatasetIndexer
from src.data.metadata import MetadataInspector
from src.data.eda import DatasetEDA

# 1. Reproducibility & logging
set_seed(42)
logger = get_logger("Agent-TT")
config = load_config("configs/config.yaml")

# 2. Programmatic validation & indexing
validator = DatasetValidator(config_path="configs/config.yaml")
val_results = validator.validate()

indexer = DatasetIndexer(config_path="configs/config.yaml")
manifests = indexer.build_all()

# 3. Clinical metadata inspection & EDA
inspector = MetadataInspector(config_path="configs/config.yaml")
meta_clean = inspector.clean_and_save()

eda = DatasetEDA(config_path="configs/config.yaml")
eda.run_all()
```

---

## 6. Phase 1 Results

### CADICA Dataset Statistics

| Metric | Measured Value | Description / Scope |
|---|:---:|---|
| **Patients** | **`42`** | `p1` to `p42` (100% matched across metadata and directory hierarchies) |
| **Total Videos** | **`668`** | All angiography sequences across 42 patients |
| **Selected Videos** | **`196`** | Sequences with keyframes selected for annotation (29.3%) |
| **Non-Selected Videos** | **`472`** | Unselected angiography sequences (70.7%) |
| **Lesion Videos** | **`236`** | Sequences containing confirmed coronary artery lesions (35.3%) |
| **Non-Lesion Videos** | **`432`** | Normal / control sequences without lesions (64.7%) |
| **Total Frames** | **`31,500`** | `5,595` selected frames + `25,905` nonselected frames |
| **Frame Resolution** | **`512 × 512`** | Exactly 100.0% uniform dimensions across all frames |
| **Annotated Frames** | **`3,996`** | Frames containing ground-truth bounding box labels (12.7% sparsity) |
| **Bounding Boxes** | **`6,161`** | Total lesion annotations (range: 1–6 boxes/frame, mean: 1.54) |
| **Stenosis Classes** | **`7`** | Lesion severity categories based on percentage lumen diameter reduction |

### Lesion Severity Class Breakdown

| Category | Stenosis Range | Bounding Boxes | Percentage | Clinical Significance |
|---|:---:|:---:|:---:|---|
| `p0_20` | < 20% | **1,944** | **31.55%** | Minimal / mild atherosclerosis |
| `p20_50` | 20% – 50% | **1,128** | **18.31%** | Moderate non-obstructive stenosis |
| `p50_70` | 50% – 70% | **999** | **16.21%** | Intermediate obstructive stenosis |
| `p70_90` | 70% – 90% | **893** | **14.49%** | Severe hemodynamically significant stenosis |
| `p90_98` | 90% – 98% | **930** | **15.09%** | Critical pre-occlusive stenosis |
| `p99` | 99% (subtotal) | **63** | **1.02%** | Subtotal occlusion (hairline lumen) |
| `p100` | 100% (total) | **204** | **3.31%** | Total coronary artery occlusion (CTO) |
| **Total** | — | **6,161** | **100.0%** | All annotated bounding boxes |

### Key Dataset Insights & Pipeline Guidelines for Phase 2–3

1. **Extreme Class Imbalance (30.9x)**:
   - The ratio between the largest category (`p0_20`: 1,944) and smallest category (`p99`: 63) is **30.86x** (exceeding the 5.0x threshold).
   - *Phase 2/3 Action*: Use class-balanced sampling, category grouping (e.g. non-significant `<50%` vs significant `≥50%`), and focal/re-weighted loss functions during training.
2. **Annotation Sparsity (12.7%)**:
   - Only 3,996 of 31,500 frames have ground truth bounding boxes.
   - *Phase 2 Action*: Train primarily on annotated keyframes while leveraging unannotated sequence frames for temporal self-supervised representation learning or tracking.
3. **Tiny Bounding Boxes (2.8%)**:
   - 175 boxes have area < 400 px² (e.g. 10×11 pixels).
   - *Phase 3 Action*: Employ feature pyramid networks (FPN / PANet) and multi-scale image anchors to prevent missing small distal lesions.
4. **Zero-Lesion Patient (`p3`)**:
   - Patient `p3` has 8 videos (all non-lesion) and zero bounding boxes.
   - *Phase 2 Action*: Stratified patient-aware cross-validation must balance patients with and without lesions across folds to avoid data leakage and evaluation bias.
5. **Image Uniformity (100%)**:
   - All 31,500 frames are exactly 512×512 grayscale PNG images, eliminating the need for aspect-ratio padding or non-square resizing in Phase 2.

---

## 7. Roadmap (Phases 1-5)
- **Phase 1: Project Setup & Dataset Indexing** ✅ *(Complete)*
  - Scaffold creation, config loader, logging, dynamic dataset scanner, validator, metadata parser, frame indexing, and EDA.
- **Phase 2: Data Preprocessing & Augmentation**
  - ICA image normalization, CLAHE enhancement, vessel filtering, bounding box scaling, patient-aware k-fold cross-validation splitters.
- **Phase 3: Model Architecture & Training**
  - Baseline object detection backbone (YOLOv8 / Faster R-CNN / DETR adapted for ICA), loss functions, model trainer pipeline.
- **Phase 4: Evaluation & Performance Analysis**
  - Lesion detection metrics calculator, IoU thresholding, confusion matrix, mAP evaluation, ROC-AUC analysis.
- **Phase 5: CLI Tooling, Reporting & Deployment**
  - Inference CLI, summary reporting, automated figure generation, model export (ONNX/TorchScript).

---

## 8. Metrics Tracked
Agent-TT evaluates CAD lesion detection using standard clinical & computer vision object detection metrics:
- **Accuracy**: Overall classification accuracy on lesion vs non-lesion frames.
- **Precision**: Proportion of true CAD lesions among detected bounding boxes.
- **Recall (Sensitivity)**: Proportion of actual CAD lesions successfully detected.
- **F1 Score**: Harmonic mean of Precision and Recall.
- **IoU (Intersection over Union)**: Bounding box overlap quality measure.
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.50.
- **mAP@0.5:0.95**: Mean Average Precision averaged over IoU thresholds 0.50 to 0.95 (step 0.05).
- **Confusion Matrix**: True Positive, False Positive, True Negative, False Negative counts.
- **ROC-AUC**: Area Under Receiver Operating Characteristic Curve for lesion frame classification.

---

## 9. Changelog Link
For full version history and updates, view the [`CHANGELOG.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/CHANGELOG.md) file.

---

## 10. Update Log
- **2026-09-30**: **Phase 1 Complete (Release v0.1.0)** - Finalized dataset ingestion, integrity verification, indexing manifests, clinical metadata cleaning, exploratory data analysis, and automated execution scripts (`run_phase1.bat`, `run_phase1.sh`, `Makefile`). 53 unit tests passing clean.
- **2026-09-30**: Phase 1.5 - Exploratory Data Analysis completed ([`src/data/eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eda.py), [`scripts/run_eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/run_eda.py)). 8 analytical figures generated in [`reports/figures/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures) and summary report [`reports/eda_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/eda_summary.md).
- **2026-09-30**: Phase 1.4 - Clinical metadata inspector and cleaner implemented ([`src/data/metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/metadata.py), [`scripts/inspect_metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/inspect_metadata.py)).
- **2026-09-30**: Phase 1.3 - Dataset Indexer implemented ([`src/data/indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/indexer.py), [`scripts/build_index.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/build_index.py)). Discovered groundtruth bounding-box formats, built 4 index tables and `classes.json`.
- **2026-09-29**: Phase 1.2 - Dataset Validator module and CLI implemented ([`src/data/validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/validator.py), [`scripts/validate_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/validate_dataset.py)). Verified full dataset integrity with PASS report.
- **2026-09-29**: Phase 1.1 - Project scaffold created, CADICA dataset scanner implemented, central config system initialized, unit tests written and verified.

