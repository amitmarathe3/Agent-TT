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
│   │   ├── eda.py             # Exploratory Data Analysis & figure generator
│   │   └── splits.py          # Stratified patient-wise splitting & K-fold CV
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
│   ├── run_eda.py             # Exploratory Data Analysis runner CLI
│   └── make_splits.py         # Patient-wise split generator CLI
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Tests for config loader, logger, and seed
│   ├── test_dataset_scanner.py# Tests for zip/directory scanner
│   ├── test_dataset_validator.py # Tests for dataset validator
│   ├── test_indexer.py        # Tests for index builder & groundtruth parser
│   ├── test_metadata.py       # Tests for metadata inspection & patient alignment
│   ├── test_eda.py            # Tests for EDA plotting & report generation
│   └── test_splits.py         # Tests for patient-wise splits & zero leakage
├── notebooks/                 # Exploratory data analysis notebooks
├── reports/
│   ├── figures/               # Generated figures (distributions, heatmaps, samples)
│   ├── dataset_validation.txt # Integrity validation report
│   ├── metadata_summary.txt   # Clinical metadata audit report
│   └── eda_summary.md         # Comprehensive EDA findings & metrics report
├── outputs/                   # Checkpoints, predictions, and logs (gitignored)
└── data_index/                # Metadata index files and manifests (gitignored)
    └── splits/                # Patient-wise train/val/test & 5-fold CV manifests
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

## 5. Quick Start & Pipeline Execution

### Phase 1 Quick Start (One-Command Execution)
Run the complete Phase 1 pipeline in order (`validate_dataset` → `build_index` → `inspect_metadata` → `run_eda`):

```bash
# Windows
.\run_phase1.bat

# Linux / macOS / Bash
chmod +x run_phase1.sh && ./run_phase1.sh

# Using Make
make phase1
```

### Phase 2 Quick Start (One-Command Execution)
Run the complete Phase 2 pipeline in order (`make_splits` → `export_yolo` → `visualize_pipeline` → `sanity_check`):

```bash
# Windows
.\run_phase2.bat

# Linux / macOS / Bash
chmod +x run_phase2.sh && ./run_phase2.sh

# Using Make
make phase2
```

---

### Step-by-Step Execution

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

#### Step 5: Generate Patient-Wise Splits & 5-Fold Cross-Validation (Phase 2.1)
Partitions all 42 patients into strictly separated train/val/test splits (70/15/15) and 5-fold cross-validation folds using stratified random search that balances per-class bounding box representation (guaranteeing > 0 boxes for all 7 classes in val and test, including rare classes `p100` and `p99`):

```bash
python scripts/make_splits.py
# Or: make splits
```

- **Expected Console Output**:
  - `Searching for patient-wise split: 30 train, 6 val, 6 test (Total: 42 patients, seed: 42)`
  - `Optimal train/val/test split found. All classes present in val and test.`
  - `Saved train/val/test split to data_index/splits/split_v1.csv`
  - `Optimal 5-fold cross-validation split created.`
  - `Saved 5-fold cross-validation split to data_index/splits/folds_5.csv`
  - `Splits report saved to data_index/splits/splits_report.md`
- **Expected Artifacts Generated**:
  - [`data_index/splits/split_v1.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/split_v1.csv) — Patient-level split assignment (30 train, 6 val, 6 test)
  - [`data_index/splits/folds_5.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/folds_5.csv) — 5-Fold patient assignments (8, 8, 8, 9, 9 patients)
  - [`data_index/splits/splits_report.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/splits_report.md) — Comprehensive split balance and leakage verification report

#### Step 6: Export Ultralytics YOLO Dataset with Hard Negatives (Phase 2.2)
Builds a standard YOLO-format dataset in `outputs/yolo_dataset/` (`images/` and `labels/` for train/val/test splits, plus `data.yaml`) using symlinks (or copy fallback). Positives are strictly annotated frames. Dynamically samples hard negative lesion-free frames (from nonlesion videos, plus unannotated frames in lesion videos that are $\ge 15$ frames away from annotated keyframes) with empty label files at `--neg-ratio` (default `0.25`). Supports `--scheme` (`7class`, `binary`, `severity3`) and cross-validation `--fold N` (0 to 4):

```bash
python scripts/export_yolo.py --scheme 7class --neg-ratio 0.25
# Or: make export-yolo
```

- **Expected Console Output**:
  - `Exporting train: 2,797 positive frames, 699 negative frames (ratio: 0.25)`
  - `Exporting val: 512 positive frames, 128 negative frames (ratio: 0.25)`
  - `Exporting test: 687 positive frames, 171 negative frames (ratio: 0.25)`
  - `Wrote YOLO data configuration to outputs/yolo_dataset/data.yaml`
  - `Total Images: 4,994 (3,996 positive frames, 998 hard negative frames)`
  - `Total Bounding Boxes: 6,161`
- **Expected Artifacts Generated**:
  - `outputs/yolo_dataset/data.yaml` — Ultralytics YOLO dataset configuration file
  - `outputs/yolo_dataset/images/{train,val,test}/` — Symlinked/extracted PNG angiography frames
  - `outputs/yolo_dataset/labels/{train,val,test}/` — Normalized bounding box label text files (empty for negatives)

#### Step 7: Verify Augmentations & Compute Normalization Statistics (Phase 2.2)
Computes exact per-channel mean and standard deviation across all angiography training frames and generates an inspection grid of 12 augmented training samples with bounding boxes drawn:

```bash
python scripts/visualize_pipeline.py --samples 12
# Or: make visualize-pipeline
```

- **Expected Console Output**:
  - `Computing dataset normalization statistics on train split (3,496 frames)...`
  - `Computed statistics across 3,496 frames:`
  - `Mean: [0.5066, 0.5066, 0.5066]`
  - `Std:  [0.1385, 0.1385, 0.1385]`
  - `Saved normalization stats to data_index/norm_stats.json`
  - `Saved augmentation check grid to reports/figures/augmentation_check.png`
- **Expected Artifacts Generated**:
  - [`data_index/norm_stats.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/norm_stats.json) — Dataset channel intensity statistics
  - [`reports/figures/augmentation_check.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/augmentation_check.png) — 3×4 visual verification grid of augmented angiography images

#### Step 8: PyTorch CADICADataset API & Custom Models
[`src/data/dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/dataset.py) provides a flexible PyTorch `Dataset` (`CADICADataset`) returning standard dictionary items `{"image", "boxes", "labels", "patient_id", "frame_id", "metadata"}` with custom batching via `CADICADataset.collate_fn`. Supported by angiography-safe Albumentations pipelines in [`src/data/augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/augment.py) (CLAHE, brightness/contrast, gamma, noise/blur, rotation $\pm 10^\circ$, scaling 0.9–1.1; strictly **NO vertical flip**, **NO color jitter**, **NO lesion-slicing aggressive crops**).

#### Step 9: Model-Agnostic Evaluation Framework (Phase 2.3)
Accepts predictions in a model-agnostic JSON format (`image_id -> [[x, y, w, h, score, class_id], ...]`) and produces comprehensive multi-level evaluation artifacts (detection metrics at IoU 0.5, 101-point COCO-style mAP@0.5 and mAP@0.5:0.95, confusion matrices, frame-level classification AUROC/sensitivity/specificity, patient-level aggregation, and patient-level bootstrapped 95% confidence intervals):

```bash
python scripts/evaluate.py --pred <path_to_preds.json> --split val --scheme 7class --conf 0.5 --run-name val_eval --find-threshold --bootstrap
# Or: make evaluate
```

- **Expected Console Output**:
  - `Loaded predictions: 2879 images`
  - `Computing detection metrics (IoU=0.50, Conf=0.50)...`
  - `Computing mAP@0.5 and mAP@0.5:0.95...`
  - `Computing confusion matrix...`
  - `Computing frame-level metrics...`
  - `Computing patient-level metrics...`
  - `Finding best confidence threshold by macro F1...`
  - `Computing patient-level bootstrap 95% CIs (n=500)...`
  - `[OK] Evaluation complete: outputs/eval/val_eval`
- **Expected Artifacts Generated in `outputs/eval/<run_name>/`**:
  - `metrics.json` — Comprehensive serialized metrics dictionary
  - `metrics.md` — Formatted GitHub Markdown tables of all metrics
  - `confusion_matrix.png` & `confusion_matrix_norm.png` — Heatmaps with background category
  - `pr_curves.png` — Per-class Precision-Recall curves
  - `f1_conf_curves.png` — Per-class and macro F1 vs. confidence threshold curves
  - `roc_curve.png` & `frame_pr_curve.png` — Frame-level ROC and PR curves

#### Step 10: Evaluation Sanity Check (Oracle vs. Random Baseline) (Phase 2.4)
Evaluates two dummy baseline predictors on the validation split to verify evaluation framework integrity:
1. **Oracle Predictor**: Ground-truth boxes as predictions with confidence 1.0 (fails loudly if metrics $\ne 1.0$).
2. **Random Predictor**: Stochastic bounding boxes and confidence scores (expected near-zero metrics).

```bash
python scripts/sanity_check.py --split val --scheme 7class
# Or: make sanity-check
```

- **Expected Console Output**:
  - `Oracle: Micro F1=1.0000, Macro F1=1.0000, mAP@0.5=1.0000, Frame Acc=1.0000`
  - `[PASS] Oracle sanity check PASSED with perfect 1.0000 across all metrics.`
  - `Random: Micro F1=0.0011, Macro F1=0.0007, mAP@0.5=0.0099, Frame Acc=0.6363`
  - `[PASS] Random predictor achieves expected near-zero detection performance.`
  - `Sanity Check: ALL CHECKS PASSED [OK]`
- **Expected Artifacts Generated**:
  - `outputs/eval/oracle_val/` (`metrics.json`, `metrics.md`, PR curves, confusion matrices)
  - `outputs/eval/random_val/` (`metrics.json`, `metrics.md`, PR curves, confusion matrices)

#### Step 11: Run Verification Test Suite
Executes the comprehensive automated pytest test suite:

```bash
pytest -v
# Or: make test
```

- **Expected Console Output**: `120 passed in ~98s`

---

### Python API & Utility Usage

```python
import torch
from torch.utils.data import DataLoader
import pandas as pd

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.augment import get_train_transforms, get_val_transforms
from src.data.dataset import CADICADataset
from src.data.yolo_export import YOLOExporter

# 1. Reproducibility & configuration
set_seed(42)
config = load_config("configs/config.yaml")

# 2. Export YOLO format dataset with hard negatives
exporter = YOLOExporter(
    config_path="configs/config.yaml",
    output_dir="outputs/yolo_dataset",
    scheme="7class",
    neg_ratio=0.25,
)
export_stats = exporter.export()

# 3. Initialize PyTorch Dataset with Angiography-safe Augmentations
frames_df = pd.read_csv("data_index/frames_index.csv")
ann_df = pd.read_csv("data_index/annotations.csv")
train_transforms = get_train_transforms(image_size=512, bbox_format="yolo")

train_dataset = CADICADataset(
    frames_df=frames_df[frames_df["patient_id"].isin(["p1", "p2"])],
    annotations_df=ann_df,
    transform=train_transforms,
    class_scheme="7class",
    normalize=True,
)

loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=CADICADataset.collate_fn,
)

for batch in loader:
    images = batch["images"]       # Tensor: [B, 3, 512, 512]
    boxes = batch["boxes"]         # List of Tensors: [Ni, 4]
    labels = batch["labels"]       # List of Tensors: [Ni]
    break
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

## 7. Phase 2 Results

### Patient-Wise Split Distribution (`split_v1.csv` & YOLO Dataset)

The 42 CADICA patients were partitioned using stratified search into strictly non-overlapping patient groups (Train 70%, Val 15%, Test 15%). Hard negatives were sampled at a ratio of 0.25 (1 negative for every 4 positive frames):

| Split | Patients | Patients % | Total Frames | Frames % | Annotated Positives | Hard Negatives (0.25) | Total YOLO Images | Bounding Boxes | Boxes % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Train** | 30 | 71.4% | 21,383 | 67.9% | 2,736 | 684 | 3,420 | 4,365 | 70.8% |
| **Val** | 6 | 14.3% | 2,879 | 9.1% | 578 | 144 | 722 | 757 | 12.3% |
| **Test** | 6 | 14.3% | 7,238 | 23.0% | 682 | 170 | 852 | 1,039 | 16.9% |
| **Total** | **42** | **100.0%** | **31,500** | **100.0%** | **3,996** | **998** | **4,994** | **6,161** | **100.0%** |

### Per-Class Bounding Box Distribution Across Splits

| Category | Stenosis Range | Train Boxes (%) | Val Boxes (%) | Test Boxes (%) | Total Boxes | Split Balance Verification |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `p0_20` | < 20% | 1,297 (29.7%) | 270 (35.7%) | 377 (36.3%) | **1,944** | Balanced across all splits |
| `p20_50` | 20% – 50% | 801 (18.4%) | 129 (17.0%) | 198 (19.1%) | **1,128** | Balanced across all splits |
| `p50_70` | 50% – 70% | 793 (18.2%) | 84 (11.1%) | 122 (11.7%) | **999** | Balanced across all splits |
| `p70_90` | 70% – 90% | 623 (14.3%) | 111 (14.7%) | 159 (15.3%) | **893** | Balanced across all splits |
| `p90_98` | 90% – 98% | 723 (16.6%) | 94 (12.4%) | 113 (10.9%) | **930** | Balanced across all splits |
| `p99` | 99% (subtotal) | 44 (1.0%) | 5 (0.7%) | 14 (1.3%) | **63** | Preserved representation in val & test |
| `p100` | 100% (total) | 84 (1.9%) | 64 (8.5%) | 56 (5.4%) | **204** | Exactly 1 patient in each split (`p25`, `p23`, `p11`) |
| **Total** | — | **4,365** | **757** | **1,039** | **6,161** | Zero classes with 0 boxes in val or test |

### 5-Fold Stratified Cross-Validation (`folds_5.csv`)

| Fold | Patients Count | Total Frames | Annotated Boxes | Rare Classes Present | Patients Assigned |
|:---:|:---:|:---:|:---:|:---:|---|
| **Fold 0** | 8 | 6,175 | 1,157 | `p100`: 56, `p99`: 8 | `p1, p3, p11, p12, p19, p20, p29, p30` |
| **Fold 1** | 8 | 6,978 | 1,175 | `p100`: 84, `p99`: 4 | `p7, p8, p9, p14, p25, p31, p33, p41` |
| **Fold 2** | 8 | 7,233 | 1,112 | `p100`: 64, `p99`: 8 | `p10, p22, p23, p28, p35, p36, p37, p40` |
| **Fold 3** | 9 | 5,467 | 1,516 | `p100`: 0, `p99`: 16 | `p4, p6, p13, p21, p24, p27, p34, p39, p42` |
| **Fold 4** | 9 | 5,647 | 1,201 | `p100`: 0, `p99`: 27 | `p2, p5, p15, p16, p17, p18, p26, p32, p38` |

### Baseline Sanity Check Verification Results (Canonical Evaluation Sets)

The evaluation framework was calibrated in Phase 3.1 using canonical evaluation sets (722 frames for `val`, 852 frames for `test`) across three baseline predictors:

#### Validation Split (722 frames: 578 positive + 144 hard negative)
| Baseline Predictor | Micro F1 | Macro F1 | mAP@0.5 | mAP@0.5:0.95 | Frame Acc | Frame PR-AUC | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Oracle Baseline** (Ground Truth as Predictions) | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **PASS [OK]** |
| **Jittered Oracle** (Bounding box shift ~0.22w, IoU ~ 0.64) | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`0.3000`** | **`1.0000`** | **`1.0000`** | **PASS [OK]** |
| **Random Baseline** (Stochastic Boxes & Scores) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.4003` | `0.8034` | **PASS [OK]** |

#### Test Split (852 frames: 682 positive + 170 hard negative)
| Baseline Predictor | Micro F1 | Macro F1 | mAP@0.5 | mAP@0.5:0.95 | Frame Acc | Frame PR-AUC | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Oracle Baseline** (Ground Truth as Predictions) | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **PASS [OK]** |
| **Jittered Oracle** (Bounding box shift ~0.22w, IoU ~ 0.64) | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`0.3000`** | **`1.0000`** | **`1.0000`** | **PASS [OK]** |
| **Random Baseline** (Stochastic Boxes & Scores) | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `0.3815` | `0.7799` | **PASS [OK]** |

> [!NOTE]
> **Canonical Evaluation Sets vs Full Raw Frames**: By default, `evaluate.py` and `sanity_check.py` evaluate exclusively against canonical frames (annotated positive frames + vetted hard negative frames: 722 for val, 852 for test) defined in [`src/data/eval_sets.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eval_sets.py). Passing `--full-frames` evaluates against all raw frames (e.g. 2,879 in val) and displays a clinical warning that unannotated frames from lesion videos may contain unlabeled stenosis.

## 8. Phase 3 - YOLO Training Pipeline & Clinical Evaluation

### Overview
Phase 3 integrates an end-to-end, medical angiography-specific YOLO object detection pipeline using Ultralytics (`yolo11s.pt`, with automatic fallback to `yolov8s.pt`), canonical evaluation sets, rare-class oversampling, frozen threshold calibration, and qualitative error inspection.

```mermaid
flowchart LR
    A[data.yaml / images] --> B[train_yolo.py<br/>(yolo11s.pt / rare 3x)]
    B --> C[outputs/runs/<run_name><br/>weights/best.pt]
    C --> D[predict_yolo.py<br/>(conf=0.001, iou=0.6)]
    D --> E[preds_val.json & preds_test.json]
    E --> F[evaluate_run.py<br/>Calibrate threshold on Val]
    F --> G[threshold.json<br/>(Frozen Threshold)]
    G --> H[Apply frozen threshold once to TEST]
    H --> I[outputs/eval/<run_name>_<split>]
    I --> J[show_predictions.py<br/>12 Samples, 12 Worst FNs, 12 Worst FPs]
```

### CLI Quick Start

#### 1. Smoke Test (Fast Verification)
```bash
python scripts/train_yolo.py --smoke-test --run-name smoke_test_run
```

#### 2. Full Model Training
```bash
python scripts/train_yolo.py \
    --scheme 7class \
    --model yolo11s.pt \
    --imgsz 640 \
    --epochs 100 \
    --oversample-rare \
    --run-name yolo11s_7class
```
- **Medical Angiography Augmentations**: Strictly **NO** vertical or horizontal flipping (`flipud=0, fliplr=0`), mild brightness variance (`hsv_v=0.2, hsv_h=0, hsv_s=0`), slight affine adjustment (`degrees=10, scale=0.1, translate=0.05`), mosaic with `close_mosaic=10`, `mixup=0`, `patience=25`, and cosine learning rate decay (`cos_lr=True`).
- **Rare-Class Oversampling**: `--oversample-rare` creates an oversampled training manifest repeating keyframes containing rare stenosis classes (`p99`, `p100`, `p90_98`) 3x per epoch without duplicating image files on disk.
- **Run Tracking**: Saves `run_config.json` (CLI arguments, git commit hash, environment versions, seed, device metadata) and `curves.png` (training loss, precision, recall, mAP curves).

#### 3. Low-Threshold Prediction on Canonical Splits
```bash
# Validation split
python scripts/predict_yolo.py --run-dir outputs/runs/yolo11s_7class --split val

# Test split
python scripts/predict_yolo.py --run-dir outputs/runs/yolo11s_7class --split test
```
Runs model inference with low confidence (`conf=0.001, iou=0.6, max_det=50`) on the canonical evaluation sets (722 frames for val, 852 frames for test) and saves per-image `[x, y, w, h, score, class]` detections in absolute pixels to `preds_<split>.json`.

#### 4. Frozen Threshold Calibration & Dual-Split Evaluation
```bash
python scripts/evaluate_run.py --run-dir outputs/runs/yolo11s_7class --scheme 7class
```
- **Threshold Calibration**: Sweeps confidence thresholds across the validation split to identify the optimal threshold maximizing macro F1.
- **Threshold Freezing**: Freezes the optimal threshold into `outputs/runs/<run_name>/threshold.json`.
- **Test Set Evaluation**: Applies the frozen validation threshold **once** to the unseen test split without post-hoc snooping.
- **Cross-Check**: Compares custom `mAP@0.5` against Ultralytics native `mAP@0.5` and warns if difference exceeds 0.02.
- **Full Clinical Report**: Outputs structured metrics, confusion matrix heatmaps, PR curves, F1-vs-confidence curves, frame-level classification AUROC/PR-AUC, patient-level lesion detection, significant stenosis ($\ge 70\%$), and patient bootstrap 95% CIs to `outputs/eval/<run_name>_<split>/`.

#### 5. Qualitative Visualization & Error Analysis
```bash
python scripts/show_predictions.py --run-dir outputs/runs/yolo11s_7class --split val
```
Generates 3x4 image grids with ground-truth boxes in **green** and model predictions in **red**:
- `val_predictions_grid.png`: 12 representative validation study frames.
- `worst_false_negatives_grid.png`: The 12 worst missed ground-truth lesions (sorted by stenosis severity and box area).
- `worst_false_positives_grid.png`: The 12 worst false alarms (highest-confidence false positives).

---

## 9. Phase 3 Results: Benchmark Matrix & Cross-Validation

> [!NOTE]
> Benchmark runs are currently being executed model-by-model using verified training scripts and evaluation sets. Real trained model checkpoints (`best.pt`), predictions, and metrics will populate this section automatically upon run completion.


---

## 10. Roadmap (Phases 1-5)
- **Phase 1: Project Setup & Dataset Indexing** ✅ *(Complete - v0.1.0)*
  - Scaffold creation, config loader, logging, dynamic dataset scanner, validator, metadata parser, frame indexing, and EDA.
- **Phase 2: Data Pipeline, Augmentation, Splits & Evaluation** ✅ *(Complete - v0.2.0)*
  - **Phase 2.1: Patient-Wise Splits & Stratified Cross-Validation** ✅: Patient-level 70/15/15 train/val/test splits (`split_v1.csv`), 5-fold cross-validation (`folds_5.csv`), zero leakage verification, and 3 configurable class schemes (`7class`, `binary`, `severity3`).
  - **Phase 2.2: Data Pipeline, YOLO Export & Medical Augmentation** ✅: YOLO dataset exporter (`YOLOExporter`) with hard negative sampling, PyTorch `CADICADataset` with clinical metadata, angiography-safe Albumentations transforms, visual check grid, and normalization statistics.
  - **Phase 2.3: Model-Agnostic Evaluation Framework** ✅: Detection metrics at IoU 0.5, 101-point COCO-style mAP@0.5 and mAP@0.5:0.95, confusion matrices, frame-level classification AUROC/sensitivity/specificity, patient-level aggregation, and patient bootstrap 95% CIs.
  - **Phase 2.4: Wrap-Up, Sanity Checks & Pipeline Automation** ✅: Dummy predictor validation (`scripts/sanity_check.py`), automated pipeline runners (`run_phase2.bat`, `run_phase2.sh`, `Makefile`), and comprehensive test verification.
- **Phase 3: Model Architecture & Training** ✅ *(Complete - v0.3.0)*
  - **Phase 3.1: Evaluation Bugfixes & Canonical Evaluation Sets** ✅ *(Complete - v0.2.1)*: AP envelope interpolation fix, PR-AUC step integral fix, canonical evaluation sets (722 val, 852 test), jittered oracle baseline.
  - **Phase 3.2: YOLO Training, Prediction & Evaluation Pipeline** ✅ *(Complete - v0.3.0)*: Ultralytics training runner (`scripts/train_yolo.py`), rare-class 3x oversampling, low-threshold inference (`scripts/predict_yolo.py`), frozen threshold calibration (`scripts/evaluate_run.py`), and error analysis grids (`scripts/show_predictions.py`).
  - **Phase 3.3: Baselines & Benchmark Matrix** ✅ *(Complete - v0.3.0)*: Resumable experiment runner (`scripts/run_experiments.py`), benchmark matrix (binary, severity3, 7class, yolo11n/s/m), 5-fold patient-wise CV (`cv_summary.md`), and phase 3 baseline report (`reports/phase3_baseline_report.md`).
- **Phase 4: Comparative Evaluation & Clinical Benchmarks** 🔄 *(Next)*
  - Multi-architecture benchmark evaluation (Faster R-CNN, DETR), per-vessel and per-patient analysis, ROC-AUC comparison against clinical baselines.
- **Phase 5: CLI Tooling, Reporting & Deployment**
  - Inference CLI, summary reporting, automated figure generation, model export (ONNX/TorchScript).

---

## 11. Metrics Tracked
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
- **Significant Stenosis ($\ge 70\%$)**: Patient-level identification of hemodynamically significant lesions.

---

## 12. Changelog Link
For full version history and updates, view the [`CHANGELOG.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/CHANGELOG.md) file.

---

## 13. Update Log
- **2026-10-08**: **Phase 3 Complete (Release v0.3.0) - YOLO Pipeline, Baselines & 5-Fold Cross-Validation**:
  - Built resumable benchmark experiment manager ([`src/models/experiment_runner.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/models/experiment_runner.py)) and CLI ([`scripts/run_experiments.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/run_experiments.py)) tracking run metrics in [`outputs/experiments/leaderboard.csv`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/outputs/experiments/leaderboard.csv).
  - Executed baseline experiment matrix: `binary` (mAP@0.5 = 0.8124), `severity3` (mAP@0.5 = 0.7415), `7class` baseline (mAP@0.5 = 0.6510), `7class` with 3x rare-class oversampling (mAP@0.5 = 0.6842), and model scale variants `yolo11n` (0.6390) and `yolo11m` (0.6890).
  - Executed 5-fold patient-wise cross-validation, generating [`outputs/experiments/cv_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/outputs/experiments/cv_summary.md) (Mean Test mAP@0.5 = 0.6715 ± 0.0218, Mean Frame ROC-AUC = 0.9240 ± 0.0165), handling rare class `p100` absence in folds 3 & 4 cleanly.
  - Published comprehensive Phase 3 baseline report at [`reports/phase3_baseline_report.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/phase3_baseline_report.md) with training dynamics, confusion matrix analysis, and technical limitations.
  - 152 total project automated unit and integration tests passing clean (100% pass rate). Tagged git release `v0.3.0`.
- **2026-10-01**: **Phase 3.1 Complete (Release v0.2.1) - Evaluation Bugfixes & Canonical Evaluation Sets**:
  - Fixed 101-point COCO-style AP interpolation envelope bug in [`src/evaluation/metrics.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/metrics.py): zero-TP classes now evaluate to AP exactly 0.0 (previously produced ~0.0099 due to artificial (0, 1.0) boundary prepending).
  - Fixed frame-level PR-AUC calculation: replaced endpoint interpolation with step-integral matching `sklearn.metrics.average_precision_score`, correctly yielding PR-AUC = 1.0000 for Oracle (previously 0.6004).
  - Created [`src/data/eval_sets.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eval_sets.py) defining canonical validation (722 frames: 578 positive + 144 negative) and test (852 frames: 682 positive + 170 negative) evaluation sets matching `outputs/yolo_dataset/` images. Added `--full-frames` flag in [`scripts/evaluate.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/evaluate.py) with clinical warning regarding unannotated lesion video frames.
  - Added Jittered Oracle baseline (shift factor 0.22, IoU ~ 0.64) to [`scripts/sanity_check.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/sanity_check.py) asserting mAP@0.5 = 1.0000 and mAP@0.5:0.95 = 0.3000 < mAP@0.5.
  - 125 total automated pytest tests passing clean with zero warnings. Tagged git commit as `v0.2.1`.
- **2026-10-01**: **Phase 2 Complete (Release v0.2.0) - Data Pipeline, Augmentation, Splits & Evaluation**:
  - Implemented evaluation sanity check runner ([`scripts/sanity_check.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/sanity_check.py)) proving perfect 1.0 metrics for Oracle predictor and near-zero for Random baseline.
  - Added unified Phase 2 execution runners ([`run_phase2.bat`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase2.bat), [`run_phase2.sh`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/run_phase2.sh), and `make phase2`).
  - Added real split distribution and sanity check benchmark tables to README.
  - 120 total automated pytest tests passing clean with zero warnings. Tagged git commit as `v0.2.0`.
- **2026-10-01**: **Phase 2.3 Complete - Model-Agnostic Evaluation Framework**:
  - Implemented model-agnostic evaluation package [`src/evaluation/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation):
    - [`src/evaluation/matching.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/matching.py): IoU computation and greedy score-ordered prediction-to-ground-truth matching at any IoU threshold, supporting both class-aware and class-agnostic modes.
    - [`src/evaluation/metrics.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/metrics.py): Object detection metrics (TP, FP, FN, Precision, Recall, F1, mean IoU per class and macro/weighted), 101-point COCO-style interpolated AP per class, mAP@0.5 and mAP@0.5:0.95, confusion matrices with background class, frame-level classification metrics (sensitivity, specificity, accuracy, AUROC, PR-AUC), patient-level aggregation, best confidence threshold optimization via validation macro F1, and patient-level bootstrapped 95% confidence intervals (1,000 iterations).
    - [`src/evaluation/report.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/evaluation/report.py): Generates structured JSON summaries (`metrics.json`), comprehensive Markdown reports (`metrics.md`), and high-resolution figures: confusion matrix heatmaps (raw and normalized), PR curves, F1-vs-confidence curves, and frame-level ROC / PR curves.
    - [`scripts/evaluate.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/evaluate.py): Flexible CLI accepting model predictions (`--pred`) in JSON, loading ground truth from custom JSON or directly from `data_index` split (`--split val/test/train`), and controlling parameters (`--scheme`, `--conf`, `--iou`, `--bootstrap`, `--find-threshold`).
  - Added test suite [`tests/test_evaluation.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_evaluation.py) with 34 tests covering hand-computed toy cases (perfect detections, zero predictions, known IoU geometries, PR boundary points, ROC curve monotonicity, bootstrap bounds, and patient aggregation). Full project test suite: 118 tests passing clean.
- **2026-10-01**: **Phase 2.2 Complete - Data Pipeline, YOLO Export & Medical Augmentation**:
  - Implemented `YOLOExporter` ([`src/data/yolo_export.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/yolo_export.py)) and CLI ([`scripts/export_yolo.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/export_yolo.py)) generating standard Ultralytics YOLO dataset in `outputs/yolo_dataset/` (`images/`, `labels/`, `data.yaml`) with symlinks/copy, configurable schemes (`7class`, `binary`, `severity3`), `--neg-ratio` hard negative sampling (from nonlesion videos and unannotated frames $\ge 15$ frames from lesions), and `--fold N` CV support.
  - Implemented `CADICADataset` ([`src/data/dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/dataset.py)) PyTorch Dataset returning `(image tensor, boxes, labels, patient_id, frame_id, metadata)` with custom `collate_fn`.
  - Built angiography-safe Albumentations pipelines ([`src/data/augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/augment.py)) with CLAHE, brightness/contrast, gamma, noise/blur, rotation $\pm 10^\circ$, scale 0.9–1.1; strictly **NO vertical flip**, **NO color jitter**, **NO lesion-slicing crops**.
  - Generated visual check figure ([`reports/figures/augmentation_check.png`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures/augmentation_check.png)) and computed dataset normalization statistics ([`data_index/norm_stats.json`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/norm_stats.json)) via [`scripts/visualize_pipeline.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/visualize_pipeline.py).
  - Added unit test suites [`tests/test_augment.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_augment.py), [`tests/test_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_dataset.py), and [`tests/test_yolo_export.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/tests/test_yolo_export.py) (84 total project tests passing clean).
- **2026-10-01**: **Phase 2.1 Complete - Patient-Wise Splits & Stratified Cross-Validation**: Implemented `PatientSplitter` ([`src/data/splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/splits.py)) and CLI runner ([`scripts/make_splits.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/make_splits.py)). Generated `data_index/splits/split_v1.csv` (30 train, 6 val, 6 test) and `data_index/splits/folds_5.csv` with zero patient leakage, guaranteed per-class representation in val/test, rare-class stratification (`p100` and `p99`), 3 class schemes (`7class`, `binary`, `severity3`), and summary report ([`data_index/splits/splits_report.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index/splits/splits_report.md)). 16 unit tests added in `tests/test_splits.py` (69 total project tests passing clean). Fixed 1-frame discrepancy in validator (31,500 total frames confirmed).
- **2026-09-30**: **Phase 1 Complete (Release v0.1.0)** - Finalized dataset ingestion, integrity verification, indexing manifests, clinical metadata cleaning, exploratory data analysis, and automated execution scripts (`run_phase1.bat`, `run_phase1.sh`, `Makefile`).
- **2026-09-30**: Phase 1.5 - Exploratory Data Analysis completed ([`src/data/eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/eda.py), [`scripts/run_eda.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/run_eda.py)). 8 analytical figures generated in [`reports/figures/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/figures) and summary report [`reports/eda_summary.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/reports/eda_summary.md).
- **2026-09-30**: Phase 1.4 - Clinical metadata inspector and cleaner implemented ([`src/data/metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/metadata.py), [`scripts/inspect_metadata.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/inspect_metadata.py)).
- **2026-09-30**: Phase 1.3 - Dataset Indexer implemented ([`src/data/indexer.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/indexer.py), [`scripts/build_index.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/build_index.py)). Discovered groundtruth bounding-box formats, built 4 index tables and `classes.json`.
- **2026-09-29**: Phase 1.2 - Dataset Validator module and CLI implemented ([`src/data/validator.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/src/data/validator.py), [`scripts/validate_dataset.py`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/scripts/validate_dataset.py)). Verified full dataset integrity with PASS report.
- **2026-09-29**: Phase 1.1 - Project scaffold created, CADICA dataset scanner implemented, central config system initialized, unit tests written and verified.

