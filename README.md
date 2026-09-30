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
│   │   └── dataset_scanner.py # Dataset discovery & zip scanning module
│   ├── models/
│   │   └── __init__.py        # Lesion detection model definitions
│   ├── evaluation/
│   │   └── __init__.py        # Evaluation metrics & loss functions
│   └── utils/
│       ├── __init__.py
│       └── config.py          # Configuration loader & logging utilities
├── scripts/                   # Execution scripts for training and preprocessing
├── tests/
│   ├── __init__.py
│   ├── test_config.py         # Unit tests for config loader
│   └── test_dataset_scanner.py# Unit tests for dataset scanner
├── notebooks/                 # Exploratory data analysis notebooks
├── reports/figures/           # Output evaluation figures and confusion matrices
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

## 5. Configuration & Utilities Usage

### Central Configuration Validation
All key project options are centralized in [`configs/config.yaml`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/configs/config.yaml):
- `dataset_root`: Path to dataset zip or extracted directory.
- `metadata_file`: Clinical metadata file name (`metadata.xlsx`).
- `folders`: Dictionary containing `selected` ("selectedVideos") and `nonselected` ("nonselectedVideos").
- `index_dir`: Path for data indexing manifests (`data_index`).
- `reports_dir`: Output reports directory (`reports`).
- `outputs_dir`: Output logs and checkpoints directory (`outputs`).
- `random_seed`: Deterministic random seed (`42`).
- `image_extension`: Default image file extension (`.png`).

Load and validate configuration programmatically:
```python
from src.utils.config import load_config

config = load_config("configs/config.yaml")
```

### Central Logging Utility
Retrieve a dual console/file logger writing to `outputs/logs/run.log`:
```python
from src.utils.logger import get_logger

logger = get_logger(name="Agent-TT", log_file="outputs/logs/run.log", level="INFO")
logger.info("Initializing Agent-TT pipeline...")
```

### Reproducibility & Random Seeding
Enforce deterministic randomness across Python, NumPy, and PyTorch:
```python
from src.utils.seed import set_seed

set_seed(42)
```

### Dataset Indexing CLI & Manifest Generation
Build structured CSV index tables and class mappings in [`data_index/`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/data_index):

```bash
# Build full index across all 42 patients
python -m scripts.build_index

# Build index for a subset of patients (quick testing)
python -m scripts.build_index --limit-patients 5

# Override dataset root path
python -m scripts.build_index --dataset_root CADICA.zip
```

#### Generated Index Schemas (`data_index/`):

1. **`frames_index.csv`** — one row per image frame:

   | Column | Description |
   |---|---|
   | `split_type` | `selected` or `nonselected` |
   | `patient_id` | Patient identifier (e.g. `p10`) |
   | `video_id` | Video sequence identifier (e.g. `v1`) |
   | `frame_id` | Frame stem (e.g. `p10_v1_00016`) |
   | `frame_path` | Path to frame PNG within zip/directory |
   | `is_selected_keyframe` | `True` if listed in `selectedFrames.txt` |
   | `has_lesion_video` | `True` if video is in `lesionVideos.txt` |
   | `has_groundtruth` | `True` if a matching groundtruth `.txt` exists |
   | `image_width` | Image width in pixels |
   | `image_height` | Image height in pixels |

2. **`annotations.csv`** — one row per bounding box:

   | Column | Description |
   |---|---|
   | `patient_id`, `video_id`, `frame_id`, `frame_path` | Frame identifiers |
   | `x`, `y`, `w`, `h` | Bounding box in absolute pixels `[x, y, w, h]` |
   | `category` | Stenosis-percentage class label (e.g. `p0_20`, `p50_70`, `p100`) |
   | `x_center`, `y_center` | YOLO-normalised box centre coordinates |
   | `w_norm`, `h_norm` | YOLO-normalised box width and height |

3. **`videos_index.csv`** — one row per video sequence:

   `patient_id`, `video_id`, `split_type`, `num_frames`, `num_selected_frames`, `is_lesion_video`, `num_annotated_frames`, `num_boxes`

4. **`patients_index.csv`** — one row per patient:

   `patient_id`, `num_videos`, `num_selected_videos`, `num_nonselected_videos`, `num_frames`, `num_lesion_videos`, `num_boxes`

5. **`classes.json`** — auto-discovered stenosis category → integer ID mapping:
   ```json
   { "p0_20": 0, "p100": 1, "p20_50": 2, "p50_70": 3, "p70_90": 4, "p90_98": 5, "p99": 6 }
   ```

### Dataset Validation CLI
Validate dataset structural integrity, check metadata, patient video folders (`pX/vY`), frame input files, groundtruth labels, and generate a clear PASS/WARN/FAIL report:
```bash
# Validate dataset root specified in configs/config.yaml
python -m scripts.validate_dataset

# Validate specific zip archive or extracted directory and output custom report path
python -m scripts.validate_dataset --dataset_root CADICA.zip --output reports/dataset_validation.txt
```

### Dataset Scanner CLI
Scan `CADICA.zip` or raw folder structures dynamically without copying dataset files into the repository:
```bash
# Scan dataset zip archive specified in configs/config.yaml
python -m src.data.dataset_scanner

# Scan custom zip path and save output report
python -m src.data.dataset_scanner --zip_path CADICA.zip --output data/dataset_summary.json

# Scan extracted directory
python -m src.data.dataset_scanner --dir_path data/extracted
```

### Running Unit Tests
Execute the pytest suite:
```bash
python -m pytest tests/
```

---

## 6. Roadmap (Phases 1-5)
- **Phase 1: Project Setup & Dataset Indexing**
  - Scaffold creation, config loader, logging, dynamic dataset scanner, metadata parser, and frame indexing.
- **Phase 2: Data Preprocessing & Augmentation**
  - ICA image normalization, CLAHE enhancement, vessel filtering, bounding box scaling, dataset splitters (patient-aware k-fold CV).
- **Phase 3: Model Architecture & Training**
  - Baseline object detection backbone (YOLO / Faster R-CNN / DETR adapted for ICA), loss functions, model trainer pipeline.
- **Phase 4: Evaluation & Performance Analysis**
  - Lesion detection metrics calculator, IoU thresholding, confusion matrix, mAP evaluation, ROC-AUC analysis.
- **Phase 5: CLI Tooling, Reporting & Deployment**
  - Inference CLI, summary reporting, automated figure generation, model export (ONNX/TorchScript).

---

## 7. Metrics Tracked
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

## 8. Changelog Link
For full version history and updates, view the [`CHANGELOG.md`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/CHANGELOG.md) file.

---

## 9. Update Log
- **2026-09-29**: Phase 1.1 - Project scaffold created, CADICA dataset scanner implemented, central config system initialized, unit tests written and verified.
