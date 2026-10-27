# CADICA YOLO & Clinical Reporting System — Full Project Methodology & Evaluation

> **Source of Truth**: All metrics and counts are read directly from `data_index/` and `outputs/eval/exp_7class_yolo11s_oversample_run1_544_{val,test}/metrics.json`.  
> **Regulatory Disclaimer**: Research prototype only. Not a medical device or diagnostic instrument. Machine predictions and candidate counts do not constitute medical diagnosis or evidence of disease.

---

## 1. Executive Summary & Objective

Automated detection and severity classification of coronary artery stenosis in X-ray coronary angiography (XCA) using the public **CADICA** dataset. The primary clinical objective is to assist interventional cardiologists by localizing potential stenotic lesions and categorizing vessel diameter reduction into established clinical categories, while maintaining strict adherence to biomedical software safety, patient-level split separation, and honest reporting of diagnostic uncertainties.

---

## 2. Dataset Architecture & Patient Partitioning

### 2.1 CADICA Dataset Overview
The CADICA dataset comprises fluoroscopic coronary angiograms captured during routine catheterization procedures. Annotations provide expert-delineated lesion bounding boxes and clinical stenosis percentages across selected video sequences.

### 2.2 Patient-Wise Split Protocol (`data_index/split_v1.csv`)
To prevent data leakage caused by anatomical and procedural correlation across frames of the same patient, partitioning is strictly patient-wise:

| Split | Patient Count | Patient Identifiers | Total Raw Frames | Selected Videos | Ground-Truth Lesions |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Train** | 24 | p1–p4, p6, p7, p10, p12–p14, p18, p19, p21, p23–p29, p31–p33, p35 | 74,013 | 263 | 1,939 |
| **Validation** | 6 | p5, p9, p15, p17, p20, p30 | 16,839 | 56 | 644 |
| **Test** | 6 | p8, p11, p16, p22, p34, p36 | 19,005 | 63 | 574 |
| **Total** | **36** | — | **109,857** | **382** | **3,157** |

*Integrity Constraint*: Zero patient intersection between splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). Enforced deterministically by audit check 2.

### 2.3 Clinical Stenosis Classification Scheme (7 Classes)

| Class ID | Technical Label | Clinical Stenosis Range | Clinical Severity Description |
| :---: | :--- | :--- | :--- |
| **0** | `p0_20` | 0% – 20% | Mild / Minimal luminal irregularity |
| **1** | `p20_50` | 20% – 50% | Mild-to-moderate stenosis |
| **2** | `p50_70` | 50% – 70% | Moderate / Hemodynamically intermediate stenosis |
| **3** | `p70_90` | 70% – 90% | Severe stenosis (frequently ischemia-inducing) |
| **4** | `p90_98` | 90% – 98% | Critical / Sub-occlusive stenosis |
| **5** | `p99` | 99% | Near-total occlusion / Pre-occlusive |
| **6** | `p100` | 100% | Total occlusion (TIMI 0 flow) |

---

## 3. Machine Learning Methodology

### 3.1 Model Architecture
- **Detector**: YOLOv11-Small (`yolo11s`).
- **Trained Artifact**: `outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt` (72.48 MB).
- **Runtime Framework**: PyTorch 2.14.1 with CUDA 12.6 acceleration on NVIDIA GeForce GTX 1650.
- **Python Binary**: Python 3.14.8 (`C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe`).

### 3.2 Preprocessing Standardization (`544_padded`)
Comparative ablations demonstrated that image preprocessing significantly impacts small-vessel feature preservation:
1. **Direct Resize (512×512)**: Bilinear interpolation distorts aspect ratio, degrading validation mAP@0.5.
2. **Standardized Padded Letterbox (`544_padded`)**: Letterboxing to 544×544 with uniform neutral padding (value 114) improved validation mAP@0.5 by approximately 2.5× over direct 512 input. All final inference pipelines, evaluators, and reporting tools strictly enforce `544_padded`.

### 3.3 Inference Pipeline (`src/models/yolo_infer.py`)
1. Read input angiogram frame (RGB / grayscale array or image file).
2. Apply letterbox padding to 544×544 px while maintaining native aspect ratio.
3. Model forward pass at base inference confidence `conf = 0.001` with Non-Maximum Suppression (NMS IoU = 0.70, `max_det = 100`).
4. Inverse-letterbox predicted coordinate boxes back to original image space.
5. Filter candidate detections using the frozen clinical threshold $\tau = 0.0500$.
6. Clinical spatial deduplication (`iou_thresh = 0.35`): Suppresses redundant anchor boxes around the identical physical lesion site, retaining the highest confidence candidate.

---

## 4. Screening Investigation & Scientific Invalidation

### 4.1 Initial Frame/Video Screening Hypothesis
To reduce detector computation, a two-stage pipeline was investigated: a lightweight ResNet-18 classifier would screen frames/videos for lesion presence before invoking the YOLO detector.

### 4.2 Shortcut Discovery & Investigation
- **Initial Run (`exp_frame_resnet18`)**: Appeared highly promising with Val ROC-AUC = 0.8872, Test ROC-AUC = 0.8804.
- **Audit Findings**: The negative pool contained unselected non-lesion video frames with systematic visual artifacts (background catheter movement and contrast wash differences), providing a non-anatomical keyframe selection shortcut.
- **Matched Negative Control (`exp_frame_matched_negatives`)**: When retrained strictly on true matched negative keyframes, Test ROC-AUC dropped sharply to 0.7077.
- **Detector-Only Control Benchmark**: Evaluating the maximum box confidence from the YOLO detector alone (`conf_max_box`) yielded Test ROC-AUC = 0.7049 (Balanced Accuracy = 0.6558) without any classifier training.
- **Scientific Determination**: The frame/video screening classifier was declared **INVALID / UNVALIDATED**. All frame/video screening scores were removed from reports and production inference.

---

## 5. Quantitative Evaluation Matrix & Diagnostic Reliability

All evaluation metrics are computed on the held-out test split (6 patients, 574 ground-truth lesions) using 500-iteration patient-level cluster bootstrap for 95% confidence intervals:

### 5.1 Object Detection Reliability Metrics (Test Set, Conf $\ge 0.0500$, IoU $\ge 0.50$)

| Evaluation Metric | Test Point Estimate | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Precision** | **0.0839** (8.39%) | [0.0320, 0.1898] |
| **Recall / Sensitivity** | **0.0462** (4.62%) | [0.0233, 0.0645] |
| **F1 Score** | **0.0596** | [0.0270, 0.0892] |
| **False-Alarm Share** $\frac{\text{FP}}{\text{TP} + \text{FP}}$ | **0.9161** (91.61%) | [0.8102, 0.9680] |
| **mAP@0.5** | **0.0107** | [0.0029, 0.0438] |
| **mAP@0.5:0.95** | **0.0019** | — |
| **Mean IoU (Matched TP)** | **0.5823** | — |
| **Frame-Level ROC-AUC** | **0.7750** | [0.7185, 0.8180] |

### 5.2 Per-Class Detection Breakdown (Test Split, IoU $\ge 0.50$)

| Class | Label | Ground Truth | TP | FP | FN | Precision | Recall | F1 Score | AP@0.5 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `p0_20` | 377 | 25 | 220 | 352 | 0.1020 | 0.0663 | 0.0804 | 0.0252 |
| **1** | `p20_50` | 198 | 2 | 171 | 196 | 0.0116 | 0.0101 | 0.0108 | 0.0008 |
| **2** | `p50_70` | 122 | 0 | 33 | 122 | 0.0000 | 0.0000 | 0.0000 | 0.0026 |
| **3** | `p70_90` | 159 | 20 | 84 | 139 | 0.1923 | 0.1258 | 0.1521 | 0.0457 |
| **4** | `p90_98` | 113 | 1 | 15 | 112 | 0.0625 | 0.0088 | 0.0155 | 0.0008 |
| **5** | `p99` | 14 | 0 | 0 | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **6** | `p100` | 56 | 0 | 1 | 56 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Total** | **All Classes** | **1,039\*** | **48** | **524** | **991** | **0.0839** | **0.0462** | **0.0596** | **0.0107** |

*\*Note: Multi-label evaluation counts total box evaluations across multi-label keyframes; TP + FN = 1,039 matches exact multi-class evaluation targets.*

---

## 6. Clinical Patient Reporting System (`scripts/patient_report.py`)

### 6.1 Design Principles
1. **Medical Ethics & Regulatory Compliance**: Absolute prohibition of automated binary verdicts ("diseased", "healthy", "positive", "negative").
2. **Transparent Uncertainty**: Runtime extraction of detector performance metrics with 95% bootstrap confidence intervals directly displayed in a prominent Reliability Box.
3. **Mandatory False-Alarm Warning**: Every report includes the statement:
   > *"Candidates can appear in patients without disease, so their presence or number is not evidence of disease."*
4. **Human-Friendly Clinical Findings**:
   - Technical class codes (`p70_90`) translated to descriptive medical nomenclature (`Severe Stenosis (70% - 90%)`).
   - Unrounded confidence floats (`0.8306`) converted to clinical percentage and confidence tiers (`83% (High)`).
   - Raw coordinate bounding box tuples (`[200.2, 526.8, 198.2, 55.5]`) replaced by numbered visual marker references (`Box #1 (Red)`).
   - Overlapping bounding boxes around the same lesion merged via spatial deduplication.
5. **Output Deliverables**:
   - Standalone HTML report (`report.html`) containing inline base64 thumbnails of keyframes.
   - Machine-readable JSON export (`report.json`) for downstream EHR integration.

---

## 7. Interactive Clinical Assistant UI (`app.py`)

- **Interface Framework**: Gradio web interface serving locally at `http://127.0.0.1:7860`.
- **Purpose**: Interactive visual assistant for angiogram frame exploration.
- **Visual Features**:
  - Direct image upload or random sample selection from test/validation splits.
  - Interactive threshold slider initialized to the frozen validation threshold ($0.0500$).
  - High-visibility red bounding box overlays with labeled badges (`#1: 70-90% (83%)`).
  - Non-technical findings table displaying:
    `["Finding", "Narrowing Severity", "Confidence Level", "Visual Marker"]`.

---

## 8. Continuous Quality Assurance & Audit Suite (`scripts/audit_project.py`)

Project integrity is guaranteed by 11 automated checks:

| Check | Name | Scope & Assertion | Status |
| :---: | :--- | :--- | :---: |
| **1** | Label files & counts | Verifies per-class box counts match `data_index` | **PASS** |
| **2** | Patient split leakage | Confirms zero patient overlap across train/val/test | **PASS** |
| **2b** | Experiment completeness | Ensures all registered experiments are fully trained | **PASS** |
| **3** | Leaderboard integrity | Confirms all leaderboard rows are backed by disk weights and metrics | **PASS** |
| **4** | `n_gt` guard | Asserts `TP + FN == n_gt == label box count` | **PASS** |
| **5** | Document provenance | Verifies every 3+ decimal number in docs originates from data | **PASS** |
| **5b** | Version claims | Confirms version tags exist in git history | **PASS** |
| **6** | Suspicious results guard | Asserts no metric $>0.95$ and no val/test gap $>0.15$ | **PASS** |
| **7** | Environment audit | Validates Python 3.14, CUDA torch, and clean git state | **PASS** |
| **8** | Frame-classifier splits | Enforces patient split separation on frame classifiers | **PASS** |
| **9** | Report reliability numbers | Verifies report HTML/JSON match `metrics.json` at runtime | **PASS** |
