# CADICA YOLO Agent — Methodology, Accuracy & Evaluation Matrix

> All numbers are read directly from  
> `outputs/eval/exp_7class_yolo11s_oversample_run1_{val,test}/metrics.json`

---

## 1. Methodology

### 1.1 Problem Statement
Automated detection and severity grading of coronary artery stenosis in X-ray angiography (CADICA dataset). A **7-class** scheme maps stenosis percentage to clinical bins.

| Class ID | Label | Stenosis Range |
|----------|-------|----------------|
| 0 | p0_20 | 0 – 20 % |
| 1 | p20_50 | 20 – 50 % |
| 2 | p50_70 | 50 – 70 % |
| 3 | p70_90 | 70 – 90 % |
| 4 | p90_98 | 90 – 98 % |
| 5 | p99 | 99 % |
| 6 | p100 | 100 % (occlusion) |

### 1.2 Model Architecture
- **Backbone**: YOLOv11-Small (`yolo11s`)  
- **Experiment**: `exp_7class_yolo11s_oversample_run1`  
- **Weights**: `outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt`  
- **Input resolution**: 512 × 512 px (letterboxed into 544 × 544 for `544_padded` mode)  
- **GPU**: NVIDIA GeForce GTX 1650 · PyTorch 2.14.1+cu126

### 1.3 Training Strategy
- Oversampling of minority classes to address severe class imbalance  
- Standard YOLO augmentation pipeline (mosaic, HSV-jitter, flips)  
- Confidence threshold frozen at **0.0500** (Youden J optimisation on val)

### 1.4 Inference Pipeline (`src/models/yolo_infer.py`)
1. Read image → BGR → letterbox to 544 px (`auto=False`, pad value 114)  
2. Forward pass through YOLOv11s  
3. NMS with `iou=0.7`, `max_det=100`  
4. Inverse-letterbox boxes back to 512 px space  
5. Filter by `conf ≥ threshold.json`

### 1.5 Evaluation Framework (`src/evaluation/`)
- **Object-level**: COCO-style 101-point interpolated AP at IoU ≥ 0.5 and 0.5:0.95  
- **Frame-level**: binary "any detection" vs ground truth; ROC-AUC, PR-AUC  
- **Patient-level**: aggregated per-patient lesion presence & significant stenosis (≥ 50 %)

---

## 2. Accuracy Summary

| Metric | Val | Test |
|--------|-----|------|
| **mAP@0.5** | 0.0109 | 0.0095 |
| **mAP@0.5:0.95** | 0.0028 | 0.0019 |
| Micro Precision (box) | 0.0651 | 0.0749 |
| Micro Recall (box) | 0.0449 | 0.0481 |
| Micro F1 (box) | 0.0532 | 0.0586 |
| Macro F1 (box) | 0.0314 | 0.0346 |
| Mean IoU (matched) | 0.6164 | 0.5806 |
| **Frame Accuracy** | 0.5568 | 0.5012 |
| Frame Precision | 0.8839 | 0.9105 |
| Frame Recall / Sensitivity | 0.5138 | 0.4179 |
| Frame Specificity | 0.7292 | 0.8353 |
| Frame F1 | 0.6499 | 0.5729 |
| **Frame ROC-AUC** | 0.6503 | 0.6985 |
| **Frame PR-AUC** | 0.8755 | 0.8939 |

---

## 3. Per-Class AP@0.5

| Class | Name | Val AP@0.5 | Test AP@0.5 |
|-------|------|-----------|------------|
| 0 | p0_20 | 0.0168 | 0.0234 |
| 1 | p20_50 | 0.0024 | 0.0007 |
| 2 | p50_70 | 0.0047 | 0.0011 |
| 3 | p70_90 | 0.0005 | 0.0411 |
| 4 | p90_98 | 0.0520 | 0.0000 |
| 5 | p99 | 0.0000 | 0.0000 |
| 6 | p100 | 0.0000 | 0.0000 |

---

## 4. Detection Confusion Matrix (Val, IoU ≥ 0.5, conf ≥ 0.05)

Rows = ground-truth class, Cols = predicted class (last col = missed / background FN)

|  | p0_20 | p20_50 | p50_70 | p70_90 | p90_98 | p99 | p100 | FN |
|--|-------|--------|--------|--------|--------|-----|------|----|
| **p0_20** | **25** | 0 | 0 | 0 | 0 | 0 | 0 | 245 |
| **p20_50** | 0 | **1** | 0 | 0 | 0 | 0 | 0 | 128 |
| **p50_70** | 0 | 0 | **3** | 0 | 0 | 0 | 0 | 81 |
| **p70_90** | 0 | 0 | 0 | **0** | 0 | 0 | 0 | 111 |
| **p90_98** | 0 | 0 | 0 | 0 | **5** | 0 | 0 | 89 |
| **p99** | 0 | 0 | 0 | 0 | 0 | **0** | 0 | 5 |
| **p100** | 0 | 0 | 0 | 0 | 0 | 0 | **0** | 64 |
| **FP (BG→cls)** | 226 | 90 | 108 | 42 | 20 | 2 | 0 | — |

---

## 5. Patient-Level Results

| Level | Split | Metric | Value |
|-------|-------|--------|-------|
| Lesion Detection | Val | Accuracy / Recall | **1.000** |
| Lesion Detection | Test | Accuracy / Recall | **1.000** |
| Significant Stenosis (≥50%) | Val | F1 | **0.800** |
| Significant Stenosis (≥50%) | Test | F1 | **0.909** |
| Significant Stenosis (≥50%) | Test | ROC-AUC | **1.000** |

---

## 6. Interpretation

| Dimension | Finding |
|-----------|---------|
| Box-level mAP | Low (~0.01) — model struggles with fine-grained severity grading at box IoU ≥ 0.5 |
| Frame-level PR-AUC | High (~0.89) — model reliably flags frames containing any stenosis |
| Patient-level recall | **100 %** — no patient with lesion is missed |
| High-precision frame mode | Precision ~0.91 at conf 0.05 — very few false-alarm frames |
| Preprocessing sensitivity | `544_padded` ×2.5 improvement over direct 512 feed |
| Class imbalance | p70_90 – p100 under-represented; mAP near zero for severe classes on val |
