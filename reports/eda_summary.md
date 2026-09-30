# CADICA Dataset — Exploratory Data Analysis Report

**Generated**: 2026-09-30 23:25:16
**Dataset**: CADICA.zip
**Total Patients**: 42
**Total Videos**: 668
**Total Frames**: 31,500
**Total Bounding Boxes**: 6,161

---

## Key Findings & Potential Issues

- **Class imbalance detected**: largest / smallest class ratio = 30.9x (threshold 5.0x). Largest: `p0_20` (1,944), smallest: `p99` (63).
- **Tiny boxes**: 175 boxes (2.8%) have area < 400 px² — may be hard for models to detect.
- **Annotation sparsity**: only 3,996 / 31,500 frames (12.7%) have ground-truth annotations.

---

## 1. Per-Patient Statistics

![Per-Patient Statistics](reports/figures/per_patient_stats.png)

| Patient | Videos | Frames | Lesion Videos | Boxes |
|---------|--------|--------|---------------|-------|
| p1 | 14 | 702 | 12 | 89 |
| p2 | 13 | 601 | 8 | 49 |
| p3 | 8 | 391 | 8 | 22 |
| p4 | 9 | 425 | 8 | 19 |
| p5 | 9 | 471 | 8 | 112 |
| p6 | 8 | 332 | 8 | 43 |
| p7 | 10 | 537 | 8 | 33 |
| p8 | 13 | 775 | 4 | 27 |
| p9 | 10 | 527 | 8 | 57 |
| p10 | 8 | 402 | 8 | 175 |
| p11 | 22 | 978 | 13 | 265 |
| p12 | 31 | 1657 | 11 | 479 |
| p13 | 17 | 1067 | 5 | 102 |
| p14 | 45 | 1486 | 3 | 122 |
| p15 | 8 | 468 | 8 | 289 |
| p16 | 33 | 963 | 13 | 155 |
| p17 | 19 | 1035 | 13 | 143 |
| p18 | 17 | 831 | 12 | 215 |
| p19 | 9 | 376 | 5 | 68 |
| p20 | 25 | 1041 | 12 | 82 |
| p21 | 10 | 329 | 8 | 86 |
| p22 | 38 | 2381 | 14 | 186 |
| p23 | 20 | 663 | 14 | 195 |
| p24 | 23 | 1360 | 13 | 726 |
| p25 | 17 | 869 | 13 | 190 |
| p26 | 5 | 211 | 5 | 17 |
| p27 | 9 | 348 | 8 | 59 |
| p28 | 8 | 323 | 7 | 33 |
| p29 | 14 | 636 | 8 | 72 |
| p30 | 13 | 394 | 6 | 80 |
| p31 | 25 | 982 | 16 | 295 |
| p32 | 15 | 730 | 10 | 146 |
| p33 | 4 | 182 | 4 | 55 |
| p34 | 13 | 564 | 9 | 223 |
| p35 | 11 | 391 | 6 | 83 |
| p36 | 26 | 1577 | 10 | 183 |
| p37 | 11 | 433 | 4 | 51 |
| p38 | 10 | 337 | 8 | 75 |
| p39 | 18 | 662 | 11 | 156 |
| p40 | 16 | 1063 | 10 | 206 |
| p41 | 23 | 1620 | 15 | 396 |
| p42 | 11 | 380 | 8 | 102 |

---

## 2. Lesion Category Distribution

![Class Distribution](reports/figures/class_distribution.png)

| Category | Count | Percentage |
|----------|-------|------------|
| `p0_20` | 1,944 | 31.55% |
| `p20_50` | 1,128 | 18.31% |
| `p50_70` | 999 | 16.21% |
| `p70_90` | 893 | 14.49% |
| `p90_98` | 930 | 15.09% |
| `p99` | 63 | 1.02% |
| `p100` | 204 | 3.31% |

> [!WARNING]
> **Class imbalance detected**: largest / smallest class ratio = 30.9x (threshold 5.0x). Largest: `p0_20` (1,944), smallest: `p99` (63).

---

## 3. Bounding Box Geometry Statistics

![Bbox Statistics](reports/figures/bbox_statistics.png)

| Metric | Min | Median | Mean | Max | Std |
|--------|-----|--------|------|-----|-----|
| Width (px) | 10.0 | 41.0 | 46.0 | 162.0 | 24.8 |
| Height (px) | 11.0 | 39.0 | 44.5 | 181.0 | 23.2 |
| Area (px²) | 165.0 | 1540.0 | 2136.0 | 13200.0 | 1784.9 |
| Aspect Ratio | 0.2 | 1.0 | 1.3 | 6.3 | 0.9 |

---

## 4. Image Resolution Distribution

![Resolution Distribution](reports/figures/resolution_distribution.png)

| Resolution | Count | Percentage |
|------------|-------|------------|
| 512×512 | 31,500 | 100.0% |

---

## 5. Lesion vs Non-Lesion & Selected vs Nonselected Splits

![Split Analysis](reports/figures/lesion_selected_splits.png)

---

## 6. Boxes per Frame Histogram

![Boxes per Frame](reports/figures/boxes_per_frame.png)

- **Annotated frames**: 3,996
- **Boxes per frame range**: 1 – 6
- **Mean boxes per frame**: 1.54
- **Frames with multiple boxes**: 1,457 (36.5%)

---

## 7. Sample Annotated Frames

![Sample Annotations](reports/figures/sample_annotations.png)

---

## 8. Metadata–Lesion Correlations

![Metadata Correlations](reports/figures/metadata_correlations.png)

The heatmap above shows Pearson correlations between numeric clinical metadata features (age, BMI, comorbidities, etc.) and per-patient lesion statistics (number of boxes, lesion videos, etc.).
