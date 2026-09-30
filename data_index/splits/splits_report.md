# CADICA Dataset — Patient-Wise Splits Report

**Random Seed**: `42`  
**Split Strategy**: Patient-Level Separation (Zero Patient Overlap)  
**Target Ratios**: Train 70% / Val 15% / Test 15%  
**Cross-Validation**: 5-Fold Patient-Wise Stratified CV  

---

## 1. Train / Validation / Test Split (`split_v1.csv`)

### Split Overview

| Split | Patients | Patients % | Frames | Frames % | Boxes | Boxes % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **train** | 30 | 71.4% | 21,383 | 67.9% | 4,365 | 70.8% |
| **val** | 6 | 14.3% | 2,879 | 9.1% | 757 | 12.3% |
| **test** | 6 | 14.3% | 7,238 | 23.0% | 1,039 | 16.9% |

### Per-Class Bounding Box Distribution across Splits

| Category | Stenosis Range | Train Boxes (Split %) | Val Boxes (Split %) | Test Boxes (Split %) | Total Boxes |
|---|:---:|:---:|:---:|:---:|:---:|
| `p0_20` | < 20% | 1297 (29.7%) | 270 (35.7%) | 377 (36.3%) | **1,944** |
| `p100` | 100% (total) | 84 (1.9%) | 64 (8.5%) | 56 (5.4%) | **204** |
| `p20_50` | 20% - 50% | 801 (18.4%) | 129 (17.0%) | 198 (19.1%) | **1,128** |
| `p50_70` | 50% - 70% | 793 (18.2%) | 84 (11.1%) | 122 (11.7%) | **999** |
| `p70_90` | 70% - 90% | 623 (14.3%) | 111 (14.7%) | 159 (15.3%) | **893** |
| `p90_98` | 90% - 98% | 723 (16.6%) | 94 (12.4%) | 113 (10.9%) | **930** |
| `p99` | 99% (subtotal) | 44 (1.0%) | 5 (0.7%) | 14 (1.3%) | **63** |

> [!NOTE]
> All 7 stenosis categories have **> 0 bounding boxes** in both validation and test sets.
> In particular, rare class `p100` (present in only 3 patients: p11, p23, p25) has exactly one patient in each split.

### Patient Allocation per Split

- **train** (30 patients): `p1, p2, p3, p4, p5, p7, p9, p10, p12, p13, p14, p15, p17, p18, p19, p20, p24, p25, p26, p27, p28, p29, p30, p32, p37, p38, p39, p40, p41, p42`
- **val** (6 patients): `p6, p21, p23, p31, p33, p35`
- **test** (6 patients): `p8, p11, p16, p22, p34, p36`

---

## 2. 5-Fold Patient-Wise Cross-Validation (`folds_5.csv`)

| Fold | Patients | Frames | Boxes | Boxes % | Rare Classes Included | Patients Assigned |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Fold 0** | 8 | 6,175 | 1,157 | 18.8% | p100: 56, p99: 8 | `p1, p3, p11, p12, p19, p20, p29, p30` |
| **Fold 1** | 8 | 6,978 | 1,175 | 19.1% | p100: 84, p99: 4 | `p7, p8, p9, p14, p25, p31, p33, p41` |
| **Fold 2** | 8 | 7,233 | 1,112 | 18.0% | p100: 64, p99: 8 | `p10, p22, p23, p28, p35, p36, p37, p40` |
| **Fold 3** | 9 | 5,467 | 1,516 | 24.6% | p100: 0, p99: 16 | `p4, p6, p13, p21, p24, p27, p34, p39, p42` |
| **Fold 4** | 9 | 5,647 | 1,201 | 19.5% | p100: 0, p99: 27 | `p2, p5, p15, p16, p17, p18, p26, p32, p38` |

---

## 3. Configured Class Schemes

The pipeline supports 3 severity aggregation schemes configured in [`configs/config.yaml`](file:///c:/Users/HP/Documents/Final%20project/New%20folder/configs/config.yaml):

1. **`7class`** (Original CADICA Granularity):
   - `p0_20`: `0` | `p20_50`: `1` | `p50_70`: `2` | `p70_90`: `3` | `p90_98`: `4` | `p99`: `5` | `p100`: `6`
2. **`binary`** (Lesion vs Normal Detection):
   - Normal / Background: `0` | All Lesion categories (`p0_20` .. `p100`): `1`
3. **`severity3`** (Clinical Tri-Tier Classification):
   - **Mild** (`0`): `p0_20` + `p20_50` (Atherosclerosis without hemodynamic compromise)
   - **Moderate** (`1`): `p50_70` (Borderline / intermediate stenosis)
   - **Severe** (`2`): `p70_90` + `p90_98` + `p99` + `p100` (Hemodynamically significant / occlusive)

---

## 4. Leakage Verification

- **Overlap check**: `len(set(train) & set(val)) == 0` — **CONFIRMED**
- **Overlap check**: `len(set(train) & set(test)) == 0` — **CONFIRMED**
- **Overlap check**: `len(set(val) & set(test)) == 0` — **CONFIRMED**
- **K-Fold overlap check**: Zero patient overlap across all 5 folds — **CONFIRMED**
