# Changelog

All notable changes to the Agent-TT (CADICA-Agent) project will be documented in this file.

## [Unreleased] - Phase 3 Reset & Real Training Runs Execution

### Added (Shortcut Check, Detector Control & Invalidation of Screening)
- **Invalidation Notice**: ResNet18 frame classifier (AUC 0.8781 val / 0.9289 test) and video screening (AUC 0.9107 val / 0.9339 test) results are **INVALID** due to an angiographer keyframe-selection shortcut. The model learned contrast presence and vessel density rather than lesion pathology.
- **Detector Frame AUC Not Credible**: The previously reported detector frame ROC-AUC (0.6503 val, 0.6985 test) was evaluated against canonical hard negatives drawn primarily from distant non-keyframes. Against matched nonlesion keyframes, detector frame discrimination drops to test ROC-AUC 0.6356 [0.5292, 0.7616] and balanced accuracy 0.5830 [0.5303, 0.6470].
- **Detector Control Evaluation (`outputs/eval/detector_control/metrics.json`)**:
  - Keyframe vs Matched Nonlesion Keyframe: Val ROC-AUC 0.5850, BalAcc 0.5691 (threshold 0.0458); Test ROC-AUC 0.6356 [0.5292, 0.7616], BalAcc 0.5830 [0.5303, 0.6470], Sensitivity 0.3988, Specificity 0.7673.
  - Keyframe vs Distant Non-Keyframe: Val ROC-AUC 0.7287, BalAcc 0.6866 (threshold 0.0019); Test ROC-AUC 0.7794 [0.7075, 0.8284], BalAcc 0.7094 [0.6534, 0.7469], Sensitivity 0.8666, Specificity 0.5521.
- **Indexer Bug Fix & Raw Data Re-Index**:
  - Identified bug in `src/data/indexer.py` where `name.endswith("lesionVideos.txt")` erroneously matched `nonlesionVideos.txt`, misclassifying 113 nonlesion selected videos as lesion videos and reporting 0 nonlesion keyframes.
  - Corrected matcher to `Path(name).name == "lesionVideos.txt"`, updating `frames_index.csv` and `videos_index.csv`. Recovered 2,130 nonlesion keyframes (1,616 train, 239 val, 275 test) across 382 selected videos (269 lesion, 113 nonlesion).
- **Trivial-Feature Baseline (`scripts/trivial_baseline.py`)**:
  - Logistic regression on 13 simple global image statistics (mean, std, 5th/10th/25th/50th/75th/90th/95th percentiles, CLAHE dark vessel pixels, edge densities) achieves Val ROC-AUC 0.7749 and Test ROC-AUC 0.8869 (within 0.05 of the ResNet18 classifier's 0.9289), confirming that simple frame-level contrast and vessel density explain most of the apparent discriminative power.
- **Score-by-Group on Original Classifier (`scripts/score_by_group.py`)**:
  - Keyframe vs Keyframe (Lesion vs Nonlesion): Val ROC-AUC = 0.5692 (mean pos 0.7561 vs neg 0.6597); Test ROC-AUC = 0.5176 (mean pos 0.6559 vs neg 0.6234), confirming near-chance separation.
  - Keyframe vs Distant Non-Keyframe: Val ROC-AUC = 0.9254 (mean pos 0.7561 vs neg 0.1241); Test ROC-AUC = 0.9428 (mean pos 0.6559 vs neg 0.0890).
  - By Video Position (Test): first third ROC-AUC = 0.9541, middle third ROC-AUC = 0.8379, last third ROC-AUC = 0.9442.
- **Matched-Negative Retrain (`exp_frame_matched_negatives`)**:
  - Retrained ResNet18 classifier with positives = annotated keyframes and negatives = selected nonlesion keyframes only (both keyframes). Best validation checkpoint at Epoch 2 (early stopped after Epoch 5, threshold 0.1100).
  - Val: ROC-AUC = 0.6027 [0.4368, 0.7031], BalAcc = 0.6037 [0.4762, 0.6765], Sens = 0.8183 [0.6691, 0.9597], Spec = 0.3891 [0.0104, 0.5563], Prevalence = 0.7075, Majority Baseline = 0.7075.
  - Test: ROC-AUC = 0.4894 [0.1801, 0.7079], BalAcc = 0.4848 [0.3042, 0.6227], Sens = 0.7405 [0.5915, 0.9324], Spec = 0.2291 [0.0, 0.4378], Prevalence = 0.7126, Majority Baseline = 0.7126.
- **Video-Level Screening Redo (`outputs/eval/video_level_redo_selected/metrics.json`)**:
  - Evaluated screening restricted to selected videos (56 val: 44 lesion, 12 nonlesion; 63 test: 48 lesion, 15 nonlesion) using top-10 keyframe mean scores with threshold 0.1000:
  - Val: ROC-AUC = 0.5189 [0.3988, 0.6365], BalAcc = 0.6212 [0.4733, 0.7123], Sens = 0.9091 [0.8208, 1.0], Spec = 0.3333 [0.0, 0.5], F1 = 0.8696 [0.8, 0.9333].
  - Test: ROC-AUC = 0.4333 [0.1330, 0.6626], BalAcc = 0.4917 [0.4222, 0.6128], Sens = 0.9167 [0.8444, 0.9790], Spec = 0.0667 [0.0, 0.3040], F1 = 0.8302 [0.7284, 0.9043].
- **Clinical Reality (Dataset Limitation)**:
  - Frame/video lesion screening is NOT supported by the CADICA dataset as constructed: (1) 41 of 42 patients have confirmed CAD lesions, meaning 'nonlesion' videos are alternate views of diseased patients; (2) lesion visibility depends entirely on angiographic projection angle, so unannotated views cannot serve as negative screening controls without high clinical label noise.

### Added (ResNet18 Frame & Video Screening - INVALIDATED)
- **src/models/frame_classifier.py**:
  - ResNet18 binary classifier with ImageNet-pretrained weights, 384x384 input resolution, AdamW (lr 0.0003), class-weighted BCE loss, and seed ensembling (seeds 42 and 1337).
  - Scoring variants: classifier alone, detector max box score, validation-tuned weighted ensemble, and temporal smoothed variants over neighbouring frames in the same video.
  - Video-level aggregation using top-k frame mean (=10$ selected on validation).
  - Validation-chosen decision thresholds (0.1900 for frame level, 0.0800 for video level).
  - 500-iteration patient cluster bootstrap 95% confidence intervals.
- **scripts/train_frame_classifier.py**:
  - End-to-end training, dual-seed ensembling, variant sweep, video aggregation, and JSON metric exporter (outputs/eval/exp_frame_resnet18_{val,test}/metrics.json).
- **	ests/test_frame_classifier.py**:
  - Unit tests verifying 3 synthetic images through ResNet18, pairwise patient split disjointness, and temporal smoothing.
- **Screening Performance**:
  - Frame Winner (classifier_alone, threshold 0.1900): Val ROC-AUC 0.8781 [0.8354, 0.9228], BalAcc 0.8413 [0.7805, 0.8942]; Test ROC-AUC 0.9289 [0.8436, 0.9739], BalAcc 0.8759 [0.7800, 0.9473].
  - Video Winner (classifier_alone, threshold 0.0800, top-k=10): Val ROC-AUC 0.9107 [0.7704, 0.9856], BalAcc 0.8911 [0.7250, 0.9706]; Test ROC-AUC 0.9339 [0.9047, 0.9790], BalAcc 0.8974 [0.8387, 0.9626].

### Added (research prototype)
- Extended **`app.py`** to integrate `predict_single_image` from `src.models.yolo_infer` defaulting to `544_padded` letterbox preprocessing.
- **`app.py`** — Gradio demo (GPU if the 3.14 Python reports CUDA, else CPU; imgsz 512, iou 0.7, max_det 100) loading `outputs/runs/exp_7class_yolo11s_oversample_run1/weights/best.pt`. Inputs: image upload + *Random test sample* (with ground-truth overlay) + confidence slider frozen at the run's `threshold.json`. Outputs: annotated image, detections table (class, stenosis range, score, x/y/w/h), frame score (max box score), FLAGGED / NOT FLAGGED, and a banner whose mAP/AUC are read at runtime from `outputs/eval/*run1*_val/metrics.json`. Bad/blank inputs return "no detections" or a clear error, never a crash. Start with **`scripts/run_app.bat`**. Research prototype, not a diagnostic device.

### Important Notice: Purge of Unverified Results
- **Purge of Earlier Phase 3 Metrics**: All metric numbers previously reported for Phase 3 baseline runs and cross-validation summaries were purged. The fake leaderboard and CV artifacts (`leaderboard.csv`, `cv_summary.md`, `phase3_baseline_report.md`) were moved to `outputs/_invalid/`. Previous git release tags for these phases were deleted because they were never created/verified.
- **Strict Verification Policy**: All metrics in this phase and later are programmatically extracted strictly from verifiable on-disk training runs (`best.pt`), predictions (`preds_*.json`), and metrics reports (`metrics.json`).

### Superseded: All Previously Reported Bootstrap Confidence Intervals
**All bootstrap 95% CIs reported before this change were computed over the wrong patient universe and are superseded.**

`bootstrap_confidence_intervals` grouped frames using `frame_to_patient` loaded from `frames_index.csv`, which covers the **entire dataset (31,499 frames / 42 patients)** — not the frames the model was actually evaluated on. 36 of those 42 patients contribute **no eval frames at all**. Their tens of thousands of empty frames became score-0 / no-GT negatives:

- **Inert for mAP, macro F1 and sensitivity** (empty frames contribute no boxes), so those point estimates were unaffected.
- **Fatal for frame ROC-AUC**: the extra negatives pushed AUC from the true 0.6503 (val) / 0.6985 (test) to a meaningless ~0.99.
- **Corrupted every CI** by resampling a 42-patient universe the model was never evaluated on.

**Fix**: the bootstrap universe is now built from the canonical eval set (`get_canonical_eval_frame_ids(split)` in [`src/data/eval_sets.py`](src/data/eval_sets.py)) — 722 frames / 6 patients for `val`, 852 frames / 6 patients for `test` — asserted at runtime via the new `EXPECTED_EVAL_UNIVERSE` table. All four metrics (mAP@0.5, macro F1, sensitivity, frame AUC) are computed by one shared aggregator over that single universe, and the returned payload now carries `universe` (split / n_frames / n_patients) and `full_sample` (point estimates) so every CI is auditable against `metrics.json`.

| Split | Metric | Old CI (wrong 42-pt universe, n=100) | **New CI (canonical 6-pt universe, n=500)** |
|---|---|---|---|
| val | macro F1 | 0.0374 [0.0099, 0.1028] | **0.0359 [0.0112, 0.0797]** |
| val | mAP@0.5 | 0.0132 [0.0022, 0.0315] | **0.0138 [0.0025, 0.0307]** |
| val | sensitivity | 0.0510 [0.0108, 0.1457] | **0.0510 [0.0117, 0.1271]** |
| val | frame ROC-AUC | *not computed* (inflated ~0.99 when computed) | **0.6520 [0.5693, 0.7537]** |
| test | macro F1 | 0.0367 [0.0159, 0.0721] | **0.0358 [0.0150, 0.0641]** |
| test | mAP@0.5 | 0.0131 [0.0021, 0.0428] | **0.0125 [0.0021, 0.0389]** |
| test | sensitivity | 0.0470 [0.0145, 0.0651] | **0.0468 [0.0227, 0.0677]** |
| test | frame ROC-AUC | *not computed* (inflated ~0.99 when computed) | **0.6887 [0.5921, 0.7671]** |

### Added
- **`EXPECTED_EVAL_UNIVERSE`** table and canonical-universe bootstrap parameter (`split`, `yolo_dir`) in [`src/data/eval_sets.py`](src/data/eval_sets.py) and [`src/evaluation/metrics.py`](src/evaluation/metrics.py).
- **Regression tests `TestBootstrapUniverse`** in [`tests/test_evaluation.py`](tests/test_evaluation.py): (a) universe size and patient count per split (722/6 val, 852/6 test, and explicitly `< 42` patients so `frames_index.csv` cannot leak back in); (b) full-sample frame AUC equals `compute_frame_metrics` within 1e-6 and lies inside its own CI; (c) all four CIs share one universe, verified by matching each full-sample value against the direct metric on the canonical frames.
- **Point-estimate column and universe header** in the bootstrap CI table of [`src/evaluation/report.py`](src/evaluation/report.py).
- **[`reports/superseded_values.json`](reports/superseded_values.json)**: records every historical / superseded / purged number that the docs intentionally quote as wrong, old or removed, each with its reason and replacement. Audit check 5 now loads this file so such values are treated as sourced. Nothing in it may be presented as a current result.

### Corrected (documentation, verified against on-disk artifacts)
- **Normalization statistics** in the README (`Step 7` expected console output) claimed `Mean: 0.5066`, `Std: 0.1385`. `data_index/norm_stats.json` actually records **`0.5076` / `0.1395`** — the README values were digit transpositions.
- **Random-baseline sanity numbers** in the README were stale and not reproducible from any on-disk artifact (they even included `mAP@0.5=0.0099`, the pre-v0.2.1 zero-TP AP artifact). Both tables were regenerated from fresh `sanity_check.py` runs (seed 42):
  - Validation (722 frames): Random Frame Acc **0.3781**, Frame PR-AUC **0.7815**.
  - Test (852 frames): Random Frame Acc **0.4026**, Frame PR-AUC **0.8077**. The test-split baseline had never actually been run; `sanity_check.py --split test` was executed to produce `outputs/eval/{oracle,jittered_oracle,random}_test/`.
- Audit check 5 now reports **0 FAIL / 0 WARN**.

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

## [0.3.3] - 2026-10-15 - Phase 3 Real Evaluation (run1)

### Added
- Real inference/evaluation artifacts for exp_7class_yolo11s_oversample_run1 in outputs/eval/..._val and _test.
- Per-class summaries and caveats appended to metrics.md.
- Leaderboard rebuilt from metrics.json with only the existing real run.

### Notes
- Bootstrap used 100 iterations.
- Ultralytics cross-check warning remains: custom mAP@0.5 differs from native by >0.02.
