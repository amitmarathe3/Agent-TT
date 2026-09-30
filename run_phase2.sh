#!/usr/bin/env bash
# ==============================================================================
# CADICA Phase 2 Pipeline Runner (Linux / macOS / Bash)
# Executes: make_splits -> export_yolo -> visualize_pipeline -> sanity_check
# ==============================================================================

set -euo pipefail

echo "========================================================"
echo "  Starting CADICA Agent Phase 2 Pipeline"
echo "========================================================"

echo "[1/4] Step 1: Generating Patient-Wise Splits and 5-Fold Cross-Validation..."
python scripts/make_splits.py

echo ""
echo "[2/4] Step 2: Exporting Ultralytics YOLO Dataset with Hard Negatives..."
python scripts/export_yolo.py --scheme 7class --neg-ratio 0.25

echo ""
echo "[3/4] Step 3: Computing Normalization Statistics and Generating Visualizations..."
python scripts/visualize_pipeline.py

echo ""
echo "[4/4] Step 4: Running Evaluation Sanity Check (Oracle vs. Random Baseline)..."
python scripts/sanity_check.py --split val --scheme 7class

echo ""
echo "========================================================"
echo "  Phase 2 Pipeline Completed Successfully!"
echo "  Outputs generated:"
echo "    - data_index/splits/split_v1.csv"
echo "    - data_index/splits/folds_5.csv"
echo "    - data_index/splits/splits_report.md"
echo "    - outputs/yolo_dataset/ (data.yaml, images/, labels/)"
echo "    - data_index/norm_stats.json"
echo "    - reports/figures/augmentation_check.png"
echo "    - outputs/eval/oracle_val/"
echo "    - outputs/eval/random_val/"
echo "========================================================"
