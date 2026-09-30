#!/usr/bin/env bash
# ==============================================================================
# CADICA Phase 1 Pipeline Runner (Linux / macOS / Bash)
# Executes: validate_dataset -> build_index -> inspect_metadata -> run_eda
# ==============================================================================

set -euo pipefail

echo "========================================================"
echo "  Starting CADICA Agent Phase 1 Pipeline"
echo "========================================================"

echo "[1/4] Step 1: Validating Dataset Integrity..."
python scripts/validate_dataset.py

echo ""
echo "[2/4] Step 2: Building Index and Ground-Truth Manifests..."
python scripts/build_index.py

echo ""
echo "[3/4] Step 3: Inspecting and Cleaning Metadata..."
python scripts/inspect_metadata.py

echo ""
echo "[4/4] Step 4: Running Exploratory Data Analysis (EDA)..."
python scripts/run_eda.py

echo ""
echo "========================================================"
echo "  Phase 1 Pipeline Completed Successfully!"
echo "  Outputs generated:"
echo "    - reports/dataset_validation.txt"
echo "    - data_index/*.csv & data_index/classes.json"
echo "    - reports/metadata_summary.txt & data_index/metadata_clean.csv"
echo "    - reports/figures/*.png & reports/eda_summary.md"
echo "========================================================"
