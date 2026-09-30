# ==============================================================================
# CADICA Agent-TT Makefile
# ==============================================================================

.PHONY: all help test validate index metadata eda phase1 splits export-yolo visualize-pipeline clean

# Default target
all: phase1

help:
	@echo "CADICA Agent-TT Makefile Commands:"
	@echo "  make phase1             - Run full Phase 1 pipeline (validate -> index -> metadata -> eda)"
	@echo "  make validate           - Step 1: Validate dataset folder structure and files"
	@echo "  make index              - Step 2: Build frames, annotations, videos, and patients indices"
	@echo "  make metadata           - Step 3: Inspect and clean clinical metadata"
	@echo "  make eda                - Step 4: Run exploratory data analysis and generate charts"
	@echo "  make splits             - Phase 2.1: Generate patient-wise train/val/test and 5-fold CV splits"
	@echo "  make export-yolo        - Phase 2.2: Export YOLO dataset with hard negatives"
	@echo "  make visualize-pipeline - Phase 2.2: Generate augmentation check figure and norm stats"
	@echo "  make test               - Run full pytest test suite"
	@echo "  make clean              - Clean generated outputs, index files, and caches"

test:
	pytest -v

validate:
	python scripts/validate_dataset.py

index:
	python scripts/build_index.py

metadata:
	python scripts/inspect_metadata.py

eda:
	python scripts/run_eda.py

splits:
	python scripts/make_splits.py

export-yolo:
	python scripts/export_yolo.py

visualize-pipeline:
	python scripts/visualize_pipeline.py

phase1: validate index metadata eda
	@echo "Phase 1 pipeline completed successfully!"

clean:
	python -c "import shutil, os, glob; [os.remove(f) for f in glob.glob('data_index/*') if os.path.isfile(f)]; [os.remove(f) for f in glob.glob('reports/figures/*.png')]; [os.remove(f) for f in glob.glob('reports/*.*') if os.path.isfile(f)]"
