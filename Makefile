# ==============================================================================
# CADICA Agent-TT Makefile
# ==============================================================================

.PHONY: all help test validate index metadata eda phase1 clean

# Default target
all: phase1

help:
	@echo "CADICA Agent-TT Makefile Commands:"
	@echo "  make phase1    - Run full Phase 1 pipeline (validate -> index -> metadata -> eda)"
	@echo "  make validate  - Step 1: Validate dataset folder structure and files"
	@echo "  make index     - Step 2: Build frames, annotations, videos, and patients indices"
	@echo "  make metadata  - Step 3: Inspect and clean clinical metadata"
	@echo "  make eda       - Step 4: Run exploratory data analysis and generate charts"
	@echo "  make test      - Run full pytest test suite"
	@echo "  make clean     - Clean generated outputs, index files, and caches"

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

phase1: validate index metadata eda
	@echo "Phase 1 pipeline completed successfully!"

clean:
	python -c "import shutil, os, glob; [os.remove(f) for f in glob.glob('data_index/*') if os.path.isfile(f)]; [os.remove(f) for f in glob.glob('reports/figures/*.png')]; [os.remove(f) for f in glob.glob('reports/*.*') if os.path.isfile(f)]"
