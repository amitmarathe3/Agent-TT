# Changelog

All notable changes to the Agent-TT (CADICA-Agent) project will be documented in this file.

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
