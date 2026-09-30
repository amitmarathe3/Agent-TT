@echo off
REM ==============================================================================
REM CADICA Phase 1 Pipeline Runner (Windows)
REM Executes: validate_dataset -> build_index -> inspect_metadata -> run_eda
REM ==============================================================================

setlocal enabledelayedexpansion

echo ========================================================
echo   Starting CADICA Agent Phase 1 Pipeline
echo ========================================================

echo [1/4] Step 1: Validating Dataset Integrity...
python scripts/validate_dataset.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Dataset validation failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Step 2: Building Index and Ground-Truth Manifests...
python scripts/build_index.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Index generation failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Step 3: Inspecting and Cleaning Metadata...
python scripts/inspect_metadata.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Metadata inspection failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Step 4: Running Exploratory Data Analysis (EDA)...
python scripts/run_eda.py
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Exploratory data analysis failed with exit code %ERRORLEVEL%!
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================================
echo   Phase 1 Pipeline Completed Successfully!
echo   Outputs generated:
echo     - reports/dataset_validation.txt
echo     - data_index/*.csv and data_index/classes.json
echo     - reports/metadata_summary.txt and data_index/metadata_clean.csv
echo     - reports/figures/*.png and reports/eda_summary.md
echo ========================================================

endlocal
