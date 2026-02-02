@echo off
REM Atlas Seeding Execution Script for Windows
REM Global-BioScan Deep-Sea eDNA Reference Atlas

echo ================================================================================
echo GLOBAL-BIOSCAN ATLAS SEEDING UTILITY
echo ================================================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [FAIL] Could not activate virtual environment
        echo [INFO] Please run: python -m venv .venv
        pause
        exit /b 1
    )
)

echo [PASS] Virtual environment active
echo.

REM Check dependencies
echo [INFO] Checking dependencies...
python -c "import pyobis; import Bio; import pyarrow; print('[PASS] All dependencies installed')" 2>nul
if errorlevel 1 (
    echo [WARN] Missing dependencies detected
    echo [INFO] Installing seeding requirements...
    pip install -r seeding_requirements.txt
    if errorlevel 1 (
        echo [FAIL] Could not install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Starting Atlas Seeding...
echo [INFO] This may take 1-4 hours depending on target species count
echo [INFO] Press Ctrl+C to interrupt (progress will be saved)
echo.

REM Run seeding script
python src\edge\seed_atlas.py

if errorlevel 1 (
    echo.
    echo [FAIL] Seeding encountered errors
    echo [INFO] Check seed_atlas.log for details
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo [SUCCESS] SEEDING COMPLETE
echo ================================================================================
echo.
echo Next Steps:
echo   1. Launch Streamlit: streamlit run src\interface\app.py --port 8504
echo   2. Navigate to Configuration tab
echo   3. Click "Verify Database Integrity"
echo   4. Check sequence count in System Diagnostics
echo.
echo Files Generated:
echo   - E:\GlobalBioScan_DB\lancedb_store\  (Database)
echo   - E:\GlobalBioScan_DB\seeding_manifest.json  (Metadata)
echo   - E:\GlobalBioScan_DB\checkpoint.json  (Resume point)
echo   - seed_atlas.log  (Execution log)
echo.
pause
