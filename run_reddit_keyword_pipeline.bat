@echo off
setlocal

REM Move to project root (location of this .bat file)
cd /d "%~dp0"

echo ===============================================
echo Multimodal Sarcasm Detection - Reddit Pipeline
echo ===============================================
echo.

REM Activate local virtual environment if present
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment: venv
    call "venv\Scripts\activate.bat"
) else (
    echo [INFO] No local venv found at .\venv
    echo [INFO] Using current Python interpreter from PATH
)

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python is not available in PATH.
    echo Install Python or activate your environment manually, then retry.
    pause
    exit /b 1
)

echo.
echo Starting Reddit keyword pipeline...
echo The script will ask: "Enter keywords separated by comma"
echo.

python social_media_pipeline.py
if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline execution failed.
    pause
    exit /b 1
)

echo.
echo Done. Check results in:
echo   results\reddit_multimodal_results.json
echo.
pause
