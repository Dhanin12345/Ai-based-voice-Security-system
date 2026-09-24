@echo off
REM VoiceGuard - Automated Data Update & Application Launcher for Windows

echo ============================================================
echo      AI VOICE CLONING DETECTION - DATA PIPELINE
echo ============================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment 'venv' not detected. Running with system python...
)

echo Select an action:
echo  [1] Update Programming Data (Generate + Extract Features + Train Model)
echo  [2] View Current Data Status and Model Metrics
echo  [3] Run Automated Test Suite (pytest)
echo  [4] Run Web Server and Dashboard
echo  [5] Full Pipeline (Update Data + Test + Run Server)
echo  [6] Exit
echo.

set /p choice="Enter your choice [1-6]: "

if "%choice%"=="1" (
    echo.
    echo Running data update pipeline...
    python ml\update_data.py
    pause
) else if "%choice%"=="2" (
    echo.
    python ml\update_data.py --status
    pause
) else if "%choice%"=="3" (
    echo.
    pytest tests\ -v
    pause
) else if "%choice%"=="4" (
    echo.
    python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
) else if "%choice%"=="5" (
    echo.
    python run_pipeline.py --all
) else (
    echo Exiting.
)
