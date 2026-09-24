@echo off
title VoiceGuard AI - Voice Impersonation Defense System
echo ============================================================
echo   Starting VoiceGuard AI - Real-Time Voice Cloning Defense
echo ============================================================
echo.

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

echo Starting FastAPI Server on http://127.0.0.1:8000 ...
echo   - Caller Protection: http://127.0.0.1:8000/caller_protection.html
echo   - Dashboard:         http://127.0.0.1:8000/dashboard.html
echo   - Overview:          http://127.0.0.1:8000/index.html
echo   - History:           http://127.0.0.1:8000/history.html
echo.

start http://127.0.0.1:8000/caller_protection.html

python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
pause
