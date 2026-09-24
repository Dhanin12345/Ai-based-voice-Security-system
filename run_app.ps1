Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting VoiceGuard AI - Real-Time Voice Cloning Defense" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting FastAPI Server on http://127.0.0.1:8000 ..." -ForegroundColor Yellow
Write-Host "  - Caller Protection: http://127.0.0.1:8000/caller_protection.html" -ForegroundColor Green
Write-Host "  - Dashboard:         http://127.0.0.1:8000/dashboard.html" -ForegroundColor Green
Write-Host "  - Overview:          http://127.0.0.1:8000/index.html" -ForegroundColor Green
Write-Host "  - History:           http://127.0.0.1:8000/history.html" -ForegroundColor Green
Write-Host ""

Start-Process "http://127.0.0.1:8000/caller_protection.html"

& ".\venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
