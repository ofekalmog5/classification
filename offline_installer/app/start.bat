@echo off
cd /d "%~dp0"

echo Starting ClassificationWebApp...

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set HF_HOME=%~dp0models\hf_cache

start "" http://127.0.0.1:8000

.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 1
