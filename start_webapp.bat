@echo off
cd /d "%~dp0"

echo Starting Classification Web App...
echo.

REM ── Backend (FastAPI) ─────────────────────────────────────────────
echo [1/2] Starting backend server...
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

start "Backend - FastAPI" cmd /k "%PYTHON% -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

REM Give the backend a moment to start
timeout /t 2 /nobreak >nul

REM ── Frontend (Vite) ───────────────────────────────────────────────
echo [2/2] Starting frontend...
cd web_app

if exist "..\node_modules\.bin\vite.cmd" (
    start "Frontend - Vite" cmd /k "..\node_modules\.bin\vite.cmd"
) else if exist "node_modules\.bin\vite.cmd" (
    start "Frontend - Vite" cmd /k "node_modules\.bin\vite.cmd"
) else (
    start "Frontend - Vite" cmd /k "npm run dev"
)

cd ..

echo.
echo Both servers are starting in separate windows.
echo   Backend:  http://127.0.0.1:8000
echo   Frontend: http://localhost:5173
echo.
echo Close this window or press any key to exit.
pause >nul
