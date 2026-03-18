@echo off
cd /d "%~dp0"

echo ============================================
echo Building ClassificationWebApp (standalone)
echo ============================================
echo.

REM ── Find Python ──────────────────────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo Using venv Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python
)

REM ── Step 1: Build the Vite frontend ──────────────────────────────────────
echo.
echo [1/2] Building frontend (Vite)...
cd web_app
call npm run build
if errorlevel 1 (
    echo ERROR: npm run build failed
    cd ..
    pause
    exit /b 1
)
cd ..
echo Frontend built to web_app\dist\

REM ── Step 2: Build the tiny launcher exe with PyInstaller ──────────────────
echo.
echo [2/2] Building launcher exe (PyInstaller)...
if exist "build" rmdir /s /q build

REM Output exe directly to project root so it sits next to .venv and backend/
"%PYTHON%" -m PyInstaller ^
    --onefile ^
    --name ClassificationWebApp ^
    --distpath . ^
    --workpath build ^
    --noconfirm ^
    --clean ^
    launcher.py

echo.
if exist "ClassificationWebApp.exe" (
    echo ============================================
    echo BUILD SUCCESSFUL
    echo ============================================
    echo.
    echo Launcher exe:   ClassificationWebApp.exe  (project root)
    echo Frontend files: web_app\dist\
    echo.
    echo To deploy, copy this entire folder to the target machine.
    echo Both .venv\ and models\hf_cache\ must be present.
    echo.
    echo Run: ClassificationWebApp.exe
) else (
    echo ============================================
    echo BUILD FAILED - check output above
    echo ============================================
)

echo.
pause
