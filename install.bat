@echo off
cd /d "%~dp0"
title Classification Web App - Installer

echo.
echo ================================================================
echo   Classification Web App - Full Installer
echo ================================================================
echo.
echo This will install EVERYTHING needed to run the app:
echo   - Python 3.11  (skipped if already installed)
echo   - Python virtual environment + all packages
echo   - PyTorch  (CUDA or CPU, auto-detected)
echo   - Node.js  (skipped if already installed)
echo   - Frontend build  (skipped if already built)
echo.
echo Requirements:
echo   - Internet connection
echo   - ~15 GB free disk space  (Python packages + models)
echo   - Windows 10 / 11  (64-bit)
echo.
echo Press any key to start, or Ctrl+C to cancel...
pause >nul
echo.

REM ── Verify PowerShell is available ───────────────────────────────
where powershell >nul 2>&1
if errorlevel 1 (
    echo ERROR: PowerShell is not available on this machine.
    echo Please install PowerShell and re-run this installer.
    pause
    exit /b 1
)

REM ── Parse optional flags ──────────────────────────────────────────
REM   install.bat --cpu         force CPU-only PyTorch
REM   install.bat --skip-torch  skip PyTorch installation
REM   install.bat --skip-sam    skip segment-geospatial / SAM
set EXTRA_ARGS=
:parse_args
if "%~1"=="" goto run
set EXTRA_ARGS=%EXTRA_ARGS% %~1
shift
goto parse_args

:run
REM ── Launch the PowerShell installer ──────────────────────────────
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %EXTRA_ARGS%
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% equ 0 (
    echo ================================================================
    echo   INSTALLATION COMPLETE
    echo ================================================================
    echo.
    echo To launch the app, double-click:  start.bat
    echo   or run it from a terminal.
    echo.
    echo The app will open in your browser at http://127.0.0.1:8000
    echo.
) else (
    echo ================================================================
    echo   INSTALLATION FAILED  (exit code %EXIT_CODE%)
    echo ================================================================
    echo.
    echo Check the output above for error details.
    echo Common fixes:
    echo   - Make sure you have an internet connection
    echo   - Free up disk space (need ~15 GB)
    echo   - Run as Administrator if you see permission errors
    echo.
)

pause
exit /b %EXIT_CODE%
