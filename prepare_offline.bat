@echo off
cd /d "%~dp0"
title Preparing Offline Installer  —  Classification Web App

echo.
echo ================================================================
echo   Prepare Offline Installer  ^(run on the DEV machine only^)
echo ================================================================
echo.
echo This script downloads everything needed to install the app on a
echo machine with NO internet and NO Python / Node.js installed.
echo.
echo What will be downloaded:
echo   - Python 3.11.9 embeddable package
echo   - get-pip.py
echo   - All pip packages  (core + optional AI / GPU)
echo   - Pre-built frontend  (npm install + npm run build)
echo.
echo Output:  offline_installer\   ^(~5-15 GB^)
echo Copy that entire folder to USB and run  Setup.bat  on the target.
echo.
echo Press any key to start, or Ctrl+C to cancel...
pause >nul

:: ══════════════════════════════════════════════════════════════════
:: Find Python 3.11
:: ══════════════════════════════════════════════════════════════════
set PYTHON=
echo.
echo Locating Python 3.11...

:: 1) Try plain 'python' in PATH
python --version 2>nul | findstr "3.11" >nul
if not errorlevel 1 ( set "PYTHON=python" & goto :python_found )

:: 2) Try Python Launcher: py -3.11
py -3.11 --version >nul 2>&1
if not errorlevel 1 ( set "PYTHON=py -3.11" & goto :python_found )

:: 3) Try the path visible in this project's .venv config
if exist ".venv\pyvenv.cfg" (
    for /f "tokens=3" %%A in ('findstr /i "^home" .venv\pyvenv.cfg') do (
        if exist "%%A\python.exe" (
            "%%A\python.exe" --version 2>nul | findstr "3.11" >nul
            if not errorlevel 1 (
                set "PYTHON=%%A\python.exe"
                goto :python_found
            )
        )
    )
)

:: 4) Common install locations (checked individually to avoid for-loop paren issues)
set "_P1=%LocalAppData%\Python\pythoncore-3.11-64\python.exe"
set "_P2=%LocalAppData%\Programs\Python\Python311\python.exe"
set "_P3=C:\Python311\python.exe"
set "_P4=C:\Program Files\Python311\python.exe"

if exist "%_P1%" ( "%_P1%" --version 2>nul | findstr "3.11" >nul & if not errorlevel 1 ( set "PYTHON=%_P1%" & goto :python_found ) )
if exist "%_P2%" ( "%_P2%" --version 2>nul | findstr "3.11" >nul & if not errorlevel 1 ( set "PYTHON=%_P2%" & goto :python_found ) )
if exist "%_P3%" ( "%_P3%" --version 2>nul | findstr "3.11" >nul & if not errorlevel 1 ( set "PYTHON=%_P3%" & goto :python_found ) )
if exist "%_P4%" ( "%_P4%" --version 2>nul | findstr "3.11" >nul & if not errorlevel 1 ( set "PYTHON=%_P4%" & goto :python_found ) )

echo.
echo ERROR: Python 3.11 not found.
echo.
echo Tried:  python,  py -3.11,  .venv\pyvenv.cfg path,  common locations.
echo.
echo Solutions:
echo   A) Add Python 3.11 to PATH and re-run
echo   B) Activate your venv first:  .venv\Scripts\activate  then re-run
echo   C) Edit this file and set PYTHON= to your python.exe path manually
echo.
pause & exit /b 1

:python_found
echo    Found Python 3.11: %PYTHON%
%PYTHON% --version

:: ══════════════════════════════════════════════════════════════════
:: Find Node.js (optional if dist/ already built)
:: ══════════════════════════════════════════════════════════════════
set NEED_NODE=1
if exist "web_app\dist\index.html" (
    echo.
    echo    Frontend already built ^(web_app\dist\index.html exists^).
    echo    Skip npm build? [Y/n]:
    set /p SKIP_NPM="   > "
    if /i not "%SKIP_NPM%"=="n" set NEED_NODE=0
)

if "%NEED_NODE%"=="1" (
    node --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo ERROR: Node.js not found in PATH and frontend is not pre-built.
        echo Install Node.js ^(https://nodejs.org^) or run  npm run build  in web_app\
        echo manually first, then re-run this script.
        pause & exit /b 1
    )
    echo    Found: Node && node --version
)

echo.

:: ══════════════════════════════════════════════════════════════════
:: Create folder structure
:: ══════════════════════════════════════════════════════════════════
echo Cleaning previous output...
if exist "offline_installer" rmdir /s /q "offline_installer"

echo Creating folders...
mkdir "offline_installer\prerequisites"
mkdir "offline_installer\offline_packages"
mkdir "offline_installer\offline_packages_torch"
mkdir "offline_installer\offline_packages_gpu"
mkdir "offline_installer\app"

copy /y "Setup.bat"  "offline_installer\Setup.bat"  >nul
copy /y "Setup.ps1"  "offline_installer\Setup.ps1"  >nul

:: ── [1 / 7]  Copy Python installation from this machine ────────────
echo.
echo [1/7] Copying Python 3.11 installation from this machine...

:: Get the Python base directory (works with both 'python' path and 'py -3.11')
for /f "usebackq delims=" %%D in (`%PYTHON% -c "import sys; print(sys.base_prefix)"`) do set "PYTHON_DIR=%%D"
echo    Source: %PYTHON_DIR%

if not exist "%PYTHON_DIR%\python.exe" (
    echo FAILED: python.exe not found at %PYTHON_DIR%
    pause & exit /b 1
)

mkdir "offline_installer\prerequisites\python311" >nul 2>&1
xcopy /e /i /y /q "%PYTHON_DIR%" "offline_installer\prerequisites\python311\"
if errorlevel 1 ( echo FAILED: xcopy returned error. & pause & exit /b 1 )
echo    OK.

:: [2/7] — get-pip.py is no longer needed (pip is included in the full Python install)
echo.
echo [2/7] Skipping get-pip.py download — pip is already included in Python installation.

:: ── [3 / 7]  Build frontend ───────────────────────────────────────
echo.
if "%NEED_NODE%"=="1" (
    echo [3/7] Building frontend ^(npm install + npm run build^)...
    cd web_app
    call npm install
    if errorlevel 1 ( echo FAILED: npm install. & cd .. & pause & exit /b 1 )
    call npm run build
    if errorlevel 1 ( echo FAILED: npm run build. & cd .. & pause & exit /b 1 )
    cd ..
    echo    OK.
) else (
    echo [3/7] Skipping frontend build ^(dist already present^).
)

:: ── [4 / 7]  Core pip packages (no torch, no SAM) ─────────────────
echo.
echo [4/7] Downloading core Python packages (no torch, no SAM)...

:: Filter requirements.txt: drop comment-only lines, blank lines, SAM packages
powershell -NoProfile -Command ^
 "$lines = Get-Content 'backend\requirements.txt';" ^
 "$f = $lines | Where-Object { $_ -notmatch '^\s*#' -and $_ -notmatch 'segment.geospatial' -and $_ -notmatch 'triton.windows' -and ($_.Trim() -ne '') };" ^
 "$f | Set-Content 'offline_installer\_core_req_tmp.txt'"

%PYTHON% -m pip download ^
    -r "offline_installer\_core_req_tmp.txt" ^
    -d "offline_installer\offline_packages" ^
    --prefer-binary
if errorlevel 1 echo WARNING: Some core packages failed. Installer may still work.

:: Also grab pip/setuptools/wheel so we can bootstrap on the target
%PYTHON% -m pip download pip setuptools wheel ^
    -d "offline_installer\offline_packages" ^
    --prefer-binary

del "offline_installer\_core_req_tmp.txt" >nul 2>&1
echo    Core packages done.

:: ── [5 / 7]  SAM / AI packages (no torch) ─────────────────────────
echo.
echo [5/7] Downloading AI / SAM packages...
%PYTHON% -m pip download ^
    "segment-geospatial[samgeo3]" ^
    triton-windows ^
    opencv-python-headless ^
    --exclude torch ^
    --exclude torchvision ^
    -d "offline_installer\offline_packages" ^
    --prefer-binary
if errorlevel 1 echo WARNING: Some AI packages failed. SAM features may be unavailable.
echo    SAM packages done.

:: ── [6 / 7]  PyTorch ──────────────────────────────────────────────
echo.
echo [6/7] PyTorch download — choose a variant:
echo.
echo    [1]  CPU only   (~200 MB,  works on any machine)
echo    [2]  CUDA 12.1  (~2.5 GB, needs NVIDIA RTX 20xx/30xx/40xx)
echo    [3]  Both       (CPU in offline_packages, CUDA in offline_packages_torch)
echo    [4]  Skip       (AI extraction will not work)
echo.
set /p TORCH_CHOICE="   Your choice [1/2/3/4]: "

if "%TORCH_CHOICE%"=="1" goto :torch_cpu
if "%TORCH_CHOICE%"=="2" goto :torch_cuda
if "%TORCH_CHOICE%"=="3" goto :torch_both
if "%TORCH_CHOICE%"=="4" goto :torch_skip
echo    Invalid — defaulting to CPU.

:torch_cpu
echo    Downloading PyTorch CPU...
%PYTHON% -m pip download torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu ^
    -d "offline_installer\offline_packages" ^
    --prefer-binary
echo    PyTorch CPU done.
goto :torch_done

:torch_cuda
echo    Downloading PyTorch CUDA 12.1 (~2.5 GB)...
%PYTHON% -m pip download "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    -d "offline_installer\offline_packages_torch" ^
    --prefer-binary
echo    PyTorch CUDA done.
goto :torch_done

:torch_both
echo    Downloading PyTorch CPU...
%PYTHON% -m pip download torch torchvision ^
    --index-url https://download.pytorch.org/whl/cpu ^
    -d "offline_installer\offline_packages" ^
    --prefer-binary
echo    Downloading PyTorch CUDA 12.1 (~2.5 GB)...
%PYTHON% -m pip download "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" ^
    --index-url https://download.pytorch.org/whl/cu121 ^
    -d "offline_installer\offline_packages_torch" ^
    --prefer-binary
echo    PyTorch CPU + CUDA done.
goto :torch_done

:torch_skip
echo    Skipping PyTorch.

:torch_done

:: ── [7 / 7]  GPU KMeans packages (cupy / nvidia-cuda-*) ──────────
echo.
echo [7/7] GPU KMeans acceleration packages?
echo    (Optional — cupy + nvidia-cuda-*  for GPU KMeans on NVIDIA machines)
echo.
set /p GPU_CHOICE="   Download GPU packages? [y/N]: "
if /i "%GPU_CHOICE%"=="y" (
    echo    Downloading GPU packages...
    %PYTHON% -m pip download ^
        -r "backend\requirements-gpu.txt" ^
        -d "offline_installer\offline_packages_gpu" ^
        --prefer-binary
    echo    GPU packages done.
) else (
    echo    Skipping GPU packages.
)

:: ── Copy HuggingFace model weights ────────────────────────────────
echo.
echo Copying HuggingFace model weights (~5 GB)...
set "HF_SRC=%USERPROFILE%\.cache\huggingface\hub"
if not exist "%HF_SRC%" (
    echo    WARNING: %HF_SRC% not found — skipping model weights.
    echo    AI extraction features will not work until you copy them manually.
) else (
    echo    Source: %HF_SRC%
    echo    Destination: offline_installer\app\models\hf_cache\hub\
    mkdir "offline_installer\app\models\hf_cache\hub" >nul 2>&1
    xcopy /e /i /y /q "%HF_SRC%" "offline_installer\app\models\hf_cache\hub\"
    echo    Model weights copied.
)

:: ── Copy app files ─────────────────────────────────────────────────
echo.
echo Copying application source files...
xcopy /e /i /y /q "backend"       "offline_installer\app\backend\"      >nul
xcopy /e /i /y /q "web_app\dist"  "offline_installer\app\web_app\dist\" >nul
xcopy /e /i /y /q "web_app\src"   "offline_installer\app\web_app\src\"  >nul
xcopy /e /i /y /q "shared"        "offline_installer\app\shared\"       >nul

for %%F in (launcher.py start.bat) do (
    if exist "%%F" copy /y "%%F" "offline_installer\app\%%F" >nul
)
copy /y "backend\requirements.txt"     "offline_installer\app\requirements.txt"     >nul
copy /y "backend\requirements-gpu.txt" "offline_installer\app\requirements-gpu.txt" >nul
copy /y "STANDALONE_DEPLOYMENT.md"     "offline_installer\STANDALONE_DEPLOYMENT.md" >nul 2>&1

echo 1.0.0 > "offline_installer\app\version.txt"

(
echo Classification Web App — Offline Installer Package
echo ==================================================
echo.
echo TARGET MACHINE:
echo   Double-click  Setup.bat  to launch the installer wizard.
echo   No internet needed. No Python or Node.js needed.
echo.
echo   AI model weights are bundled automatically from:
echo     %USERPROFILE%\.cache\huggingface\hub\
echo.
echo DEV MACHINE content:
echo   prerequisites\          Python 3.11.9 embeddable + get-pip.py
echo   offline_packages\       core pip wheels  (+ CPU torch if chosen)
echo   offline_packages_torch\ CUDA torch override  (if chosen)
echo   offline_packages_gpu\   CuPy + NVIDIA packages  (if chosen)
echo   app\                    pre-built application files
) > "offline_installer\README.txt"

:: ── Done ───────────────────────────────────────────────────────────
echo.
echo ================================================================
echo   DONE!   offline_installer\  is ready.
echo ================================================================
echo.
echo Next steps:
echo   1. Copy the entire  offline_installer\  folder to a USB drive.
echo   2. On the target machine: double-click  Setup.bat
echo.
pause
