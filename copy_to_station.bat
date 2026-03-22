@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Classification App - Copy to Standalone Station
echo ============================================================
echo.

REM ── Get destination path from argument or prompt ─────────────────────────
if "%~1"=="" (
    set /p DEST="Enter destination path (e.g. E:\ClassificationApp): "
) else (
    set DEST=%~1
)

if "%DEST%"=="" (
    echo ERROR: No destination path provided.
    pause
    exit /b 1
)

echo.
echo Destination: %DEST%
echo.

REM ── Confirm ──────────────────────────────────────────────────────────────
set /p CONFIRM="This will copy ~8 GB. Continue? (Y/N): "
if /i not "%CONFIRM%"=="Y" (
    echo Cancelled.
    pause
    exit /b 0
)

echo.

REM ── Create destination folder ─────────────────────────────────────────────
if not exist "%DEST%" (
    echo Creating %DEST%...
    mkdir "%DEST%"
    if errorlevel 1 (
        echo ERROR: Could not create destination folder.
        pause
        exit /b 1
    )
)

REM ── Helper: print section header ─────────────────────────────────────────
set STEP=0

REM ================================================================
REM  1. Build frontend if dist\ doesn't exist
REM ================================================================
set /a STEP+=1
echo [%STEP%] Checking frontend build...
if not exist "web_app\dist\index.html" (
    echo     web_app\dist\ not found - building now...
    cd web_app
    call npm run build
    if errorlevel 1 (
        echo ERROR: Frontend build failed. Fix errors above and retry.
        cd ..
        pause
        exit /b 1
    )
    cd ..
    echo     Frontend built OK.
) else (
    echo     web_app\dist\ already exists - skipping build.
)

REM ================================================================
REM  2. Copy backend source (skip __pycache__)
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying backend source...
robocopy "backend" "%DEST%\backend" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
echo     Done.

REM ================================================================
REM  3. Copy web_app\dist (compiled frontend only - not node_modules)
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying compiled frontend (web_app\dist)...
robocopy "web_app\dist" "%DEST%\web_app\dist" /MIR /NFL /NDL /NJH /NJS /nc /ns /np
echo     Done.

REM ================================================================
REM  3b. Copy shared/ contracts
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying shared contracts...
if exist "shared" (
    robocopy "shared" "%DEST%\shared" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
    echo     Done.
) else (
    echo     No shared/ folder - skipping.
)

REM ================================================================
REM  4. Copy .venv (skip __pycache__)
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying Python venv (~1.5 GB, this may take a while)...
robocopy ".venv" "%DEST%\.venv" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
echo     Done.

REM ================================================================
REM  5. Copy HuggingFace model cache
REM    Source: %USERPROFILE%\.cache\huggingface\hub
REM    Dest:   <station>\models\hf_cache\hub
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying HuggingFace model cache (~5 GB, this will take a while)...
set HF_SRC=%USERPROFILE%\.cache\huggingface\hub
if not exist "%HF_SRC%" (
    echo     WARNING: HuggingFace cache not found at %HF_SRC%
    echo     Skipping model cache copy. SAM feature extraction will NOT work offline.
) else (
    robocopy "%HF_SRC%" "%DEST%\models\hf_cache\hub" /MIR /NFL /NDL /NJH /NJS /nc /ns /np
    echo     Done.
)

REM ================================================================
REM  6. Copy SAM3 local repo (if it exists next to project)
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Checking for SAM3 local repo...
set SAM3_SRC=%~dp0..\sam3-main
if exist "%SAM3_SRC%\sam3" (
    echo     Found sam3-main - copying...
    robocopy "%SAM3_SRC%" "%DEST%\sam3-main" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
    echo     Done.
) else (
    echo     sam3-main not found next to project - skipping.
    echo     (OWLv2+SAM2 will be used instead, which works fine.)
)

REM ================================================================
REM  7. Copy launcher files and root scripts
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Copying launchers and config files...
for %%F in (
    start.bat
    start_webapp.bat
    launcher.py
    server_launcher.py
    setup_and_validate.py
    app_config.json
    copy_to_station.bat
    build_exe.bat
    STANDALONE_DEPLOYMENT.md
) do (
    if exist "%%F" (
        copy /Y "%%F" "%DEST%\%%F" >nul
        echo     Copied %%F
    )
)

REM Copy exe if it was already built
if exist "ClassificationWebApp.exe" (
    copy /Y "ClassificationWebApp.exe" "%DEST%\ClassificationWebApp.exe" >nul
    echo     Copied ClassificationWebApp.exe
)
REM Copy .spec files for PyInstaller builds on station
for %%S in (*.spec) do (
    copy /Y "%%S" "%DEST%\%%S" >nul
    echo     Copied %%S
)
REM Copy pyinstaller hooks if present
if exist "pyinstaller_hooks" (
    robocopy "pyinstaller_hooks" "%DEST%\pyinstaller_hooks" /MIR /XD __pycache__ /NFL /NDL /NJH /NJS /nc /ns /np
    echo     Copied pyinstaller_hooks\
)

REM ================================================================
REM  8. Copy Resources folder (raster data) - optional
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Resources folder...
if exist "Resources" (
    set /p COPY_RES="     Copy Resources folder (may be large)? (Y/N): "
    if /i "!COPY_RES!"=="Y" (
        robocopy "Resources" "%DEST%\Resources" /MIR /NFL /NDL /NJH /NJS /nc /ns /np
        echo     Done.
    ) else (
        echo     Skipped.
    )
) else (
    echo     No Resources folder found - skipping.
)

REM ================================================================
REM  9. Write app_config.json at project root (where config.py reads it)
REM ================================================================
set /a STEP+=1
echo.
echo [%STEP%] Writing app config for offline model path...
(
echo {
echo   "sam3_local_dir": "sam3-main",
echo   "hf_cache_dir": "models/hf_cache",
echo   "offline_mode": true
echo }
) > "%DEST%\app_config.json"
echo     Written app_config.json (project root)

REM ================================================================
REM  Done
REM ================================================================
echo.
echo ============================================================
echo  COPY COMPLETE
echo ============================================================
echo.
echo  Destination : %DEST%
echo.
echo  To run on the station:
echo    - Double-click  start.bat
echo    - OR run        ClassificationWebApp.exe  (if built)
echo    - Browser opens to http://127.0.0.1:8000 automatically
echo.
echo  See STANDALONE_DEPLOYMENT.md for full instructions.
echo ============================================================
echo.
pause
