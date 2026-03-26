@echo off
cd /d "%~dp0"
title Building ClassificationApp_Setup.exe

echo.
echo ================================================================
echo   Build ClassificationApp_Setup.exe  (run on DEV machine)
echo ================================================================
echo.
echo  What this produces:
echo    ClassificationApp_Setup.exe  - single self-extracting EXE
echo    Double-click it on any machine to launch the installer GUI.
echo.

:: ── Prerequisites check ────────────────────────────────────────────────────
if not exist "offline_installer\" (
    echo ERROR: offline_installer\ not found.
    echo Run  prepare_offline.bat  first, then re-run this script.
    pause & exit /b 1
)
if not exist "offline_installer\offline_packages\" (
    echo ERROR: offline_installer\offline_packages\ not found.
    echo Run  prepare_offline.bat  first to download all packages.
    pause & exit /b 1
)

:: ── Compile the GUI installer EXE if not up to date ───────────────────────
echo [1/5] Compiling ClassificationInstaller.exe...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0compile_installer.ps1"
if not exist "ClassificationInstaller.exe" (
    echo ERROR: compile_installer.ps1 failed to produce ClassificationInstaller.exe
    pause & exit /b 1
)
echo    Done.

:: ── Build frontend if not present ─────────────────────────────────────────
echo.
echo [2/5] Checking frontend build...
if exist "web_app\dist\index.html" goto :frontend_ok
echo    Building frontend (Vite)...
cd web_app
call npm install --silent
call npm run build
cd ..
if not exist "web_app\dist\index.html" echo    WARNING: Vite build failed - frontend will not be bundled.
if exist "web_app\dist\index.html" echo    Frontend built OK.
goto :frontend_done
:frontend_ok
echo    web_app\dist already present - skipping.
:frontend_done

:: ── Stage bundle folder ────────────────────────────────────────────────────
echo.
echo [3/5] Staging bundle...
if exist "_build_bundle\" rd /s /q "_build_bundle\"
mkdir "_build_bundle"

:: Copy the GUI installer EXE
copy /b "ClassificationInstaller.exe" "_build_bundle\ClassificationInstaller.exe" >nul
if errorlevel 1 ( echo ERROR: Could not copy ClassificationInstaller.exe & pause & exit /b 1 )

:: Copy the offline_installer folder (packages + app files)
echo    Copying offline_installer...
xcopy "offline_installer\" "_build_bundle\offline_installer\" /E /I /Q /Y >nul
if errorlevel 1 ( echo ERROR: xcopy of offline_installer failed. & pause & exit /b 1 )

:: Refresh app files from current source (so latest main.py etc. are bundled)
echo    Refreshing app sources...
if not exist "_build_bundle\offline_installer\app\" mkdir "_build_bundle\offline_installer\app"
if exist "backend\"      xcopy "backend\"      "_build_bundle\offline_installer\app\backend\"  /E /I /Q /Y >nul
if exist "web_app\dist\" xcopy "web_app\dist\" "_build_bundle\offline_installer\app\web_app\dist\" /E /I /Q /Y >nul
if exist "shared\"       xcopy "shared\"       "_build_bundle\offline_installer\app\shared\"   /E /I /Q /Y >nul
if exist "models\"       xcopy "models\"       "_build_bundle\offline_installer\app\models\"   /E /I /Q /Y >nul
if exist "backend\requirements.txt"     copy "backend\requirements.txt"     "_build_bundle\offline_installer\app\" /Y >nul
if exist "backend\requirements-gpu.txt" copy "backend\requirements-gpu.txt" "_build_bundle\offline_installer\app\" /Y >nul
echo    Done.

:: ── Find 7-Zip ─────────────────────────────────────────────────────────────
echo.
echo [4/5] Locating 7-Zip...
set SZ=
7z.exe >nul 2>&1
if not errorlevel 1 ( set "SZ=7z.exe" & goto :sz_found )
7za.exe >nul 2>&1
if not errorlevel 1 ( set "SZ=7za.exe" & goto :sz_found )
if exist "%ProgramFiles%\7-Zip\7z.exe"  ( set "SZ=%ProgramFiles%\7-Zip\7z.exe" & goto :sz_found )
if exist "_build_tools\7zip\7z.exe"     ( set "SZ=_build_tools\7zip\7z.exe"    & goto :sz_found )
if exist "_build_tools\7zip\7za.exe"    ( set "SZ=_build_tools\7zip\7za.exe"   & goto :sz_found )

echo    7-Zip not found - downloading 7-Zip 23.01...
mkdir "_build_tools" >nul 2>&1
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('https://www.7-zip.org/a/7z2301-x64.exe','_build_tools\7z_installer.exe')"
if errorlevel 1 ( echo ERROR: 7-Zip download failed. & pause & exit /b 1 )
"_build_tools\7z_installer.exe" /S /D="%CD%\_build_tools\7zip"
timeout /t 4 /nobreak >nul
if exist "_build_tools\7zip\7z.exe"  ( set "SZ=_build_tools\7zip\7z.exe"  & goto :sz_found )
if exist "_build_tools\7zip\7za.exe" ( set "SZ=_build_tools\7zip\7za.exe" & goto :sz_found )
echo ERROR: 7-Zip setup failed.
pause & exit /b 1

:sz_found
echo    7-Zip: %SZ%

:: ── Locate SFX module ──────────────────────────────────────────────────────
set SFX=
for %%D in ("%SZ%") do set "_SZ_DIR=%%~dpD"
if exist "%_SZ_DIR%7zSD.sfx"     ( set "SFX=%_SZ_DIR%7zSD.sfx"   & goto :sfx_found )
if exist "%_SZ_DIR%7z.sfx"       ( set "SFX=%_SZ_DIR%7z.sfx"     & goto :sfx_found )
if exist "_build_tools\7zSD.sfx" ( set "SFX=_build_tools\7zSD.sfx" & goto :sfx_found )

echo    Downloading 7-Zip SFX module...
powershell -NoProfile -Command "(New-Object System.Net.WebClient).DownloadFile('https://www.7-zip.org/a/7z2301-extra.7z','_build_tools\7zextra.7z')"
"%SZ%" e "_build_tools\7zextra.7z" "7zSD.sfx" -o"_build_tools\" -y >nul
if exist "_build_tools\7zSD.sfx" ( set "SFX=_build_tools\7zSD.sfx" & goto :sfx_found )
echo ERROR: Could not obtain 7zSD.sfx.
pause & exit /b 1

:sfx_found
echo    SFX module: %SFX%

:: ── Write SFX config ───────────────────────────────────────────────────────
(
echo ;!@Install@!UTF-8!
echo Title="Classification Web App Setup"
echo BeginPrompt="Classification Web App Installer\nClick OK to launch the installation wizard."
echo RunProgram="ClassificationInstaller.exe"
echo GUIFlags="8"
echo ;!@InstallEnd@!
) > "_build_tools\sfx_config.txt"

:: ── Compress bundle ────────────────────────────────────────────────────────
echo.
echo [5/5] Compressing bundle (may take several minutes)...
if exist "_build_tools\payload.7z" del "_build_tools\payload.7z"

"%SZ%" a -t7z "_build_tools\payload.7z" "_build_bundle\*" ^
    -mx=5 -mmt=on -ms=on ^
    -xr!"*.pyc" -xr!"__pycache__" -xr!"*.pdb"
if errorlevel 1 ( echo ERROR: Compression failed. & pause & exit /b 1 )

:: ── Build final EXE ────────────────────────────────────────────────────────
if exist "ClassificationApp_Setup.exe" del "ClassificationApp_Setup.exe"
copy /b "%SFX%" + "_build_tools\sfx_config.txt" + "_build_tools\payload.7z" "ClassificationApp_Setup.exe" >nul
if errorlevel 1 ( echo ERROR: Failed to create setup exe. & pause & exit /b 1 )

:: ── Cleanup staging ────────────────────────────────────────────────────────
rd /s /q "_build_bundle\" >nul 2>&1

:: ── Summary ────────────────────────────────────────────────────────────────
echo.
echo ================================================================
echo   DONE!
echo ================================================================
echo.
for %%F in ("ClassificationApp_Setup.exe") do (
    set /a MB=%%~zF/1048576
    echo   Output: ClassificationApp_Setup.exe
    echo   Size:   %%~zF bytes
)
echo.
echo   Copy to USB and double-click on the target machine.
echo   The C# installer GUI launches automatically.
echo.
echo   NOTE: SAM3 model weights are NOT bundled.
echo   After install, copy:
echo     %%USERPROFILE%%\.cache\huggingface\hub\
echo   Into the install folder under:
echo     ^<install dir^>\models\hf_cache\hub\
echo.
pause
