@echo off
cd /d "%~dp0"

echo ============================================
echo Building ClassificationWebApp.exe
echo ============================================
echo.

REM Activate virtual environment
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
    echo Using venv Python: %PYTHON%
) else (
    set PYTHON=python
    echo Using system Python
)

echo.
echo Cleaning previous build...
if exist "build" rmdir /s /q build
if exist "dist\ClassificationWebApp" rmdir /s /q dist\ClassificationWebApp

echo.
echo Running PyInstaller...
"%PYTHON%" -m PyInstaller WebApp.spec --clean

echo.
if exist "dist\ClassificationWebApp.exe" (
    echo ============================================
    echo BUILD SUCCESSFUL
    echo ============================================
    echo Executable: dist\ClassificationWebApp.exe
    echo.
    echo To run: dist\ClassificationWebApp.exe
) else (
    echo ============================================
    echo BUILD FAILED
    echo ============================================
    echo Check the output above for errors
)

echo.
pause
