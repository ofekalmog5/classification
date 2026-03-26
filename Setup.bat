@echo off
cd /d "%~dp0"
title Classification Web App — Setup

:: Check if we're running from the offline_installer folder
:: (Setup.bat lives next to Setup.ps1, prerequisites/, offline_packages/, app/)
if not exist "%~dp0Setup.ps1" (
    echo ERROR: Setup.ps1 not found next to this file.
    echo Make sure you are running Setup.bat from inside the offline_installer folder.
    pause
    exit /b 1
)

if not exist "%~dp0prerequisites\python-3.11.9-embed-amd64.zip" (
    echo ERROR: prerequisites\python-3.11.9-embed-amd64.zip not found.
    echo The installer package appears to be incomplete.
    echo Run prepare_offline.bat on the dev machine first.
    pause
    exit /b 1
)

:: Require 64-bit Windows
if /i "%PROCESSOR_ARCHITECTURE%" == "x86" (
    if not defined PROCESSOR_ARCHITEW6432 (
        echo ERROR: This installer requires 64-bit Windows.
        pause
        exit /b 1
    )
)

:: Launch the PowerShell GUI wizard
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Setup.ps1"

exit /b 0
