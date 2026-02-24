@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
	.venv\Scripts\python.exe tkinter_app.py
) else (
	python tkinter_app.py
)
pause
