#!/usr/bin/env python3
"""Launcher script to start the tkinter app without terminal encoding issues"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).parent / "tkinter_app.py"
    
    # Launch the tkinter app
    print(f"Launching {app_path}...")
    subprocess.Popen([sys.executable, str(app_path)])
