#!/usr/bin/env python3
"""
MEA Calibration Tool launcher.
Starts the FastAPI backend and opens the browser.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PORT = int(os.getenv("MEA_CAL_PORT", "8001"))
HOST = os.getenv("MEA_CAL_HOST", "127.0.0.1")

_ROOT = Path(__file__).parent


def _find_python() -> str:
    return sys.executable


def main() -> None:
    python = _find_python()
    backend_cmd = [
        python, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", HOST,
        "--port", str(PORT),
        "--log-level", "info",
    ]

    print(f"[MEA Cal] Starting backend on http://{HOST}:{PORT}")
    proc = subprocess.Popen(backend_cmd, cwd=str(_ROOT))

    # Wait for backend to be ready
    import urllib.request
    for _ in range(30):
        try:
            urllib.request.urlopen(f"http://{HOST}:{PORT}/profile", timeout=1)
            break
        except Exception:
            time.sleep(0.5)

    url = f"http://{HOST}:{PORT}"
    print(f"[MEA Cal] Opening {url}")
    webbrowser.open(url)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\n[MEA Cal] Stopped.")


if __name__ == "__main__":
    main()
