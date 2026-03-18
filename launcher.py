"""
Standalone launcher for ClassificationWebApp.
- Finds project root (folder containing .venv and backend/)
- Starts the FastAPI backend
- Opens the browser automatically
"""
import os
import sys
import time
import threading
import webbrowser
import subprocess
from pathlib import Path

PORT = 8000
URL = f"http://127.0.0.1:{PORT}"


def _find_project_root() -> Path:
    """Walk upward from the exe/script to find the project root.
    Project root is the directory that contains both 'backend' and '.venv'.
    """
    # When frozen by PyInstaller, start from the exe's directory
    start = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

    for candidate in [start, start.parent, start.parent.parent]:
        if (candidate / "backend").is_dir() and (candidate / ".venv").is_dir():
            return candidate

    # Fallback: use exe/script directory and hope for the best
    return start


def _open_browser():
    time.sleep(3.0)
    webbrowser.open(URL)


def main():
    project_root = _find_project_root()
    python = str(project_root / ".venv" / "Scripts" / "python.exe")

    if not Path(python).exists():
        python = sys.executable  # last resort

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HOME", str(project_root / "models" / "hf_cache"))
    env["PYTHONPATH"] = str(project_root)

    print(f"Project root : {project_root}")
    print(f"Python       : {python}")
    print(f"Opening      : {URL}")

    threading.Thread(target=_open_browser, daemon=True).start()

    cmd = [
        python, "-m", "uvicorn",
        "backend.app.main:app",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--workers", "1",
    ]

    proc = subprocess.run(cmd, cwd=str(project_root), env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
