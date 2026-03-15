"""
Web app launcher: starts the FastAPI backend and opens the browser.

In production (PyInstaller exe): serves static React frontend from embedded web_app_dist/
In development: serves from web_app/dist/

All API routes are mounted under /api so the React frontend's /api/* calls work
without a Vite proxy.
"""
import sys
import os
import threading
import webbrowser
import time
from pathlib import Path

# Force UTF-8 on Windows to avoid encoding errors in console output
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

HOST = "127.0.0.1"
PORT = 8000


def get_dist_path() -> Path:
    """Return path to built React frontend.

    When running as a PyInstaller frozen exe, files are extracted to sys._MEIPASS.
    In dev, they live at web_app/dist relative to this file.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS) / 'web_app_dist'
    return Path(__file__).parent / 'web_app' / 'dist'


def create_app():
    """Build the combined FastAPI app: API under /api + static frontend at /."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from backend.app.main import app as api_app

    wrapper = FastAPI(title="Classification Web App")

    # API routes are available at /api/* (Vite proxy stripped /api in dev,
    # here we re-add it so the production build works without a proxy)
    wrapper.mount("/api", api_app)

    dist = get_dist_path()
    if dist.exists():
        # html=True makes StaticFiles return index.html for unknown paths (SPA routing)
        wrapper.mount("/", StaticFiles(directory=str(dist), html=True), name="static")
    else:
        print(f"WARNING: Frontend dist not found at {dist}. Only API will be served.")

    return wrapper


def _open_browser():
    time.sleep(2.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    print(f"Starting Classification Web App on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.\n")

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
