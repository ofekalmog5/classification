"""
Web app launcher: starts the FastAPI backend and opens the browser.

In production (PyInstaller exe): serves static React frontend from embedded web_app_dist/
In development: serves from web_app/dist/

All API routes are mounted under /api so the React frontend's /api/* calls work
without a Vite proxy.
"""
import sys
import os
import re
import subprocess
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

# Persistent directory for GPU packages (survives between EXE runs)
_GPU_PKG_DIR = Path(os.environ.get('APPDATA', Path.home())) / 'ClassificationApp' / 'gpu_packages'


# ─── GPU / faiss-gpu auto-setup ───────────────────────────────────────────────

def _get_cuda_major() -> int | None:
    """Return host CUDA major version (e.g. 12) from nvidia-smi, or None."""
    try:
        r = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        m = re.search(r'CUDA Version:\s*(\d+)', r.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _has_nvidia_gpu() -> bool:
    """Return True if at least one NVIDIA GPU is present."""
    try:
        r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        return r.returncode == 0 and 'GPU' in r.stdout
    except Exception:
        return False


def _faiss_gpu_works() -> bool:
    """Return True if faiss-gpu is importable and can open GPU resources."""
    try:
        import faiss
        res = faiss.StandardGpuResources()
        del res
        return True
    except Exception:
        return False


def _install_faiss_gpu(pkg_name: str) -> bool:
    """Install pkg_name to _GPU_PKG_DIR using pip's Python API. Returns success."""
    try:
        from pip._internal.cli.main import main as pip_main
        code = pip_main([
            'install',
            '--target', str(_GPU_PKG_DIR),
            '--quiet',
            '--no-deps',        # faiss-gpu has no Python deps — much faster
            '--disable-pip-version-check',
            pkg_name,
        ])
        return code == 0
    except Exception as e:
        print(f"  [GPU setup] pip error: {e}")
        return False


def _ensure_faiss_gpu():
    """
    Auto-install the right faiss-gpu wheel when an NVIDIA GPU is present.

    Priority:
      faiss-gpu-cu12  (CUDA 12.x)
      faiss-gpu-cu11  (CUDA 11.x)
      faiss-gpu       (legacy fallback, CUDA 11.4 wheels)

    Packages are installed once to %APPDATA%\\ClassificationApp\\gpu_packages
    and reused on every subsequent launch (no internet needed after first run).
    """
    # Always prepend persistent pkg dir so previously installed faiss-gpu is found
    pkg_str = str(_GPU_PKG_DIR)
    if pkg_str not in sys.path:
        sys.path.insert(0, pkg_str)

    # Already working? Nothing to do.
    if _faiss_gpu_works():
        print("[GPU] faiss-gpu already installed and active")
        return

    # No NVIDIA GPU? Skip.
    if not _has_nvidia_gpu():
        print("[GPU] No NVIDIA GPU detected — using faiss-cpu")
        return

    cuda = _get_cuda_major()
    print(f"[GPU] NVIDIA GPU detected (CUDA {cuda}.x)" if cuda else "[GPU] NVIDIA GPU detected (CUDA version unknown)")

    # Pick candidate packages in priority order
    if cuda and cuda >= 12:
        candidates = ['faiss-gpu-cu12', 'faiss-gpu-cu11', 'faiss-gpu']
    else:
        candidates = ['faiss-gpu-cu11', 'faiss-gpu']

    _GPU_PKG_DIR.mkdir(parents=True, exist_ok=True)

    for pkg in candidates:
        print(f"[GPU] Installing {pkg} → {_GPU_PKG_DIR} ...")
        if _install_faiss_gpu(pkg):
            # Flush any stale faiss imports so the new install is found
            for mod in list(sys.modules):
                if 'faiss' in mod:
                    del sys.modules[mod]
            if _faiss_gpu_works():
                print(f"[GPU] {pkg} active ✓")
                return
            print(f"[GPU] {pkg} installed but GPU probe failed — trying next")
        else:
            print(f"[GPU] {pkg} install failed — trying next")

    print("[GPU] Could not activate faiss-gpu — falling back to faiss-cpu")


# ─── App creation ──────────────────────────────────────────────────────────────

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

    # ── GPU setup runs before the backend imports core.py ─────────────────────
    # This ensures _probe_acceleration() in core.py sees faiss-gpu if available.
    _ensure_faiss_gpu()

    print(f"\nStarting Classification Web App on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.\n")

    threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(create_app(), host=HOST, port=PORT, log_level="info")
