# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ClassificationWebApp.exe

Bundles:
  - FastAPI backend (backend/app/main.py + core.py)
  - Built React frontend (web_app/dist -> web_app_dist inside exe)
  - All geospatial libraries (rasterio, fiona, pyproj, shapely, geopandas)
  - faiss-cpu (always bundled — 3-8x faster KMeans than sklearn, fully offline)
  - CuPy GPU KMeans (bundled automatically if cupy-cuda12x is installed in venv)
  - uvicorn + aiofiles for serving

Build steps:
  1. Activate venv:       .venv\\Scripts\\Activate.ps1
  2. Install base deps:   pip install -r backend/requirements.txt aiofiles pyinstaller pynvml
  3. GPU support          pip install -r backend/requirements-gpu.txt
     (RTX A4000/30xx/40xx — no CUDA Toolkit needed, ~700 MB nvidia-* DLL packages)
  4. Build frontend:      cd web_app && npm run build && cd ..
  5. Build exe:           pyinstaller WebApp.spec --noconfirm

Output: dist/ClassificationWebApp.exe

GPU note:
  CuPy (cupy-cuda12x + nvidia-*-cu12) gives GPU KMeans on Windows without conda.
  Works on RTX A4000, RTX 30xx/40xx (Ampere/Ada, compute cap >= 8.6, driver >= 520).
  The EXE auto-bundles CuPy if installed; falls back to faiss-cpu on CPU-only machines.
  For faiss-gpu (even faster, conda only):
    conda install -c pytorch faiss-gpu cudatoolkit=11.8
"""
import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
hiddenimports = []
binaries = []

# ── Geospatial library data files (GDAL drivers, PROJ data, etc.) ─────────
for pkg in ['rasterio', 'fiona', 'pyproj', 'shapely', 'geopandas']:
    datas += collect_data_files(pkg)

# ── Hidden imports for geo + ML packages (dynamic loaders) ────────────────
hiddenimports += collect_submodules('rasterio')
hiddenimports += collect_submodules('skimage')
hiddenimports += collect_submodules('sklearn')
hiddenimports += [
    'rasterio.sample',
    'rasterio._shim',
    'fiona._shim',
]

# ── faiss-cpu (always bundled — fast CPU KMeans, fully offline) ────────────
try:
    hiddenimports += collect_submodules('faiss')
    datas         += collect_data_files('faiss')
except Exception:
    pass  # faiss not installed — app falls back to sklearn at runtime

# ── CuPy GPU KMeans (bundled if installed: pip install cupy-cuda12x) ───────
# Provides real GPU KMeans on Windows without conda.
# If not installed the app uses faiss-cpu automatically.
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    hiddenimports += collect_submodules('cupy')
    datas         += collect_data_files('cupy')
    binaries      += collect_dynamic_libs('cupy')
    hiddenimports += [
        'cupy._core',
        'cupy.cuda',
        'cupy.cuda.memory',
        'cupy.random',
        'cupy.linalg',
    ]
    print("[spec] CuPy found — bundling GPU KMeans support")
except Exception as _e:
    print(f"[spec] CuPy not installed ({_e}) — GPU KMeans via CuPy skipped")

# ── pynvml (GPU detection) ─────────────────────────────────────────────────
hiddenimports += ['pynvml']

# ── uvicorn uses importlib to load its components dynamically ─────────────
hiddenimports += [
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.loops.asyncio',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'uvicorn.lifespan.off',
    'h11',
    'aiofiles',            # required by fastapi.staticfiles.StaticFiles
    'aiofiles.os',
    'aiofiles.threadpool',
]

# ── Built React frontend ───────────────────────────────────────────────────
# Source: web_app/dist  →  destination inside exe: web_app_dist
datas += [('web_app/dist', 'web_app_dist')]

# ──────────────────────────────────────────────────────────────────────────

a = Analysis(
    ['server_launcher.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ClassificationWebApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,               # Keep console so users see startup/error output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
