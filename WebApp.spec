# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ClassificationWebApp.exe

Bundles:
  - FastAPI backend (backend/app/main.py + core.py)
  - Built React frontend (web_app/dist -> web_app_dist inside exe)
  - All geospatial libraries (rasterio, fiona, pyproj, shapely, geopandas)
  - faiss-cpu (always bundled — 3-8x faster KMeans than sklearn)
  - faiss-gpu used automatically at runtime if NVIDIA CUDA is installed on the
    host machine (NOT bundled — requires system CUDA + pip install faiss-gpu)
  - uvicorn + aiofiles for serving

Build steps:
  1. Activate venv:      .venv\\Scripts\\Activate.ps1
  2. Install deps:       pip install -r backend/requirements.txt aiofiles pyinstaller
  3. Build frontend:     cd web_app && npm run build && cd ..
  4. Build exe:          pyinstaller WebApp.spec --noconfirm

Output: dist/ClassificationWebApp.exe

GPU note:
  The EXE always ships faiss-cpu for fast CPU KMeans.
  For GPU acceleration, the end-user installs CUDA + faiss-gpu separately:
    pip install faiss-gpu
  The app detects this automatically — no rebuild needed.
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

# ── faiss (CPU — always bundled) ───────────────────────────────────────────
# faiss-cpu ships its native .pyd/.so alongside the Python package.
# collect_data_files picks up the shared libraries; collect_submodules ensures
# all submodules are importable inside the frozen exe.
try:
    hiddenimports += collect_submodules('faiss')
    datas += collect_data_files('faiss')
except Exception:
    pass  # faiss not installed — app falls back to sklearn at runtime

# ── pynvml (GPU detection, optional) ──────────────────────────────────────
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
