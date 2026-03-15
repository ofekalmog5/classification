# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for ClassificationWebApp.exe

Bundles:
  - FastAPI backend (backend/app/main.py + core.py)
  - Built React frontend (web_app/dist -> web_app_dist inside exe)
  - All geospatial libraries (rasterio, fiona, pyproj, shapely, geopandas)
  - uvicorn + aiofiles for serving

Build steps:
  1. Activate venv:      .venv\\Scripts\\Activate.ps1
  2. Install deps:       pip install -r backend/requirements.txt aiofiles pyinstaller
  3. Build frontend:     cd web_app && npm run build && cd ..
  4. Build exe:          pyinstaller WebApp.spec --noconfirm

Output: dist/ClassificationWebApp.exe
"""
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = []
hiddenimports = []

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
    binaries=[],
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
