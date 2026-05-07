# Image Material Classification

Unsupervised classification of orthophoto / multispectral imagery into MEA (Multi-Element Aperture) material classes,
with optional AI-driven feature extraction (roads, buildings, trees, fields, water) and a sidecar
material table for downstream simulation tools.

The project ships **three runnable applications**:

| App | Path | Purpose |
|-----|------|---------|
| **Web app** (primary) | `backend/` + `web_app/` | FastAPI + React UI — main pipeline, AI extraction, batch mode |
| **MEA Calibration Tool** | `mea_calibration_tool/` | Stand-alone FastAPI + React tool for sampling material reference colors and writing a calibration profile that the main app consumes |
| **Tkinter app** (legacy) | `tkinter_app.py` | Desktop GUI — kept for offline / quick-test workflows; not actively maintained |

For an operational reference (commands, troubleshooting), see [RUNNING_GUIDE.md](RUNNING_GUIDE.md).
For deployment to a no-internet station, see [STANDALONE_DEPLOYMENT.md](STANDALONE_DEPLOYMENT.md).

---

## Pipeline Overview

The classification pipeline is split into **two independent steps** so the (slow) clustering does
not need to be re-run when only vector overlays change:

```
                 ┌──────────────────────┐    ┌─────────────────────────────────┐
   raster ──►   │ Step 1               │ ── │ Step 2                          │── classified.tif
   (vrt/tif)    │ classify_and_export  │    │ rasterize_vectors_onto_class…   │   (+ companion XML)
                │ KMeans + RGB export  │    │ Vector overlays                 │
                └──────────────────────┘    └─────────────────────────────────┘
                          │                                │
                  POST /classify-step1               POST /classify-step2
                                  POST /classify   (= step1 + step2)
```

Both steps live in [backend/app/core.py](backend/app/core.py) (~6.4 kLoC); the wrapper `classify()`
calls them sequentially and stays backward-compatible.

### MEA mode

When the chosen class set matches the 13 MEA materials (see `MEA_CLASSES` in `core.py` and
[shared/mea_classes.json](shared/mea_classes.json)), the pipeline:

- Picks per-class colors from the calibration profile (or factory defaults).
- Enforces material diversity (one cluster cannot dominate every class).
- Writes a companion `<output>.xml` `<Composite_Material_Table>` with `_MEA_COMPOSITE_NAMES`
  composite names and ARGB colors.
- Optionally writes `.txr` / `.txs` sidecars for the simulator.
- Pads the raster on disk to power-of-2 dimensions and reprojects to EPSG:4326.

Material reference colors are read from
`%ProgramData%\MaterialClassification\mea_calibration_profile.json` if present,
falling back to [shared/mea_defaults.json](shared/mea_defaults.json). See
[docs/MEA_CALIBRATION_TOOL.md](docs/MEA_CALIBRATION_TOOL.md).

### MEA classes (6)

`BM_ASPHALT`, `BM_CONCRETE`, `BM_VEGETATION`, `BM_WATER`, `BM_SAND`, `BM_SOIL`.

The pipeline is SAM3-first: water comes from a shapefile (if provided),
roads/buildings come from SAM3 (or shapefiles), and KMeans only handles the
three "earth" classes — vegetation, sand, soil. Each kmeans-source material
has a single anchor color (vegetation `[0,100,0]`, sand `[230,200,130]`,
soil `[85,55,30]`) for clean spectral separation.

---

## AI Feature Extraction

The web app can refine the classification with text-prompted segmentation. See
[docs/AI_FEATURE_EXTRACTION.md](docs/AI_FEATURE_EXTRACTION.md) for the full description.

| Feature | Backend prompts (per tile) | Default merge color |
|---------|----------------------------|---------------------|
| Roads | `road, highway, asphalt path` | `BM_ASPHALT` (`#2D2D30`) |
| Buildings | `building, house, roof, rooftop, structure` | `BM_CONCRETE` (`#B4B4B4`) |
| Trees | `tree, trees, forest, woodland, grove` | `BM_VEGETATION` (`#006400`) |
| Fields | `grass, lawn, field, meadow, pasture` + `crop, farmland, agriculture, cultivated field` | `BM_VEGETATION` (`#006400`) |
| Water | `water, lake, pond, reservoir, pool` + `river, stream, canal, waterway, channel` + `sea, ocean, fish pond, swimming pool` | `BM_WATER` (`#1C6BA0`) |

Each feature uses a two-strategy detector — OWLv2+SAM2/3 text segmentation **OR'd** with a
color+geometry CV detector — plus a per-feature linearity / RGB pre-filter so the model is
not invoked on terrain that obviously contains nothing of the target type.

Outputs are written to `_<feature_type>/` subfolders next to the input raster and merged onto
the classification with `merge_feature_masks_onto_classification()`.

---

## Repository Layout

```
classification-master/
├── backend/                  FastAPI service (KMeans + AI extraction)
│   ├── app/
│   │   ├── main.py             FastAPI endpoints
│   │   ├── core.py             KMeans pipeline, tiling, MEA, XML/TXR/TXS export
│   │   ├── road_extraction.py  OWLv2+SAM2/SAM3 + color/geometry detectors
│   │   ├── config.py           Persistent JSON app config
│   │   └── mea_profile.py      Reader for the calibration profile
│   ├── requirements.txt
│   └── requirements-gpu.txt    Adds CuPy + nvidia-cuda-* wheels
│
├── web_app/                  Vite + React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx
│   │   ├── api/client.ts        FastAPI client + SSE progress
│   │   ├── store/               Reducer-based state
│   │   ├── constants/mea.ts
│   │   └── components/
│   │       ├── MapView.tsx, LayerPanel.tsx, Layout.tsx, …
│   │       └── sidebar/         InputSection, MaterialsSection,
│   │                            FeaturesSection, VectorsSection,
│   │                            PerformanceSection, ClassificationSection,
│   │                            ActionsSection, SettingsSection,
│   │                            MeaProfileStatus
│   └── package.json
│
├── mea_calibration_tool/     Stand-alone calibration app
│   ├── backend/app/             FastAPI: profile / sampling / raster_io
│   ├── web_app/                 Companion React UI
│   └── launcher.py
│
├── shared/                   Cross-app constants
│   ├── mea_classes.json         13 MEA classes + composite names
│   ├── mea_defaults.json        Factory-default reference colors
│   └── contracts/               JSON-Schema request contracts
│
├── offline_installer/        Self-contained installer for offline stations
│   ├── Setup.bat / Setup.ps1    Installer wizard
│   ├── prerequisites/           Embedded Python + pip
│   ├── offline_packages*/       Bundled wheels (core / torch / GPU)
│   └── app/                     Pre-built application files
│
├── tkinter_app.py            Legacy desktop GUI
├── ClassificationApp.iss     Inno Setup installer script
├── *.spec                    PyInstaller specs (CLI / web app / app)
├── prepare_offline.bat       Builds the offline_installer/ payload
├── build_installer.bat       Builds ClassificationApp_Setup.exe (Inno)
├── start_webapp.bat          One-click launch (backend + Vite dev)
└── docs/                     Topic deep-dives (see below)
```

### Topic deep-dives

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — request/response flow, communities, god nodes
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — every FastAPI endpoint
- [docs/AI_FEATURE_EXTRACTION.md](docs/AI_FEATURE_EXTRACTION.md) — OWLv2 + SAM, model setup
- [docs/MEA_CALIBRATION_TOOL.md](docs/MEA_CALIBRATION_TOOL.md) — calibrating reference colors
- [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md) — CuPy / FAISS engine selection
- [docs/TILING.md](docs/TILING.md) — tile mode & `suggest_tile_size`
- [SHADOW_DETECTION_FEATURE.md](SHADOW_DETECTION_FEATURE.md) — shadow → adjacent-material inference
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) — historical: two-step split
- [FIXES_APPLIED.md](FIXES_APPLIED.md) — historical: PROJ / CRS fixes
- [RASTERIZE_DEBUG.md](RASTERIZE_DEBUG.md) — historical: rasterization debugging notes (Hebrew)

---

## Quick Start (web app, dev machine)

```powershell
# One-time setup
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd web_app && npm install && cd ..

# Run (two terminals, or use start_webapp.bat)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend     # backend  → 127.0.0.1:8000
cd web_app && npm run dev                                                         # frontend → 127.0.0.1:5174
```

Or simply:

```bat
start_webapp.bat
```

Browser opens automatically at the Vite URL; it talks to FastAPI at `http://127.0.0.1:8000`
through `/api/...` (rewritten by `_StripApiPrefix` middleware in `main.py`).

---

## Build / Distribution

| Goal | Script |
|------|--------|
| One-click web app `.exe` | `build_exe.bat` → `dist/ClassificationWebApp.exe` (PyInstaller, `WebApp.spec`) |
| Inno Setup installer | `build_installer.bat` → `ClassificationApp_Setup.exe` (`ClassificationApp.iss`) |
| Offline payload (USB) | `prepare_offline.bat` → `offline_installer/` |

See [STANDALONE_DEPLOYMENT.md](STANDALONE_DEPLOYMENT.md) for the offline-station workflow.

---

## Notes

- Output GeoTIFFs are saved next to the input raster with a `_classified.tif` suffix and are
  reprojected to **EPSG:4326** before writing.
- Every output gets a companion `<stem>.xml` material table; MEA outputs additionally get
  `.txr` / `.txs` sidecars.
- All AI extraction outputs go into organized `_roads/`, `_buildings/`, `_trees/`,
  `_fields/`, `_water/` subfolders next to the source.
- The web app can group multiple input images and cascade-delete their layers as a unit.
- Tile mode is recommended for any raster > ~5000×5000 pixels; the **Auto** tile-size choice
  queries `/suggest-tile-size` and picks a side that fits in current RAM.
- GPU KMeans is automatic — install `requirements-gpu.txt` (CuPy + nvidia-cuda-* wheels) and
  the engine flips from `faiss-cpu` to `cupy` on next start.
