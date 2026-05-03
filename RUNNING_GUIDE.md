# Running the Project — Operational Guide

This file is the day-to-day "how do I start things up and what's broken" reference. For
architecture and feature description, read [README.md](README.md).

There are three runnable applications. Most users only need #1.

| # | App | When to use |
|---|-----|-------------|
| 1 | **Web app** (FastAPI + React) — primary | Anything other than calibration |
| 2 | **MEA Calibration Tool** (separate FastAPI + React) | Sampling material reference colors before a classification run |
| 3 | **Tkinter app** (legacy) | Quick local test, no Node, no browser |

---

## 1. Web App

### Prerequisites
- Python 3.11+ (3.11.9 is what the offline installer ships)
- Node.js 16+ and npm
- Optional: NVIDIA GPU with CUDA 11.x or 12.x drivers (CuPy)

### One-click

```bat
start_webapp.bat
```

Starts uvicorn (backend) and `npm run dev` (frontend) in two terminals and opens the browser.

### Manual

Terminal 1 — backend:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
# Optional: pip install -r backend\requirements-gpu.txt   (CuPy + CUDA wheels)
python -m uvicorn app.main:app --reload --app-dir backend
```

Backend is at `http://127.0.0.1:8000`.
- Swagger UI: `http://127.0.0.1:8000/docs`
- All endpoints are summarised in [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

Terminal 2 — frontend:

```powershell
cd web_app
npm install
npm run dev
```

Vite serves at `http://127.0.0.1:5173` (or whatever port it prints) and proxies `/api/*`
to the backend. The backend's `_StripApiPrefix` middleware strips the `/api` so the same
client also works against a packaged `dist/` build served from FastAPI directly.

### Production build of the frontend

```powershell
cd web_app
npm run build      # writes web_app/dist/
```

Once `dist/` exists, the FastAPI server mounts it as static files — open `http://127.0.0.1:8000`
directly, no Vite required.

---

## 2. MEA Calibration Tool

Independent app — uses its own FastAPI backend on a different port and writes the calibration
profile to `%ProgramData%\MaterialClassification\mea_calibration_profile.json`. The main app
picks it up automatically via [backend/app/mea_profile.py](backend/app/mea_profile.py).

```powershell
cd mea_calibration_tool
python -m uvicorn app.main:app --app-dir backend --port 8100
cd web_app && npm install && npm run dev
```

Or use [mea_calibration_tool/launcher.py](mea_calibration_tool/launcher.py) (and the
`launcher.py` shipped inside the offline installer package) which boots both processes.

See [docs/MEA_CALIBRATION_TOOL.md](docs/MEA_CALIBRATION_TOOL.md) for the workflow.

---

## 3. Tkinter App (legacy)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe tkinter_app.py
```

No backend / Node required.

---

## AI Feature Extraction Setup

`extract-features` (roads/buildings/trees/fields/water) and `extract-roads` need
HuggingFace model weights cached locally:

| Model | Purpose | Approx size |
|-------|---------|-------------|
| `google/owlv2-base-patch16-ensemble` | Open-vocabulary detection | 593 MB |
| `facebook/sam2-hiera-large` | SAM 2 mask refinement | 857 MB |
| `facebook/sam3` | SAM 3 (preferred when available) | 3.3 GB |
| `ShilongLiu/GroundingDINO` | LangSAM fallback | ~300 MB |

The first time the backend runs they download to `~\.cache\huggingface\hub\`. On offline
stations, copy the `hub\` folder to `<install dir>\models\hf_cache\hub\` — the launcher sets
`HF_HUB_OFFLINE=1` and `HF_HOME` so the models resolve from local snapshots.

Triton (used by SAM3) is mocked on Windows when `triton-windows` isn't available, so OWLv2+SAM2
remains usable as a fallback.

---

## GPU vs CPU

`backend/app/core.py` probes a fixed priority list: **CuPy → faiss-gpu → faiss-cpu → sklearn**.
The selected engine name shows up in the run log and on the status bar. Install
`backend/requirements-gpu.txt` to enable CuPy.

See [docs/GPU_ACCELERATION.md](docs/GPU_ACCELERATION.md) for installation specifics.

---

## Tile Processing

Big rasters (> ~5000×5000) blow up memory. Enable **Tile Processing** in the sidebar's
*Performance* panel.

- **Auto** queries `POST /suggest-tile-size` and picks a side length that fits comfortably
  in current available RAM.
- Sizes that wouldn't fit are hidden from the dropdown.
- Tile workers are limited by `tile_workers` in `PerformanceSection`; you can also cap the
  total via the **Limit max threads** option (`max_threads` parameter).

---

## Building Distributables

| Output | Script | Spec |
|--------|--------|------|
| `dist/ClassificationWebApp.exe` | `build_exe.bat` | `WebApp.spec` |
| `ClassificationApp_Setup.exe` (Inno Setup) | `build_installer.bat` | `ClassificationApp.iss` |
| `offline_installer/` payload (USB) | `prepare_offline.bat` | `offline_installer/Setup.ps1` |

The PyInstaller specs (`WebApp.spec`, `ClassificationApp.spec`,
`MaterialClassification_CLI.spec`, `ClassificationWebApp.spec`) include CuPy CUDA DLLs via
the runtime hook so the frozen EXE keeps GPU acceleration.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---------|-------------------|
| Blank page in browser | Frontend not built — `cd web_app && npm run build` |
| `ModuleNotFoundError: uvicorn` | `pip install -r backend/requirements.txt` |
| Backend port 8000 already in use | `netstat -ano \| findstr :8000` and kill the PID |
| Frontend can't reach `/api/...` | Confirm backend is on 127.0.0.1:8000; check `vite.config.ts` proxy |
| `ERROR 1: PROJ: proj_identify: Cannot find proj.db` | core.py runs `_setup_proj_lib()` — the venv must contain `pyproj`'s data; reinstall `pyproj` |
| GPU not used | `backend/requirements-gpu.txt` not installed, or NVIDIA drivers missing on the target |
| AI extraction fails | Model weights missing or `triton` import error — see *AI Feature Extraction Setup* above |
| `GeoSeries already has a CRS …` | Old tip — `set_crs(..., allow_override=True)` is now used everywhere |
| Vector overlay produces 0 pixels | CRS/transform mismatch — see [RASTERIZE_DEBUG.md](RASTERIZE_DEBUG.md) |
| `Set-ExecutionPolicy` blocks venv activation | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| `OutOfMemory` on large image | Enable Tile Processing; pick a smaller tile size from the dropdown |

---

## Stopping Services

- Backend / frontend / tkinter app: `Ctrl+C` in the terminal that owns it.
- `start_webapp.bat`: close both spawned terminals.
- Frozen EXE: close the launched browser tab and the console window.
