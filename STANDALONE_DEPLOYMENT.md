# Standalone Station Deployment Guide

Deploy the Classification Web App to a machine with **no internet, no Python, no Node.js**.

There are three deployment methods. Pick whichever matches the target environment.

| # | Method | Distribution artifact | When to use |
|---|--------|------------------------|-------------|
| A | **Inno Setup `.exe` installer** | `ClassificationApp_Setup.exe` (single file) | Easiest hand-off — one self-extracting installer |
| B | **`Setup.bat` wizard** (offline_installer) | `offline_installer/` folder on USB | Network-isolated machines, custom install path, advanced options |
| C | **Manual copy** | Project folder + `.venv\` + HF cache | Dev-to-dev migration, no installer needed |

All three include both the **classification web app** and the **MEA Calibration Tool** as a
secondary launchable from the same install root.

---

## Method A — Inno Setup `.exe` Installer (recommended)

A single ~8 GB installer built from [ClassificationApp.iss](ClassificationApp.iss).

### On the dev machine

```bat
build_installer.bat
```

This invokes Inno Setup against `ClassificationApp.iss` and produces
`ClassificationApp_Setup.exe` at the project root. The installer embeds:

- The pre-built web app (`web_app/dist/`) and backend source.
- Embedded Python 3.11.9 + all wheels (no system Python required).
- The MEA Calibration Tool.
- All HuggingFace model weights (so AI extraction works offline out of the box).

### On the target machine

Double-click `ClassificationApp_Setup.exe`, accept the install path, optionally pick
GPU vs CPU package set, finish. Desktop shortcuts are created for:

- **Classification Web App** → `start.bat`
- **MEA Calibration Tool** → `mea_calibration_tool/launcher.py`

The browser opens automatically at `http://127.0.0.1:8000`.

---

## Method B — `Setup.bat` Wizard (offline_installer/)

A PowerShell-based GUI wizard (`Setup.ps1`) drives the install. Lower-level than Method A —
useful when you need to choose torch flavours separately or audit each package set.

### On the dev machine

```bat
prepare_offline.bat
```

Prompts for PyTorch variant (CPU / CUDA) and GPU packages, then assembles:

```
offline_installer/
  Setup.bat                        ← double-click this on the target
  Setup.ps1                        ← wizard
  prerequisites/
    python-3.11.9-embed-amd64.zip
    get-pip.py
  offline_packages/                ← core pip wheels + (optional) CPU torch
  offline_packages_torch/          ← CUDA torch override (if chosen)
  offline_packages_gpu/            ← CuPy / nvidia-cuda-* (if chosen)
  app/
    backend/                       backend source
    web_app/dist/                  built frontend
    shared/                        MEA classes + factory defaults
    mea_calibration_tool/          companion calibration app
    launcher.py                    boots backend + opens browser
    start.bat
    requirements.txt
    requirements-gpu.txt
    version.txt
  STANDALONE_DEPLOYMENT.md         ← copy of this guide
  README.txt
```

Copy the entire `offline_installer/` folder to a USB stick (16 GB+ recommended).

### On the target machine

1. Double-click **`Setup.bat`**.
2. The wizard asks for:
   - Install folder (e.g. `C:\ClassificationApp`).
   - Feature pack — *Full AI* (with HF models) or *Core only* (KMeans only).
   - PyTorch variant and GPU options.
3. Click **Install** — takes 10–30 minutes.
4. Shortcuts are placed on the Desktop and Start Menu.

### Copy AI model weights after install

If *Core only* was chosen, AI extraction features need model weights (~5 GB) that aren't
bundled. On the dev machine:

```
C:\Users\<name>\.cache\huggingface\hub\
```

Copy that `hub\` folder to the target machine at:

```
<install dir>\models\hf_cache\hub\
```

Expected sub-folders:

```
hub/
  models--google--owlv2-base-patch16-ensemble/   593 MB
  models--facebook--sam2-hiera-large/            857 MB
  models--facebook--sam3/                        3.3 GB
  models--ShilongLiu--GroundingDINO/             ~300 MB
```

The launcher exports `HF_HUB_OFFLINE=1` and `HF_HOME=<install>\models\hf_cache` so the
models resolve from disk only.

### Installed layout (Method B)

```
<install dir>/
  python/                     embedded Python 3.11.9 (self-contained)
    python.exe
    python311.zip             standard library
    Lib/site-packages/        all installed packages
  backend/                    FastAPI backend
  web_app/dist/               pre-built frontend
  mea_calibration_tool/       companion app + its launcher
  shared/                     MEA classes + factory defaults
  models/hf_cache/hub/        AI model weights (manual copy unless Full AI)
  start.bat                   launches the web app
  launcher.py                 used by start.bat
  version.txt
```

### Running the app

Use the Desktop / Start Menu shortcut, or:

```
<install dir>\start.bat
```

Browser opens automatically at `http://127.0.0.1:8000`.

---

## Method C — Manual Copy *(no installer)*

Copy the project folder + venv from the dev machine directly.

### What to copy

| Item | Source (dev) | Size | Required |
|------|--------------|------|----------|
| Project folder | `classification-master/` | ~200 MB | yes |
| Python venv | inside project: `.venv/` | **7.5 GB** | yes |
| HF model cache | `C:\Users\<name>\.cache\huggingface\hub\` | ~5.1 GB | for AI features |

Total: **~13 GB** — use USB 3.0 (16 GB+).

### Prepare on the dev machine

1. Build the frontend (once, or after UI changes):
   ```bat
   cd web_app
   npm run build
   ```
2. Copy model cache into the project:
   ```
   C:\Users\<name>\.cache\huggingface\hub\  →  <project>\models\hf_cache\hub\
   ```

### Copy to target machine

Place the entire project folder anywhere, e.g. `C:\ClassificationApp\`.
The `.venv\` must remain inside the project folder.

```
ClassificationApp/
  start.bat
  backend/
  web_app/dist/
  mea_calibration_tool/
  shared/
  .venv/                  ← 7.5 GB — Python + all packages
  models/hf_cache/hub/    ← 5.1 GB — AI model weights
```

### Run on the target machine

```
start.bat
```

No Python installation needed — `.venv\Scripts\python.exe` is used.

---

## MEA Calibration Tool

All three deployment methods include the calibration tool. After install, launch:

```
<install dir>\mea_calibration_tool\launcher.py
```

(or via the Start Menu shortcut "MEA Calibration Tool"). It writes the active profile to:

```
%ProgramData%\MaterialClassification\mea_calibration_profile.json
```

The main classification app reads it on every run via `backend/app/mea_profile.py`.
See [docs/MEA_CALIBRATION_TOOL.md](docs/MEA_CALIBRATION_TOOL.md).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Blank page in browser | Frontend not built — run `npm run build` in `web_app\` |
| Server won't start | `.venv\` (or `python\`) missing from the install root |
| AI extraction fails | Model weights not in `models\hf_cache\hub\` |
| GPU not used | Install NVIDIA drivers on target; app falls back to CPU (slower) |
| Very slow on large images | Enable **Tile Processing** in the *Performance* sidebar panel |
| `ModuleNotFoundError: uvicorn` (Method B) | Package install failed — re-run Setup, check installer log |
| MEA profile shows "Factory Default" | No user profile yet — run the MEA Calibration Tool first |
| Browser doesn't open automatically | Visit `http://127.0.0.1:8000` manually |

---

## Size reference

| Item | Size |
|------|------|
| `.venv/` (Method C) | 7.5 GB |
| `python/` embedded (Method B) | ~1.5 GB |
| `models/hf_cache/hub/` | ~5.1 GB |
| `web_app/dist/` | ~400 KB |
| `backend/` source | ~1.2 MB |
| **Total Method A (Inno EXE)** | **~8 GB installed** |
| **Total Method B (Setup.bat — Full AI)** | **~8 GB installed** |
| **Total Method C (manual copy)** | **~13 GB** |
