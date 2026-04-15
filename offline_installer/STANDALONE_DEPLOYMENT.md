# Standalone Station Deployment Guide

Deploy Classification Web App to a machine with **no internet, no Python, no Node.js**.

---

## Method A — Installer Wizard  *(recommended)*

A proper GUI installer (`Setup.bat`) guides you through the process.
It uses an **embedded Python** — no system Python installation needed.

### On the Dev Machine (has internet)

```bat
prepare_offline.bat
```

Prompts for PyTorch variant (CPU / CUDA) and GPU packages, then creates:

```
offline_installer\
  Setup.bat                    ← double-click this on the target
  Setup.ps1                    ← wizard code
  prerequisites\
    python-3.11.9-embed-amd64.zip
    get-pip.py
  offline_packages\            ← core pip wheels + (optional) CPU torch
  offline_packages_torch\      ← CUDA torch override  (if chosen)
  offline_packages_gpu\        ← CuPy / nvidia-cuda-* (if chosen)
  app\                         ← pre-built application files
  README.txt
```

Copy the entire `offline_installer\` folder to USB (16 GB+ recommended).

### On the Target Machine (no internet)

1. Double-click **`Setup.bat`**
2. The wizard asks for:
   - Install folder (e.g. `C:\ClassificationApp`)
   - Feature pack (Full AI / Core only)
   - PyTorch variant and GPU options
3. Click **Install** — takes 10–30 minutes
4. Shortcuts are created on Desktop and/or Start Menu

### Copy AI Model Weights After Install

AI extraction features need model weights (~5 GB) that are too large to bundle.

On the **dev machine**:
```
C:\Users\<name>\.cache\huggingface\hub\
```

Copy that `hub\` folder to the **target machine** at:
```
<install dir>\models\hf_cache\hub\
```

Expected content inside `hub\`:
```
hub\
  models--google--owlv2-base-patch16-ensemble\   (593 MB)
  models--facebook--sam2-hiera-large\            (857 MB)
  models--facebook--sam3\                        (3.3 GB)
  models--ShilongLiu--GroundingDINO\             (~300 MB)
```

### Installed Layout

```
<install dir>\
  python\              — embedded Python 3.11.9 (self-contained)
    python.exe
    python311.zip      — standard library
    Lib\site-packages\ — all installed packages
  backend\             — FastAPI backend
  web_app\dist\        — pre-built frontend
  shared\
  models\hf_cache\hub\ — AI model weights (copy manually, see above)
  start.bat            — launch script  (uses python\python.exe)
```

### Running the App

Use the Desktop/Start Menu shortcut, or:
```
<install dir>\start.bat
```

Browser opens automatically at `http://127.0.0.1:8000`.

---

## Method B — Manual Copy  *(no installer needed)*

Copy the project folder + venv from the dev machine directly.

### What to Copy

| Item | Source (dev machine) | Size | Required |
|------|---------------------|------|----------|
| Project folder | `classification-master\` | ~200 MB | Yes |
| Python venv | Inside project: `.venv\` | **7.5 GB** | Yes |
| HF model cache | `C:\Users\<name>\.cache\huggingface\hub\` | ~5.1 GB | For AI features |

Total: **~13 GB** — use USB 3.0 (16 GB+).

### Prepare on the Dev Machine

1. **Build the frontend** (once, or after UI changes):
   ```bat
   cd web_app
   npm run build
   ```
2. **Copy model cache** into the project:
   ```
   C:\Users\<name>\.cache\huggingface\hub\  →  <project>\models\hf_cache\hub\
   ```

### Copy to Target Machine

Place the entire project folder anywhere, e.g. `C:\ClassificationApp\`.
The `.venv\` must remain inside the project folder.

```
ClassificationApp\
  start.bat
  backend\
  web_app\dist\
  .venv\                ← 7.5 GB — Python + all packages
  models\hf_cache\hub\  ← 5.1 GB — AI model weights
```

### Run on the Target Machine

```
start.bat
```

No Python installation needed — `.venv\Scripts\python.exe` is used.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Blank page in browser | Frontend not built — run `npm run build` in `web_app\` |
| Server won't start | `.venv\` missing or not inside the project folder |
| AI extraction fails | Model weights not in `models\hf_cache\hub\` |
| GPU not used | Install NVIDIA drivers on target; app works on CPU (slower) |
| Very slow on large images | Enable **Tile Processing** in the app UI |
| `ModuleNotFoundError: uvicorn` (Method A) | Package install failed — re-run Setup, check installer log |

---

## Size Reference

| Item | Size |
|------|------|
| `.venv\` (Method B) | 7.5 GB |
| `python\` embedded (Method A) | ~1.5 GB |
| `models\hf_cache\hub\` | ~5.1 GB |
| `web_app\dist\` | ~400 KB |
| `backend\` source | ~1.2 MB |
| **Total (Method A — full)** | **~8 GB install dir** |
| **Total (Method B)** | **~13 GB** |
