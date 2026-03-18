# Standalone Station Deployment Guide

This guide explains exactly what to copy and how to set up the Classification Web App on a standalone (offline) station.

---

## What You Need to Copy

### Total size: ~13 GB — use a USB 3.0 drive with at least 16 GB free

| Item | Source path (dev machine) | Size | Required |
|------|--------------------------|------|----------|
| Project folder | `C:\...\classification-master\` | ~200 MB | ✅ Yes |
| Python venv | Inside project: `.venv\` | **7.5 GB** | ✅ Yes |
| HuggingFace model cache | `C:\Users\B\.cache\huggingface\hub\` | ~5.1 GB | ✅ Yes (for SAM road/building/vegetation extraction) |
| SAM3 local repo *(optional)* | `C:\Users\B\Desktop\ofek\sam3-main\` | ~50 MB | ⚠️ Only if you have the `sam3.pt` checkpoint |

> **Note:** If you skip the SAM3 repo, the system falls back to OWLv2+SAM2 (already in the HF cache) which works fine.

---

## Step-by-Step Instructions

### On the Development Machine (before copying)

#### 1. Build the frontend
Open a terminal in the project folder and run:
```bat
cd web_app
npm run build
```
This creates `web_app\dist\` (~5 MB). It only needs to be done once (or after UI changes).

#### 2. Build the launcher exe *(optional — or just use start.bat)*
```bat
build_exe.bat
```
This produces `ClassificationWebApp.exe` at the project root (~10 MB).
Alternatively, skip this — `start.bat` works without building.

#### 3. Copy the HuggingFace model cache
Copy the **entire** folder:
```
C:\Users\B\.cache\huggingface\hub\
```
into the project folder as:
```
<project root>\models\hf_cache\hub\
```

So the final structure inside the project is:
```
classification-master\
└── models\
    └── hf_cache\
        └── hub\
            ├── models--google--owlv2-base-patch16-ensemble\   (593 MB)
            ├── models--facebook--sam2-hiera-large\            (857 MB)
            ├── models--facebook--sam3\                        (3.3 GB)
            └── models--ShilongLiu--GroundingDINO\             (~300 MB)
```

---

### What to Copy to the Target Machine

Copy the **entire project folder** to any location on the target machine, for example:
```
C:\ClassificationApp\
```

The folder must contain:
```
ClassificationApp\
├── ClassificationWebApp.exe     ← launcher (if built)
├── start.bat                    ← alternative launcher (no build needed)
├── backend\                     ← Python API server
├── web_app\
│   └── dist\                    ← compiled frontend (must be built first)
├── .venv\                       ← Python + all packages (~1.5 GB)
├── models\
│   └── hf_cache\
│       └── hub\                 ← AI model weights (~5.1 GB)
└── Resources\                   ← user data / raster files
```

> **.venv must stay inside the project folder.** Do not move it separately.

---

### On the Target (Standalone) Machine

#### Prerequisites
- **Windows 10 or 11** (64-bit)
- **No internet required** after setup
- **GPU (NVIDIA):** Optional but recommended for faster classification. CUDA drivers must be installed if you want GPU support.
- **RAM:** Minimum 8 GB. 16 GB+ recommended for large images.
- **No Python installation needed** — the `.venv` folder contains everything.

#### To Run the App

**Option A — Double-click the exe:**
```
ClassificationWebApp.exe
```

**Option B — Double-click the bat file:**
```
start.bat
```

Both will:
1. Start the backend server at `http://127.0.0.1:8000`
2. Open the browser automatically after ~3 seconds

> If the browser doesn't open automatically, go to: **http://127.0.0.1:8000**

---

## SAM3 Local Path (Optional)

If you have the `sam3.pt` checkpoint and the `sam3-main` folder, copy it to the target machine and configure the path in the app:

1. Copy `sam3-main\` folder anywhere on the target machine
2. Open the app → expand **SAM backend** section → enter the path to `sam3-main\` → click **Set**

The path is saved and remembered between sessions.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Browser opens but page is blank | Frontend not built | Run `npm run build` in `web_app\`, then restart |
| Server doesn't start | `.venv` missing or wrong path | Make sure `.venv\` is inside the project folder |
| Road/building/vegetation extraction fails | HF model cache missing or wrong path | Check `models\hf_cache\hub\` exists and contains the model folders |
| GPU not used | CUDA drivers not installed | Install NVIDIA drivers; app works on CPU too (slower) |
| Classification very slow on large images | Image too large for RAM | Enable **Tile Processing** in the Performance section |
| `Permission denied` on output path | Output path is a directory | Set a specific output file path or leave blank for auto |

---

## Folder Size Reference

| Folder | Size |
|--------|------|
| `.venv\` | **7.5 GB** |
| `models\hf_cache\hub\models--google--owlv2-base-patch16-ensemble\` | 593 MB |
| `models\hf_cache\hub\models--facebook--sam2-hiera-large\` | 857 MB |
| `models\hf_cache\hub\models--facebook--sam3\` | 3.3 GB |
| `models\hf_cache\hub\models--ShilongLiu--GroundingDINO\` | ~300 MB |
| `web_app\dist\` | ~400 KB |
| `backend\` source code | ~1.2 MB |
| **Total** | **~13 GB** |

> Use a USB 3.0 drive with at least 16 GB free space.
