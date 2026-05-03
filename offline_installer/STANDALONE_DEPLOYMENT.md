# Standalone Station Deployment — Method B Reference

This is the in-package copy of the deployment guide. It documents only the **Setup.bat
wizard** delivered alongside this folder. For the full multi-method deployment matrix
(Inno installer, manual copy, etc.) see the parent project's
[STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md).

---

## On the target machine (no internet)

1. Double-click **`Setup.bat`** in this folder.
2. The wizard (PowerShell — `Setup.ps1`) asks for:
   - Install folder (e.g. `C:\ClassificationApp`).
   - Feature pack — *Full AI* (HF models bundled) or *Core only* (KMeans only).
   - PyTorch variant (CPU / CUDA) and GPU options.
3. Click **Install** — takes 10–30 minutes.
4. Desktop / Start-Menu shortcuts are created for:
   - **Classification Web App** → `<install>\start.bat`
   - **MEA Calibration Tool** → `<install>\mea_calibration_tool\launcher.py`

The browser opens automatically at `http://127.0.0.1:8000`.

---

## Folder contents (this directory)

```
offline_installer/
  Setup.bat                     ← entry point
  Setup.ps1                     wizard implementation
  prerequisites/
    python-3.11.9-embed-amd64.zip
    get-pip.py
  offline_packages/             core wheels (FastAPI, rasterio, …)
  offline_packages_torch/       CUDA-enabled torch (optional)
  offline_packages_gpu/         CuPy + nvidia-cuda-* (optional)
  app/                          pre-built application files
    backend/
    web_app/dist/
    shared/
    mea_calibration_tool/
    launcher.py
    start.bat
    requirements.txt
    requirements-gpu.txt
    version.txt
  STANDALONE_DEPLOYMENT.md      this file
  README.txt
```

---

## Copy AI model weights (Core-only installs)

If *Core only* was chosen, AI extraction needs ~5 GB of HuggingFace models. On the **dev
machine**:

```
C:\Users\<name>\.cache\huggingface\hub\
```

Copy that `hub\` folder to the **target machine** at:

```
<install dir>\models\hf_cache\hub\
```

Expected content:

```
hub/
  models--google--owlv2-base-patch16-ensemble/   593 MB
  models--facebook--sam2-hiera-large/            857 MB
  models--facebook--sam3/                        3.3 GB
  models--ShilongLiu--GroundingDINO/             ~300 MB
```

The launcher exports `HF_HUB_OFFLINE=1` and `HF_HOME=<install>\models\hf_cache` so the
models resolve from disk only.

---

## Installed layout

```
<install dir>/
  python/                     embedded Python 3.11.9
  backend/                    FastAPI backend
  web_app/dist/               pre-built frontend
  mea_calibration_tool/       companion app
  shared/                     MEA classes + factory defaults
  models/hf_cache/hub/        AI model weights (auto if Full AI; manual otherwise)
  start.bat                   launches the web app
  launcher.py
  version.txt
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Blank page in browser | Frontend not built — run `npm run build` in `web_app\` (dev machine) and rebuild the offline payload |
| Server won't start | `python\` folder missing — re-run `Setup.bat`, check log |
| AI extraction fails | Model weights not in `models\hf_cache\hub\` |
| GPU not used | NVIDIA drivers missing on the target — app falls back to CPU |
| Very slow on large images | Enable **Tile Processing** in the *Performance* sidebar panel |
| `ModuleNotFoundError: uvicorn` | Package install failed — re-run Setup, inspect installer log |
