# offline_installer/

USB-friendly payload that installs the Classification Web App + MEA Calibration
Tool on a machine with **no internet, no Python, and no Node.js**.

The deployment workflow itself is documented in two places:

- [STANDALONE_DEPLOYMENT.md](STANDALONE_DEPLOYMENT.md) — the in-package
  reference (read this on the target machine).
- [../STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md) — the project-wide
  deployment guide that compares all three methods (Inno installer / Setup.bat
  / manual copy).

This README only describes how the folder is built and what each subdirectory
contains.

---

## How it's built

From the project root, on a machine that *does* have internet:

```bat
prepare_offline.bat
```

The script:

1. Downloads `python-3.11.9-embed-amd64.zip` and `get-pip.py` into
   `prerequisites/`.
2. Downloads core wheels into `offline_packages/`.
3. Optionally downloads CUDA torch into `offline_packages_torch/` and
   CuPy + nvidia-cuda-* into `offline_packages_gpu/`.
4. Builds the frontend (`npm run build` inside `web_app/`).
5. Copies backend, frontend, MEA calibration tool, shared assets, and the
   launcher into `offline_installer/app/`.
6. Writes `Setup.bat`, `Setup.ps1`, and `README.txt`.

The output folder is around **8 GB** — copy it whole to a USB stick.

---

## Layout

```
offline_installer/
├── Setup.bat                      ← double-click this on the target
├── Setup.ps1                      PowerShell wizard (folder picker, options)
├── prerequisites/
│   ├── python-3.11.9-embed-amd64.zip
│   └── get-pip.py
├── offline_packages/              core pip wheels (FastAPI, rasterio, …)
├── offline_packages_torch/        CUDA torch override (optional)
├── offline_packages_gpu/          CuPy + nvidia-cuda-* (optional)
├── app/                           pre-built application files
│   ├── backend/
│   ├── web_app/dist/
│   ├── mea_calibration_tool/
│   ├── shared/
│   ├── launcher.py                main app launcher (sets HF_HUB_OFFLINE)
│   ├── start.bat
│   ├── requirements.txt
│   ├── requirements-gpu.txt
│   └── version.txt
├── STANDALONE_DEPLOYMENT.md
└── README.txt                     short user-facing instructions
```

---

## Notes

- The wizard installs **embedded Python 3.11.9** into `<install>\python\`. The
  target machine doesn't need any other Python install.
- HuggingFace model weights are not bundled by default. The wizard offers a
  *Full AI* feature pack that copies the dev machine's
  `~\.cache\huggingface\hub\` into `<install>\models\hf_cache\hub\`. If skipped,
  copy the `hub\` folder manually after install — see
  [STANDALONE_DEPLOYMENT.md](STANDALONE_DEPLOYMENT.md).
- The launcher exports `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
  `HF_HOME=<install>\models\hf_cache` so the models resolve from disk only —
  no network calls are attempted at runtime.
- `requirements-gpu.txt` opts in to CuPy KMeans. With the matching
  NVIDIA drivers on the target, the engine probe in `core.py` automatically
  picks GPU. Without GPU, the app falls back to FAISS CPU then sklearn KMeans.

---

## Updating the payload

Re-run `prepare_offline.bat` from the project root after any backend / frontend
change. The script overwrites the existing `offline_installer/app/` and any
selected `offline_packages_*/` folders. The `prerequisites/` folder is reused
across builds (the embedded Python doesn't change unless you bump the Python
version in the script).
