# GPU Acceleration

KMeans dominates pipeline runtime, so the project supports four engines and
picks the fastest one available at startup. The chosen engine is logged at the
top of every classification run and is exposed via `GET /gpu-info`.

| Priority | Engine | Wrapper class | Install |
|----------|--------|---------------|---------|
| 1 | CuPy KMeans (GPU) | `_CupyKMeans` | `pip install -r backend/requirements-gpu.txt` |
| 2 | FAISS GPU | `_FaissKMeans` | `conda install -c pytorch faiss-gpu cudatoolkit=11.8` (not on PyPI) |
| 3 | FAISS CPU | `_FaissKMeans` | `pip install faiss-cpu` (in `backend/requirements.txt`) |
| 4 | scikit-learn KMeans | (default) | always available |

Probe code lives at the top of [backend/app/core.py](../backend/app/core.py) —
search for `_ACCEL_ENGINE`.

---

## CuPy (recommended on Windows)

`backend/requirements-gpu.txt`:

```
cupy-cuda12x
nvidia-cuda-runtime-cu12
nvidia-cublas-cu12
nvidia-cuda-nvrtc-cu12
nvidia-curand-cu12
```

The `nvidia-*` wheels supply the CUDA runtime DLLs via pip, so **no system
CUDA Toolkit install is needed**. Just NVIDIA drivers ≥ 525 (CUDA 12.x).

For older drivers (CUDA 11.x), swap `cupy-cuda12x` for `cupy-cuda11x` and the
matching `nvidia-*-cu11` packages.

---

## FAISS GPU

`faiss-gpu` is conda-only and was dropped from PyPI. The project still detects
it if a user installs via conda, but the offline installer does not bundle it.
CuPy is the supported GPU path.

---

## Verifying the active engine

```bash
curl http://127.0.0.1:8000/gpu-info
```

Returns `{"engine": "cupy", "gpu": true, "info": "NVIDIA RTX A4000 (16 GB)"}`
or similar. The web app's status bar shows the same thing.

The progress log line near the start of every classification reads e.g.
`KMeans engine: cupy (GPU: NVIDIA RTX A4000)` — quick check that GPU was
actually used.

---

## Frozen EXE / PyInstaller

`WebApp.spec` (and the related specs) include a runtime hook that pre-loads the
CuPy CUDA DLLs by full path. Without it, CuPy fails to find `cudart64_12.dll`
inside the `_MEIxxxxx` temp directory at runtime. This is why the bundled
`ClassificationWebApp.exe` keeps GPU support — just plug an NVIDIA-equipped
machine.

---

## Tile mode + GPU

Tile mode shells out to a `ProcessPoolExecutor` (`_classify_tile_worker`).
Each worker re-probes the engine, so GPU acceleration applies per tile. With
`tile_workers > 1` and a single GPU, workers serialize on the device — set
`tile_workers = 1` if you observe contention.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cupy.cuda.runtime.CUDARuntimeError: cudaErrorInsufficientDriver` | Update NVIDIA driver |
| `ImportError: DLL load failed while importing _runtime` | Re-install `nvidia-cuda-runtime-cu12` (correct major version) |
| Engine probe picks `faiss-cpu` despite CuPy installed | The probe gracefully falls back if any CuPy import fails — run `python -c "import cupy; cupy.cuda.runtime.getDeviceCount()"` to see the real error |
| GPU works in dev, fails in EXE | Make sure the runtime hook in `WebApp.spec` is present and the CUDA DLLs are bundled |
