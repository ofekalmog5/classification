# Backend (FastAPI)

The classification service. Owns the KMeans pipeline, AI feature extraction,
file/raster helpers, persistent app config, and SSE progress streaming.

For the high-level project picture see [../README.md](../README.md).
For the API surface see [../docs/API_REFERENCE.md](../docs/API_REFERENCE.md).

---

## Module map

| File | Lines | Role |
|------|------:|------|
| `app/main.py` | ~1160 | FastAPI app — every endpoint, the `_StripApiPrefix` middleware, SSE progress queues, cancellation flags |
| `app/core.py` | ~6380 | KMeans pipeline, two-step split, tile worker, MEA color logic, XML / TXR / TXS sidecar export, shadow detection, GPU engine probe |
| `app/road_extraction.py` | ~1990 | OWLv2 + SAM 2 / SAM 3 / LangSAM extraction, color-and-geometry detectors, `FEATURE_CONFIGS`, `should_extract_feature` pre-filter, mask merge |
| `app/config.py` | ~110 | Persistent JSON config (last paths, performance settings) |
| `app/mea_profile.py` | ~110 | Read-only consumer of the calibration profile written by the MEA Calibration Tool |
| `requirements.txt` | — | Core dependencies (FastAPI, rasterio, faiss-cpu, samgeo, transformers …) |
| `requirements-gpu.txt` | — | Adds `cupy-cuda12x` + `nvidia-*-cu12` wheels for GPU KMeans |

---

## Running

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --app-dir backend
```

Listens on `http://127.0.0.1:8000`.

For GPU acceleration also:
```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-gpu.txt
```

---

## Public functions worth knowing

[app/core.py](app/core.py):

- `classify_and_export(raster_path, classes, smoothing, feature_flags, ...)` — Step 1.
- `rasterize_vectors_onto_classification(classification_path, vector_layers, classes, ...)` — Step 2.
- `classify(raster_path, classes, vector_layers, smoothing, feature_flags, ...)` — backward-compat wrapper.
- `train_kmeans_model(...)` + `build_shared_color_table(...)` — batch shared model.
- `suggest_tile_size(raster_path, workers=4)` — RAM-aware tile size picker.
- `_detect_structures_mask` + `_detect_shadows_and_infer` — shadow detection.
- `_write_composite_material_xml(output_path, classes)` — MEA companion XML.

[app/road_extraction.py](app/road_extraction.py):

- `extract_roads(...)` — legacy single-feature road extraction.
- `extract_feature_masks(raster, feature_type, ...)` — unified extraction.
  Valid `feature_type`: `roads`, `buildings`, `trees`, `fields`, `water`.
- `merge_feature_masks_onto_classification(...)` — chains across all
  classification files in a run.
- `should_extract_feature(raster_path, feature_type) -> (bool, reason)` — fast
  pre-filter on a 256×256 thumbnail.
- `FEATURE_CONFIGS` — per-feature prompts, suffixes, default colors,
  thresholds, and color-detect callbacks.
- `set_sam3_local_dir(...)` — override SAM3 weights path.

[app/mea_profile.py](app/mea_profile.py):

- `load_active_profile()` — merge user profile onto factory defaults.
- `profile_status()` — summary used by the sidebar panel.
- Read-only — the calibration tool is the only writer.

---

## SSE progress

`_PHASE_WEIGHTS` and `_PIPELINE_TOTALS` at the top of `main.py` define how
each phase is weighted in the global 0–100 % progress percentage. To add a
phase, edit those tables and emit `progress_callback(phase, done, total)` from
your function.

`_ProgressTracker` raises `TaskCancelledError` if `_CANCEL_FLAGS[task_id]` is
set, so a long pipeline aborts cleanly when the user clicks *Cancel*.

---

## Adding a new endpoint

1. Define a Pydantic request model (the file uses `BaseModel` everywhere).
2. Decorate the handler with `@app.<verb>("/path")`.
3. If the handler is long-running, allocate a `task_id`, instantiate
   `_ProgressTracker(task_id, "<pipeline-name>")`, and pass it as
   `progress_callback`. Update `_PIPELINE_TOTALS` if the new pipeline has a
   different total weight.
4. The frontend client lives at
   [../web_app/src/api/client.ts](../web_app/src/api/client.ts) — add the
   matching `fetch` wrapper there.

---

## Testing

There is no automated test suite — verification is manual via Swagger UI
(`/docs`) or the web app. The Tkinter app and `cli.py` exercise the same
`core.py` entry points.
