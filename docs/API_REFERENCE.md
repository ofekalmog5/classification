# API Reference

Every FastAPI endpoint exposed by [backend/app/main.py](../backend/app/main.py).
Backend listens on `http://127.0.0.1:8000` by default. Interactive Swagger UI at
`/docs`, ReDoc at `/redoc`.

The backend installs a `_StripApiPrefix` middleware so every path is also reachable
under `/api/<path>` — that is what the dev frontend uses through Vite's proxy.

The MEA Calibration Tool runs as a **separate** FastAPI service (default port
8100). Its endpoints are listed at the bottom.

---

## Pipeline

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/classify` | Full pipeline: KMeans + RGB export + vector overlay |
| POST | `/classify-step1` | Step 1 only — KMeans + RGB export |
| POST | `/classify-step2` | Step 2 only — vector rasterisation onto an existing classification |
| POST | `/classify-batch` | Train one shared KMeans model and apply it to many rasters |

All four return `{"status": "ok", "outputPath": "..."}` on success and
`{"status": "error", "message": "..."}` on failure. Long-running calls are
keyed by a `taskId` so the client can stream progress.

### Common request fields
- `rasterPath` — absolute path on the server filesystem
- `classes` — `[{ "id": "class-1", "name": "BM_ASPHALT", "color": "#2D2D30" }, …]`
- `vectorLayers` — `[{ "layerPath": "...", "classId": "class-1" }, …]`
- `featureFlags` — `{ "spectral": true, "texture": true, "indices": true }`
- `smoothing` — `"none"`, `"median_1"`, `"median_2"`, …
- `tileMode`, `tileMaxPixels`, `tileWorkers`
- `detectShadows: bool`, `maxThreads: int | null`
- `outputPath` — optional override; defaults to `<input>_classified.tif`

The full request schema lives at
[shared/contracts/classification-request.schema.json](../shared/contracts/classification-request.schema.json).

---

## AI Feature Extraction

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/extract-roads` | OWLv2+SAM road extraction (legacy single-feature endpoint) |
| POST | `/merge-road-mask` | Merge a road mask onto a classification |
| POST | `/extract-features` | Unified feature extraction (`feature_type` ∈ `roads`, `buildings`, `trees`, `fields`, `water`) |
| POST | `/merge-feature-masks` | Merge AI feature masks onto a classification — chains across all classifications |
| POST | `/set-sam3-path` | Override the local SAM3 weights directory |
| GET | `/road-extract-config` | Return the SAM3 path + current model status |

`extract-features` writes outputs to `_<feature_type>/<sub-suffix>.tif` next to the
source. Sub-suffixes come from `FEATURE_CONFIGS` (e.g. water has
`water_bodies`, `water_channels`, `water_other`). See
[AI_FEATURE_EXTRACTION.md](AI_FEATURE_EXTRACTION.md).

---

## Progress & Cancellation

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/progress/{task_id}` | Server-Sent Events stream — `{"phase", "done", "total"}` updates |
| POST | `/cancel/{task_id}` | Set the cancel flag — long pipelines raise `TaskCancelledError` at the next progress callback |

Phases and their relative weights are defined in `_PHASE_WEIGHTS` /
`_PIPELINE_TOTALS` at the top of `main.py`.

---

## File / Raster Helpers

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/raster-info` | Returns CRS, bounds, dtype, band count, transform |
| POST | `/raster-as-png` | Render a raster (full or window) to PNG for the map |
| GET | `/raster-as-png/{file_path:path}` | Same, but for tile URLs Leaflet can request |
| POST | `/list-dir` | List entries in a directory (filtered by `.vrt`, `.tif`, vectors, …) |
| POST | `/scan-folder` | Recursive scan, returns hierarchy for the file browser modal |
| POST | `/suggest-tile-size` | Returns a memory-safe tile side length for the given raster |

---

## System / Health

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | `{"status": "ok"}` |
| GET | `/gpu-info` | Acceleration engine name + GPU info (CuPy / FAISS / sklearn) |
| GET | `/app-config` | Persistent JSON config (last paths, last performance settings) |
| POST | `/app-config` | Update persistent config |

In production builds the static frontend is mounted at `/` and a fallback
`@app.get("/{full_path:path}")` returns `index.html` so React's client-side
routing works.

---

## MEA Calibration Tool API

Separate process, default port 8100.
Endpoints in [mea_calibration_tool/backend/app/main.py](../mea_calibration_tool/backend/app/main.py):

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/profile` | Return the active user profile (or `null` if none) + profile path |
| GET | `/profile/factory-defaults` | Return the bundled factory defaults |
| POST | `/profile` | Save a new / updated user profile |
| DELETE | `/profile` | Delete the user profile (reverts the main app to factory defaults) |
| POST | `/profile/import` | Import a profile JSON file |
| POST | `/profile/export` | Export the active profile to a path |
| POST | `/sample-pixels` | Sample raster pixels in a given polygon — returns reference colors |
| POST | `/geo-to-raster` | Geographic → raster pixel coordinate transform |
| POST | `/raster-info`, `/raster-as-png`, `/list-dir` | Same helpers as the main backend |
| GET | `/pick-file`, `/pick-save-path` | Native file dialogs for the standalone tool |

---

## Notes
- Endpoint paths above use the literal forms from `@app.<method>(...)` decorators.
- All non-`GET` endpoints accept JSON bodies.
- The `_StripApiPrefix` middleware means `/api/classify` and `/classify` are
  equivalent — both clients (dev proxy + production static) hit the same routes.
