# Architecture

High-level map of how the classification project's pieces fit together. For
operational commands see [../RUNNING_GUIDE.md](../RUNNING_GUIDE.md);
for deployment see [../STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md).

The graphify knowledge graph at [../graphify-out/](../graphify-out/) was used to
identify the "god nodes" and community structure summarised below.

---

## God Nodes (most-connected functions)

| Node | Edges | Where |
|------|-------|-------|
| `classify_and_export()` | 44 | [backend/app/core.py](../backend/app/core.py) |
| `get()` (web client wrapper) | 35 | [web_app/src/api/client.ts](../web_app/src/api/client.ts) |
| `rasterize_vectors_onto_classification()` | 17 | [backend/app/core.py](../backend/app/core.py) |
| `build_shared_color_table()` | 16 | [backend/app/core.py](../backend/app/core.py) |
| FastAPI app at `backend/app/main.py` | 16 | [backend/app/main.py](../backend/app/main.py) |
| `_classify_tile_worker()` | 15 | [backend/app/core.py](../backend/app/core.py) |
| `classify()` (back-compat wrapper) | 14 | [backend/app/core.py](../backend/app/core.py) |
| Reducer (app state) | 14 | [web_app/src/store/index.tsx](../web_app/src/store/index.tsx) |
| `AppState` interface | 12 | [web_app/src/types.ts](../web_app/src/types.ts) |
| `ActionsSection` sidebar panel | 12 | [web_app/src/components/sidebar/ActionsSection.tsx](../web_app/src/components/sidebar/ActionsSection.tsx) |

---

## Process / Service Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  React + Vite (web_app)                                              │  │
│  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │  │
│  │  │  Sidebar     │  │  MapView    │  │  LayerPanel / StatusBar  │ │  │
│  │  │  panels      │  │  (Leaflet)  │  └──────────────────────────┘ │  │
│  │  └──────────────┘  └─────────────┘                                 │  │
│  │           │                                                          │  │
│  │           ▼  api/client.ts ──── /api/* via Vite proxy ────────┐    │  │
│  └────────────────────────────────────────────────────────────────┼────┘  │
└────────────────────────────────────────────────────────────────────┼───────┘
                                                                     │
                                                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│  FastAPI backend  (uvicorn :8000)                                            │
│  backend/app/main.py                                                         │
│   ├── _StripApiPrefix middleware  (strips /api so the same client works     │
│   │                                  against dev proxy and prod static)    │
│   ├── /classify, /classify-step1, /classify-step2, /classify-batch          │
│   ├── /extract-roads, /merge-road-mask                                      │
│   ├── /extract-features, /merge-feature-masks                               │
│   ├── /progress/{task_id}  (SSE), /cancel/{task_id}                         │
│   ├── /raster-info, /raster-as-png, /list-dir, /scan-folder                 │
│   ├── /suggest-tile-size, /gpu-info, /app-config, /set-sam3-path …          │
│   │                                                                         │
│   ├── core.py        ← classify_and_export, rasterize_vectors,              │
│   │                    train_kmeans_model, build_shared_color_table,        │
│   │                    suggest_tile_size, _classify_tile_worker,            │
│   │                    _detect_shadows_and_infer, _MEA_COMPOSITE_NAMES,     │
│   │                    _write_composite_material_xml, .txr/.txs writers    │
│   ├── road_extraction.py  ← extract_roads / extract_feature_masks /         │
│   │                          merge_*, FEATURE_CONFIGS                       │
│   ├── config.py      ← persistent JSON app config                           │
│   └── mea_profile.py ← reads %ProgramData%\…\mea_calibration_profile.json   │
└────────────────────────────────────────────────────────────────────────────┘
            ▲
            │  reads
            │
%ProgramData%\MaterialClassification\mea_calibration_profile.json
            ▲
            │  written by
            │
┌────────────────────────────────────────────────────────────────────────────┐
│  MEA Calibration Tool  (separate FastAPI :8100 + React)                     │
│  mea_calibration_tool/backend/app/{main,profile,raster_io,sampling}.py      │
└────────────────────────────────────────────────────────────────────────────┘
```

The two apps are intentionally decoupled — the calibration tool only writes the
profile JSON, the main app only reads it. There is no IPC.

---

## Two-step pipeline (Step 1 + Step 2)

```
raster.tif ──► classify_and_export()  ──►  classified.tif (RGB)
                                            └─► <stem>.xml (MEA only)
                                            └─► <stem>.txr / .txs (MEA only)
                                            └─► EPSG:4326 reprojection
                                            └─► power-of-2 padding

classified.tif + vectors ──► rasterize_vectors_onto_classification() ──► merged.tif
```

`classify()` is a thin wrapper that calls both steps in order — kept for backward
compatibility with the Tkinter app and `cli.py`.

The `/classify` endpoint runs the full pipeline; `/classify-step1` runs step 1 only;
`/classify-step2` runs step 2 only on an existing classification.

---

## Batch shared-model classification

`POST /classify-batch` trains one KMeans model on a representative sample, then
re-uses the trained centroids and a **shared color table** (`build_shared_color_table()`)
to classify every raster in the batch. This avoids each image getting its own
unrelated cluster IDs.

Used when the user picks multiple files at once.

---

## AI feature extraction pipeline

```
raster.tif
   │
   ▼
should_extract_feature(raster, feature_type)        ← RGB / linearity pre-filter
   │
   ▼ pass
FEATURE_CONFIGS[feature_type] (1+ sub-prompts)
   │
   ▼ for each sub-prompt
extract_feature_masks( … )
   │   ├── OWLv2 + SAM2/3 text-prompted segmentation
   │   ├── color_detect: color+geometry CV detector
   │   └── union (OR) per-tile masks → <suffix>.tif
   │
   ▼
_<feature_type>/<sub-suffix>.tif        ← organized subfolders
   │
   ▼
merge_feature_masks_onto_classification(...)        ← chains across all classifications
```

See [AI_FEATURE_EXTRACTION.md](AI_FEATURE_EXTRACTION.md) for prompts, feature colors,
and model fallback chain.

---

## Web-app state

`web_app/src/store/index.tsx` owns the reducer and exposes
`StoreProvider`, `useAppState`, `useAppDispatch`. Sidebar sections dispatch actions;
`MapView` and `LayerPanel` read state.

The `AppState` interface in `web_app/src/types.ts` is the single source of truth for
what the UI tracks: active raster, vector layers, MEA-mode toggle, performance
settings, classification result, progress events, …

---

## Build / packaging

- `WebApp.spec` (PyInstaller) → `dist/ClassificationWebApp.exe`. Includes a runtime
  hook that pre-loads CuPy CUDA DLLs by full path so GPU works in the frozen exe.
- `ClassificationApp.iss` (Inno Setup) → `ClassificationApp_Setup.exe`.
- `prepare_offline.bat` → `offline_installer/` USB payload (embedded Python + all
  wheels + optional HF model cache).

---

## Extension points

- New AI feature: add an entry to `FEATURE_CONFIGS` in
  [backend/app/road_extraction.py](../backend/app/road_extraction.py) — prompts,
  default merge color, optional `color_detect` callback, optional `threshold`.
  Add a checkbox in `web_app/src/components/sidebar/FeaturesSection.tsx`.
- New post-processing step: insert in `classify_and_export()` between
  *Pixel assignment* and *Saving output* phases (see `_PHASE_WEIGHTS` in
  [backend/app/main.py](../backend/app/main.py)).
- New MEA material: edit [shared/mea_classes.json](../shared/mea_classes.json),
  [shared/mea_defaults.json](../shared/mea_defaults.json),
  [web_app/src/constants/mea.ts](../web_app/src/constants/mea.ts), and
  `MEA_CLASSES` / `_MEA_COMPOSITE_NAMES` in `core.py`.
