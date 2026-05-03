# Web App (Vite + React + TypeScript)

The primary user interface for the classification service. Talks to the FastAPI
backend at `http://127.0.0.1:8000` via a Vite proxy on `/api/*`.

For the project overview see [../README.md](../README.md).
For backend endpoints see [../docs/API_REFERENCE.md](../docs/API_REFERENCE.md).

---

## Stack

- React 18 + TypeScript
- Vite 5
- Tailwind CSS 3
- Leaflet + react-leaflet for the map view
- `georaster` + `georaster-layer-for-leaflet` for tiled raster rendering

---

## Project layout

```
web_app/
├── index.html
├── package.json
├── postcss.config.ts
├── tailwind.config.ts
├── tsconfig.json
├── vite.config.ts
└── src/
    ├── main.tsx
    ├── App.tsx
    ├── index.css
    ├── types.ts                  AppState, ClassItem, VectorLayer, MapLayer, …
    ├── api/
    │   └── client.ts             fetch + SSE wrappers, /api/* base
    ├── store/
    │   └── index.tsx             Reducer + StoreProvider + useAppState / useAppDispatch
    ├── constants/
    │   └── mea.ts                MEA_CLASSES (13) + generatePalette()
    └── components/
        ├── Layout.tsx
        ├── Sidebar.tsx
        ├── MapView.tsx
        ├── LayerPanel.tsx
        ├── StatusBar.tsx
        ├── FileBrowserModal.tsx
        └── sidebar/
            ├── InputSection.tsx
            ├── MaterialsSection.tsx
            ├── FeaturesSection.tsx
            ├── VectorsSection.tsx
            ├── PerformanceSection.tsx
            ├── ClassificationSection.tsx
            ├── ActionsSection.tsx
            ├── SettingsSection.tsx
            └── MeaProfileStatus.tsx
```

---

## Running

```powershell
cd web_app
npm install
npm run dev          # http://127.0.0.1:5173
```

Vite proxies `/api/*` to the FastAPI backend on port 8000. Make sure the
backend is running first (`uvicorn` — see [../backend/README.md](../backend/README.md)
or [../RUNNING_GUIDE.md](../RUNNING_GUIDE.md)).

Production build:

```powershell
npm run build        # writes web_app/dist/
```

`web_app/dist/` is what the FastAPI backend serves as static files in standalone
deployments — open `http://127.0.0.1:8000` directly in that case (no Vite).

---

## State (`src/store/index.tsx`)

Reducer-based `AppState` covers:

- Active raster + scanned input folder
- Vector layers, MEA mode toggle, classes (custom or MEA preset)
- Performance settings (tile mode, tile size, workers, max threads, detect shadows)
- Per-feature extraction checkboxes
- Map layers + render preferences
- Progress events (live phase + done %)
- Last classification result

Sidebar sections dispatch actions; `MapView`, `LayerPanel`, and `StatusBar`
read state. Hooks are `useAppState()` and `useAppDispatch()`.

---

## API client (`src/api/client.ts`)

One module wraps every backend endpoint:

- `runStep1`, `runStep2`, `runFullPipeline`, `runMeaPipeline`, `runBatchClassify`
- `extractRoads`, `mergeRoadMask`, `extractFeatures`, `mergeFeatureMasks`
- `getRasterInfo`, `getRasterAsPng`, `listDir`, `scanFolder`
- `suggestTileSize`, `fetchGpuInfo`, `getAppConfig`, `setAppConfig`,
  `setSam3Path`, `getRoadExtractConfig`
- `getMeaProfileStatus`
- `startProgressStream(taskId, onEvent)` — Server-Sent Events listener
- `cancelTask(taskId)`

---

## Sidebar sections

| Section | What it does |
|---------|--------------|
| **Input** | Pick raster + scan input folder. Group support — multiple images uploaded together cascade-delete as one unit. |
| **Materials** | Edit class list. Toggling MEA mode loads the 13 MEA classes from `constants/mea.ts`. |
| **Features** | Per-feature checkboxes (roads, buildings, trees, fields, water) for unified extraction. |
| **Vectors** | Add vector overlays mapped to class IDs. |
| **Performance** | Tile mode, auto / manual tile size, tile workers, max threads, detect shadows. |
| **Classification** | Smoothing, feature flags (spectral / texture / indices), shadow detection on/off. |
| **Actions** | Buttons: Step 1, Step 2, Full, MEA, Batch, Extract, Merge. Triggers all backend pipelines. |
| **Settings** | Misc app settings, SAM3 path override. |
| **MeaProfileStatus** | Read-only display of the active calibration profile (Custom badge if user profile exists). |

---

## Map view

`MapView.tsx` initialises Leaflet, fetches raster info via `/raster-info`, and
overlays the rendered PNG (`/raster-as-png`) as a tile layer. The UI prompts
before rendering many images at once to avoid hammering the backend.

`LayerPanel.tsx` is the layer toggle; layers belong to groups so deleting a
group removes all derived layers in one click.

---

## Conventions

- All API calls go through `client.ts` — never `fetch` directly from a component.
- Component-local UI state lives in `useState`; cross-cutting state lives in
  the store.
- New material classes must match
  [../shared/mea_classes.json](../shared/mea_classes.json) and the backend's
  `MEA_CLASSES` constant.
- Tailwind utility classes everywhere; no styled-components or CSS modules.

---

## Build artefacts

- `web_app/dist/` — production frontend, mounted by the FastAPI backend in
  standalone builds.
- `tsconfig.tsbuildinfo` — TypeScript incremental build cache (gitignored).
