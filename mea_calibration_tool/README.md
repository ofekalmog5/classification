# MEA Calibration Tool

A separate web app for sampling material reference colors from a representative
raster and saving the result as a calibration profile that the main
classification app reads on every run.

> For the workflow, profile schema, and merge rules see
> [../docs/MEA_CALIBRATION_TOOL.md](../docs/MEA_CALIBRATION_TOOL.md).

---

## Layout

```
mea_calibration_tool/
├── launcher.py              boots backend (port 8001) and opens the browser
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py            FastAPI: /profile, /sample-pixels, /geo-to-raster, …
│       ├── profile.py         Read/write the v2 profile JSON
│       ├── sampling.py        Region sampling — per-material RGB stats
│       └── raster_io.py       Raster I/O helpers
└── web_app/
    ├── package.json           Companion React UI
    ├── index.html
    └── src/
        ├── App.tsx
        ├── main.tsx
        ├── api/
        ├── components/
        ├── constants/
        ├── store/
        └── types.ts
```

---

## Running (dev)

```powershell
cd mea_calibration_tool
python launcher.py
```

`launcher.py` starts uvicorn on `127.0.0.1:8001` and opens the browser. Override
with `MEA_CAL_PORT=8888 python launcher.py` if 8001 is taken.

For the web UI alongside the backend (separate Vite process):

```powershell
cd mea_calibration_tool/web_app
npm install
npm run dev
```

---

## Profile location

```
%ProgramData%\MaterialClassification\mea_calibration_profile.json
```

Same path on every host. The main app reads it via
[../backend/app/mea_profile.py](../backend/app/mea_profile.py); this tool is the
**only** writer.

Factory defaults (used when no user profile exists) live at
[../shared/mea_defaults.json](../shared/mea_defaults.json).

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/profile` | Active user profile + path |
| GET | `/profile/factory-defaults` | Factory defaults |
| POST | `/profile` | Save / update profile |
| DELETE | `/profile` | Remove user profile (revert to factory) |
| POST | `/profile/import` | Import a profile JSON file |
| POST | `/profile/export` | Export to a path |
| POST | `/sample-pixels` | Sample raster pixels in a polygon |
| POST | `/geo-to-raster` | Geographic ↔ raster pixel transform |
| POST | `/raster-info`, `/raster-as-png`, `/list-dir` | Same helpers as the main backend |
| GET | `/pick-file`, `/pick-save-path` | Native file dialogs |

---

## Standalone install

The offline installer (Method A and B in
[../STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md)) ships this folder
under `<install dir>\mea_calibration_tool\`. The Start-Menu shortcut "MEA
Calibration Tool" runs `launcher.py` against the same embedded Python that
powers the main app.
