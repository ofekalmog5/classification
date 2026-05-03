# MEA Calibration Tool

The MEA Calibration Tool is a small, separate web app that lets a domain expert
sample reference colors for the 13 MEA material classes from a representative
raster, then save the result as a **calibration profile** that the main
classification app reads on every run.

---

## Why it exists

Out of the box the main app uses [shared/mea_defaults.json](../shared/mea_defaults.json) —
generic factory colors. Real-world imagery (lighting, sensor calibration, paint
choices, season) drifts from these defaults, which leads the KMeans assignment
to lock onto the wrong material per cluster. The calibration tool fixes this by:

1. Loading a representative raster.
2. Letting the user click polygons / sample regions for each material that
   appears in that raster.
3. Computing per-material reference RGB, anchor colors, sample std and a
   tolerance radius.
4. Writing a v2 profile JSON to a system-wide location the main app reads.

When a profile exists, the main app's MEA mode merges the user's overrides on
top of factory defaults — so partially-calibrated profiles still work.

---

## Profile location

```
%ProgramData%\MaterialClassification\mea_calibration_profile.json
```

This is the **only** path used; both the calibration tool (writer) and the main
app's [backend/app/mea_profile.py](../backend/app/mea_profile.py) (reader) point
here. There is no other IPC between the two apps.

`mea_profile.profile_status()` is what the sidebar's
[MeaProfileStatus](../web_app/src/components/sidebar/MeaProfileStatus.tsx)
panel displays — *Custom* badge when a user profile is active, otherwise
*Factory defaults*.

---

## Profile schema (v2)

```json
{
  "version": 2,
  "name": "Site A — winter",
  "created_at": "2026-04-23T12:00:00+00:00",
  "raster_path": "C:/data/site-a/orthophoto.tif",
  "material_overrides": {
    "BM_ASPHALT": {
      "reference_color": "#2D2D30",
      "reference_rgb": [45, 45, 48],
      "anchors": [[44, 44, 56], [52, 55, 72], [91, 91, 101]],
      "tolerance_radius": 34,
      "sample_count": 1247,
      "sample_std_rgb": [12.4, 11.9, 13.1]
    },
    "BM_WATER":   { ... },
    ...
  },
  "bias_overrides": {
    "BM_WATER": 1.2,
    "BM_VEGETATION": 0.9
  },
  "frequency_prior_overrides": { ... }
}
```

- `material_overrides` — per-material reference + anchor colors, tolerance
  radius (computed from `sample_std_rgb` if absent: `max(30, round(2 * max_std))`).
- `bias_overrides` — multiplier on per-material distance score (push KMeans
  toward / away from a material).
- `frequency_prior_overrides` — per-material frequency prior used by the
  diversity-enforcement pass.

Materials missing from the profile fall back to the factory defaults entry.

---

## Running the tool

### Dev machine

```powershell
cd mea_calibration_tool
python launcher.py
```

`launcher.py` boots `mea_calibration_tool/backend/app/main.py` on
`http://127.0.0.1:8001` (overridable via `MEA_CAL_PORT`) and opens the browser.

### Standalone install

The offline installer (Method A and B in
[STANDALONE_DEPLOYMENT.md](../STANDALONE_DEPLOYMENT.md)) ships the calibration
tool as a sibling app under `<install dir>\mea_calibration_tool\`. A Start-Menu
shortcut "MEA Calibration Tool" runs `launcher.py` against the same embedded
Python that the main app uses.

---

## Workflow

1. Open the tool and pick a raster (`/list-dir`, `/raster-info`, `/raster-as-png`
   helpers — same shapes as the main backend).
2. For each material that appears in the raster, draw a polygon over a
   representative area. The tool calls `POST /sample-pixels` which returns the
   mean RGB, std RGB and per-pixel sample.
3. Adjust per-material tolerance / bias in the UI as needed.
4. Click *Save profile* → `POST /profile` writes `mea_calibration_profile.json`.
5. Restart (or just reload) the main classification app — its sidebar's
   *MEA Profile* panel will now show the profile name and material count.

---

## Endpoints (calibration backend, port 8001 by default)

See [API_REFERENCE.md](API_REFERENCE.md) for the full list. The frequently-used ones:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/profile` | Active user profile (or `null`) + path |
| GET | `/profile/factory-defaults` | Factory defaults from `shared/mea_defaults.json` |
| POST | `/profile` | Save a profile |
| DELETE | `/profile` | Remove the user profile (revert to factory) |
| POST | `/profile/import` | Load a profile JSON file |
| POST | `/profile/export` | Save the active profile to a path |
| POST | `/sample-pixels` | Sample raster pixels in a polygon |
| POST | `/geo-to-raster` | Geographic ↔ raster pixel transform |

---

## Reading the profile from the main app

[backend/app/mea_profile.py](../backend/app/mea_profile.py):

- `load_active_profile()` — merges user overrides on top of factory defaults
  and returns the merged dict (with `_source` set to `"user"` or `"factory"`).
- `profile_status()` — light-weight call used by the sidebar status panel.
- The reader **never writes**. Edit profiles only via the calibration tool.

The merge rule for `material_overrides` is per-key shallow merge: a user entry
overrides individual fields of the matching factory entry, but unspecified
materials keep their factory values entirely.
