# AI Feature Extraction

The web app can refine a KMeans classification with text-prompted segmentation
masks. The masks come from two cooperating detectors per feature:

1. **OWLv2 + SAM 2 / SAM 3** — open-vocabulary detection followed by mask
   refinement (HuggingFace `transformers` pipeline).
2. **Color + geometry CV detector** — a hand-tuned detector that catches what
   OWLv2 misses (e.g. small water bodies, dim roof tones, dirt roads).

Per tile, the two outputs are unioned (logical OR) before the mask is written
out. This is implemented in [backend/app/road_extraction.py](../backend/app/road_extraction.py).

---

## Supported feature types

`FEATURE_CONFIGS` in [road_extraction.py:1302](../backend/app/road_extraction.py)
defines one or more sub-prompts per feature. Each sub-prompt produces its own
output raster with a unique suffix.

| Feature | Sub-prompt(s) | Output suffix(es) | Default merge color |
|---------|---------------|-------------------|---------------------|
| `roads` | `road, highway, asphalt path` | `roads` | `BM_ASPHALT` (`#2D2D30`) |
| `buildings` | `building, house, roof, rooftop, structure` | `buildings` | `BM_CONCRETE` (`#B4B4B4`) |
| `trees` | `tree, trees, forest, woodland, grove` | `trees` | `BM_VEGETATION` (`#228B22`) |
| `fields` | `grass, lawn, field, meadow, pasture` | `fields_grass` | `BM_LAND_GRASS` (`#7CFC00`) |
| | `crop, farmland, agriculture, cultivated field` | `fields_agriculture` | `BM_LAND_DRY_GRASS` (`#BDB76B`) |
| `water` | `water, lake, pond, reservoir, pool` | `water_bodies` | `BM_WATER` (`#1C6BA0`) |
| | `river, stream, canal, waterway, channel` | `water_channels` | `BM_WATER` (`#1C6BA0`) |
| | `sea, ocean, fish pond, swimming pool` | `water_other` | `BM_WATER` (`#1C6BA0`) |

A water-only image runs through three sub-prompts; a typical orthophoto with
mixed land cover may run all 8 prompt variants.

Each entry can also override the per-feature OWLv2 score `threshold` and supply
a `color_detect` callable. Buildings and water use the lowest thresholds because
OWLv2 struggles with them on aerial imagery.

---

## Pre-filter

Before invoking any model, `should_extract_feature(raster_path, feature_type)`
reads a thumbnail (≤256×256 overview) and decides whether the image plausibly
contains the feature at all. Examples:

- **Roads** require a minimum linearity score (skips open terrain / forest).
- **Buildings** require enough mid-grey pixels.
- **Water** requires a minimum blue / dark-region pixel count.
- **Fields** look for green or yellow / brown homogeneous areas.
- **Trees** look for green vegetation patches.

The pre-filter runs in milliseconds even on huge rasters and short-circuits the
full extraction when there's nothing to find.

---

## Output organisation

```
input_image.tif
input_image_classified.tif
_roads/
  input_image.roads.tif
_buildings/
  input_image.buildings.tif
_trees/
  input_image.trees.tif
_fields/
  input_image.fields_grass.tif
  input_image.fields_agriculture.tif
_water/
  input_image.water_bodies.tif
  input_image.water_channels.tif
  input_image.water_other.tif
```

`merge_feature_masks_onto_classification()` walks every classification file in
the run and chains the merges so a single click in the UI updates them all.

---

## Model fallback chain

`backend/app/road_extraction.py` tries models in this order:

1. **SAM 3** (`facebook/sam3`) via `samgeo` — preferred when available. Requires
   `triton` (Linux) or `triton-windows` (Windows).
2. **OWLv2 + SAM 2** (`google/owlv2-base-patch16-ensemble` +
   `facebook/sam2-hiera-large`) — works on Windows without compilation. This is
   the default fallback.
3. **LangSAM (GroundingDINO + SAM)** (`ShilongLiu/GroundingDINO`) — last-resort
   fallback. Slower and lower-quality on aerial imagery but no triton dep.

When a model isn't installed or the weights aren't on disk, the code falls
through silently and only the color+geometry detector runs.

---

## Model weight installation

On a dev machine the first request downloads ~5 GB to `~\.cache\huggingface\hub\`.
On an offline station, copy that folder to `<install dir>\models\hf_cache\hub\`
and the launcher exports `HF_HUB_OFFLINE=1` + `HF_HOME=<install>\models\hf_cache`
so the models resolve from disk. Sizes (approx):

| Model | Size |
|-------|------|
| `models--google--owlv2-base-patch16-ensemble` | 593 MB |
| `models--facebook--sam2-hiera-large` | 857 MB |
| `models--facebook--sam3` | 3.3 GB |
| `models--ShilongLiu--GroundingDINO` | ~300 MB |

`POST /set-sam3-path` lets advanced users point the backend at an alternative
SAM3 weights directory.

---

## Progress

Phase weights from `backend/app/main.py`:

| Phase | Weight |
|-------|--------|
| Loading SAM 3 model | 10 |
| Extracting roads | 70 |
| Morphological closing | 10 |
| Merging road mask | 80 |

The unified feature extraction reuses the same machinery and emits the same
SSE progress events.

---

## Adding a new feature type

1. Add an entry to `FEATURE_CONFIGS` in
   [backend/app/road_extraction.py](../backend/app/road_extraction.py). The
   minimum keys are `prompt`, `suffix`, `color`. Optional: `threshold`,
   `color_detect` callable.
2. Extend `should_extract_feature()` with the appropriate pre-filter (or fall
   through to "no filter for feature_type=…" — the default is to always run).
3. Add a checkbox in
   [web_app/src/components/sidebar/FeaturesSection.tsx](../web_app/src/components/sidebar/FeaturesSection.tsx).
4. The `/extract-features` endpoint accepts the new `feature_type` value
   automatically — no backend route change needed.
