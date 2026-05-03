# Tile Processing

Big rasters (> ~5000×5000 pixels) won't fit in memory if processed in one shot.
Tile mode chops the input into square tiles, classifies each in a separate
process, and stitches the results back together.

---

## Enabling tile mode

In the UI: *Performance* sidebar panel → check **Tile Processing** → pick a
tile size.

Programmatically:

```python
from backend.app.core import classify

classify(
    raster_path="big.tif",
    classes=[...],
    vector_layers=[...],
    smoothing="median_1",
    feature_flags={"spectral": True, "texture": True, "indices": True},
    tile_mode=True,
    tile_max_pixels=512 * 512,
    tile_workers=4,
)
```

The CLI exposes the same flags (`--tile-mode`, `--tile-max-pixels`,
`--tile-workers`).

---

## Tile size selection

The dropdown in the UI is populated from
`POST /suggest-tile-size`. The implementation in
[backend/app/core.py:suggest_tile_size](../backend/app/core.py):

1. Reads `height`, `width`, `bands`, `dtype` from the raster.
2. Computes a per-worker memory budget from available RAM.
3. Picks the largest power-of-2 tile side length from
   `{256, 512, 1024, 2048, 4096}` that fits the budget at ~8× scratch copies
   (GDAL read + feature extraction + normalization + labels + color).
4. Caps to `min(image_height, image_width)`.

Sizes that wouldn't fit at the current worker count are hidden from the
dropdown — the user only sees safe options. The **Auto** entry shows the
suggested size in parentheses, e.g. *Auto (2048×2048)*.

---

## Worker pool

`_classify_tile_worker()` runs in a `ProcessPoolExecutor`. Worker count is:

```
max_workers = min(tile_workers or os.cpu_count(),
                  max_threads if max_threads else infinity)
```

`max_threads` is the global cap (Performance panel → *Limit max threads*).
`tile_workers` is specifically how many tiles to process in parallel.

---

## Overlap & stitching

`_generate_tile_windows()` produces overlapping windows when `overlap > 0`. Each
worker writes its result into a tile-sized GeoTIFF in a temporary `output/`
folder. The driver merges them back into the final raster, blending overlap
regions to avoid edge artefacts.

Default overlap is small (a few pixels) — enough to hide tile seams without
inflating runtime.

---

## Memory math (rule of thumb)

For a 4-band UInt16 image:

```
bytes_per_pixel = 4 bands * 2 bytes * 8 scratch copies = 64 B/pixel
2048 × 2048 tile  ≈ 256 MB working set
4096 × 4096 tile  ≈ 1 GB working set
```

With `tile_workers = 4` you need 4× that simultaneously, plus headroom for the
OS and GDAL caches. `suggest_tile_size` divides usable RAM by worker count
before deciding.

---

## When NOT to use tile mode

Small rasters (< 5000 px on the long edge) — full-image processing is faster
because there's no `ProcessPoolExecutor` startup cost, no stitching, and the
KMeans model trains on a richer pixel sample.

The web app's progress phase weights (in `_PHASE_WEIGHTS`) reflect this:
`Classifying tiles` carries a higher weight than `Pixel assignment`.

---

## Output reprojection / padding

Tile mode does not change the output format. Every classification raster is
still:

- Reprojected to **EPSG:4326** before being written.
- Padded to power-of-2 dimensions on disk.
- Accompanied by `<stem>.xml` (and `.txr` / `.txs` in MEA mode).

Padding happens once at stitch time — individual tiles are not padded.
