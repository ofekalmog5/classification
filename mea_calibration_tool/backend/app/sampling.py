"""Pixel sampling from raster regions, computing mean/std/tolerance_radius."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


def sample_regions(
    raster_path: str,
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sample pixels from each region and compute per-material statistics.

    Each region: {"material": str, "x": int, "y": int, "width": int, "height": int}

    Returns list of:
    {
        "material": str,
        "sample_count": int,
        "mean_rgb": [r, g, b],
        "std_rgb": [r, g, b],
        "reference_color": "#RRGGBB",
        "reference_rgb": [r, g, b],
        "tolerance_radius": int,
        "anchor": [r, g, b],   # mean of this region — one anchor per call
    }
    """
    # Group regions by material
    by_material: Dict[str, List[np.ndarray]] = {}

    try:
        with rasterio.open(raster_path) as src:
            is_palette = src.colorinterp[0] == rasterio.enums.ColorInterp.palette
            if is_palette:
                cmap = src.colormap(1)          # {index: (R, G, B, A)}
                lut = np.zeros((256, 3), dtype=np.float32)
                for idx, (r, g, b, _a) in cmap.items():
                    lut[idx] = [r, g, b]

            band_count = 3 if is_palette else min(src.count, 3)
            bands = list(range(1, (1 if is_palette else band_count) + 1))

            for region in regions:
                win = Window(region["x"], region["y"], region["width"], region["height"])
                if is_palette:
                    raw = src.read(1, window=win).astype(np.uint8)   # (H, W)
                    pixels = lut[raw].reshape(-1, 3)                  # (N, 3) RGB floats
                else:
                    data = src.read(bands, window=win).astype(np.float32)
                    pixels = data.reshape(len(bands), -1).T           # (N, bands)
                by_material.setdefault(region["material"], []).append(pixels)
    except Exception as e:
        raise RuntimeError(f"Failed to read raster: {e}") from e

    results = []
    for mat, pixel_lists in by_material.items():
        all_px = np.vstack(pixel_lists)  # (N, 3)
        mean_rgb = all_px.mean(axis=0).tolist()
        std_rgb = all_px.std(axis=0).tolist()

        r = int(round(mean_rgb[0]))
        g = int(round(mean_rgb[1])) if len(mean_rgb) > 1 else r
        b = int(round(mean_rgb[2])) if len(mean_rgb) > 2 else r
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))

        max_std = max(std_rgb[0], std_rgb[1], std_rgb[2]) if len(std_rgb) >= 3 else std_rgb[0]
        tolerance_radius = max(30, int(round(2 * max_std)))

        results.append({
            "material": mat,
            "sample_count": int(len(all_px)),
            "mean_rgb": [r, g, b],
            "std_rgb": [round(s, 2) for s in std_rgb],
            "reference_color": f"#{r:02X}{g:02X}{b:02X}",
            "reference_rgb": [r, g, b],
            "tolerance_radius": tolerance_radius,
            "anchor": [r, g, b],
        })

    return results
