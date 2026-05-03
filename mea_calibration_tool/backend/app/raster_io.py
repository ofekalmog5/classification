"""Minimal rasterio helpers — no dependency on main app's core.py."""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
import rasterio.warp as _warp
from rasterio.crs import CRS
from rasterio.windows import Window


def raster_info(file_path: str) -> Dict[str, Any]:
    with rasterio.open(file_path) as src:
        crs_str = src.crs.to_string() if src.crs else None
        epsg = src.crs.to_epsg() if src.crs else None
        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": crs_str,
            "epsg": epsg,
            "transform": list(src.transform),
            "bounds": list(src.bounds),
        }


def raster_as_png(file_path: str, max_dim: int = 1024) -> str:
    """Return base64-encoded PNG thumbnail of the raster (RGB or gray)."""
    from PIL import Image

    with rasterio.open(file_path) as src:
        scale = min(max_dim / src.width, max_dim / src.height, 1.0)
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))

        # Palette/indexed raster: apply the colormap to get true RGB
        if src.colorinterp[0] == rasterio.enums.ColorInterp.palette:
            raw = src.read(
                1, out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.nearest,
            ).astype(np.uint8)
            cmap = src.colormap(1)          # {index: (R, G, B, A)}
            lut = np.zeros((256, 3), dtype=np.uint8)
            for idx, (r, g, b, _a) in cmap.items():
                lut[idx] = [r, g, b]
            img = Image.fromarray(lut[raw], mode="RGB")

        else:
            bands = min(src.count, 3)
            data = src.read(
                list(range(1, bands + 1)),
                out_shape=(bands, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            out = np.zeros_like(data, dtype=np.uint8)
            for b in range(bands):
                p2, p98 = np.percentile(data[b], (2, 98))
                span = max(p98 - p2, 1.0)
                out[b] = np.clip((data[b] - p2) / span * 255, 0, 255).astype(np.uint8)
            if bands == 1:
                img = Image.fromarray(out[0], mode="L")
            else:
                img = Image.fromarray(np.transpose(out, (1, 2, 0)), mode="RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def geo_to_raster(
    file_path: str,
    lat_top: float,
    lng_left: float,
    lat_bot: float,
    lng_right: float,
) -> Dict[str, int]:
    """Convert geographic lat/lng bounds to raster pixel coordinates."""
    with rasterio.open(file_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            xs, ys = _warp.transform(
                CRS.from_epsg(4326), src.crs,
                [lng_left, lng_right], [lat_top, lat_bot],
            )
        else:
            xs = [lng_left, lng_right]
            ys = [lat_top, lat_bot]

        row0, col0 = src.index(xs[0], ys[0])
        row1, col1 = src.index(xs[1], ys[1])

        r0 = max(0, min(row0, row1))
        c0 = max(0, min(col0, col1))
        r1 = min(src.height, max(row0, row1))
        c1 = min(src.width, max(col0, col1))
        return {"x": c0, "y": r0, "width": max(1, c1 - c0), "height": max(1, r1 - r0)}


def list_dir(directory: str) -> List[Dict[str, Any]]:
    p = Path(directory)
    entries = []
    for item in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        entries.append({
            "name": item.name,
            "path": str(item),
            "is_dir": item.is_dir(),
            "size": item.stat().st_size if item.is_file() else 0,
        })
    return entries
