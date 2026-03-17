"""
Road extraction from GeoTIFFs using SAM 3 (Segment Anything Model 3)
via the samgeo (segment-geospatial) library.

Standalone usage:
    python road_extraction.py input.tif -o roads_mask.tif

Integration:
    from road_extraction import extract_roads, merge_road_mask_onto_classification
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# SAM 3 model loading
# ---------------------------------------------------------------------------

_sam_model = None  # lazy singleton


def _load_sam3(device: str = "auto"):
    """Load SamGeo3 model (singleton)."""
    global _sam_model
    if _sam_model is not None:
        return _sam_model

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from samgeo import SamGeo3

    _sam_model = SamGeo3(device=device)
    return _sam_model


# ---------------------------------------------------------------------------
# Tile generation helpers
# ---------------------------------------------------------------------------

def _generate_tiles(
    width: int,
    height: int,
    tile_size: int = 1024,
    overlap: int = 102,
) -> List[Tuple[Window, Window]]:
    """Generate (read_window, write_window) pairs covering the full raster.

    *read_window* includes overlap margins for context.
    *write_window* is the inner (non-overlapping) region that gets written to
    the output — guaranteeing no duplicate pixels.
    """
    tiles: List[Tuple[Window, Window]] = []

    for row_off in range(0, height, tile_size - overlap):
        for col_off in range(0, width, tile_size - overlap):
            # --- write region (inner, no overlap) ---
            w_col = col_off
            w_row = row_off
            w_width = min(tile_size - overlap, width - col_off)
            w_height = min(tile_size - overlap, height - row_off)

            if w_width <= 0 or w_height <= 0:
                continue

            # --- read region (expanded by overlap on all sides) ---
            r_col = max(col_off - overlap // 2, 0)
            r_row = max(row_off - overlap // 2, 0)
            r_right = min(col_off + tile_size - overlap + overlap // 2, width)
            r_bottom = min(row_off + tile_size - overlap + overlap // 2, height)

            read_win = Window(r_col, r_row, r_right - r_col, r_bottom - r_row)

            # Offset of the write region inside the read window
            inner_col = w_col - r_col
            inner_row = w_row - r_row

            write_win = Window(w_col, w_row, w_width, w_height)

            tiles.append((read_win, write_win, inner_col, inner_row))

    return tiles


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_roads(
    input_path: str,
    output_path: str | None = None,
    text_prompt: str = "road, highway, asphalt path",
    tile_size: int = 1024,
    overlap_pct: float = 0.1,
    closing_kernel_size: int = 15,
    device: str = "auto",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Extract roads from a GeoTIFF using SAM 3 text-prompted segmentation.

    Parameters
    ----------
    input_path : str
        Path to the input GeoTIFF (any CRS / resolution).
    output_path : str, optional
        Path for the output binary mask GeoTIFF.
        Defaults to ``<input_stem>_roads.tif``.
    text_prompt : str
        Open-vocabulary prompt for SAM 3.
    tile_size : int
        Tile side length in pixels (default 1024).
    overlap_pct : float
        Fractional overlap between tiles (default 0.1 = 10 %).
    closing_kernel_size : int
        Diameter of the disk kernel for morphological closing (default 15).
    device : str
        ``"auto"``, ``"cuda"``, or ``"cpu"``.
    progress_callback : callable, optional
        ``callback(phase, done, total)`` for progress reporting.

    Returns
    -------
    dict  ``{"status": "ok", "outputPath": "..."}``
    """
    input_path = str(input_path)
    inp = Path(input_path)

    if output_path is None:
        output_path = str(inp.with_name(f"{inp.stem}_roads.tif"))
    output_path = str(output_path)

    # --- open source raster metadata ---
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform

    # --- prepare output profile (single-band uint8) ---
    profile.update(
        driver="GTiff",
        dtype="uint8",
        count=1,
        compress="deflate",
        nodata=0,
    )

    overlap = max(1, int(tile_size * overlap_pct))
    tiles = _generate_tiles(width, height, tile_size, overlap)
    total_tiles = len(tiles)

    if progress_callback:
        progress_callback("Loading SAM 3 model", 0, total_tiles + 2)

    # --- load model ---
    sam = _load_sam3(device)

    if progress_callback:
        progress_callback("Loading SAM 3 model", 1, total_tiles + 2)

    # --- process tiles into a temporary raw mask ---
    tmp_mask_path = output_path + ".tmp_raw.tif"
    try:
        with rasterio.open(tmp_mask_path, "w", **profile) as dst:
            with rasterio.open(input_path) as src:
                for idx, (read_win, write_win, inner_col, inner_row) in enumerate(tiles):
                    if progress_callback:
                        progress_callback("Extracting roads", idx, total_tiles)

                    # Read RGB tile (bands 1-3)
                    bands_to_read = min(src.count, 3)
                    tile_data = src.read(
                        list(range(1, bands_to_read + 1)),
                        window=read_win,
                    )  # shape: (C, H, W)

                    # Convert to HWC uint8 for SAM
                    tile_rgb = np.moveaxis(tile_data[:3], 0, -1).astype(np.uint8)

                    # --- Run SAM 3 text-prompted segmentation ---
                    mask_tile = _segment_tile(sam, tile_rgb, text_prompt)
                    # mask_tile: (H, W) binary uint8 {0, 1}

                    # Crop to write region (inner portion)
                    w_h = write_win.height
                    w_w = write_win.width
                    inner_mask = mask_tile[
                        inner_row : inner_row + w_h,
                        inner_col : inner_col + w_w,
                    ]

                    # Write to output
                    dst.write(
                        inner_mask.astype(np.uint8)[np.newaxis, :, :],
                        window=write_win,
                    )

        if progress_callback:
            progress_callback("Morphological closing", total_tiles, total_tiles + 2)

        # --- Post-processing: morphological closing on the full mask ---
        _morphological_close(tmp_mask_path, output_path, profile, closing_kernel_size)

    finally:
        # Clean up temp file
        tmp = Path(tmp_mask_path)
        if tmp.exists():
            tmp.unlink()

    if progress_callback:
        progress_callback("Done", total_tiles + 2, total_tiles + 2)

    return {"status": "ok", "outputPath": output_path}


def _segment_tile(sam, tile_rgb: np.ndarray, text_prompt: str) -> np.ndarray:
    """Run SAM 3 text-prompted segmentation on a single tile.

    Returns a binary mask (H, W) with 1 = road, 0 = background.
    """
    h, w = tile_rgb.shape[:2]

    # Save tile to a temporary file (samgeo expects file paths)
    tmp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_output = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_input_path = tmp_input.name
    tmp_output_path = tmp_output.name
    tmp_input.close()
    tmp_output.close()

    try:
        cv2.imwrite(tmp_input_path, cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))

        sam.set_image(tmp_input_path)
        sam.text_prompt = text_prompt
        sam.generate(output=tmp_output_path)

        # Read the generated mask
        if Path(tmp_output_path).exists():
            mask = cv2.imread(tmp_output_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Binarize — any non-zero pixel is road
                binary = (mask > 0).astype(np.uint8)
                return binary

        # If SAM produced no mask, return zeros
        return np.zeros((h, w), dtype=np.uint8)

    finally:
        for p in (tmp_input_path, tmp_output_path):
            try:
                Path(p).unlink()
            except OSError:
                pass


def _morphological_close(
    raw_mask_path: str,
    output_path: str,
    profile: dict,
    kernel_size: int,
) -> None:
    """Apply morphological closing to bridge small gaps in the road mask.

    Processes in horizontal strips to handle very large rasters without OOM.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    pad = kernel_size  # context rows needed above/below each strip

    with rasterio.open(raw_mask_path) as src:
        height = src.height
        width = src.width

        with rasterio.open(output_path, "w", **profile) as dst:
            strip_h = 2048  # rows per strip

            for row_start in range(0, height, strip_h):
                # Expand read region for morphological context
                read_start = max(0, row_start - pad)
                read_end = min(height, row_start + strip_h + pad)
                read_h = read_end - read_start

                win = Window(0, read_start, width, read_h)
                data = src.read(1, window=win)  # (H, W) uint8

                closed = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)

                # Crop back to the inner strip (remove padding)
                inner_top = row_start - read_start
                inner_h = min(strip_h, height - row_start)
                inner = closed[inner_top : inner_top + inner_h, :]

                out_win = Window(0, row_start, width, inner_h)
                dst.write(inner[np.newaxis, :, :], window=out_win)


# ---------------------------------------------------------------------------
# Merge road mask onto classification output
# ---------------------------------------------------------------------------

def merge_road_mask_onto_classification(
    classification_path: str,
    road_mask_path: str,
    output_path: str | None = None,
    asphalt_color: Tuple[int, int, int] = (45, 45, 48),  # BM_ASPHALT #2D2D30
    progress_callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Overlay the road mask as BM_ASPHALT onto a classification GeoTIFF.

    Parameters
    ----------
    classification_path : str
        Path to the RGB classification GeoTIFF.
    road_mask_path : str
        Path to the binary road mask GeoTIFF (1 = road).
    output_path : str, optional
        Output path. Defaults to ``<classification_stem>_roads_merged.tif``.
    asphalt_color : tuple
        RGB color for asphalt pixels (default BM_ASPHALT: 45, 45, 48).
    progress_callback : callable, optional
        ``callback(phase, done, total)``

    Returns
    -------
    dict  ``{"status": "ok", "outputPath": "..."}``
    """
    cls_path = Path(classification_path)
    if output_path is None:
        output_path = str(cls_path.with_name(f"{cls_path.stem}_roads_merged.tif"))
    output_path = str(output_path)

    with rasterio.open(classification_path) as cls_src:
        profile = cls_src.profile.copy()
        width = cls_src.width
        height = cls_src.height

    profile.update(driver="GTiff", compress="deflate")

    strip_h = 2048
    total_strips = math.ceil(height / strip_h)

    with rasterio.open(classification_path) as cls_src, \
         rasterio.open(road_mask_path) as mask_src, \
         rasterio.open(output_path, "w", **profile) as dst:

        for strip_idx in range(total_strips):
            if progress_callback:
                progress_callback("Merging road mask", strip_idx, total_strips)

            row_start = strip_idx * strip_h
            h = min(strip_h, height - row_start)
            win = Window(0, row_start, width, h)

            # Read RGB classification tile (C, H, W)
            rgb = cls_src.read(window=win)

            # Read mask tile (H, W)
            mask_band = mask_src.read(1, window=win)
            road_pixels = mask_band > 0

            # Apply asphalt color where mask is 1
            for band_idx in range(min(rgb.shape[0], 3)):
                rgb[band_idx][road_pixels] = asphalt_color[band_idx]

            dst.write(rgb, window=win)

    if progress_callback:
        progress_callback("Done", total_strips, total_strips)

    return {"status": "ok", "outputPath": output_path}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract roads from a GeoTIFF using SAM 3 text-prompted segmentation."
    )
    parser.add_argument("input", help="Input GeoTIFF path")
    parser.add_argument("-o", "--output", default=None, help="Output mask GeoTIFF path")
    parser.add_argument(
        "--prompt",
        default="road, highway, asphalt path",
        help="Text prompt for SAM 3 (default: 'road, highway, asphalt path')",
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size in pixels (default: 1024)")
    parser.add_argument("--overlap", type=float, default=0.1, help="Overlap fraction (default: 0.1)")
    parser.add_argument("--kernel", type=int, default=15, help="Morphological closing kernel size (default: 15)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device (default: auto)")
    parser.add_argument(
        "--merge",
        default=None,
        metavar="CLASSIFICATION_PATH",
        help="If provided, also merge road mask onto this classification GeoTIFF",
    )

    args = parser.parse_args()

    def _progress(phase, done, total):
        pct = int(done / max(total, 1) * 100)
        print(f"[{pct:3d}%%] {phase} ({done}/{total})")

    print(f"Input:  {args.input}")
    print(f"Prompt: {args.prompt}")
    print(f"Tiles:  {args.tile_size}x{args.tile_size}, overlap {args.overlap:.0%}")
    print()

    result = extract_roads(
        input_path=args.input,
        output_path=args.output,
        text_prompt=args.prompt,
        tile_size=args.tile_size,
        overlap_pct=args.overlap,
        closing_kernel_size=args.kernel,
        device=args.device,
        progress_callback=_progress,
    )

    mask_path = result["outputPath"]
    print(f"\nRoad mask saved to: {mask_path}")

    if args.merge:
        print(f"\nMerging road mask onto: {args.merge}")
        merge_result = merge_road_mask_onto_classification(
            classification_path=args.merge,
            road_mask_path=mask_path,
            progress_callback=_progress,
        )
        print(f"Merged output: {merge_result['outputPath']}")


if __name__ == "__main__":
    main()
