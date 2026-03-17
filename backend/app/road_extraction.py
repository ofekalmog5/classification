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

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

import numpy as np
import rasterio
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# SAM 3 model loading
# ---------------------------------------------------------------------------

_sam_model = None       # lazy singleton
_sam_model_type = None  # "sam3" | "langsam" | "owlv2sam2"


def _load_sam3(device: str = "auto"):
    """Load a text-prompted segmentation model (singleton).

    Priority order:
      1. SamGeo3  — SAM 3  (Linux/CUDA, needs triton)
      2. LangSAM  — SAM 2 + GroundingDINO  (needs compatible transformers)
      3. OWLv2+SAM2 — pure HuggingFace, works everywhere, no compilation

    Returns (model_bundle, model_type).
    """
    global _sam_model, _sam_model_type
    if _sam_model is not None:
        return _sam_model, _sam_model_type

    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for road extraction.\n"
            "  GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "  CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        ) from None

    if device == "auto":
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"

    import sys as _sys

    # ── 1. Try SAM3 ────────────────────────────────────────────────────────
    # On Windows, triton may not be installed. We try three paths:
    #   a) triton-windows package is installed — works natively
    #   b) inject a MagicMock for triton so sam3 can import (inference only)
    #   c) neither works — fall through to LangSAM / OWLv2+SAM2
    if _sys.platform == "win32":
        try:
            import triton  # noqa: F401  (triton-windows)
            print("[RoadExtract] triton-windows found — attempting SAM3")
        except ImportError:
            # Inject a minimal mock so sam3 can import on Windows.
            # SAM3 uses triton only for JIT-compiled CUDA kernels; at inference
            # time PyTorch's built-in SDPA is used instead, so the mock is safe.
            try:
                from unittest.mock import MagicMock as _MM
                _triton_mock = _MM()
                for _mod in ("triton", "triton.language", "triton.runtime",
                             "triton.runtime.jit", "triton.ops",
                             "triton.compiler", "triton.backends"):
                    _sys.modules.setdefault(_mod, _MM())
                print("[RoadExtract] triton mocked for Windows SAM3 inference")
            except Exception:
                pass
    try:
        from samgeo import SamGeo3
        model = SamGeo3(device=device)
        _sam_model = model
        _sam_model_type = "sam3"
        print("[RoadExtract] Loaded SAM 3 model")
        return _sam_model, _sam_model_type
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("triton", "sam3", "no module named", "importerror")):
            print(f"[RoadExtract] SAM3 unavailable: {e}")
        else:
            print(f"[RoadExtract] SAM3 failed unexpectedly: {e}")
        print("[RoadExtract] Trying LangSAM …")

    # ── 2. Try LangSAM (GroundingDINO + SAM2) ──────────────────────────────
    try:
        from samgeo.text_sam import LangSAM
        model = LangSAM(model_type="sam2-hiera-large")
        _sam_model = model
        _sam_model_type = "langsam"
        print("[RoadExtract] Loaded LangSAM (SAM 2 + GroundingDINO)")
        return _sam_model, _sam_model_type
    except Exception as e:
        print(f"[RoadExtract] LangSAM failed: {e}")
        print("[RoadExtract] Trying OWLv2 + SAM2 (pure HuggingFace) …")

    # ── 3. OWLv2 + SAM2 via HuggingFace transformers ──────────────────────
    # Requires only: pip install transformers (+ torch already installed)
    try:
        from transformers import pipeline as _hf_pipeline
        detector = _hf_pipeline(
            "zero-shot-object-detection",
            model="google/owlv2-base-patch16-ensemble",
            device=0 if device == "cuda" else -1,
        )
        segmenter = _hf_pipeline(
            "mask-generation",
            model="facebook/sam2-hiera-large",
            device=0 if device == "cuda" else -1,
        )
        _sam_model = {"detector": detector, "segmenter": segmenter}
        _sam_model_type = "owlv2sam2"
        print("[RoadExtract] Loaded OWLv2 + SAM2 (HuggingFace)")
        return _sam_model, _sam_model_type
    except Exception as e:
        raise ImportError(
            f"All road extraction backends failed. Last error: {e}\n\n"
            "Make sure these are installed:\n"
            "  pip install torch torchvision transformers segment-geospatial\n"
            "  (GPU): --index-url https://download.pytorch.org/whl/cu121"
        ) from e


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

    # Handle output path - ensure it's a valid .tif file path
    if output_path is None:
        output_path = str(inp.with_name(f"{inp.stem}_roads.tif"))
    else:
        outp = Path(output_path)
        if outp.is_dir():
            # Directory provided - create filename inside it
            output_path = str(outp / f"{inp.stem}_roads.tif")
        elif not outp.suffix or outp.suffix.lower() not in ['.tif', '.tiff']:
            # No extension or wrong extension - add .tif
            output_path = str(outp.with_suffix('.tif'))
        else:
            output_path = str(outp)

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
    sam, sam_type = _load_sam3(device)

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

                    # --- Run SAM text-prompted segmentation ---
                    mask_tile = _segment_tile(sam, sam_type, tile_rgb, text_prompt)
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


def _segment_tile(sam, sam_type: str, tile_rgb: np.ndarray, text_prompt: str) -> np.ndarray:
    """Run text-prompted segmentation on a single tile.

    Supports sam3, langsam, and owlv2sam2.
    Returns a binary mask (H, W) with 1 = road, 0 = background.
    """
    h, w = tile_rgb.shape[:2]

    if sam_type == "owlv2sam2":
        return _segment_tile_owlv2(sam, tile_rgb, text_prompt)

    # samgeo-based backends need file I/O
    tmp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_output = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
    tmp_input_path = tmp_input.name
    tmp_output_path = tmp_output.name
    tmp_input.close()
    tmp_output.close()

    try:
        if _CV2_AVAILABLE:
            cv2.imwrite(tmp_input_path, cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image as _PILImage
            _PILImage.fromarray(tile_rgb).save(tmp_input_path)

        if sam_type == "langsam":
            sam.predict(
                tmp_input_path,
                text_prompt,
                box_threshold=0.24,
                text_threshold=0.24,
                output=tmp_output_path,
            )
        else:
            # SamGeo3 API
            sam.set_image(tmp_input_path)
            sam.text_prompt = text_prompt
            sam.generate(output=tmp_output_path)

        if Path(tmp_output_path).exists():
            if _CV2_AVAILABLE:
                mask = cv2.imread(tmp_output_path, cv2.IMREAD_GRAYSCALE)
            else:
                from PIL import Image as _PILImage
                mask = np.array(_PILImage.open(tmp_output_path).convert("L"))
            if mask is not None:
                return (mask > 0).astype(np.uint8)

        return np.zeros((h, w), dtype=np.uint8)

    finally:
        for p in (tmp_input_path, tmp_output_path):
            try:
                Path(p).unlink()
            except OSError:
                pass


def _segment_tile_owlv2(sam_bundle: dict, tile_rgb: np.ndarray, text_prompt: str) -> np.ndarray:
    """Segment tile using OWLv2 (detection) + SAM2 (masks) via HuggingFace.

    Returns binary mask (H, W).
    """
    from PIL import Image as _PILImage

    h, w = tile_rgb.shape[:2]
    pil_image = _PILImage.fromarray(tile_rgb)
    detector = sam_bundle["detector"]
    segmenter = sam_bundle["segmenter"]

    # OWLv2: detect boxes matching the text prompt
    # Expects list of candidate labels
    labels = [p.strip() for p in text_prompt.replace(",", " ").split() if p.strip()]
    detections = detector(pil_image, candidate_labels=labels, threshold=0.1)

    if not detections:
        return np.zeros((h, w), dtype=np.uint8)

    # SAM2: generate masks from detected boxes
    boxes = [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]
             for d in detections]

    outputs = segmenter(pil_image, points_per_batch=32, input_boxes=[boxes])
    masks = outputs.get("masks", [])

    if not masks:
        return np.zeros((h, w), dtype=np.uint8)

    # Union all masks
    combined = np.zeros((h, w), dtype=np.uint8)
    for mask in masks:
        arr = np.array(mask)
        if arr.ndim == 3:
            arr = arr[0]
        combined = np.logical_or(combined, arr > 0).astype(np.uint8)

    return combined


def _morphological_close(
    raw_mask_path: str,
    output_path: str,
    profile: dict,
    kernel_size: int,
) -> None:
    """Apply morphological closing to bridge small gaps in the road mask.

    Processes in horizontal strips to handle very large rasters without OOM.
    Falls back to scipy binary_closing if cv2 is not installed.
    """
    if _CV2_AVAILABLE:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
    else:
        from scipy.ndimage import generate_binary_structure as _gbs
        kernel = None  # used as flag; scipy path below
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

                if _CV2_AVAILABLE:
                    closed = cv2.morphologyEx(data, cv2.MORPH_CLOSE, kernel)
                else:
                    from scipy.ndimage import binary_closing as _bc
                    import numpy as _np
                    disk = _np.zeros((kernel_size, kernel_size), dtype=bool)
                    cx = kernel_size // 2
                    Y, X = _np.ogrid[:kernel_size, :kernel_size]
                    disk[(X - cx) ** 2 + (Y - cx) ** 2 <= cx ** 2] = True
                    closed = _bc(data.astype(bool), structure=disk).astype(np.uint8)

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

    # Handle output path - ensure it's a valid .tif file path
    if output_path is None:
        output_path = str(cls_path.with_name(f"{cls_path.stem}_roads_merged.tif"))
    else:
        outp = Path(output_path)
        if outp.is_dir():
            # Directory provided - create filename inside it
            output_path = str(outp / f"{cls_path.stem}_roads_merged.tif")
        elif not outp.suffix or outp.suffix.lower() not in ['.tif', '.tiff']:
            # No extension or wrong extension - add .tif
            output_path = str(outp.with_suffix('.tif'))
        else:
            output_path = str(outp)

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
