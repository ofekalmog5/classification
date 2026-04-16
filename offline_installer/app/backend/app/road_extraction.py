"""
Road extraction from GeoTIFFs using SAM 3 (Segment Anything Model 3).

Supports multiple backends (tried in order):
  0. Local SAM3  — Meta's SAM 3 from a local directory
  1. SamGeo3    — segment-geospatial wrapper
  2. LangSAM    — SAM 2 + GroundingDINO
  3. OWLv2+SAM2 — pure HuggingFace, works everywhere

All backends work fully offline once models are downloaded.

Standalone usage:
    python road_extraction.py input.tif -o roads_mask.tif

Integration:
    from road_extraction import extract_roads, merge_road_mask_onto_classification
"""

from __future__ import annotations

import argparse
import math
import os
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
# Configuration — all paths/settings can be overridden at runtime
# ---------------------------------------------------------------------------

# For fully-offline standalone stations, set these env vars BEFORE launching:
#   set HF_HUB_OFFLINE=1
#   set TRANSFORMERS_OFFLINE=1
# This forces HuggingFace to use only cached models (no network requests).
# On a dev machine with internet, leave them unset so models can download.

# ---------------------------------------------------------------------------
# SAM 3 model loading
# ---------------------------------------------------------------------------

_sam_model = None       # lazy singleton
_sam_model_type = None  # "sam3_local" | "sam3" | "langsam" | "owlv2sam2"

# Path to locally cloned SAM3 repo (Meta's SAM 3).
# Can be overridden via:
#   - Environment variable  SAM3_LOCAL_DIR
#   - Calling  set_sam3_local_dir(path)  at runtime
#   - POST /set-sam3-path  API endpoint
_sam3_local_dir: Optional[Path] = None

def _resolve_sam3_dir() -> Optional[Path]:
    """Return the SAM3 local directory, checking env var and default paths."""
    if _sam3_local_dir is not None:
        return _sam3_local_dir

    # Check persistent config
    try:
        from . import config as _cfg
        saved = _cfg.get("sam3_local_dir")
        if saved:
            p = Path(saved)
            if p.exists() and (p / "sam3").is_dir():
                return p
    except Exception:
        pass

    # Check environment variable
    env = os.environ.get("SAM3_LOCAL_DIR")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # Default paths to search (portable — works on any station)
    _project_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        _project_root / "sam3-main",   # sibling of project root
        _project_root / "sam3",        # sibling of project root
    ]
    # When frozen, also check next to the exe
    if getattr(sys, "frozen", False):
        _exe_dir = Path(sys.executable).parent
        candidates.insert(0, _exe_dir / "sam3-main")
        candidates.insert(1, _exe_dir / "sam3")

    for c in candidates:
        if c.exists() and (c / "sam3").is_dir():
            return c

    return None


def set_sam3_local_dir(path: str | Path | None):
    """Set (or clear) the local SAM3 directory at runtime.

    Persists to app_config.json so the setting survives restarts.
    Clears the cached model so the next extraction re-loads from the new path.
    """
    global _sam3_local_dir
    _sam3_local_dir = Path(path) if path else None
    reset_model_cache()
    # Persist to config file
    try:
        from . import config as _cfg
        _cfg.save({"sam3_local_dir": str(path) if path else None})
    except Exception:
        pass


def get_road_extract_config() -> dict:
    """Return current road-extraction configuration."""
    sam3_dir = _resolve_sam3_dir()
    sam3_ckpt = None
    if sam3_dir:
        ckpt = sam3_dir / "sam3.pt"
        sam3_ckpt = str(ckpt) if ckpt.exists() else None

    # Check HuggingFace cache for OWLv2+SAM2 models
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    owlv2_cached = (hf_cache / "models--google--owlv2-base-patch16-ensemble").exists()
    sam2_cached = (hf_cache / "models--facebook--sam2-hiera-large").exists()

    return {
        "sam3LocalDir": str(sam3_dir) if sam3_dir else None,
        "sam3CheckpointFound": sam3_ckpt is not None,
        "sam3CheckpointPath": sam3_ckpt,
        "loadedBackend": _sam_model_type,
        "offlineMode": os.environ.get("HF_HUB_OFFLINE") == "1",
        "hfCacheDir": str(hf_cache),
        "owlv2Cached": owlv2_cached,
        "sam2Cached": sam2_cached,
    }


def reset_model_cache():
    """Clear the cached SAM model singleton (e.g. after code changes)."""
    global _sam_model, _sam_model_type
    _sam_model = None
    _sam_model_type = None


def _load_sam3(device: str = "auto"):
    """Load a text-prompted segmentation model (singleton).

    Priority order:
      0. Local SAM3 — Meta's SAM 3 from local directory (needs checkpoint)
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

    # ── 0. Try local SAM3 (Meta's repo) ─────────────────────────────────
    sam3_dir = _resolve_sam3_dir()
    if sam3_dir is not None:
        try:
            # Add sam3 repo to path so we can import it
            sam3_str = str(sam3_dir)
            if sam3_str not in _sys.path:
                _sys.path.insert(0, sam3_str)

            # Mock triton if on Windows (SAM3 needs it for imports but not inference)
            if _sys.platform == "win32":
                try:
                    import triton  # noqa: F401
                except ImportError:
                    from unittest.mock import MagicMock as _MM
                    for _mod in ("triton", "triton.language", "triton.runtime",
                                 "triton.runtime.jit", "triton.ops",
                                 "triton.compiler", "triton.backends"):
                        _sys.modules.setdefault(_mod, _MM())

            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            # Look for checkpoint — local file or HuggingFace cache
            ckpt_path = sam3_dir / "sam3.pt"
            if not ckpt_path.exists():
                # Check HuggingFace cache
                try:
                    from huggingface_hub import try_to_load_from_cache
                    cached = try_to_load_from_cache("facebook/sam3", "sam3.pt")
                    if cached and Path(cached).exists():
                        ckpt_path = Path(cached)
                    else:
                        raise FileNotFoundError("No local SAM3 checkpoint found")
                except Exception:
                    raise FileNotFoundError(
                        f"SAM3 checkpoint not found. Place sam3.pt in {sam3_dir} "
                        "or run: huggingface-cli login && huggingface-cli download facebook/sam3 sam3.pt"
                    )

            print(f"[RoadExtract] Loading local SAM3 from {ckpt_path} …")
            model = build_sam3_image_model(
                device=device,
                eval_mode=True,
                checkpoint_path=str(ckpt_path),
                load_from_HF=False,
            )
            processor = Sam3Processor(model, device=device)
            _sam_model = {"processor": processor, "device": device}
            _sam_model_type = "sam3_local"
            print("[RoadExtract] Loaded local SAM 3 (Meta)")
            return _sam_model, _sam_model_type
        except Exception as e:
            print(f"[RoadExtract] Local SAM3 failed: {e}")

    # ── 1. Try SamGeo3 ──────────────────────────────────────────────────
    # On Windows, triton may not be installed. We try three paths:
    #   a) triton-windows package is installed — works natively
    #   b) inject a MagicMock for triton so sam3 can import (inference only)
    #   c) neither works — fall through to LangSAM / OWLv2+SAM2
    if _sys.platform == "win32":
        try:
            import triton  # noqa: F401  (triton-windows)
            print("[RoadExtract] triton-windows found — attempting SamGeo3")
        except ImportError:
            try:
                from unittest.mock import MagicMock as _MM
                for _mod in ("triton", "triton.language", "triton.runtime",
                             "triton.runtime.jit", "triton.ops",
                             "triton.compiler", "triton.backends"):
                    _sys.modules.setdefault(_mod, _MM())
                print("[RoadExtract] triton mocked for Windows SamGeo3 inference")
            except Exception:
                pass
    try:
        from samgeo import SamGeo3
        model = SamGeo3(device=device)
        _sam_model = model
        _sam_model_type = "sam3"
        print("[RoadExtract] Loaded SamGeo3 model")
        return _sam_model, _sam_model_type
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("triton", "sam3", "no module named", "importerror",
                                   "401 client error", "gated repo", "restricted")):
            print(f"[RoadExtract] SamGeo3 unavailable: {e}")
        else:
            print(f"[RoadExtract] SamGeo3 failed unexpectedly: {e}")
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
    # NOTE: We use Sam2Processor + Sam2Model (not the mask-generation pipeline)
    # because the pipeline doesn't accept input_boxes for prompted segmentation.
    try:
        import os as _os
        from pathlib import Path as _Path
        import torch as _torch
        from transformers import pipeline as _hf_pipeline
        from transformers import Sam2Processor, Sam2Model

        def _resolve_hf_model(model_id: str) -> str:
            """Return local snapshot path when it exists (works with old or new HF cache
            format).  Falls back to the hub model ID for online download."""
            hf_home = _Path(_os.environ.get("HF_HOME", _Path.home() / ".cache" / "huggingface"))
            cache_name = "models--" + model_id.replace("/", "--")
            snap_dir = hf_home / "hub" / cache_name / "snapshots"
            if snap_dir.exists():
                snaps = sorted(snap_dir.iterdir())
                if snaps:
                    local_path = str(snaps[-1])
                    print(f"[RoadExtract] Using local snapshot for {model_id}: {local_path}")
                    return local_path
            return model_id

        _owlv2_path = _resolve_hf_model("google/owlv2-base-patch16-ensemble")
        _sam2_path  = _resolve_hf_model("facebook/sam2-hiera-large")

        _dev = 0 if device == "cuda" else -1
        detector = _hf_pipeline(
            "zero-shot-object-detection",
            model=_owlv2_path,
            device=_dev,
        )
        sam2_processor = Sam2Processor.from_pretrained(_sam2_path)
        sam2_model = Sam2Model.from_pretrained(_sam2_path)
        sam2_model.to(device)
        sam2_model.eval()
        _sam_model = {
            "detector": detector,
            "sam2_processor": sam2_processor,
            "sam2_model": sam2_model,
            "device": device,
        }
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

    # Output goes into a _roads/ subfolder
    if output_path is None:
        road_dir = inp.parent / "_roads"
    else:
        outp = Path(output_path)
        road_dir = (outp if outp.is_dir() else outp.parent) / "_roads"
    road_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(road_dir / f"{inp.stem}_roads.tif")

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
                    try:
                        mask_tile = _segment_tile(sam, sam_type, tile_rgb, text_prompt)
                    except Exception as tile_err:
                        print(f"[RoadExtract] Tile {idx+1}/{total_tiles} failed: {tile_err}")
                        mask_tile = np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
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


def _segment_tile(
    sam,
    sam_type: str,
    tile_rgb: np.ndarray,
    text_prompt: str,
    threshold: float = 0.05,
) -> np.ndarray:
    """Run text-prompted segmentation on a single tile.

    Supports sam3_local, sam3, langsam, and owlv2sam2.
    Returns a binary mask (H, W) with 1 = feature, 0 = background.
    """
    h, w = tile_rgb.shape[:2]

    if sam_type == "sam3_local":
        return _segment_tile_sam3_local(sam, tile_rgb, text_prompt)

    if sam_type == "owlv2sam2":
        return _segment_tile_owlv2(sam, tile_rgb, text_prompt, threshold=threshold)

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
            if Path(tmp_output_path).exists():
                if _CV2_AVAILABLE:
                    mask = cv2.imread(tmp_output_path, cv2.IMREAD_GRAYSCALE)
                else:
                    from PIL import Image as _PILImage
                    mask = np.array(_PILImage.open(tmp_output_path).convert("L"))
                if mask is not None:
                    return (mask > 0).astype(np.uint8)
        else:
            # SamGeo3: set_image → generate_masks (sets sam.masks as side effect)
            sam.set_image(tmp_input_path)
            sam.generate_masks(text_prompt, quiet=True)
            print(f"    [SamGeo3] found {len(sam.masks)} mask(s) for prompt={text_prompt!r}")
            if len(sam.masks) > 0:
                combined = np.zeros((h, w), dtype=np.uint8)
                for m in sam.masks:
                    arr = np.asarray(m)
                    # Handle (1, H, W) or (H, W) shapes
                    if arr.ndim == 3:
                        arr = arr[0]
                    # Resize to tile dimensions if needed
                    if arr.shape != (h, w):
                        from PIL import Image as _PILImg
                        arr = np.array(_PILImg.fromarray(arr.astype(np.uint8)).resize(
                            (w, h), resample=_PILImg.NEAREST))
                    combined |= (arr > 0).astype(np.uint8)
                return combined

        return np.zeros((h, w), dtype=np.uint8)

    finally:
        for p in (tmp_input_path, tmp_output_path):
            try:
                Path(p).unlink()
            except OSError:
                pass


def _segment_tile_sam3_local(sam_bundle: dict, tile_rgb: np.ndarray, text_prompt: str) -> np.ndarray:
    """Segment tile using local Meta SAM3 model with text prompts.

    Returns binary mask (H, W).
    """
    from PIL import Image as _PILImage

    h, w = tile_rgb.shape[:2]
    pil_image = _PILImage.fromarray(tile_rgb)
    processor = sam_bundle["processor"]

    # SAM3 API: set_image → set_text_prompt → read masks
    state = processor.set_image(pil_image)
    state = processor.set_text_prompt(text_prompt, state)

    masks = state.get("masks")  # shape: (N, 1, H, W) boolean tensor
    if masks is None or masks.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # Union all detected masks
    combined = masks.squeeze(1).any(dim=0).cpu().numpy().astype(np.uint8)
    return combined


def _segment_tile_owlv2(
    sam_bundle: dict,
    tile_rgb: np.ndarray,
    text_prompt: str,
    threshold: float = 0.05,
) -> np.ndarray:
    """Segment tile using OWLv2 (detection) + SAM2 (masks) via HuggingFace.

    Uses Sam2Processor + Sam2Model with input_boxes for prompted segmentation
    (the mask-generation pipeline does NOT support input_boxes).

    Returns binary mask (H, W).
    """
    import torch as _torch
    from PIL import Image as _PILImage

    h, w = tile_rgb.shape[:2]
    pil_image = _PILImage.fromarray(tile_rgb)
    detector = sam_bundle["detector"]
    sam2_processor = sam_bundle["sam2_processor"]
    sam2_model = sam_bundle["sam2_model"]
    device = sam_bundle["device"]

    # OWLv2: detect boxes matching the text prompt
    # Split by commas to keep multi-word labels intact
    # e.g. "road, street, highway" → ["road", "street", "highway"]
    labels = [p.strip() for p in text_prompt.split(",") if p.strip()]
    raw_detections = detector(pil_image, candidate_labels=labels, threshold=threshold)

    # Pipeline may return [[{...}]] (batched) or [{...}] (single image)
    if raw_detections and isinstance(raw_detections[0], list):
        detections = raw_detections[0]
    else:
        detections = raw_detections

    print(f"    [OWLv2] {len(detections)} detection(s) for prompt={text_prompt!r} "
          f"(top scores: {sorted([d['score'] for d in detections], reverse=True)[:5]})")

    if not detections:
        return np.zeros((h, w), dtype=np.uint8)

    # Collect detected boxes as [[xmin, ymin, xmax, ymax], ...]
    boxes = [[d["box"]["xmin"], d["box"]["ymin"], d["box"]["xmax"], d["box"]["ymax"]]
             for d in detections]

    # SAM2: prompted segmentation using detected boxes
    # input_boxes must be nested: [[[x0,y0,x1,y1], ...]] (batch=1, image boxes)
    inputs = sam2_processor(
        images=pil_image,
        input_boxes=[boxes],
        return_tensors="pt",
    ).to(device)

    with _torch.no_grad():
        outputs = sam2_model(**inputs)

    # Post-process masks back to original image size
    pred_masks = sam2_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )

    if not pred_masks or len(pred_masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    # pred_masks[0] shape: (num_boxes, num_predictions_per_box, H, W)
    # Union all predicted masks
    combined = np.zeros((h, w), dtype=np.uint8)
    masks_tensor = pred_masks[0]  # first (and only) image in batch
    if hasattr(masks_tensor, "numpy"):
        masks_arr = masks_tensor.numpy()
    else:
        masks_arr = np.array(masks_tensor)

    # Flatten all mask predictions and union them
    for box_masks in masks_arr:
        if box_masks.ndim >= 2:
            # Take the highest-confidence mask per box (first prediction)
            mask_2d = box_masks[0] if box_masks.ndim == 3 else box_masks
            combined = np.logical_or(combined, mask_2d > 0).astype(np.uint8)

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

    # If the classification path is a directory (tile output), process each
    # tile file individually and return the list of merged tile outputs.
    if cls_path.is_dir():
        tile_files = sorted(
            p for p in cls_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".img")
        )
        if not tile_files:
            return {"status": "error", "message": f"No raster tiles found in directory: {cls_path}"}
        print(f"[merge_road_mask] Directory input — processing {len(tile_files)} tiles")
        merged_outputs: list[str] = []
        for i, tf in enumerate(tile_files):
            if progress_callback:
                progress_callback(f"Merging road mask (tile {i+1}/{len(tile_files)})", i, len(tile_files))
            tile_result = merge_road_mask_onto_classification(
                classification_path=str(tf),
                road_mask_path=road_mask_path,
                output_path=None,  # auto-generate per tile
                asphalt_color=asphalt_color,
                progress_callback=None,
            )
            if tile_result.get("status") == "ok":
                merged_outputs.append(tile_result["outputPath"])
            else:
                print(f"  [warn] Tile {tf.name} merge failed: {tile_result.get('message')}")
        if progress_callback:
            progress_callback("Merging road mask", len(tile_files), len(tile_files))
        merged_dir = cls_path / "_merged"
        return {
            "status": "ok",
            "outputPath": str(merged_dir),
            "tileOutputs": merged_outputs,
            "message": f"Road mask merged onto {len(merged_outputs)}/{len(tile_files)} tiles",
        }

    # Output goes into a _merged/ subfolder
    if output_path is None:
        merge_dir = cls_path.parent / "_merged"
    else:
        outp = Path(output_path)
        merge_dir = (outp if outp.is_dir() else outp.parent) / "_merged"
    merge_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(merge_dir / f"{cls_path.stem}_roads_merged.tif")

    from rasterio.windows import from_bounds as window_from_bounds
    import numpy as np

    with rasterio.open(classification_path) as cls_src:
        profile = cls_src.profile.copy()
        width = cls_src.width
        height = cls_src.height
        cls_transform = cls_src.transform
        cls_crs = cls_src.crs
        cls_bounds = cls_src.bounds

    # Force clean 3-band uint8 RGB — strip inherited photometric/predictor
    profile.update(
        driver="GTiff", dtype="uint8", count=3,
        compress="deflate", photometric="rgb",
    )
    profile.pop("predictor", None)

    strip_h = 2048
    total_strips = math.ceil(height / strip_h)

    with rasterio.open(classification_path) as cls_src, \
         rasterio.open(road_mask_path) as mask_src, \
         rasterio.open(output_path, "w", **profile) as dst:

        # Check if mask and classification share the same grid (full-image case)
        # or if we need to use geographic alignment (tile case).
        _same_grid = (
            mask_src.width == cls_src.width
            and mask_src.height == cls_src.height
            and mask_src.transform == cls_src.transform
        )

        for strip_idx in range(total_strips):
            if progress_callback:
                progress_callback("Merging road mask", strip_idx, total_strips)

            row_start = strip_idx * strip_h
            h = min(strip_h, height - row_start)
            cls_win = Window(0, row_start, width, h)

            # Read exactly 3 RGB bands from classification strip
            src_bands = min(cls_src.count, 3)
            raw = cls_src.read(list(range(1, src_bands + 1)), window=cls_win)
            if src_bands < 3:
                rgb = np.zeros((3, h, width), dtype=np.uint8)
                rgb[:src_bands] = raw
            else:
                rgb = raw[:3].astype(np.uint8)

            if _same_grid:
                # Same dimensions — read mask with same pixel window
                mask_band = mask_src.read(1, window=cls_win)
            else:
                # Geographic alignment: compute the bounds of this classification
                # strip and read the corresponding area from the road mask.
                strip_transform = rasterio.windows.transform(cls_win, cls_transform)
                strip_bounds = rasterio.transform.array_bounds(h, width, strip_transform)
                # left, bottom, right, top
                try:
                    mask_win = window_from_bounds(
                        *strip_bounds, transform=mask_src.transform,
                    )
                    # Clamp to mask extent
                    mask_win = mask_win.intersection(
                        Window(0, 0, mask_src.width, mask_src.height)
                    )
                    raw_mask = mask_src.read(1, window=mask_win)
                    # Resize to match the classification strip if needed
                    if raw_mask.shape != (h, width):
                        from rasterio.enums import Resampling
                        mask_band = np.zeros((h, width), dtype=raw_mask.dtype)
                        rasterio.warp.reproject(
                            source=raw_mask,
                            destination=mask_band,
                            src_transform=rasterio.windows.transform(mask_win, mask_src.transform),
                            src_crs=mask_src.crs,
                            dst_transform=strip_transform,
                            dst_crs=cls_crs,
                            resampling=Resampling.nearest,
                        )
                    else:
                        mask_band = raw_mask
                except Exception as e:
                    # Strip falls outside mask extent — no roads here
                    print(f"  [merge] Strip {strip_idx}: no mask overlap ({e})")
                    mask_band = np.zeros((h, width), dtype=np.uint8)

            road_pixels = mask_band > 0

            # Apply exact asphalt color where mask is 1
            for band_idx in range(3):
                rgb[band_idx][road_pixels] = np.uint8(asphalt_color[band_idx])

            dst.write(rgb[:3], window=cls_win)

    if progress_callback:
        progress_callback("Done", total_strips, total_strips)

    return {"status": "ok", "outputPath": output_path}


# ---------------------------------------------------------------------------
# Feature extraction configs (prompt + merge color per sub-type)
# Colors are RGB tuples matching MEA class hex codes
# ---------------------------------------------------------------------------
# Color + geometry based detectors (complement OWLv2 which struggles overhead)
# ---------------------------------------------------------------------------

def _color_detect_buildings(tile_rgb: np.ndarray) -> np.ndarray:
    """Detect building roofs by color ranges + rectangular geometry.

    OWLv2 struggles to recognise rooftops from orthophoto nadir view.
    This function catches concrete, tile, greenhouse, and metal roofs
    that the text-prompted model misses.

    Returns binary mask (H, W) uint8 {0, 1}.
    """
    if not _CV2_AVAILABLE:
        return np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
    h, w = tile_rgb.shape[:2]
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # --- Roof color masks (HSV, OpenCV ranges: H 0-179, S 0-255, V 0-255) ---
    # Concrete / gray roofs: low saturation, moderate brightness
    concrete = (S < 40) & (V > 90) & (V < 215)
    # Orange / red tile roofs: warm hue, saturated
    tile = ((H < 15) | (H > 170)) & (S > 50) & (V > 80)
    # White / bright roofs (greenhouses, painted): very bright, low sat
    white = (V > 210) & (S < 35)
    # Metal roofs: slight blue-gray, moderate brightness
    metal = (S < 35) & (V > 120) & (V < 200)

    candidates = (concrete | tile | white | metal).astype(np.uint8)

    # Morphological clean-up
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, k_open)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    min_area = max(300, h * w * 0.0015)   # ~0.15 % of tile
    max_area = h * w * 0.35               # not more than 35 % of tile

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.55:         # buildings are compact
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 7:              # too elongated → road / fence
            continue
        extent = area / max(bw * bh, 1)
        if extent < 0.35:           # buildings fill their bounding box
            continue
        cv2.drawContours(result, [cnt], -1, 1, cv2.FILLED)
    return result


def _color_detect_trees(tile_rgb: np.ndarray) -> np.ndarray:
    """Detect tree canopy regions by dark-green color.

    Trees from above appear as dark-to-medium green blobs.
    Returns binary mask (H, W) uint8 {0, 1}.
    """
    if not _CV2_AVAILABLE:
        return np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
    h, w = tile_rgb.shape[:2]
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Dark-to-medium green (tree canopy, not bright lawn)
    green_mask = (H > 25) & (H < 90) & (S > 30) & (V > 25) & (V < 170)
    green = green_mask.astype(np.uint8)

    # Clean-up: open to remove speckle, close to merge canopy
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k_open)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, k_close)

    # Keep only clusters (≥ a few trees together)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    min_area = max(400, h * w * 0.002)    # cluster of trees, not a single bush

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 1, cv2.FILLED)
    return result


def _color_detect_water(tile_rgb: np.ndarray) -> np.ndarray:
    """Detect water bodies using multi-strategy color + texture analysis.

    Water from orthophoto nadir view comes in many appearances:
      - Deep blue (sea, deep lakes)
      - Dark blue-green (fish ponds, reservoirs)
      - Turquoise / cyan (treated pools, shallow coastal)
      - Green-tinted (algae, stagnant water, fish pools)
      - Dark / near-black (deep shadow on water, deep reservoirs)
      - Muddy brown-gray (rivers after rain, turbid ponds)
      - Gray-silver (reflective surfaces, overcast sky reflection)
      - Teal / muted blue-green (estuaries, irrigation channels)

    Uses a combination of:
      1) HSV color ranges for different water types
      2) RGB ratio analysis for blue/green dominance
      3) Local texture smoothness (water is uniformly smooth from above)
      4) Adaptive thresholding to catch low-contrast water

    Returns binary mask (H, W) uint8 {0, 1}.
    """
    if not _CV2_AVAILABLE:
        return np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
    h, w = tile_rgb.shape[:2]

    # ---- Convert to useful color spaces ----
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    r = tile_rgb[:, :, 0].astype(np.int16)
    g = tile_rgb[:, :, 1].astype(np.int16)
    b = tile_rgb[:, :, 2].astype(np.int16)
    brightness = (r + g + b) // 3

    # =========================================================
    # Strategy 1: HSV-based color detection (wide coverage)
    # =========================================================
    # OpenCV HSV: H 0-179, S 0-255, V 0-255

    # 1a) Blue water — classic lake/sea/pool (H ~100-130)
    blue_water = (H > 90) & (H < 135) & (S > 25) & (V > 20) & (V < 230)

    # 1b) Deep blue water — darker, more saturated (deep sea/reservoir)
    deep_blue = (H > 95) & (H < 140) & (S > 50) & (V > 15) & (V < 160)

    # 1c) Cyan / turquoise — treated pools, shallow tropical water
    cyan_water = (H > 75) & (H < 100) & (S > 30) & (V > 60) & (V < 230)

    # 1d) Green-blue water — algae-rich ponds, fish pools, eutrophic lakes
    #     Distinguish from vegetation: water has B closer to G (muted green-blue),
    #     vegetation has G >> B (pure saturated green).
    green_water = (H > 55) & (H < 95) & (S > 20) & (V > 25) & (V < 180) & \
                  (b > g * 3 // 10)   # B >= 30% of G → rules out pure green vegetation

    # 1e) Dark teal — irrigation channels, wastewater, shaded water
    dark_teal = (H > 70) & (H < 120) & (S > 15) & (V > 10) & (V < 90)

    # 1f) Muddy/turbid water — brownish (rivers, ponds after rain)
    #     Low hue (brown-yellow), low saturation, medium-low brightness
    muddy = (H > 8) & (H < 30) & (S > 15) & (S < 100) & (V > 30) & (V < 140)

    # 1g) Gray reflective water — overcast sky reflection, silvery surface
    #     Requires slight blue tint (b >= r) to avoid matching gray asphalt/concrete.
    #     Water reflection always has slight blue shift compared to neutral gray surfaces.
    gray_water = (S < 25) & (V > 50) & (V < 170) & (b > r)

    hsv_mask = (blue_water | deep_blue | cyan_water | green_water |
                dark_teal).astype(np.uint8)

    # =========================================================
    # Strategy 2: RGB ratio-based detection
    # =========================================================

    # 2a) Blue-dominant: B channel is strongest
    blue_dom = (b > r + 8) & (b > g - 15) & (b > 30)

    # 2b) Green-blue dominant: B and G both exceed R → water (not vegetation)
    #     Require B > R to ensure actual blue component (rejects pure green plants)
    gb_dom = (b > r) & (g > r) & ((g + b) > (r * 2 + 30)) & (b > 30)

    # 2c) Dark water: low total brightness, blue slightly exceeds red
    #     Require B > R (not just >=) to reject neutral-gray dark asphalt/shadow
    dark_water = (brightness < 70) & (b > r + 2) & (b > 15)

    # 2d) Medium-dark water (wider catch): moderate brightness, not red-ish
    med_dark_water = (brightness > 30) & (brightness < 110) & (b >= r - 10) & \
                     (b > g - 20) & (S > 10)

    # 2e) Turquoise / aquamarine pools: high G+B, low R
    turquoise = (g > 70) & (b > 60) & (r < g - 8) & (r < b + 10)

    rgb_mask = (blue_dom | gb_dom | dark_water | turquoise).astype(np.uint8)

    # =========================================================
    # Strategy 3: Texture smoothness (water is uniform from above)
    # =========================================================
    # Convert to grayscale and measure local standard deviation
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)

    # Valid-pixel mask — exclude black/nodata background (R+G+B < 15)
    valid_pixels = (brightness > 5).astype(np.uint8)

    # Local std dev over 11×11 window — water will have very low variance
    blur = cv2.GaussianBlur(gray.astype(np.float32), (11, 11), 0)
    blur_sq = cv2.GaussianBlur((gray.astype(np.float32)) ** 2, (11, 11), 0)
    local_std = np.sqrt(np.maximum(blur_sq - blur ** 2, 0))

    # Smooth regions: std dev < 12 (water is very uniform)
    smooth_mask = (local_std < 12).astype(np.uint8)
    # Exclude nodata/black from smooth mask
    smooth_mask = cv2.bitwise_and(smooth_mask, valid_pixels)

    # Texture-assisted water: smooth AND has clear blue dominance
    # (strict — prevents false positives on smooth concrete/asphalt/vegetation)
    water_color_hint = (
        # Blue must actually exceed red (not just close) + not too bright
        (b > r + 5) & (b > 30) & (brightness > 10) & (brightness < 200)
    ).astype(np.uint8)
    texture_water = cv2.bitwise_and(smooth_mask, water_color_hint)

    # =========================================================
    # Strategy 4: Combined muddy/gray water with texture confirmation
    # =========================================================
    # Gray and muddy water are ambiguous — only accept them when texture
    # confirms smoothness (avoids false positives on roads/buildings)
    ambiguous_color = (muddy | gray_water | med_dark_water).astype(np.uint8)
    ambiguous_color = cv2.bitwise_and(ambiguous_color, valid_pixels)
    # Require stronger smoothness for ambiguous colors
    strong_smooth = (local_std < 8).astype(np.uint8)
    strong_smooth = cv2.bitwise_and(strong_smooth, valid_pixels)
    confirmed_ambiguous = cv2.bitwise_and(ambiguous_color, strong_smooth)

    # =========================================================
    # Combine all strategies
    # =========================================================
    combined = np.maximum(hsv_mask, rgb_mask)
    combined = np.maximum(combined, texture_water)
    combined = np.maximum(combined, confirmed_ambiguous)

    # Mask out nodata/black background pixels
    combined = cv2.bitwise_and(combined, valid_pixels)

    # =========================================================
    # Morphological cleanup
    # =========================================================
    # Open: remove small noise (3×3)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)

    # Close: fill small gaps within water bodies (21×21 — water bodies are large)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    # Re-mask after closing to prevent bleeding into nodata
    combined = cv2.bitwise_and(combined, valid_pixels)

    # Dilate slightly to capture edges where water meets land
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.dilate(combined, k_dilate, iterations=1)

    # Final nodata mask
    combined = cv2.bitwise_and(combined, valid_pixels)

    # =========================================================
    # Contour filtering — keep only genuine water bodies
    # =========================================================
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    # Lower min area to catch small pools and narrow channels
    min_area = max(200, h * w * 0.001)    # 0.1% of tile (was 0.3%)
    max_area = h * w * 0.95               # almost entire tile is OK for sea/lake

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        if area > max_area:
            continue
        # For medium-sized blobs, verify they have water-like color distribution
        # (reject false positives from roads / gray buildings)
        if area < h * w * 0.01:
            # Small blob — check that average color is water-like
            blob_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)
            mean_r = np.mean(tile_rgb[:, :, 0][blob_mask > 0])
            mean_g = np.mean(tile_rgb[:, :, 1][blob_mask > 0])
            mean_b = np.mean(tile_rgb[:, :, 2][blob_mask > 0])
            # Reject if clearly red/orange dominant (rust, roof)
            if mean_r > mean_b + 30 and mean_r > mean_g + 20:
                continue
            # Reject if very bright and unsaturated (white roof/concrete)
            if mean_r > 200 and mean_g > 200 and mean_b > 200:
                continue
        cv2.drawContours(result, [cnt], -1, 1, cv2.FILLED)

    # =========================================================
    # Fill holes inside detected water bodies (islands are rare
    # at tile scale; holes are usually noise)
    # =========================================================
    if result.any():
        flood = result.copy()
        contours_fill, _ = cv2.findContours(flood, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours_fill:
            cv2.drawContours(result, [cnt], -1, 1, cv2.FILLED)

    return result


def _color_detect_fields(tile_rgb: np.ndarray) -> np.ndarray:
    """Detect green fields / agriculture by bright-green color + large uniform area.

    Fields from above: bright green (grass, lawn, crops) or yellow-green
    (dry/ripe crops).  Returns binary mask (H, W) uint8 {0, 1}.
    """
    if not _CV2_AVAILABLE:
        return np.zeros(tile_rgb.shape[:2], dtype=np.uint8)
    h, w = tile_rgb.shape[:2]
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Bright-to-medium green (lawn, crop, pasture) — brighter than tree canopy
    green = (H > 25) & (H < 85) & (S > 20) & (V > 60) & (V < 240)
    # Yellow-green (dry grass, ripe crop)
    yellow_green = (H > 18) & (H < 35) & (S > 25) & (V > 80)

    fields = (green | yellow_green).astype(np.uint8)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    fields = cv2.morphologyEx(fields, cv2.MORPH_OPEN, k_open)
    fields = cv2.morphologyEx(fields, cv2.MORPH_CLOSE, k_close)

    # Keep only large areas (fields are big, not bushes)
    contours, _ = cv2.findContours(fields, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros((h, w), dtype=np.uint8)
    min_area = max(800, h * w * 0.005)    # at least 0.5 % of tile

    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(result, [cnt], -1, 1, cv2.FILLED)
    return result


# ---------------------------------------------------------------------------

FEATURE_CONFIGS: Dict[str, List[Dict]] = {
    # Two detection strategies per feature:
    #   1) OWLv2+SAM text-prompted segmentation (prompt)
    #   2) Color+geometry CV detection (color_detect) — catches what OWLv2 misses
    # Results are unioned (OR) per tile.
    "roads": [
        {
            "prompt": "road, highway, asphalt path",
            "suffix": "roads",
            "color": (45, 45, 48),       # BM_ASPHALT #2D2D30
        },
    ],
    "buildings": [
        {
            "prompt": "building, house, roof, rooftop, structure",
            "suffix": "buildings",
            "color": (180, 180, 180),     # BM_CONCRETE #B4B4B4
            "threshold": 0.02,
            "color_detect": _color_detect_buildings,
        },
    ],
    "trees": [
        {
            "prompt": "tree, trees, forest, woodland, grove",
            "suffix": "trees",
            "color": (34, 139, 34),       # BM_VEGETATION #228B22
            "color_detect": _color_detect_trees,
        },
    ],
    "fields": [
        {
            "prompt": "grass, lawn, field, meadow, pasture",
            "suffix": "fields_grass",
            "color": (124, 252, 0),       # BM_LAND_GRASS #7CFC00
            "color_detect": _color_detect_fields,
        },
        {
            "prompt": "crop, farmland, agriculture, cultivated field",
            "suffix": "fields_agriculture",
            "color": (189, 183, 107),     # BM_LAND_DRY_GRASS #BDB76B
            "color_detect": _color_detect_fields,
        },
    ],
    "water": [
        {
            "prompt": "water, lake, pond, reservoir, pool",
            "suffix": "water_bodies",
            "color": (28, 107, 160),      # BM_WATER #1C6BA0
            "threshold": 0.03,            # lower threshold — water is hard for OWLv2
            "color_detect": _color_detect_water,
        },
        {
            "prompt": "river, stream, canal, waterway, channel",
            "suffix": "water_channels",
            "color": (28, 107, 160),      # BM_WATER #1C6BA0
            "threshold": 0.03,
            "color_detect": _color_detect_water,
        },
        {
            "prompt": "sea, ocean, fish pond, swimming pool",
            "suffix": "water_other",
            "color": (28, 107, 160),      # BM_WATER #1C6BA0
            "threshold": 0.03,
            "color_detect": _color_detect_water,
        },
    ],
}


def should_extract_feature(raster_path: str, feature_type: str) -> tuple:
    """Quick RGB analysis to decide whether to run SAM extraction on this image.

    Reads a small thumbnail (≤256×256 overview) — fast even on huge rasters.
    Returns (should_run: bool, reason: str).

    Thresholds (fraction of valid pixels matching the feature signature):
      roads     >5 %  gray, moderate brightness, R≈B (rejects sand)
      buildings >3 %  gray roof | red/tile roof (R>>B) | bright metal roof
      trees     >5 %  dark green (G dominates, low brightness)
      fields    >8 %  green or yellowish-green
    """
    try:
        from rasterio.enums import Resampling as _R
        with rasterio.open(raster_path) as src:
            target = 256
            scale = max(src.width, src.height) / target
            out_h = max(1, int(src.height / scale))
            out_w = max(1, int(src.width / scale))
            bands = min(src.count, 3)
            data = src.read(
                list(range(1, bands + 1)),
                out_shape=(bands, out_h, out_w),
                resampling=_R.average,
            ).astype(float)
    except Exception as exc:
        return True, f"pre-filter skipped (read error: {exc})"

    if data.shape[0] < 3:
        return True, "pre-filter skipped (< 3 bands)"

    r, g, b = data[0], data[1], data[2]
    valid = (r > 5) | (g > 5) | (b > 5)
    n_valid = int(valid.sum())
    if n_valid < 100:
        return True, "pre-filter skipped (too few valid pixels)"

    r, g, b = r[valid], g[valid], b[valid]
    brightness = (r + g + b) / 3.0
    sat = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)

    if feature_type == "roads":
        # Step 1: Quick gray pixel ratio — roads must have enough neutral gray pixels.
        # True gray: low saturation, moderate brightness, R≈B (rejects sandy/yellow soil)
        road_like = (sat < 30) & (np.abs(r - b) < 20) & (brightness > 30) & (brightness < 175)
        ratio = float(road_like.sum()) / n_valid
        if ratio <= 0.03:
            return False, f"no roads detected (gray ratio={ratio:.1%}, threshold=3%)"

        # Step 2: Linearity check — roads are thin LINEAR features; plowed fields,
        # shadows and bare soil are large amorphous blobs.  Build a 2D road-candidate
        # mask from the original spatial data and run Hough line detection.
        try:
            import cv2 as _cv2
            r2d, g2d, b2d = data[0], data[1], data[2]
            br2d = (r2d + g2d + b2d) / 3.0
            sa2d = np.maximum(np.maximum(r2d, g2d), b2d) - np.minimum(np.minimum(r2d, g2d), b2d)
            road_mask2d = (
                (sa2d < 30) & (np.abs(r2d - b2d) < 20) & (br2d > 30) & (br2d < 175)
            ).astype(np.uint8) * 255

            min_dim = min(road_mask2d.shape)
            # Minimum line segment length: 7% of thumbnail dimension (~18px on 256-px thumb)
            min_seg = max(8, int(min_dim * 0.07))
            lines = _cv2.HoughLinesP(
                road_mask2d, rho=1, theta=np.pi / 180,
                threshold=12, minLineLength=min_seg, maxLineGap=10,
            )
            if lines is None:
                return False, f"gray pixels ({ratio:.1%}) but no linear road structure — likely open terrain/field"
            # Any linear structure found — pass. The total-length gate was too
            # strict and rejected images where roads are narrow or cross a corner.
            total_len = float(sum(
                np.hypot(l[0][2] - l[0][0], l[0][3] - l[0][1]) for l in lines
            ))
            return True, f"road lines: {len(lines)} segments, total={total_len:.0f}px, gray={ratio:.1%}"
        except Exception:
            pass  # cv2 unavailable — fall back to ratio-only pass

        return True, f"gray pixel ratio={ratio:.1%}"

    elif feature_type == "buildings":
        # Gray/flat/concrete roofs: low saturation, any brightness
        gray_roof = (sat < 40) & (brightness > 40) & (brightness < 240)
        # Red/orange tile roofs: R clearly dominates (terracotta, clay — very common in orthophotos)
        tile_roof = (r > g + 20) & (r > b + 35) & (brightness > 50) & (brightness < 230)
        # Bright metal/industrial roofs: very bright, low saturation
        metal_roof = (sat < 25) & (brightness > 160)
        struct_like = gray_roof | tile_roof | metal_roof
        ratio = float(struct_like.sum()) / n_valid
        if ratio > 0.01:
            return True, f"structure pixel ratio={ratio:.1%}"
        return False, f"no buildings detected (structure ratio={ratio:.1%}, threshold=1%)"

    elif feature_type == "trees":
        # Dark green — G clearly above R and B, not too bright (canopy shadow)
        # Also catch medium-green canopy (greenness>8) for sparse tree coverage
        greenness = g - np.maximum(r, b)
        tree_like = (greenness > 8) & (brightness < 170)
        ratio = float(tree_like.sum()) / n_valid
        if ratio > 0.03:
            return True, f"tree pixel ratio={ratio:.1%}"
        return False, f"no trees detected (dark-green ratio={ratio:.1%}, threshold=5%)"

    elif feature_type == "fields":
        # Green fields: G dominant over R and B; dry fields: yellowish (R+G)/2 >> B
        green_field = (g > r + 5) & (g > b + 10) & (brightness > 40)
        dry_field = ((r + g) / 2 - b > 15) & (brightness > 40) & (brightness < 220) & (g > b)
        field_like = green_field | dry_field
        ratio = float(field_like.sum()) / n_valid
        if ratio > 0.05:
            return True, f"field pixel ratio={ratio:.1%}"
        return False, f"no fields detected (green/yellow ratio={ratio:.1%}, threshold=5%)"

    elif feature_type == "water":
        # Water signature: blue-dominant, dark-blue, green-blue, or dark/low-brightness
        # Very low threshold — we want to run extraction even if only a small portion is water
        blue_dom = (b > r + 5) & (b > g - 15) & (b > 20)
        dark_blue = (b >= r) & (brightness < 90) & (b > 15)
        green_blue = (g > r) & (b > r) & (brightness < 150)
        # Also catch very dark areas (deep water can appear almost black)
        very_dark = (brightness < 40)
        water_like = blue_dom | dark_blue | green_blue | very_dark
        ratio = float(water_like.sum()) / n_valid
        if ratio > 0.005:   # only 0.5% — very permissive, let color_detect decide
            return True, f"water pixel ratio={ratio:.1%}"
        return False, f"no water detected (water ratio={ratio:.1%}, threshold=0.5%)"

    return True, f"no filter for feature_type={feature_type!r}"


def extract_feature_masks(
    input_path: str,
    output_path: str | None = None,
    feature_type: str = "roads",
    tile_size: int = 1024,
    overlap_pct: float = 0.1,
    closing_kernel_size: int = 15,
    device: str = "auto",
    progress_callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Extract one or more feature masks from a GeoTIFF using SAM text prompts.

    Parameters
    ----------
    feature_type : str
        One of ``"roads"``, ``"buildings"``, ``"vegetation"``.
        Vegetation produces 4 masks (foliage / dry grass / grass / vegetation).

    Returns
    -------
    dict
        ``{"status": "ok", "maskPaths": [...], "colors": [(r,g,b), ...]}``
    """
    configs = FEATURE_CONFIGS.get(feature_type)
    if configs is None:
        return {"status": "error", "message": f"Unknown feature_type: {feature_type!r}"}

    inp = Path(input_path)
    n = len(configs)
    mask_paths: List[str] = []
    colors: List[Tuple[int, int, int]] = []

    # Load SAM model once, reuse for all sub-prompts
    if progress_callback:
        progress_callback("Loading SAM model", 0, n * 100 + 2)
    sam, sam_type = _load_sam3(device)

    for i, cfg in enumerate(configs):
        suffix = cfg["suffix"]
        color = cfg["color"]
        prompt = cfg["prompt"]
        thresh = cfg.get("threshold", 0.05)
        border_ext = cfg.get("border_extend", 0)
        color_detect_fn = cfg.get("color_detect", None)

        # Output goes into a _<feature_type>/ subfolder
        if output_path is None:
            feature_dir = inp.parent / f"_{feature_type}"
        else:
            outp = Path(output_path)
            feature_dir = (outp if outp.is_dir() else outp.parent) / f"_{feature_type}"
        feature_dir.mkdir(parents=True, exist_ok=True)
        out = str(feature_dir / f"{inp.stem}_{suffix}.tif")

        def _cb(phase: str, done: int, total: int, _i: int = i, _n: int = n) -> None:
            if progress_callback:
                progress_callback(phase, _i * 100 + done, _n * 100 + 2)

        result = _run_single_extraction(
            input_path=input_path,
            output_path=out,
            text_prompt=prompt,
            sam=sam,
            sam_type=sam_type,
            tile_size=tile_size,
            overlap_pct=overlap_pct,
            closing_kernel_size=closing_kernel_size,
            threshold=thresh,
            extend_borders_px=border_ext,
            color=color,
            progress_callback=_cb,
            color_detect_fn=color_detect_fn,
        )
        if result.get("status") == "ok":
            mask_paths.append(result["outputPath"])
            colors.append(color)
        elif result.get("status") == "empty":
            print(f"[extract_feature_masks] {suffix}: no features found — no raster exported")
        else:
            print(f"[extract_feature_masks] Sub-extraction {suffix} failed: {result.get('message')}")

    if not mask_paths:
        return {"status": "error", "message": f"All {feature_type} sub-extractions failed"}

    if progress_callback:
        progress_callback("Done", n * 100 + 2, n * 100 + 2)

    return {"status": "ok", "maskPaths": mask_paths, "colors": colors}


def _extend_at_borders(mask_path: str, border_px: int = 60, kernel_px: int = 20) -> None:
    """Dilate road pixels within the image border zone to bridge cross-image gaps.

    When a road crosses from one orthophoto tile to an adjacent one, SAM may
    detect it right up to the edge in both images but leave a visual gap because
    detection doesn't reach the very last pixels.  This function dilates the
    road mask ONLY inside the border region so that detected roads always reach
    the image edge — without creating false roads in the image interior.
    """
    with rasterio.open(mask_path) as src:
        data = src.read(1)  # (H, W) uint8
        profile = src.profile.copy()

    h, w = data.shape
    b = min(border_px, min(h, w) // 6)  # cap so we don't eat into small images

    # Border zone mask (all four edges)
    border_zone = np.zeros((h, w), dtype=bool)
    border_zone[:b, :]  = True
    border_zone[-b:, :] = True
    border_zone[:, :b]  = True
    border_zone[:, -b:] = True

    # Only bother if roads exist near a border
    if not np.any(data[border_zone] > 0):
        return

    # Dilate the full mask, then keep interior pixels unchanged
    if _CV2_AVAILABLE:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_px, kernel_px))
        dilated = cv2.dilate(data, kernel)
    else:
        from scipy.ndimage import binary_dilation as _bd
        dilated = _bd(data > 0, iterations=kernel_px // 2).astype(np.uint8)

    result = data.copy()
    result[border_zone] = np.maximum(data[border_zone], dilated[border_zone])

    tmp = mask_path + "._bext.tif"
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.write(result[np.newaxis, :, :])
        import shutil as _shutil
        _shutil.move(tmp, mask_path)
    except Exception:
        try:
            Path(tmp).unlink()
        except OSError:
            pass


def _colorize_mask(mask_path: str, color: Tuple[int, int, int]) -> None:
    """Convert a 1-band binary mask GeoTIFF to a 3-band RGB GeoTIFF in-place.

    Feature pixels (value > 0) are painted with *color*; background stays (0,0,0).
    This makes the mask file directly viewable with the correct material color.
    """
    tmp = mask_path + "._rgb_tmp.tif"
    try:
        with rasterio.open(mask_path) as src:
            binary = src.read(1)
            profile = src.profile.copy()
        # Force clean RGB profile — no inherited photometric/predictor
        # that could alter pixel values on write.
        profile.update(count=3, dtype="uint8", nodata=None, photometric="rgb")
        profile.pop("predictor", None)
        with rasterio.open(tmp, "w", **profile) as dst:
            for band_idx, c in enumerate(color):
                dst.write(
                    np.where(binary > 0, np.uint8(c), np.uint8(0)),
                    band_idx + 1,
                )
        import shutil as _shutil
        _shutil.move(tmp, mask_path)
    except Exception as e:
        print(f"[colorize_mask] Failed to colorize {mask_path}: {e}")
        try:
            Path(tmp).unlink(missing_ok=True)
        except OSError:
            pass


def _run_single_extraction(
    input_path: str,
    output_path: str,
    text_prompt: str,
    sam,
    sam_type: str,
    tile_size: int = 1024,
    overlap_pct: float = 0.1,
    closing_kernel_size: int = 15,
    threshold: float = 0.05,
    extend_borders_px: int = 0,
    color: Tuple[int, int, int] = (255, 255, 255),
    progress_callback: Optional[Callable] = None,
    color_detect_fn: Optional[Callable] = None,
) -> Dict[str, object]:
    """Run a single SAM text-prompted mask extraction (reuses an already-loaded model).

    If *color_detect_fn* is provided, it runs on each tile and its result is
    unioned (OR) with the SAM mask — this catches features that OWLv2 misses.

    Returns ``{"status": "ok", "outputPath": ...}`` when features are found,
    or ``{"status": "empty"}`` when the mask is entirely blank (no raster written).
    """
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height

    profile.update(driver="GTiff", dtype="uint8", count=1, compress="deflate", nodata=0)
    overlap = max(1, int(tile_size * overlap_pct))
    tiles = _generate_tiles(width, height, tile_size, overlap)
    total_tiles = len(tiles)

    has_any_feature = False
    tmp_mask_path = output_path + ".tmp_raw.tif"
    try:
        with rasterio.open(tmp_mask_path, "w", **profile) as dst:
            with rasterio.open(input_path) as src:
                for idx, (read_win, write_win, inner_col, inner_row) in enumerate(tiles):
                    if progress_callback:
                        progress_callback("Extracting", idx, total_tiles)
                    bands_to_read = min(src.count, 3)
                    tile_data = src.read(list(range(1, bands_to_read + 1)), window=read_win)
                    tile_rgb = np.moveaxis(tile_data[:3], 0, -1).astype(np.uint8)
                    try:
                        mask_tile = _segment_tile(sam, sam_type, tile_rgb, text_prompt, threshold=threshold)
                    except Exception as tile_err:
                        print(f"[Extract] Tile {idx+1}/{total_tiles} failed: {tile_err}")
                        mask_tile = np.zeros(tile_rgb.shape[:2], dtype=np.uint8)

                    # Union with color+geometry detection (catches what OWLv2 misses)
                    if color_detect_fn is not None:
                        try:
                            color_mask = color_detect_fn(tile_rgb)
                            mask_tile = np.logical_or(mask_tile > 0, color_mask > 0).astype(np.uint8)
                        except Exception as cd_err:
                            print(f"[Extract] Color detect tile {idx+1} failed: {cd_err}")

                    if mask_tile.any():
                        has_any_feature = True
                    w_h = write_win.height
                    w_w = write_win.width
                    inner_mask = mask_tile[inner_row: inner_row + w_h, inner_col: inner_col + w_w]
                    dst.write(inner_mask.astype(np.uint8)[np.newaxis, :, :], window=write_win)

        # No features detected anywhere — skip raster export
        if not has_any_feature:
            print(f"[Extract] No features found for prompt={text_prompt!r} — skipping raster export")
            return {"status": "empty", "message": f"No features found for: {text_prompt}"}

        if progress_callback:
            progress_callback("Closing", total_tiles, total_tiles + 1)
        _morphological_close(tmp_mask_path, output_path, profile, closing_kernel_size)
    finally:
        tmp = Path(tmp_mask_path)
        if tmp.exists():
            tmp.unlink()

    # Extend road pixels at image borders so roads connect across adjacent images
    if extend_borders_px > 0:
        _extend_at_borders(output_path, border_px=extend_borders_px)

    # Convert binary mask to RGB so the file shows the actual material color
    _colorize_mask(output_path, color)

    if progress_callback:
        progress_callback("Done", total_tiles + 1, total_tiles + 1)
    return {"status": "ok", "outputPath": output_path}


def merge_feature_masks_onto_classification(
    classification_path: str,
    mask_paths: List[str],
    colors: List[Tuple[int, int, int]],
    output_path: str | None = None,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, object]:
    """Merge one or more feature masks onto a classification GeoTIFF.

    Masks are applied in order — later masks override earlier ones where they overlap.
    Each mask is painted with its corresponding RGB color.

    Parameters
    ----------
    classification_path : str
        Path to the RGB classification GeoTIFF (or directory of tiles).
    mask_paths : list of str
        Paths to binary mask GeoTIFFs (1 = feature, 0 = background).
    colors : list of (r, g, b) tuples
        RGB color for each mask (same order as mask_paths).
    output_path : str, optional
        Output path. Defaults to ``<classification_stem>_features_merged.tif``.
    """
    cls_path = Path(classification_path)

    # Handle tiled classification directory
    if cls_path.is_dir():
        tile_files = sorted(
            p for p in cls_path.iterdir()
            if p.is_file() and p.suffix.lower() in (".tif", ".tiff", ".img")
        )
        if not tile_files:
            return {"status": "error", "message": f"No raster tiles found in: {cls_path}"}
        print(f"[merge_features] Directory — processing {len(tile_files)} tiles")
        merged_outputs: list[str] = []
        for i, tf in enumerate(tile_files):
            if progress_callback:
                progress_callback(f"Merging tile {i+1}/{len(tile_files)}", i, len(tile_files))
            r = merge_feature_masks_onto_classification(
                classification_path=str(tf),
                mask_paths=mask_paths,
                colors=colors,
                output_path=None,
                progress_callback=None,
            )
            if r.get("status") == "ok":
                merged_outputs.append(r["outputPath"])
            else:
                print(f"  [warn] Tile {tf.name} failed: {r.get('message')}")
        if progress_callback:
            progress_callback("Done", len(tile_files), len(tile_files))
        merged_dir = cls_path / "_merged"
        return {"status": "ok", "outputPath": str(merged_dir), "tileOutputs": merged_outputs}

    # Output goes into a _merged/ subfolder
    if output_path is None:
        merge_dir = cls_path.parent / "_merged"
    else:
        outp = Path(output_path)
        merge_dir = (outp if outp.is_dir() else outp.parent) / "_merged"
    merge_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(merge_dir / f"{cls_path.stem}_features_merged.tif")

    from rasterio.windows import from_bounds as window_from_bounds
    import numpy as np

    with rasterio.open(classification_path) as cls_src:
        profile = cls_src.profile.copy()
        width = cls_src.width
        height = cls_src.height
        cls_transform = cls_src.transform
        cls_crs = cls_src.crs

    # Force clean 3-band uint8 RGB output — strip inherited photometric/predictor
    profile.update(
        driver="GTiff", dtype="uint8", count=3,
        compress="deflate", photometric="rgb",
    )
    profile.pop("predictor", None)

    strip_h = 2048
    total_strips = math.ceil(height / strip_h)

    def _read_feature_binary(mask_src, window) -> np.ndarray:
        """Read a mask (1-band binary or 3-band RGB) and return a 2-D bool array."""
        if mask_src.count >= 3:
            bands = mask_src.read([1, 2, 3], window=window)
            return bands.sum(axis=0) > 0
        return mask_src.read(1, window=window) > 0

    # Pre-open all mask files
    mask_srcs = [rasterio.open(mp) for mp in mask_paths]
    try:
        same_grids = [
            ms.width == width and ms.height == height and ms.transform == cls_transform
            for ms in mask_srcs
        ]

        with rasterio.open(classification_path) as cls_src, \
             rasterio.open(output_path, "w", **profile) as dst:

            for strip_idx in range(total_strips):
                if progress_callback:
                    progress_callback("Merging", strip_idx, total_strips)

                row_start = strip_idx * strip_h
                h = min(strip_h, height - row_start)
                cls_win = Window(0, row_start, width, h)
                # Read exactly 3 RGB bands
                src_bands = min(cls_src.count, 3)
                raw = cls_src.read(list(range(1, src_bands + 1)), window=cls_win)
                if src_bands < 3:
                    rgb = np.zeros((3, h, width), dtype=np.uint8)
                    rgb[:src_bands] = raw
                else:
                    rgb = raw[:3].astype(np.uint8)

                for mask_src, color, same_grid in zip(mask_srcs, colors, same_grids):
                    if same_grid:
                        feature_pixels = _read_feature_binary(mask_src, cls_win)
                    else:
                        strip_transform = rasterio.windows.transform(cls_win, cls_transform)
                        strip_bounds = rasterio.transform.array_bounds(h, width, strip_transform)
                        try:
                            mask_win = window_from_bounds(*strip_bounds, transform=mask_src.transform)
                            mask_win = mask_win.intersection(Window(0, 0, mask_src.width, mask_src.height))
                            # Collapse RGB or binary mask to a single binary band for reprojection
                            if mask_src.count >= 3:
                                raw_bands = mask_src.read([1, 2, 3], window=mask_win)
                                raw_binary = (raw_bands.sum(axis=0) > 0).astype(np.uint8)
                            else:
                                raw_binary = mask_src.read(1, window=mask_win)
                            if raw_binary.shape != (h, width):
                                from rasterio.enums import Resampling
                                mask_band = np.zeros((h, width), dtype=np.uint8)
                                rasterio.warp.reproject(
                                    source=raw_binary,
                                    destination=mask_band,
                                    src_transform=rasterio.windows.transform(mask_win, mask_src.transform),
                                    src_crs=mask_src.crs,
                                    dst_transform=strip_transform,
                                    dst_crs=cls_crs,
                                    resampling=Resampling.nearest,
                                )
                                feature_pixels = mask_band > 0
                            else:
                                feature_pixels = raw_binary > 0
                        except Exception:
                            feature_pixels = np.zeros((h, width), dtype=bool)

                    for band_idx in range(3):
                        rgb[band_idx][feature_pixels] = np.uint8(color[band_idx])

                dst.write(rgb[:3], window=cls_win)
    finally:
        for ms in mask_srcs:
            ms.close()

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
