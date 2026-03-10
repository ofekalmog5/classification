from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Literal, Dict, Tuple, Optional
from pathlib import Path
import asyncio
import io
import json as _json
import threading
import numpy as np
from collections import OrderedDict
from queue import Queue, Empty

from .core import classify as run_classify
from .core import classify_and_export as run_classify_and_export
from .core import rasterize_vectors_onto_classification as run_rasterize_vectors
from .core import apply_road_object_removal as run_road_object_removal

# ─── SSE progress streaming infrastructure ───────────────────────────────

_PROGRESS_QUEUES: Dict[str, Queue] = {}

# Per-phase weights reflecting typical relative durations.
# Step 1 phases sum to ~100; other pipelines normalise by total weight.
_PHASE_WEIGHTS: Dict[str, float] = {
    # Step 1 phases
    "Loading raster": 3,
    "Shadow balance": 2,
    "Feature extraction": 20,
    "KMeans clustering": 20,
    "Pixel assignment": 12,
    "Classifying tiles": 35,
    "Shadow detection": 3,
    "Post-processing": 25,
    "Saving output": 15,
    # Step 2 / Step 3 phases
    "Loading classified raster": 5,
    "Converting to label raster": 10,
    "Removing road objects": 50,
    "Saving result": 15,
}

# Approximate total weight per pipeline (for normalising to 0–100 %).
_PIPELINE_TOTALS: Dict[str, float] = {
    "step1": 100,
    "full": 180,   # step1 + step2
    "mea": 180,
    "step2": 80,
    "step3": 80,
}


class _ProgressTracker:
    """Converts per-phase progress_callback(phase, done, total) into global
    weighted 0–100 % and pushes events into a ``Queue`` for SSE streaming."""

    def __init__(self, task_id: str, pipeline: str = "step1"):
        self.task_id = task_id
        self.queue: Queue = Queue()
        self.completed_weight = 0.0
        self.current_phase: Optional[str] = None
        self.current_weight: float = 0
        self.total_weight = _PIPELINE_TOTALS.get(pipeline, 100)
        _PROGRESS_QUEUES[task_id] = self.queue

    def __call__(self, phase: str, done: int, total: int):
        if phase != self.current_phase:
            # Entering a new phase → mark previous as fully complete.
            if self.current_phase is not None:
                self.completed_weight += self.current_weight
            self.current_phase = phase
            self.current_weight = _PHASE_WEIGHTS.get(phase, 5)

        frac = done / total if total > 0 else 0.0
        pct = min(99.0, (self.completed_weight + self.current_weight * frac)
                  / self.total_weight * 100.0)
        self.queue.put({
            "phase": phase,
            "done": round(pct, 1),
            "total": 100,
        })

    def finish(self):
        """Send sentinel and schedule queue cleanup."""
        self.queue.put(None)

        def _cleanup():
            import time
            time.sleep(3)
            _PROGRESS_QUEUES.pop(self.task_id, None)
        threading.Thread(target=_cleanup, daemon=True).start()


class ClassItem(BaseModel):
    id: str
    name: str
    color: str


class VectorLayer(BaseModel):
    id: str
    name: str
    filePath: str
    classId: str


class FeatureFlags(BaseModel):
    spectral: bool
    texture: bool
    indices: bool


class ClassifyRequest(BaseModel):
    rasterPath: str
    classes: List[ClassItem]
    vectorLayers: List[VectorLayer]
    smoothing: Literal["none", "median_1", "median_2", "median_3", "median_5"]
    featureFlags: FeatureFlags
    outputPath: str | None = None
    exportFormat: str = "tif"
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileOutputDir: str | None = None
    tileWorkers: int | None = None
    detectShadows: bool = False
    maxThreads: int | None = None
    taskId: str | None = None


class ClassifyStep1Request(BaseModel):
    """Step 1: Classification & Export (without vectors)"""
    rasterPath: str
    classes: List[ClassItem]
    smoothing: Literal["none", "median_1", "median_2", "median_3", "median_5"]
    featureFlags: FeatureFlags
    outputPath: str | None = None
    exportFormat: str = "tif"
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileOutputDir: str | None = None
    tileWorkers: int | None = None
    detectShadows: bool = False
    maxThreads: int | None = None
    taskId: str | None = None


class ClassifyStep2Request(BaseModel):
    """Step 2: Vector Rasterization onto existing classification"""
    classificationPath: str
    vectorLayers: List[VectorLayer]
    classes: List[ClassItem]
    outputPath: str | None = None
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileOutputDir: str | None = None
    tileWorkers: int | None = None
    maxThreads: int | None = None
    taskId: str | None = None


app = FastAPI(title="Material Classification API")

# Small in-memory cache for rendered raster previews (speeds repeated map redraws)
_RASTER_PREVIEW_CACHE: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
_RASTER_PREVIEW_CACHE_MAX_ITEMS = 64

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ─── SSE progress streaming endpoint ─────────────────────────────────────

@app.get("/progress/{task_id}")
async def progress_sse(task_id: str):
    """Stream real-time progress events for a running task via SSE."""

    async def _stream():
        # Wait for the task to register its queue (may start slightly after
        # the EventSource connection is opened).
        q: Queue | None = None
        for _ in range(50):  # up to 5 s
            q = _PROGRESS_QUEUES.get(task_id)
            if q is not None:
                break
            await asyncio.sleep(0.1)
        if q is None:
            yield f"event: error\ndata: {{\"message\":\"Task not found\"}}\n\n"
            return

        loop = asyncio.get_event_loop()
        while True:
            try:
                evt = await loop.run_in_executor(None, lambda: q.get(timeout=0.5))
                if evt is None:  # sentinel → task finished
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {_json.dumps(evt)}\n\n"
            except Empty:
                yield ": keepalive\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Classify endpoints ──────────────────────────────────────────────────

@app.post("/classify")
def classify(request: ClassifyRequest) -> dict:
    """Complete pipeline: Classify + Rasterize vectors (if provided)"""
    print(f"[API /classify] outputPath={request.outputPath!r} exportFormat={request.exportFormat!r}")
    tracker = _ProgressTracker(request.taskId, "full") if request.taskId else None
    try:
        result = run_classify(
            request.rasterPath,
            [item.model_dump() for item in request.classes],
            [layer.model_dump() for layer in request.vectorLayers],
            request.smoothing,
            request.featureFlags.model_dump(),
            request.outputPath,
            request.tileMode,
            request.tileMaxPixels or 512 * 512,
            request.tileOverlap,
            request.tileOutputDir,
            request.tileWorkers,
            request.detectShadows,
            request.maxThreads,
            export_format=request.exportFormat,
            progress_callback=tracker,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


@app.post("/classify-step1")
def classify_step1(request: ClassifyStep1Request) -> dict:
    """Step 1: KMeans classification and export to RGB (no vectors)"""
    print(f"[API /classify-step1] outputPath={request.outputPath!r} exportFormat={request.exportFormat!r}")
    tracker = _ProgressTracker(request.taskId, "step1") if request.taskId else None
    try:
        result = run_classify_and_export(
            request.rasterPath,
            [item.model_dump() for item in request.classes],
            request.smoothing,
            request.featureFlags.model_dump(),
            request.outputPath,
            request.tileMode,
            request.tileMaxPixels or 512 * 512,
            request.tileOverlap,
            request.tileOutputDir,
            request.tileWorkers,
            request.detectShadows,
            request.maxThreads,
            export_format=request.exportFormat,
            progress_callback=tracker,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


@app.post("/classify-step2")
def classify_step2(request: ClassifyStep2Request) -> dict:
    """Step 2: Rasterize vector layers onto existing classification file"""
    print(f"[API /classify-step2] outputPath={request.outputPath!r}")
    tracker = _ProgressTracker(request.taskId, "step2") if request.taskId else None
    try:
        result = run_rasterize_vectors(
            request.classificationPath,
            [layer.model_dump() for layer in request.vectorLayers],
            [item.model_dump() for item in request.classes],
            request.outputPath,
            request.tileMode,
            request.tileMaxPixels or 512 * 512,
            request.tileOverlap,
            request.tileOutputDir,
            request.tileWorkers,
            request.maxThreads,
            progress_callback=tracker,
        )
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── Step 3: Remove road objects ─────────────────────────────────────────


class RemoveRoadObjectsRequest(BaseModel):
    classificationPath: str
    taskId: str | None = None


@app.post("/remove-road-objects")
def remove_road_objects(request: RemoveRoadObjectsRequest) -> dict:
    """Step 3: Remove small objects enclosed by asphalt."""
    tracker = _ProgressTracker(request.taskId, "step3") if request.taskId else None
    try:
        out_path = run_road_object_removal(
            request.classificationPath,
            progress_callback=tracker,
        )
        return {"status": "ok", "outputPath": out_path}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── File/folder browser (for web UI) ────────────────────────────────────

import os
import string


class ListDirRequest(BaseModel):
    path: Optional[str] = None


@app.post("/list-dir")
def list_dir(request: ListDirRequest) -> dict:
    """List contents of a directory. If no path given, list drive roots (Windows) or /."""
    try:
        # No path → list drive roots on Windows, or / on Unix
        if not request.path:
            if os.name == "nt":
                drives = []
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:\\"
                    if os.path.isdir(drive):
                        drives.append({"name": drive, "type": "dir", "size": 0})
                return {"path": "", "parent": None, "entries": drives}
            else:
                request.path = "/"

        p = Path(request.path).resolve()
        if not p.is_dir():
            return JSONResponse(status_code=400, content={"error": "Not a directory"})

        parent = str(p.parent) if p.parent != p else None
        entries = []

        try:
            items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return {"path": str(p), "parent": parent, "entries": [], "error": "Permission denied"}

        for item in items:
            try:
                is_dir = item.is_dir()
                size = 0 if is_dir else item.stat().st_size
                entries.append({
                    "name": item.name,
                    "type": "dir" if is_dir else "file",
                    "size": size,
                })
            except (PermissionError, OSError):
                continue

        return {"path": str(p), "parent": parent, "entries": entries}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── Scan folder for raster images ────────────────────────────────────────


class ScanFolderRequest(BaseModel):
    folderPath: str
    extensions: Optional[List[str]] = None


RASTER_EXTENSIONS = {
    ".tif", ".tiff", ".jpg", ".jpeg", ".png",
    ".img", ".bmp", ".jp2", ".ecw", ".sid",
}


@app.post("/scan-folder")
def scan_folder(request: ScanFolderRequest) -> dict:
    """Recursively scan a folder for raster image files."""
    import os

    folder = Path(request.folderPath)
    if not folder.is_dir():
        return JSONResponse(status_code=400, content={"error": f"Not a directory: {request.folderPath}"})

    allowed = set(request.extensions) if request.extensions else RASTER_EXTENSIONS
    files: list[dict] = []

    for root, _dirs, filenames in os.walk(folder):
        for fname in sorted(filenames):
            ext = Path(fname).suffix.lower()
            if ext in allowed:
                full_path = str(Path(root) / fname)
                files.append({
                    "name": fname,
                    "path": full_path,
                    "relativePath": str(Path(root).relative_to(folder) / fname),
                })

    return {"folder": str(folder), "count": len(files), "files": files}


# ─── Raster info (bounds for map view) ───────────────────────────────────


class RasterInfoRequest(BaseModel):
    filePath: str


@app.post("/raster-info")
def raster_info(request: RasterInfoRequest) -> dict:
    """Return the WGS84 bounding box of a raster file for Leaflet display."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(request.filePath) as ds:
            src_crs = ds.crs
            bounds = ds.bounds

            if src_crs and not src_crs.is_geographic:
                west, south, east, north = transform_bounds(
                    src_crs, "EPSG:4326",
                    bounds.left, bounds.bottom, bounds.right, bounds.top,
                )
            else:
                west, south, east, north = (
                    bounds.left, bounds.bottom, bounds.right, bounds.top,
                )

            return {
                "bounds": [[south, west], [north, east]],
                "crs": str(src_crs) if src_crs else "unknown",
                "width": ds.width,
                "height": ds.height,
            }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ─── Serve raster as browser-friendly PNG ─────────────────────────────────


class RasterAsPngRequest(BaseModel):
    filePath: str
    maxDim: int = 1536


def _render_raster_as_image(file_path: str, max_dim: int = 1536):
    """
    Read any raster, reproject to EPSG:4326, and return bytes + media_type.
    Supports GeoTIFF, ERDAS Imagine (.img), and any GDAL-readable format.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject
    from rasterio.crs import CRS
    from PIL import Image as PILImage

    p = Path(file_path)
    if not p.exists():
        return None, None, f"File not found: {file_path}"

    MAX_DIM = max(256, min(max_dim, 4096))
    DST_CRS = CRS.from_epsg(4326)

    # Cache key by file identity + render params
    stat = p.stat()
    cache_key = f"{p}|{int(stat.st_mtime)}|{stat.st_size}|{MAX_DIM}|4326"
    cached = _RASTER_PREVIEW_CACHE.get(cache_key)
    if cached is not None:
        payload, media_type = cached
        _RASTER_PREVIEW_CACHE.move_to_end(cache_key)
        return payload, media_type, None

    with rasterio.open(str(p)) as ds:
        src_crs = ds.crs or DST_CRS
        src_dtype = ds.dtypes[0]  # e.g. 'uint8', 'float32'
        is_uint8_source = (src_dtype == 'uint8')

        # ── Compute destination transform and dimensions in EPSG:4326 ──
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, DST_CRS,
            ds.width, ds.height,
            *ds.bounds,
        )

        # Down-sample if exceeding MAX_DIM
        scale = min(MAX_DIM / dst_w, MAX_DIM / dst_h, 1.0)
        dst_w = max(1, int(dst_w * scale))
        dst_h = max(1, int(dst_h * scale))
        dst_transform = rasterio.transform.from_bounds(
            *rasterio.warp.transform_bounds(src_crs, DST_CRS, *ds.bounds),
            dst_w, dst_h,
        )

        band_count = min(ds.count, 3)
        bands = list(range(1, band_count + 1))

        # Use nearest resampling for classification results (uint8 RGB)
        # to preserve exact class colors; bilinear for continuous imagery.
        resamp = Resampling.nearest if is_uint8_source else Resampling.bilinear

        # ── Reproject each band into EPSG:4326 ──
        dst_data = np.zeros((band_count, dst_h, dst_w), dtype=np.float64)
        for i, band_idx in enumerate(bands):
            reproject(
                source=rasterio.band(ds, band_idx),
                destination=dst_data[i],
                src_transform=ds.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=DST_CRS,
                resampling=resamp,
            )

    # ── Normalise to uint8 ──
    arr = dst_data
    if band_count == 1:
        arr = arr[0]  # squeeze to (H, W)

    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    # If source was already uint8, values are in [0,255] — just clip.
    # Otherwise stretch to full 0–255 range.
    if is_uint8_source:
        arr = np.clip(arr, 0, 255)
    elif mx > 255 or mn < 0 or mx > mn:
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255.0
        else:
            arr = np.zeros_like(arr)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # ── Build output image ──
    media_type = "image/png"
    if arr.ndim == 3:
        rgb = np.transpose(arr, (1, 2, 0))  # (C,H,W) → (H,W,C)
        # For classification results, add alpha channel to hide nodata (black)
        if is_uint8_source:
            # Pixel is nodata when all bands are 0
            alpha = np.where(
                (rgb[:, :, 0] == 0) & (rgb[:, :, 1] == 0) & (rgb[:, :, 2] == 0),
                0, 255,
            ).astype(np.uint8)
            img = PILImage.fromarray(rgb, mode="RGB").convert("RGBA")
            img.putalpha(PILImage.fromarray(alpha, mode="L"))
            media_type = "image/png"  # PNG for transparency
        else:
            img = PILImage.fromarray(rgb, mode="RGB")
            media_type = "image/jpeg"
    else:
        alpha = np.where(arr > 0, 255, 0).astype(np.uint8)
        img = PILImage.fromarray(arr, mode="L").convert("RGBA")
        img.putalpha(PILImage.fromarray(alpha, mode="L"))

    buf = io.BytesIO()
    if media_type == "image/jpeg":
        img.save(buf, format="JPEG", quality=80, optimize=True)
    else:
        img.save(buf, format="PNG", compress_level=2)
    buf.seek(0)

    payload = buf.getvalue()
    _RASTER_PREVIEW_CACHE[cache_key] = (payload, media_type)
    _RASTER_PREVIEW_CACHE.move_to_end(cache_key)
    while len(_RASTER_PREVIEW_CACHE) > _RASTER_PREVIEW_CACHE_MAX_ITEMS:
        _RASTER_PREVIEW_CACHE.popitem(last=False)

    return payload, media_type, None


@app.post("/raster-as-png")
def raster_as_png_post(request: RasterAsPngRequest):
    """POST version — accepts filePath in body (robust for Windows paths)."""
    try:
        payload, media_type, err = _render_raster_as_image(request.filePath, request.maxDim)
        if err:
            return JSONResponse(status_code=404, content={"error": err})
        return StreamingResponse(
            io.BytesIO(payload),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/raster-as-png/{file_path:path}")
def raster_as_png(file_path: str, max_dim: int = 1536):
    """GET version (legacy) — file path encoded in URL."""
    try:
        payload, media_type, err = _render_raster_as_image(file_path, max_dim)
        if err:
            return JSONResponse(status_code=404, content={"error": err})
        return StreamingResponse(
            io.BytesIO(payload),
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

