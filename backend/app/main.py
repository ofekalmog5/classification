import sys
import os

# Force UTF-8 for stdout/stderr on Windows so that print() calls with
# special characters (arrows, checkmarks, etc.) don't throw UnicodeEncodeError.
# PYTHONIOENCODING is inherited by child processes (ProcessPoolExecutor workers).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
from .core import train_kmeans_model, build_shared_color_table, suggest_tile_size
from .core import _ACCEL_ENGINE, _ACCEL_GPU, _ACCEL_GPU_INFO
from .core import _is_mea_classes, MEA_CLASSES
from .pipeline import classify_v6 as run_classify_v6
from .road_extraction import extract_roads as run_extract_roads
from .road_extraction import merge_road_mask_onto_classification as run_merge_road_mask
from .road_extraction import extract_feature_masks as run_extract_feature_masks
from .road_extraction import merge_feature_masks_onto_classification as run_merge_feature_masks
from .road_extraction import set_sam3_local_dir, get_road_extract_config

# ─── SSE progress streaming infrastructure ───────────────────────────────

_PROGRESS_QUEUES: Dict[str, Queue] = {}
_CANCEL_FLAGS: Dict[str, bool] = {}


class TaskCancelledError(Exception):
    """Raised when the user cancels a running task."""

# Per-phase weights reflecting typical relative durations.
# Step 1 phases sum to ~100; other pipelines normalise by total weight.
_PHASE_WEIGHTS: Dict[str, float] = {
    # Step 1 phases
    "Loading raster": 5,
    "Feature extraction": 35,
    "KMeans clustering": 20,
    "Pixel assignment": 15,
    "Classifying tiles": 35,
    "Saving output": 15,
    # Step 2 phases
    "Loading classified raster": 5,
    "Converting to label raster": 10,
    "Saving result": 15,
    # Batch phases
    "Training shared model": 15,
    "Classifying": 5,
    # Road extraction phases
    "Loading SAM 3 model": 10,
    "Extracting roads": 70,
    "Morphological closing": 10,
    "Merging road mask": 80,
}

# Approximate total weight per pipeline (for normalising to 0–100 %).
_PIPELINE_TOTALS: Dict[str, float] = {
    "step1": 100,
    "full": 180,   # step1 + step2
    "mea": 180,
    "step2": 80,
    "road_extract": 100,
    "road_merge": 100,
    # v6: 2x SAM3 (~80 each) + KMeans step1 (100) + mask fusion (80)
    "v6_step1": 340,
    "v6_full": 420,  # v6_step1 + step2 vector rasterisation
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
        # Check for user-requested cancellation at every progress update.
        if _CANCEL_FLAGS.get(self.task_id):
            _CANCEL_FLAGS.pop(self.task_id, None)
            raise TaskCancelledError("Task cancelled by user")

        if phase != self.current_phase:
            # Entering a new phase -> mark previous as fully complete.
            if self.current_phase is not None:
                self.completed_weight += self.current_weight
            self.current_phase = phase
            # The v6 pipeline prefixes phase names with a label
            # (e.g. "Roads (SAM3): Loading SAM 3 model").  Strip the prefix
            # before looking up the canonical weight.
            weight_key = phase.split(": ", 1)[1] if ": " in phase else phase
            self.current_weight = (
                _PHASE_WEIGHTS.get(phase)
                or _PHASE_WEIGHTS.get(weight_key)
                or 5
            )

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
            _CANCEL_FLAGS.pop(self.task_id, None)
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
    overrideColor: list | None = None


class FeatureFlags(BaseModel):
    spectral: bool
    texture: bool
    indices: bool
    colorIndices: bool = True
    entropy: bool = False
    morphCleanup: bool = True


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
    # 6-material SAM3-first pipeline controls (v6).  When the request's classes
    # match the 6-material MEA schema, these fields decide how road/building
    # pixels are produced.
    sam3Enabled: bool = True
    roadShapefile: str | None = None
    buildingShapefile: str | None = None


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
    sam3Enabled: bool = True
    roadShapefile: str | None = None
    buildingShapefile: str | None = None


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


# ─── /api prefix middleware (standalone production mode) ──────────────────
# In development, Vite's proxy strips "/api" before forwarding to FastAPI.
# In production (uvicorn serving static files directly), the browser calls
# /api/xxx so we strip the prefix here instead.
from starlette.middleware.base import BaseHTTPMiddleware

class _StripApiPrefix(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.scope.get("path", "").startswith("/api"):
            stripped = request.scope["path"][4:] or "/"
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)

app.add_middleware(_StripApiPrefix)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/gpu-info")
def gpu_info() -> dict:
    return {
        "available": _ACCEL_GPU,
        "info": _ACCEL_GPU_INFO,
        "engine": _ACCEL_ENGINE,   # "faiss-gpu" | "faiss-cpu" | "cuml" | "sklearn"
    }


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
                if evt is None:  # sentinel -> task finished
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


# ─── Cancel endpoint ─────────────────────────────────────────────────────

@app.post("/cancel/{task_id}")
async def cancel_task(task_id: str):
    """Signal a running task to stop at its next progress checkpoint."""
    if task_id in _PROGRESS_QUEUES:
        _CANCEL_FLAGS[task_id] = True
        return {"status": "ok", "message": "Cancel signal sent"}
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Task not found or already finished"},
    )


# ─── Classify endpoints ──────────────────────────────────────────────────

def _route_to_v6(classes_dict: list, sam3_enabled: bool) -> bool:
    """Decide whether the new SAM3-first 6-material pipeline should handle
    this request.  True only when the request carries ALL 6 canonical MEA
    classes AND the caller hasn't disabled SAM3.  Partial-MEA payloads (1–5
    of the canonical names) fall through to the legacy KMeans path."""
    return (
        bool(sam3_enabled)
        and len(classes_dict) == len(MEA_CLASSES)
        and _is_mea_classes(classes_dict)
    )


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict:
    """Complete pipeline: Classify + Rasterize vectors (if provided).

    When the request's classes match the 6-material MEA schema and
    ``sam3Enabled`` is True (default), the new SAM3-first pipeline runs first
    to produce road and building masks, then KMeans handles the remaining
    natural materials, then user vectors are rasterised on top.
    """
    print(f"[API /classify] outputPath={request.outputPath!r} exportFormat={request.exportFormat!r}")
    classes_dict = [item.model_dump() for item in request.classes]
    vector_layers = [layer.model_dump() for layer in request.vectorLayers]
    use_v6 = _route_to_v6(classes_dict, request.sam3Enabled)
    pipeline_kind = "v6_full" if use_v6 else "full"
    tracker = _ProgressTracker(request.taskId, pipeline_kind) if request.taskId else None
    try:
        if use_v6:
            print("[API /classify] routing -> classify_v6 (SAM3-first 6-material pipeline)")
            result = run_classify_v6(
                raster_path=request.rasterPath,
                classes=classes_dict,
                smoothing=request.smoothing,
                feature_flags=request.featureFlags.model_dump(),
                output_path=request.outputPath,
                sam3_enabled=request.sam3Enabled,
                road_shapefile=request.roadShapefile,
                building_shapefile=request.buildingShapefile,
                tile_mode=request.tileMode,
                tile_max_pixels=request.tileMaxPixels or 512 * 512,
                tile_overlap=request.tileOverlap,
                tile_output_dir=request.tileOutputDir,
                tile_workers=request.tileWorkers,
                detect_shadows=request.detectShadows,
                max_threads=request.maxThreads,
                export_format=request.exportFormat,
                progress_callback=tracker,
            )
            # Rasterize user-supplied vector layers on top of the v6 output.
            if result.get("status") == "ok" and vector_layers:
                cls_path = result.get("outputPath")
                if cls_path:
                    rasterize_result = run_rasterize_vectors(
                        classification_path=cls_path,
                        vector_layers=vector_layers,
                        classes=classes_dict,
                        output_path=request.outputPath,
                        tile_mode=request.tileMode,
                        tile_max_pixels=request.tileMaxPixels or 512 * 512,
                        tile_overlap=request.tileOverlap,
                        tile_output_dir=request.tileOutputDir,
                        tile_workers=request.tileWorkers,
                        max_threads=request.maxThreads,
                        progress_callback=tracker,
                    )
                    if rasterize_result.get("status") == "ok":
                        result["outputPath"] = rasterize_result.get("outputPath", cls_path)
            return result

        result = run_classify(
            raster_path=request.rasterPath,
            classes=classes_dict,
            vector_layers=vector_layers,
            smoothing=request.smoothing,
            feature_flags=request.featureFlags.model_dump(),
            output_path=request.outputPath,
            tile_mode=request.tileMode,
            tile_max_pixels=request.tileMaxPixels or 512 * 512,
            tile_overlap=request.tileOverlap,
            tile_output_dir=request.tileOutputDir,
            tile_workers=request.tileWorkers,
            detect_shadows=request.detectShadows,
            max_threads=request.maxThreads,
            export_format=request.exportFormat,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


@app.post("/classify-step1")
def classify_step1(request: ClassifyStep1Request) -> dict:
    """Step 1: KMeans classification and export to RGB (no vectors).

    Routes to ``classify_v6`` when the request matches the 6-material MEA
    schema and SAM3 is enabled (default) — same behaviour as ``/classify``
    minus the vector-rasterisation step."""
    print(f"[API /classify-step1] outputPath={request.outputPath!r} exportFormat={request.exportFormat!r}")
    classes_dict = [item.model_dump() for item in request.classes]
    use_v6 = _route_to_v6(classes_dict, request.sam3Enabled)
    pipeline_kind = "v6_step1" if use_v6 else "step1"
    tracker = _ProgressTracker(request.taskId, pipeline_kind) if request.taskId else None
    try:
        if use_v6:
            print("[API /classify-step1] routing -> classify_v6 (SAM3-first 6-material pipeline)")
            result = run_classify_v6(
                raster_path=request.rasterPath,
                classes=classes_dict,
                smoothing=request.smoothing,
                feature_flags=request.featureFlags.model_dump(),
                output_path=request.outputPath,
                sam3_enabled=request.sam3Enabled,
                road_shapefile=request.roadShapefile,
                building_shapefile=request.buildingShapefile,
                tile_mode=request.tileMode,
                tile_max_pixels=request.tileMaxPixels or 512 * 512,
                tile_overlap=request.tileOverlap,
                tile_output_dir=request.tileOutputDir or request.outputPath,
                tile_workers=request.tileWorkers,
                detect_shadows=request.detectShadows,
                max_threads=request.maxThreads,
                export_format=request.exportFormat,
                progress_callback=tracker,
            )
            return result

        result = run_classify_and_export(
            raster_path=request.rasterPath,
            classes=classes_dict,
            smoothing=request.smoothing,
            feature_flags=request.featureFlags.model_dump(),
            output_path=request.outputPath,
            tile_mode=request.tileMode,
            tile_max_pixels=request.tileMaxPixels or 512 * 512,
            tile_overlap=request.tileOverlap,
            tile_output_dir=request.tileOutputDir or request.outputPath,
            tile_workers=request.tileWorkers,
            detect_shadows=request.detectShadows,
            max_threads=request.maxThreads,
            export_format=request.exportFormat,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── Batch classify (shared model across all files) ──────────────────────


class BatchClassifyRequest(BaseModel):
    """Classify multiple rasters with a single shared KMeans model."""
    rasterPaths: List[str]
    classes: List[ClassItem]
    vectorLayers: List[VectorLayer] = []
    smoothing: str = "none"
    featureFlags: FeatureFlags
    outputPath: str | None = None
    exportFormat: str = "tif"
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileWorkers: int | None = None
    detectShadows: bool = False
    maxThreads: int | None = None
    taskId: str | None = None


@app.post("/classify-batch")
def classify_batch(request: BatchClassifyRequest) -> dict:
    """Train ONE shared model on all rasters, then classify each with it."""
    import time as _time

    n = len(request.rasterPaths)
    print(f"\n[API /classify-batch] {n} raster(s), shared model")
    tracker = _ProgressTracker(request.taskId, "full") if request.taskId else None
    classes_raw = [item.model_dump() for item in request.classes]
    vectors_raw = [layer.model_dump() for layer in request.vectorLayers]
    ff = request.featureFlags.model_dump()
    has_vectors = len(vectors_raw) > 0
    print(f"  has_vectors={has_vectors}, vectors_raw count={len(vectors_raw)}")
    if vectors_raw:
        for _dv in vectors_raw:
            print(f"  vector: id={_dv.get('id')}, classId={_dv.get('classId')}, filePath={_dv.get('filePath')}")
    import sys; sys.stdout.flush()

    try:
        # ── Phase 1: Train shared model ──
        if tracker:
            tracker("Training shared model", 0, 1)
        t0 = _time.perf_counter()
        scaler, kmeans = train_kmeans_model(
            request.rasterPaths, classes_raw, ff,
            detect_shadows=request.detectShadows,
        )
        # Build ONE shared color table from all rasters
        mea_mapping, color_table = build_shared_color_table(
            request.rasterPaths, scaler, kmeans, classes_raw, ff,
        )
        if tracker:
            tracker("Training shared model", 1, 1)
        train_time = _time.perf_counter() - t0
        print(f"[Batch] Shared model trained in {train_time:.1f}s")

        # ── Phase 2: Classify each raster (Step 1 only) ──
        all_results: list[dict] = []
        all_tile_outputs: list[str] = []
        all_output_paths: list[str] = []
        errors: list[tuple[str, str]] = []
        # Track classified outputs for later vector rasterization
        classified_outputs: list[tuple[str, str]] = []  # (raster_path, classified_output_path)

        for i, rpath in enumerate(request.rasterPaths):
            fname = Path(rpath).name
            if tracker:
                tracker(f"Classifying {fname}", i, n)
            print(f"\n[Batch] ({i+1}/{n}) {fname}")

            try:
                result = run_classify_and_export(
                    raster_path=rpath,
                    classes=classes_raw,
                    smoothing=request.smoothing,
                    feature_flags=ff,
                    output_path=request.outputPath,
                    tile_mode=request.tileMode,
                    tile_max_pixels=request.tileMaxPixels or 512 * 512,
                    tile_overlap=request.tileOverlap,
                    tile_output_dir=request.outputPath,
                    tile_workers=request.tileWorkers,
                    detect_shadows=request.detectShadows,
                    max_threads=request.maxThreads,
                    pretrained_scaler=scaler,
                    pretrained_kmeans=kmeans,
                    pretrained_color_table=color_table,
                    pretrained_mea_mapping=mea_mapping,
                    export_format=request.exportFormat,
                    progress_callback=tracker,
                )

                all_results.append(result)
                if result.get("status") == "ok":
                    op = result.get("outputPath", "")
                    tiles = result.get("tileOutputs")
                    if op:
                        classified_outputs.append((rpath, str(op)))
                    if tiles:
                        # Tile mode: add individual tile files (not the directory)
                        all_tile_outputs.extend(tiles)
                        if not has_vectors:
                            all_output_paths.extend(str(t) for t in tiles)
                    elif op:
                        # Single-file mode: add the output file
                        if not has_vectors:
                            all_output_paths.append(str(op))
                else:
                    errors.append((rpath, result.get("message", "unknown error")))
            except Exception as e:
                import traceback
                error_log = Path("classification_error.log")
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(f"--- Error classifying {fname} ---\n")
                    f.write(f"Exception: {e}\n")
                    f.write(traceback.format_exc())
                    f.write("-" * 50 + "\n")
                errors.append((rpath, str(e)))
                print(f"[Batch][error] {fname}: {e}")
        if tracker:
            tracker(f"Classifying", n, n)

        # ── Phase 3: Rasterize vectors onto ALL classified outputs ──
        if has_vectors and classified_outputs:
            import copy, sys
            n_vec = len(classified_outputs)
            print(f"\n[Batch] Phase 3: Rasterizing vectors onto {n_vec} classified image(s)")
            print(f"  vectors_raw count={len(vectors_raw)}, classes_raw count={len(classes_raw)}")
            for _dv in vectors_raw:
                print(f"  vector: filePath={_dv.get('filePath')}, classId={_dv.get('classId')}, exists={Path(_dv.get('filePath','')).exists()}")
            sys.stdout.flush()
            for vi, (rpath, classif_output) in enumerate(classified_outputs):
                fname = Path(rpath).name
                if tracker:
                    tracker(f"Rasterizing vectors {fname}", vi, n_vec)
                print(f"\n[Batch][Vec] ({vi+1}/{n_vec}) {fname}")
                print(f"  classif_output={classif_output}")
                print(f"  classif_output exists={Path(classif_output).exists()}")
                sys.stdout.flush()
                try:
                    _classif_path = Path(classif_output)
                    _vec_dir = _classif_path.parent / "with_vectors"
                    _vec_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  _vec_dir={_vec_dir}, exists={_vec_dir.exists()}")
                    # Deep copy vectors_raw so mutations from one iteration
                    # don't affect the next.
                    _vectors_copy = copy.deepcopy(vectors_raw)
                    vec_result = run_rasterize_vectors(
                        classification_path=classif_output,
                        vector_layers=_vectors_copy,
                        classes=classes_raw,
                        output_path=str(_vec_dir),
                        raster_stem_hint=Path(rpath).stem,
                        tile_mode=request.tileMode,
                        tile_max_pixels=request.tileMaxPixels or 512 * 512,
                        tile_overlap=request.tileOverlap,
                        tile_output_dir=request.outputPath,
                        tile_workers=request.tileWorkers,
                        max_threads=request.maxThreads,
                        progress_callback=tracker,
                    )
                    print(f"  vec_result={vec_result}")
                    sys.stdout.flush()
                    if vec_result.get("status") == "ok":
                        vec_op = vec_result.get("outputPath", "")
                        vec_tiles = vec_result.get("tileOutputs")
                        if vec_tiles:
                            # Tile mode: add individual tile files (not the directory)
                            all_tile_outputs.extend(vec_tiles)
                            all_output_paths.extend(str(t) for t in vec_tiles)
                        elif vec_op:
                            # Single-file mode: add the output file
                            all_output_paths.append(str(vec_op))
                            if not Path(vec_op).exists():
                                print(f"  [WARN] outputPath={vec_op} does NOT exist on disk!")
                        print(f"  [OK] Vectors rasterized -> {vec_op}")
                    else:
                        print(f"  [WARN] Vector rasterize failed: {vec_result.get('message')}")
                        errors.append((rpath, f"vector rasterize: {vec_result.get('message', 'unknown')}"))
                except Exception as e:
                    import traceback as _tb
                    print(f"  [ERROR] Vector rasterize: {e}")
                    _tb.print_exc()
                    errors.append((rpath, f"vector rasterize: {e}"))
                sys.stdout.flush()
            if tracker:
                tracker(f"Rasterizing vectors", n_vec, n_vec)
            # List all with_vectors directories for debugging
            for _rp, _co in classified_outputs:
                _vd = Path(_co).parent / "with_vectors"
                if _vd.exists():
                    _files = list(_vd.iterdir())
                    print(f"  [DEBUG] {_vd} contains {len(_files)} files: {[f.name for f in _files]}")
                else:
                    print(f"  [DEBUG] {_vd} does NOT exist")
            sys.stdout.flush()
        elif has_vectors and not classified_outputs:
            print(f"\n[Batch] Phase 3 SKIPPED: has_vectors={has_vectors} but classified_outputs is EMPTY")
            sys.stdout.flush()

        ok_count = sum(1 for r in all_results if r.get("status") == "ok")
        return {
            "status": "ok" if ok_count > 0 else "error",
            "message": f"Batch complete: {ok_count}/{n} succeeded",
            "outputPaths": all_output_paths,
            "tileOutputs": sorted(all_tile_outputs) if all_tile_outputs else None,
            "results": all_results,
            "errors": errors if errors else None,
            "meaMapping": mea_mapping,
        }
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        import traceback
        # Write full traceback to file (stdout/stderr may fail with charmap on Windows)
        try:
            with open("classification_error.log", "a", encoding="utf-8") as _ef:
                _ef.write(f"\n{'='*60}\n")
                _ef.write(f"Exception in /classify-batch: {e!r}\n")
                traceback.print_exc(file=_ef)
                _ef.write(f"{'='*60}\n")
        except Exception:
            pass
        try:
            traceback.print_exc()
        except Exception:
            pass
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
            classification_path=request.classificationPath,
            vector_layers=[layer.model_dump() for layer in request.vectorLayers],
            classes=[item.model_dump() for item in request.classes],
            output_path=request.outputPath,
            tile_mode=request.tileMode,
            tile_max_pixels=request.tileMaxPixels or 512 * 512,
            tile_overlap=request.tileOverlap,
            tile_output_dir=request.tileOutputDir or request.outputPath,
            tile_workers=request.tileWorkers,
            max_threads=request.maxThreads,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── Road extraction (SAM 3) ────────────────────────────────────────────


class ExtractRoadsRequest(BaseModel):
    rasterPath: str
    outputPath: str | None = None
    textPrompt: str = "road, highway, asphalt path"
    tileSize: int = 1024
    overlapPct: float = 0.1
    closingKernelSize: int = 15
    device: str = "auto"
    taskId: str | None = None


@app.post("/extract-roads")
def extract_roads(request: ExtractRoadsRequest) -> dict:
    """Extract roads from a GeoTIFF using SAM 3 text-prompted segmentation."""
    print(f"[API /extract-roads] rasterPath={request.rasterPath!r}")
    tracker = _ProgressTracker(request.taskId, "road_extract") if request.taskId else None
    try:
        result = run_extract_roads(
            input_path=request.rasterPath,
            output_path=request.outputPath,
            text_prompt=request.textPrompt,
            tile_size=request.tileSize,
            overlap_pct=request.overlapPct,
            closing_kernel_size=request.closingKernelSize,
            device=request.device,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


class MergeRoadMaskRequest(BaseModel):
    classificationPath: str
    roadMaskPath: str
    outputPath: str | None = None
    taskId: str | None = None


@app.post("/merge-road-mask")
def merge_road_mask(request: MergeRoadMaskRequest) -> dict:
    """Merge a road mask onto a classification output as BM_ASPHALT."""
    print(f"[API /merge-road-mask] classification={request.classificationPath!r} mask={request.roadMaskPath!r}")
    tracker = _ProgressTracker(request.taskId, "road_merge") if request.taskId else None
    try:
        result = run_merge_road_mask(
            classification_path=request.classificationPath,
            road_mask_path=request.roadMaskPath,
            output_path=request.outputPath,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── Generic feature extraction (buildings, vegetation, roads) ───────────


class ExtractFeaturesRequest(BaseModel):
    rasterPath: str
    featureType: str  # "roads" | "buildings" | "trees" | "fields" | "water"
    outputPath: str | None = None
    tileSize: int = 1024
    overlapPct: float = 0.1
    closingKernelSize: int = 15
    device: str = "auto"
    taskId: str | None = None


@app.post("/extract-features")
def extract_features(request: ExtractFeaturesRequest) -> dict:
    """Extract feature masks (roads/buildings/vegetation) using SAM text prompts."""
    print(f"[API /extract-features] featureType={request.featureType!r} rasterPath={request.rasterPath!r}")
    tracker = _ProgressTracker(request.taskId, "feature_extract") if request.taskId else None
    try:
        result = run_extract_feature_masks(
            input_path=request.rasterPath,
            output_path=request.outputPath,
            feature_type=request.featureType,
            tile_size=request.tileSize,
            overlap_pct=request.overlapPct,
            closing_kernel_size=request.closingKernelSize,
            device=request.device,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


class MergeFeatureMasksRequest(BaseModel):
    classificationPath: str
    maskPaths: list[str]
    colors: list[list[int]]  # [[r, g, b], ...]
    outputPath: str | None = None
    taskId: str | None = None


@app.post("/merge-feature-masks")
def merge_feature_masks(request: MergeFeatureMasksRequest) -> dict:
    """Merge feature masks (buildings/vegetation/roads) onto a classification output."""
    print(f"[API /merge-feature-masks] classification={request.classificationPath!r} masks={len(request.maskPaths)}")
    tracker = _ProgressTracker(request.taskId, "feature_merge") if request.taskId else None
    colors = [tuple(c) for c in request.colors]
    try:
        result = run_merge_feature_masks(
            classification_path=request.classificationPath,
            mask_paths=request.maskPaths,
            colors=colors,
            output_path=request.outputPath,
            progress_callback=tracker,
        )
        return result
    except TaskCancelledError:
        return JSONResponse(status_code=499, content={"status": "cancelled", "message": "Task cancelled by user"})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        if tracker:
            tracker.finish()


# ─── App config (persistent settings) ────────────────────────────────────

from .config import load as _load_config, save as _save_config, config_path as _config_path


class AppConfigUpdate(BaseModel):
    sam3_local_dir: str | None = None
    hf_cache_dir: str | None = None
    offline_mode: bool | None = None


@app.get("/app-config")
def app_config_get() -> dict:
    """Return all persistent app settings + config file path."""
    cfg = _load_config()
    return {**cfg, "configPath": _config_path()}


@app.post("/app-config")
def app_config_set(request: AppConfigUpdate) -> dict:
    """Update persistent app settings. Only provided fields are changed."""
    updates = {k: v for k, v in request.model_dump().items() if v is not None}
    # If sam3 path changed, also update the road extraction module
    if "sam3_local_dir" in updates:
        set_sam3_local_dir(updates["sam3_local_dir"])
    cfg = _save_config(updates)
    return {**cfg, "configPath": _config_path()}


# ─── Road extraction config ──────────────────────────────────────────────


class SetSam3PathRequest(BaseModel):
    path: str | None = None   # None → clear override, revert to auto-detection


@app.post("/set-sam3-path")
def set_sam3_path(request: SetSam3PathRequest) -> dict:
    """Set (or clear) the local SAM3 directory. Clears cached model."""
    set_sam3_local_dir(request.path)
    return get_road_extract_config()


@app.get("/road-extract-config")
def road_extract_config() -> dict:
    """Return current road-extraction backend configuration."""
    return get_road_extract_config()


# ─── Suggest tile size ───────────────────────────────────────────────────


class SuggestTileSizeRequest(BaseModel):
    rasterPath: str
    workers: int = 4


@app.post("/suggest-tile-size")
def suggest_tile_size_endpoint(request: SuggestTileSizeRequest) -> dict:
    """Return the memory-safe suggested tile side length for a raster."""
    try:
        side = suggest_tile_size(request.rasterPath, workers=max(1, request.workers))
        return {"side": side, "label": f"{side}×{side}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── File/folder browser (for web UI) ────────────────────────────────────

import os
import string


class ListDirRequest(BaseModel):
    path: Optional[str] = None


@app.post("/list-dir")
def list_dir(request: ListDirRequest) -> dict:
    """List contents of a directory. If no path given, list drive roots (Windows) or /."""
    try:
        # No path -> list drive roots on Windows, or / on Unix
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

        p = Path(request.filePath)
        if not p.exists():
            return JSONResponse(status_code=400, content={"error": f"File not found: {request.filePath}"})
        if p.is_dir():
            return JSONResponse(status_code=400, content={"error": f"Path is a directory, not a file: {request.filePath}"})

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
    if p.is_dir():
        return None, None, f"Path is a directory, not a file: {file_path}"

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
    # If source was already uint8, values are in [0,255] - just clip.
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
        rgb = np.transpose(arr, (1, 2, 0))  # (C,H,W) -> (H,W,C)
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
    """POST version - accepts filePath in body (robust for Windows paths)."""
    try:
        payload, media_type, err = _render_raster_as_image(request.filePath, request.maxDim)
        if err:
            return JSONResponse(status_code=404, content={"error": err})
        return StreamingResponse(
            io.BytesIO(payload),
            media_type=media_type,
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/raster-as-png/{file_path:path}")
def raster_as_png(file_path: str, max_dim: int = 1536):
    """GET version (legacy) - file path encoded in URL."""
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


# ─── Serve compiled frontend (for standalone exe deployment) ──────────────
# When the Vite frontend is built (`npm run build` → web_app/dist/), FastAPI
# serves it directly so no separate dev server is needed.
# In development the Vite dev server handles the frontend instead.
_STATIC_DIR = Path(__file__).parent.parent.parent / "web_app" / "dist"
if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        return FileResponse(str(_STATIC_DIR / "index.html"))