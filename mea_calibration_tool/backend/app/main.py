"""MEA Calibration Tool — FastAPI backend."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .profile import (
    delete_profile as _del,
    export_profile as _export,
    import_profile as _import,
    load_factory_defaults,
    load_profile as _load,
    profile_path as _path,
    save_profile as _save,
)
from .raster_io import geo_to_raster as _geo_to_raster, list_dir, raster_as_png, raster_info
from .sampling import sample_regions

app = FastAPI(title="MEA Calibration Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strip /api prefix so the frontend's Vite-proxy convention works in production too
from starlette.middleware.base import BaseHTTPMiddleware

class _StripApiPrefix(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        p = request.scope.get("path", "")
        if p.startswith("/api"):
            s = p[4:] or "/"
            request.scope["path"] = s
            request.scope["raw_path"] = s.encode()
        return await call_next(request)

app.add_middleware(_StripApiPrefix)


# ─── Profile endpoints ────────────────────────────────────────────────────────

@app.get("/profile")
def get_profile():
    profile = _load()
    return {
        "profile": profile,
        "profile_path": str(_path()),
        "active": profile is not None,
    }


@app.get("/profile/factory-defaults")
def get_factory_defaults():
    return {"defaults": load_factory_defaults()}


class SaveProfileRequest(BaseModel):
    name: str
    raster_path: str
    material_overrides: Dict[str, Any]
    bias_overrides: Optional[Dict[str, float]] = None
    frequency_prior_overrides: Optional[Dict[str, float]] = None


@app.post("/profile")
def save_profile_endpoint(req: SaveProfileRequest):
    profile = _save(req.model_dump())
    return {"status": "ok", "profile": profile}


@app.delete("/profile")
def delete_profile_endpoint():
    return {"status": "ok", "deleted": _del()}


class ImportRequest(BaseModel):
    src_path: str


@app.post("/profile/import")
def import_profile_endpoint(req: ImportRequest):
    try:
        profile = _import(req.src_path)
        return {"status": "ok", "profile": profile}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


class ExportRequest(BaseModel):
    dest_path: str


@app.post("/profile/export")
def export_profile_endpoint(req: ExportRequest):
    try:
        dest = _export(req.dest_path)
        return {"status": "ok", "dest_path": dest}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ─── Raster endpoints ─────────────────────────────────────────────────────────

class RasterPathRequest(BaseModel):
    path: str


@app.post("/raster-info")
def get_raster_info(req: RasterPathRequest):
    try:
        return raster_info(req.path)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/raster-as-png")
def get_raster_png(req: RasterPathRequest):
    try:
        data = raster_as_png(req.path)
        return {"image_base64": data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class ListDirRequest(BaseModel):
    directory: str


@app.post("/list-dir")
def list_directory(req: ListDirRequest):
    try:
        return {"entries": list_dir(req.directory)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── Sampling endpoints ───────────────────────────────────────────────────────

class SampleRegion(BaseModel):
    material: str
    x: int
    y: int
    width: int
    height: int


class SamplePixelsRequest(BaseModel):
    rasterPath: str
    regions: List[SampleRegion]


@app.post("/sample-pixels")
def sample_pixels(req: SamplePixelsRequest):
    try:
        regions = [r.model_dump() for r in req.regions]
        samples = sample_regions(req.rasterPath, regions)
        return {"status": "ok", "samples": samples}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


class GeoRectRequest(BaseModel):
    filePath: str
    latLngTopLeft: List[float]
    latLngBottomRight: List[float]


@app.post("/geo-to-raster")
def geo_to_raster_endpoint(req: GeoRectRequest):
    try:
        lat_top, lng_left = req.latLngTopLeft
        lat_bot, lng_right = req.latLngBottomRight
        result = _geo_to_raster(req.filePath, lat_top, lng_left, lat_bot, lng_right)
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ─── Native file picker (tkinter dialog — runs on server, returns path) ──────

@app.get("/pick-file")
def pick_file(filter: str = ""):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    if filter == ".tif":
        filetypes = [("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")]
    elif filter == ".json":
        filetypes = [("JSON profile", "*.json"), ("All files", "*.*")]
    else:
        filetypes = [("All files", "*.*")]
    path = filedialog.askopenfilename(parent=root, filetypes=filetypes)
    root.destroy()
    return {"path": path or None}


@app.get("/pick-save-path")
def pick_save_path(default_name: str = "calibration_profile.json"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    path = filedialog.asksaveasfilename(
        parent=root,
        defaultextension=".json",
        filetypes=[("JSON profile", "*.json"), ("All files", "*.*")],
        initialfile=default_name,
    )
    root.destroy()
    return {"path": path or None}


# ─── Serve built frontend ─────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent.parent.parent / "web_app" / "dist"
if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        from fastapi.responses import FileResponse, Response
        # Don't intercept API calls that slipped through (would return HTML instead of JSON)
        if full_path.startswith("api/"):
            return Response(status_code=404)
        return FileResponse(str(_STATIC_DIR / "index.html"))
