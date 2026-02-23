from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Literal, Dict, Tuple
from .core import classify as run_classify
from .core import classify_and_export as run_classify_and_export
from .core import rasterize_vectors_onto_classification as run_rasterize_vectors


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
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileOutputDir: str | None = None
    tileWorkers: int | None = None
    detectShadows: bool = False
    maxThreads: int | None = None


class ClassifyStep1Request(BaseModel):
    """Step 1: Classification & Export (without vectors)"""
    rasterPath: str
    classes: List[ClassItem]
    smoothing: Literal["none", "median_1", "median_2", "median_3", "median_5"]
    featureFlags: FeatureFlags
    outputPath: str | None = None
    tileMode: bool = False
    tileMaxPixels: int | None = None
    tileOverlap: int = 0
    tileOutputDir: str | None = None
    tileWorkers: int | None = None
    detectShadows: bool = False
    maxThreads: int | None = None


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


app = FastAPI(title="Material Classification API")

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


@app.post("/classify")
def classify(request: ClassifyRequest) -> dict:
    """Complete pipeline: Classify + Rasterize vectors (if provided)"""
    return run_classify(
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
        request.maxThreads
    )


@app.post("/classify-step1")
def classify_step1(request: ClassifyStep1Request) -> dict:
    """Step 1: KMeans classification and export to RGB (no vectors)"""
    return run_classify_and_export(
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
        request.maxThreads
    )


@app.post("/classify-step2")
def classify_step2(request: ClassifyStep2Request) -> dict:
    """Step 2: Rasterize vector layers onto existing classification file"""
    return run_rasterize_vectors(
        request.classificationPath,
        [layer.model_dump() for layer in request.vectorLayers],
        [item.model_dump() for item in request.classes],
        request.outputPath,
        request.tileMode,
        request.tileMaxPixels or 512 * 512,
        request.tileOverlap,
        request.tileOutputDir,
        request.tileWorkers,
        request.maxThreads
    )

