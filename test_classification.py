#!/usr/bin/env python3
"""Direct test of classification without API"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core import classify

# Test data
raster_path = "data/data.0.tif"
vector_layers = [
    {"filePath": "data/tlv_buildings.shp"}
]
classes = [
    {"id": "class1", "name": "Class 1", "color": "#FF0000"},
    {"id": "class2", "name": "Class 2", "color": "#00FF00"},
    {"id": "class3", "name": "Class 3", "color": "#0000FF"},
]
smoothing = "none"
feature_flags = {
    "spectral": True,
    "texture": True,
    "indices": True
}
output_path = "output_debug.tif"

print("\n" + "="*70)
print("STARTING CLASSIFICATION TEST")
print("="*70)
print(f"Raster: {raster_path}")
print(f"Vectors: {[v['filePath'] for v in vector_layers]}")
print(f"Classes: {len(classes)}")
print(f"Output: {output_path}")
print("="*70 + "\n")

result = classify(
    raster_path=raster_path,
    classes=classes,
    vector_layers=vector_layers,
    smoothing=smoothing,
    feature_flags=feature_flags,
    output_path=output_path
)

print("\n" + "="*70)
print("RESULT:")
print(result)
print("="*70)
