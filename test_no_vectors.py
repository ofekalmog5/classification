#!/usr/bin/env python3
"""Minimal test - just classify without vectors"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.core import classify

# Test data - NO VECTORS
raster_path = "data/data.0.tif"
vector_layers = []  # <-- Empty!
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
output_path = "output_no_vectors.tif"

print("\n" + "="*70)
print("MINIMAL TEST - CLASSIFICATION WITHOUT VECTORS")
print("="*70)
print(f"Raster: {raster_path}")
print(f"Vectors: NONE")
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
