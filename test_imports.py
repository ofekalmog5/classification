#!/usr/bin/env python3
"""
Test script to verify core.py can be imported and functions work
"""

import sys
import os

print("[TEST] Python environment:")
print(f"  Python: {sys.version}")
print(f"  Executable: {sys.executable}")
print(f"  Working dir: {os.getcwd()}")

print("\n[TEST] Attempting to import geopandas...")
try:
    import geopandas as gpd
    print(f"  ✓ geopandas version: {gpd.__version__}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n[TEST] Attempting to import rasterio...")
try:
    import rasterio
    print(f"  ✓ rasterio available")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n[TEST] Attempting to import core module...")
try:
    from backend.app.core import classify_and_export, rasterize_vectors_onto_classification, classify
    print(f"  ✓ All functions imported successfully!")
    print(f"    - classify()")
    print(f"    - classify_and_export()")
    print(f"    - rasterize_vectors_onto_classification()")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[TEST] ✓ All tests passed! The module is ready to use.")
