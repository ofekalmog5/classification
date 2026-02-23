#!/usr/bin/env python3
"""Test the refactored API endpoints"""

import sys
from pathlib import Path
import json

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.core import classify_and_export, rasterize_vectors_onto_classification, classify


def test_step1_only():
    """Test Step 1: Classification and export only"""
    print("\n" + "="*70)
    print("TEST 1: Step 1 Only (Classification + Export)")
    print("="*70)
    
    raster_path = str(Path(__file__).parent / "data" / "data.vrt")
    classes = [
        {"id": "1", "name": "Building", "color": "#FF0000"},
        {"id": "2", "name": "Water", "color": "#0000FF"},
        {"id": "3", "name": "Vegetation", "color": "#00FF00"},
    ]
    
    output_path = str(Path(__file__).parent / "output" / "test_step1.tif")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    result = classify_and_export(
        raster_path=raster_path,
        classes=classes,
        smoothing="median_1",
        feature_flags={
            "spectral": True,
            "texture": True,
            "indices": True
        },
        output_path=output_path
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
    if result["status"] == "ok":
        print(f"✓ Step 1 SUCCESS - Output: {result['outputPath']}")
        return True
    else:
        print(f"✗ Step 1 FAILED - {result.get('message', 'Unknown error')}")
        return False


def test_complete_pipeline():
    """Test complete pipeline with wrapper"""
    print("\n" + "="*70)
    print("TEST 2: Complete Pipeline (Classification + Vectors)")
    print("="*70)
    
    raster_path = str(Path(__file__).parent / "data" / "data.vrt")
    classes = [
        {"id": "1", "name": "Building", "color": "#FF0000"},
        {"id": "2", "name": "Water", "color": "#0000FF"},
        {"id": "3", "name": "Vegetation", "color": "#00FF00"},
    ]
    vector_layers = []  # Start with no vectors
    
    output_path = str(Path(__file__).parent / "output" / "test_complete.tif")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    result = classify(
        raster_path=raster_path,
        classes=classes,
        vector_layers=vector_layers,
        smoothing="median_1",
        feature_flags={
            "spectral": True,
            "texture": True,
            "indices": True
        },
        output_path=output_path
    )
    
    print(f"\nResult: {json.dumps(result, indent=2)}")
    
    if result["status"] == "ok":
        print(f"✓ Complete Pipeline SUCCESS - Output: {result['outputPath']}")
        return True
    else:
        print(f"✗ Complete Pipeline FAILED - {result.get('message', 'Unknown error')}")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING REFACTORED CLASSIFICATION API")
    print("="*70)
    
    test1_passed = test_step1_only()
    test2_passed = test_complete_pipeline()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Step 1 Only:        {'PASS ✓' if test1_passed else 'FAIL ✗'}")
    print(f"Complete Pipeline:  {'PASS ✓' if test2_passed else 'FAIL ✗'}")
    print("="*70 + "\n")
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)
