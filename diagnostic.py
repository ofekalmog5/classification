#!/usr/bin/env python3
"""
Diagnostic script to check vector and raster coordinate systems
"""

import geopandas as gpd
import rasterio
from pathlib import Path

# Correct paths based on actual directory structure
raster_path = Path(r"C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_ortho.tiff\out.tif").resolve()
vector_path = Path(r"C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_buildings.shp").resolve()

print("=" * 70)
print("COORDINATE SYSTEM DIAGNOSTIC")
print("=" * 70)

if not raster_path.exists():
    print(f"✗ Raster not found: {raster_path}")
    # Try alternative path
    raster_path = Path(r"C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\data.vrt").resolve()
    print(f"  Trying: {raster_path}")

if not vector_path.exists():
    print(f"✗ Vector not found: {vector_path}")
    print()
else:
    print(f"\n✓ Vector file: {vector_path}")
    print(f"  Exists: {vector_path.exists()}")

print("\n--- RASTER INFO ---")
if raster_path.exists():
    with rasterio.open(raster_path) as src:
        print(f"File: {raster_path.name}")
        print(f"Shape: {src.height}x{src.width}, {src.count} bands")
        print(f"Transform:\n{src.transform}")
        print(f"CRS: {src.crs}")
        print(f"CRS WKT:\n{src.crs.to_wkt()}")
        
        # Calculate bounds
        from rasterio.windows import bounds
        window = rasterio.windows.Window(0, 0, src.width, src.height)
        minx, miny, maxx, maxy = rasterio.windows.bounds(window, src.transform)
        print(f"Bounds: minx={minx:.2f}, miny={miny:.2f}, maxx={maxx:.2f}, maxy={maxy:.2f}")
else:
    print("✗ Raster file not found!")

print("\n--- VECTOR INFO ---")
if vector_path.exists():
    gdf = gpd.read_file(vector_path)
    print(f"File: {vector_path.name}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"CRS WKT:\n{gdf.crs.to_wkt()}")
    
    bounds = gdf.total_bounds
    print(f"Bounds: minx={bounds[0]:.2f}, miny={bounds[1]:.2f}, maxx={bounds[2]:.2f}, maxy={bounds[3]:.2f}")
    
    print(f"\nFirst 3 geometries:")
    for idx, (_, row) in enumerate(gdf.head(3).iterrows()):
        geom = row.geometry
        print(f"  {idx}: {geom.geom_type} - bounds: {geom.bounds}")
else:
    print("✗ Vector file not found!")

print("\n--- CRS COMPARISON ---")
if raster_path.exists() and vector_path.exists():
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
    gdf = gpd.read_file(vector_path)
    vector_crs = gdf.crs
    
    print(f"CRS match: {str(raster_crs) == str(vector_crs)}")
    print(f"CRS equal: {raster_crs == vector_crs}")
    
    if str(raster_crs) != str(vector_crs):
        print(f"\nCRS MISMATCH!")
        print(f"Raster CRS: {raster_crs}")
        print(f"Vector CRS: {vector_crs}")
        
        print(f"\nTrying to transform vector to raster CRS...")
        try:
            gdf_transformed = gdf.to_crs(raster_crs)
            bounds = gdf_transformed.total_bounds
            print(f"✓ Transformation successful!")
            print(f"Transformed bounds: minx={bounds[0]:.2f}, miny={bounds[1]:.2f}, maxx={bounds[2]:.2f}, maxy={bounds[3]:.2f}")
        except Exception as e:
            print(f"✗ Transformation failed: {e}")

print("\n" + "=" * 70)
