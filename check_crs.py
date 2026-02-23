#!/usr/bin/env python3
"""Quick CRS and bounds check"""

import os
os.environ['PROJ_LIB'] = r'C:\Users\ofeka\anaconda3\envs\py37\Library\share\proj'

import geopandas as gpd
import rasterio
from pathlib import Path

raster_path = r"C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_ortho.tiff\out.tif"
vector_path = r"C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_buildings.shp"

print("\n=== RASTER ===")
with rasterio.open(raster_path) as src:
    print(f"CRS (str): {str(src.crs)}")
    print(f"CRS (repr): {repr(src.crs)}")
    print(f"Size: {src.width}x{src.height}")
    print(f"Transform: {src.transform}")

print("\n=== VECTOR ===")
gdf = gpd.read_file(vector_path)
print(f"CRS (str): {str(gdf.crs)}")
print(f"CRS (repr): {repr(gdf.crs)}")
print(f"Bounds: {gdf.total_bounds}")
print(f"Features: {len(gdf)}")

print("\n=== COMPARISON ===")
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    # Calculate raster bounds
    from rasterio.windows import bounds
    minx, miny, maxx, maxy = bounds(
        rasterio.windows.Window(0, 0, src.width, src.height),
        src.transform
    )
    print(f"Raster bounds: {minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}")

print(f"Vector bounds: {gdf.total_bounds}")

print(f"\nCRS exact match: {str(raster_crs) == str(gdf.crs)}")
print(f"CRS equal: {raster_crs == gdf.crs}")

# Try transforming
print(f"\n=== TRYING TO TRANSFORM ===")
try:
    gdf_trans = gdf.to_crs(raster_crs)
    print(f"✓ Transformed successfully")
    print(f"Transformed bounds: {gdf_trans.total_bounds}")
except Exception as e:
    print(f"✗ Transform failed: {e}")
    try:
        gdf_forced = gdf.set_crs(raster_crs, allow_override=True)
        print(f"✓ Forced CRS override")
        print(f"Force-override bounds: {gdf_forced.total_bounds}")
    except Exception as e2:
        print(f"✗ Force override also failed: {e2}")
