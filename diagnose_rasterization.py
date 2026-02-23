#!/usr/bin/env python3
"""
Comprehensive diagnostic script for rasterization issues.
This script will identify why rasterio.features.rasterize() produces 0 pixels.
"""

import os
import sys

# Set up PROJ_LIB before importing geopandas
proj_paths = [
    r'C:\Users\ofeka\anaconda3\envs\py37\Library\share\proj',
    r'C:\Users\ofeka\anaconda3\Library\share\proj',
    os.path.expandvars(r'%CONDA_PREFIX%\Library\share\proj'),
]

for proj_path in proj_paths:
    if os.path.exists(proj_path):
        os.environ['PROJ_LIB'] = proj_path
        print(f"✓ Set PROJ_LIB to: {proj_path}")
        break

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box

# Paths
raster_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_ortho.tiff\out.tif'
vector_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_buildings.shp'

print("\n" + "="*70)
print("RASTERIZATION DIAGNOSTIC TOOL")
print("="*70 + "\n")

# 1. Check if files exist
print("[1] FILE VERIFICATION")
print(f"    Raster exists: {os.path.exists(raster_path)}")
print(f"    Vector exists: {os.path.exists(vector_path)}")

if not os.path.exists(raster_path) or not os.path.exists(vector_path):
    print("\n[ERROR] Files not found!")
    sys.exit(1)

# 2. Read raster metadata
print("\n[2] RASTER METADATA")
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    raster_transform = src.transform
    raster_shape = src.shape
    raster_bounds = src.bounds
    
    print(f"    CRS: {raster_crs}")
    print(f"    Transform: {raster_transform}")
    print(f"    Shape (bands, height, width): {raster_shape}")
    print(f"    Bounds: {raster_bounds}")
    print(f"    Transform details:")
    print(f"      a (pixel width):  {raster_transform.a}")
    print(f"      e (pixel height): {raster_transform.e}")
    print(f"      c (x origin):     {raster_transform.c}")
    print(f"      f (y origin):     {raster_transform.f}")

# 3. Read vector data
print("\n[3] VECTOR DATA")
gdf = gpd.read_file(vector_path)
print(f"    Features: {len(gdf)}")
print(f"    CRS: {gdf.crs}")
print(f"    Bounds: {gdf.total_bounds}")  # [minx, miny, maxx, maxy]
print(f"    Bounds X: [{gdf.total_bounds[0]:.6f}, {gdf.total_bounds[2]:.6f}]")
print(f"    Bounds Y: [{gdf.total_bounds[1]:.6f}, {gdf.total_bounds[3]:.6f}]")

# 4. Check geometry validity
print("\n[4] GEOMETRY VALIDATION")
valid_geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
print(f"    Valid geometries: {len(valid_geoms)}/{len(gdf)}")
print(f"    Empty geometries: {len(gdf) - len(valid_geoms)}")

if valid_geoms:
    bounds_list = [g.bounds for g in valid_geoms]
    all_minx = min(b[0] for b in bounds_list)
    all_miny = min(b[1] for b in bounds_list)
    all_maxx = max(b[2] for b in bounds_list)
    all_maxy = max(b[3] for b in bounds_list)
    print(f"    Geometry bounds: X=[{all_minx:.6f}, {all_maxx:.6f}], Y=[{all_miny:.6f}, {all_maxy:.6f}]")

# 5. CRS comparison
print("\n[5] CRS ANALYSIS")
print(f"    Raster CRS type: {type(raster_crs)}")
print(f"    Vector CRS type: {type(gdf.crs)}")
print(f"    CRS match (string): {str(raster_crs) == str(gdf.crs)}")
print(f"    Raster CRS string:\n      {str(raster_crs)}")
print(f"    Vector CRS string:\n      {str(gdf.crs)}")

# 6. Check if transformation is needed
print("\n[6] CRS TRANSFORMATION CHECK")
if str(raster_crs) != str(gdf.crs):
    print("    ⚠ CRS mismatch detected. Attempting transformation...")
    try:
        gdf_transformed = gdf.to_crs(raster_crs)
        print("    ✓ Transformation successful!")
        print(f"    Original bounds: {gdf.total_bounds}")
        print(f"    Transformed bounds: {gdf_transformed.total_bounds}")
        gdf = gdf_transformed  # Use transformed version for rest of diagnostics
    except Exception as e:
        print(f"    ✗ Transformation failed: {e}")
        print(f"    Attempting force override...")
        try:
            gdf = gdf.set_crs(raster_crs, allow_override=True)
            print(f"    ✓ Force override successful (WARNING: coordinates may be wrong)")
        except Exception as e2:
            print(f"    ✗ Force override failed: {e2}")
else:
    print("    ✓ CRS already match")

# 7. Bounds overlap analysis
print("\n[7] BOUNDS OVERLAP ANALYSIS")
gdf_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
raster_minx = raster_bounds.left
raster_miny = raster_bounds.bottom
raster_maxx = raster_bounds.right
raster_maxy = raster_bounds.top

print(f"    Vector bounds: X=[{gdf_bounds[0]:.6f}, {gdf_bounds[2]:.6f}], Y=[{gdf_bounds[1]:.6f}, {gdf_bounds[3]:.6f}]")
print(f"    Raster bounds: X=[{raster_minx:.6f}, {raster_maxx:.6f}], Y=[{raster_miny:.6f}, {raster_maxy:.6f}]")

# Check overlap component-wise
x_overlap = not (gdf_bounds[2] < raster_minx or gdf_bounds[0] > raster_maxx)
y_overlap = not (gdf_bounds[3] < raster_miny or gdf_bounds[1] > raster_maxy)
total_overlap = x_overlap and y_overlap

print(f"    X-overlap: {x_overlap}")
print(f"    Y-overlap: {y_overlap}")
print(f"    Total overlap: {total_overlap}")

if not total_overlap:
    print("    ✗ CRITICAL: Bounds DO NOT overlap!")
    print("    This is why rasterization produces 0 pixels!")
else:
    print("    ✓ Bounds overlap")

# 8. Test rasterization
print("\n[8] RASTERIZATION TEST")
shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
print(f"    Input shapes: {len(shapes)}")

try:
    height, width = raster_shape[1], raster_shape[2]
    burned = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=raster_transform,
        fill=0,
        all_touched=True
    )
    pixels_burned = np.sum(burned > 0)
    print(f"    ✓ Rasterization executed")
    print(f"    Pixels rasterized: {pixels_burned}/{width * height}")
    
    if pixels_burned == 0:
        print("    ✗ No pixels rasterized - PROBLEM CONFIRMED")
    else:
        print(f"    ✓ {pixels_burned} pixels successfully rasterized")
        
except Exception as e:
    print(f"    ✗ Rasterization failed: {e}")

# 9. Detailed recommendations
print("\n[9] RECOMMENDATIONS")
if not total_overlap:
    print("    ⚠ PRIMARY ISSUE: Geometry and raster bounds do not overlap")
    print("    ")
    print("    Possible causes:")
    print("    1. Vector data is in a different coordinate system than the raster")
    print("    2. CRS transformation failed silently")
    print("    3. Raster transform is incorrect")
    print("    ")
    print("    Solutions to try:")
    print("    1. Verify both files are in the same geographic area")
    print("    2. Check the raster's georeference (TIFF tags)")
    print("    3. Manually inspect coordinate values with a GIS tool (QGIS)")
    print("    4. Try setting the raster CRS explicitly if it's missing")
elif pixels_burned == 0:
    print("    ⚠ Bounds overlap but no pixels rasterized")
    print("    Possible causes:")
    print("    1. Transform matrix doesn't match actual geometry coordinates")
    print("    2. Geometry coordinates are in a different precision/units")
    print("    3. Hidden CRS mismatch despite matching CRS strings")
    print("    ")
    print("    Solutions to try:")
    print("    1. Print first geometry coordinates and manually calculate pixel location")
    print("    2. Verify transform matrix matches raster's actual georeference")
    print("    3. Try buffering geometries to ensure they're at least 1 pixel in size")
else:
    print("    ✓ Everything looks good - rasterization should work!")

print("\n" + "="*70 + "\n")
