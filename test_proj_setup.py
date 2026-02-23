#!/usr/bin/env python3
"""
Test script to verify PROJ setup and transformation
"""
import os
import sys

# Try multiple PROJ_LIB paths
proj_paths = [
    r'C:\Users\ofeka\anaconda3\envs\py37\Library\share\proj',
    r'C:\Users\ofeka\anaconda3\envs\py311\Library\share\proj', 
    r'C:\Users\ofeka\anaconda3\Library\share\proj',
    r'C:\Users\ofeka\AppData\Local\Mini conda\envs\py37\Library\share\proj',
]

print("Searching for PROJ database...")
found_path = None
for path in proj_paths:
    if os.path.exists(path):
        print(f"✓ Found: {path}")
        os.environ['PROJ_LIB'] = path
        found_path = path
        break
    else:
        db_file = os.path.join(path, 'proj.db')
        print(f"✗ Not found: {path}")

if found_path:
    print(f"\nUsing PROJ_LIB={found_path}")
    proj_db = os.path.join(found_path, 'proj.db')
    print(f"Checking proj.db: {os.path.exists(proj_db)}")
    print(f"Directory contents: {os.listdir(found_path)[:5]}...")  # Show first 5 files

# Now try to do the transformation
import geopandas as gpd
from pyproj import CRS, Transformer

print("\n" + "="*70)
print("TESTING TRANSFORMATION")  
print("="*70)

vector_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_buildings.shp'
raster_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_ortho.tiff\out.tif'

# Read data
gdf = gpd.read_file(vector_path)
print(f"\nVector CRS: {gdf.crs}")
print(f"Vector bounds: {gdf.total_bounds}")

import rasterio
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    print(f"\nRaster CRS: {raster_crs}")
    print(f"Raster bounds: {src.bounds}")

# Try transformation
print("\n" + "-"*70)
print("Attempting CRS transformation...")
print("-"*70)

try:
    print(f"From: {gdf.crs}")
    print(f"To: {raster_crs}")
    
    gdf_transformed = gdf.to_crs(raster_crs)
    
    print("✓ Transformation successful!")
    print(f"Original bounds: {gdf.total_bounds}")
    print(f"Transformed bounds: {gdf_transformed.total_bounds}")
    print(f"Transformed CRS: {gdf_transformed.crs}")
    
except Exception as e:
    print(f"✗ Transformation failed: {e}")
    print(f"\nAttempting manual transformation with Transformer...")
    
    try:
        # Try to create a Transformer directly
        transformer = Transformer.from_crs(gdf.crs, raster_crs, always_xy=True)
        print(f"✓ Transformer created successfully")
        
        # Transform bounds as a test
        bounds = gdf.total_bounds
        x1, y1 = transformer.transform(bounds[0], bounds[1])
        x2, y2 = transformer.transform(bounds[2], bounds[3])
        print(f"✓ Sample transformation: ({bounds[0]}, {bounds[1]}) -> ({x1:.2f}, {y1:.2f})")
        
    except Exception as e2:
        print(f"✗ Transformer creation failed: {e2}")
