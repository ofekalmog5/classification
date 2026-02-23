#!/usr/bin/env python3
"""Debug rasterization issue"""

import sys
from pathlib import Path
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# Load raster
raster_path = Path("data/data.0.tif")
vector_path = Path("data/tlv_buildings.shp")

# Check raster
print("=== RASTER INFO ===")
with rasterio.open(raster_path) as src:
    print(f"Shape: {src.height} x {src.width}")
    print(f"Bounds: {src.bounds}")
    print(f"CRS: {src.crs}")
    print(f"Transform: {src.transform}")
    raster_array = src.read()
    print(f"Data dtype: {raster_array.dtype}, shape: {raster_array.shape}")
    print(f"Data range: min={np.min(raster_array)}, max={np.max(raster_array)}")

print("\n=== VECTOR INFO ===")
gdf = gpd.read_file(vector_path)
print(f"Features: {len(gdf)}")
print(f"CRS: {gdf.crs}")
print(f"Bounds: {gdf.total_bounds}")
print(f"First geometry type: {type(gdf.geometry.iloc[0])}")

# Try rasterization
print("\n=== RASTERIZATION TEST ===")
shapes = [(geom, 4) for geom in gdf.geometry if geom is not None and not geom.is_empty]
print(f"Valid geometries: {len(shapes)}")

with rasterio.open(raster_path) as src:
    transform = src.transform
    burned = rasterize(
        shapes=shapes,
        out_shape=(src.height, src.width),
        transform=transform,
        fill=0,
        all_touched=True
    )
    print(f"Burned pixels: {np.sum(burned > 0)}")
    print(f"Burned array range: min={np.min(burned)}, max={np.max(burned)}")
    print(f"Unique values: {np.unique(burned)}")

print("\n=== CHECKING CRS TRANSFORMATION ===")
# Transform vector to match raster CRS
with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    
print(f"Vector original CRS: {gdf.crs}")
print(f"Raster CRS: {raster_crs}")
print(f"CRS match: {str(gdf.crs) == str(raster_crs)}")

# Try transforming
if str(gdf.crs) != str(raster_crs):
    print("Attempting transformation...")
    gdf_transformed = gdf.to_crs(raster_crs)
    print(f"Transformed bounds: {gdf_transformed.total_bounds}")
    
    # Try rasterization with transformed
    shapes_trans = [(geom, 4) for geom in gdf_transformed.geometry if geom is not None and not geom.is_empty]
    with rasterio.open(raster_path) as src:
        transform = src.transform
        burned_trans = rasterize(
            shapes=shapes_trans,
            out_shape=(src.height, src.width),
            transform=transform,
            fill=0,
            all_touched=True
        )
        print(f"Burned pixels after transform: {np.sum(burned_trans > 0)}")
