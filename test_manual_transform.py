#!/usr/bin/env python3
"""
Test manual Web Mercator transformation
"""
import os
import sys
import math

# Set up environment
proj_paths = [
    r'C:\Users\ofeka\anaconda3\envs\py37\Library\share\proj',
    r'C:\Users\ofeka\anaconda3\envs\py311\Library\share\proj', 
    r'C:\Users\ofeka\anaconda3\Library\share\proj',
]

for proj_path in proj_paths:
    if os.path.exists(proj_path):
        os.environ['PROJ_LIB'] = proj_path
        break

# Import geospatial libraries
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
import numpy as np

# Manual transformation functions
def _web_mercator_forward(lon: float, lat: float) -> tuple:
    """Convert WGS84 (lon, lat) to Web Mercator (x, y) coordinates"""
    EARTH_RADIUS = 20037508.34  # meters
    x = lon * EARTH_RADIUS / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * EARTH_RADIUS / math.pi
    return x, y


def _transform_geometries_to_web_mercator(gdf):
    """Transform GeoDataFrame from EPSG:4326 to Web Mercator projection"""
    
    def transform_geom(geom):
        """Recursively transform a geometry"""
        if geom.is_empty:
            return geom
        
        if isinstance(geom, Point):
            x, y = _web_mercator_forward(geom.x, geom.y)
            return Point(x, y)
        
        elif isinstance(geom, LineString):
            coords = [_web_mercator_forward(x, y) for x, y in geom.coords]
            return LineString(coords)
        
        elif isinstance(geom, Polygon):
            exterior = [_web_mercator_forward(x, y) for x, y in geom.exterior.coords]
            interiors = [[_web_mercator_forward(x, y) for x, y in interior.coords] 
                        for interior in geom.interiors]
            return Polygon(exterior, interiors)
        
        elif isinstance(geom, (MultiPoint, MultiLineString, MultiPolygon)):
            return type(geom)([transform_geom(g) for g in geom.geoms])
        
        else:
            return geom
    
    gdf_copy = gdf.copy()
    gdf_copy['geometry'] = gdf_copy['geometry'].apply(transform_geom)
    
    # Update CRS
    local_crs_str = 'LOCAL_CS["WGS 84 / Pseudo-Mercator",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    try:
        gdf_copy = gdf_copy.set_crs(local_crs_str, allow_override=True)
    except:
        gdf_copy.crs = local_crs_str
    
    return gdf_copy


# Paths
vector_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_buildings.shp'
raster_path = r'C:\Users\ofeka\OneDrive\Desktop\Jobs\mil\classification\data\tlv_ortho.tiff\out.tif'

print("="*70)
print("Testing Manual Web Mercator Transformation")
print("="*70)

# Read data
print("\n[1] Reading data...")
gdf = gpd.read_file(vector_path)
print(f"    Loaded {len(gdf)} geometries from vector")
print(f"    Original CRS: {gdf.crs}")
print(f"    Original bounds: {gdf.total_bounds}")

with rasterio.open(raster_path) as src:
    raster_crs = src.crs
    raster_bounds = src.bounds
    print(f"    Raster CRS: {raster_crs}")
    print(f"    Raster bounds: {raster_bounds}")

# Test transformation
print("\n[2] Applying manual Web Mercator transformation...")
try:
    gdf_transformed = _transform_geometries_to_web_mercator(gdf)
    print(f"    ✓ Transformation successful!")
    print(f"    Transformed CRS: {gdf_transformed.crs}")
    print(f"    Transformed bounds: {gdf_transformed.total_bounds}")
    
    # Check if bounds now overlap with raster
    gdf_bounds = gdf_transformed.total_bounds
    raster_minx = raster_bounds.left
    raster_maxx = raster_bounds.right
    raster_miny = raster_bounds.bottom
    raster_maxy = raster_bounds.top
    
    x_overlap = not (gdf_bounds[2] < raster_minx or gdf_bounds[0] > raster_maxx)
    y_overlap = not (gdf_bounds[3] < raster_miny or gdf_bounds[1] > raster_maxy)
    total_overlap = x_overlap and y_overlap
    
    print(f"\n[3] Bounds overlap check:")
    print(f"    Transformed bounds: X=[{gdf_bounds[0]:.0f}, {gdf_bounds[2]:.0f}], Y=[{gdf_bounds[1]:.0f}, {gdf_bounds[3]:.0f}]")
    print(f"    Raster bounds:      X=[{raster_minx:.0f}, {raster_maxx:.0f}], Y=[{raster_miny:.0f}, {raster_maxy:.0f}]")
    print(f"    X-overlap: {x_overlap}, Y-overlap: {y_overlap}")
    print(f"    Total overlap: {total_overlap}")
    
    if total_overlap:
        print(f"    ✓ SUCCESS! Bounds now overlap correctly")
    else:
        print(f"    ✗ Still no overlap - check coordinate transformation")
        
except Exception as e:
    print(f"    ✗ Transformation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
