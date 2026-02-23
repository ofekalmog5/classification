#!/usr/bin/env python3
"""Create a simple test RGB file to check QGIS rendering"""

import numpy as np
import rasterio
from rasterio.transform import from_bounds

# Create a simple test image with 4 quadrants of different colors
height, width = 100, 100
rgb = np.zeros((3, height, width), dtype=np.uint8)

# Red quadrant (top-left)
rgb[0, :50, :50] = 255  # R=255

# Green quadrant (top-right)
rgb[1, :50, 50:] = 255  # G=255

# Blue quadrant (bottom-left)
rgb[2, 50:, :50] = 255  # B=255

# White quadrant (bottom-right)
rgb[:, 50:, 50:] = 255  # All channels = 255

# Write the file
profile = {
    'driver': 'GTiff',
    'height': height,
    'width': width,
    'count': 3,
    'dtype': 'uint8',
    'transform': from_bounds(-180, -90, 180, 90, width, height),
    'interleave': 'band'
}

with rasterio.open('test_colors.tif', 'w', **profile) as dst:
    dst.write(rgb)

print("Created test_colors.tif with:")
print("  Top-left (RED): R=255, G=0, B=0")
print("  Top-right (GREEN): R=0, G=255, B=0")
print("  Bottom-left (BLUE): R=0, G=0, B=255")
print("  Bottom-right (WHITE): R=255, G=255, B=255")
print("Open in QGIS to verify colors are displayed correctly")
