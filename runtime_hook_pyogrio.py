"""Runtime hook for PyInstaller: set GDAL_DATA and PROJ_LIB for pyogrio."""
import os
import sys

if getattr(sys, 'frozen', False):
    _meipass = sys._MEIPASS
    # pyogrio bundles gdal_data and proj_data under its package directory
    _gdal_data = os.path.join(_meipass, 'pyogrio', 'gdal_data')
    _proj_data = os.path.join(_meipass, 'pyogrio', 'proj_data')
    if os.path.isdir(_gdal_data):
        os.environ.setdefault('GDAL_DATA', _gdal_data)
    if os.path.isdir(_proj_data):
        os.environ.setdefault('PROJ_LIB', _proj_data)
    # Also check rasterio's bundled data (fallback)
    _rio_gdal = os.path.join(_meipass, 'rasterio', 'gdal_data')
    _rio_proj = os.path.join(_meipass, 'rasterio', 'proj_data')
    if os.path.isdir(_rio_gdal) and 'GDAL_DATA' not in os.environ:
        os.environ['GDAL_DATA'] = _rio_gdal
    if os.path.isdir(_rio_proj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _rio_proj
