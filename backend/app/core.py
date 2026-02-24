from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os
import sys
import site
from concurrent.futures import ProcessPoolExecutor, as_completed
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.transform import array_bounds

# Configure PROJ library for geopandas/rasterio BEFORE importing geopandas
# This fixes "PROJ: proj_identify: Cannot find proj.db" errors
def _setup_proj_lib():
    """Setup PROJ_LIB environment variable for geopandas/rasterio"""
    def _layout_minor(proj_dir: Path) -> Optional[int]:
        try:
            import sqlite3
            db_path = proj_dir / 'proj.db'
            if not db_path.exists():
                return None
            con = sqlite3.connect(str(db_path))
            cur = con.execute(
                "select value from metadata where key='DATABASE.LAYOUT.VERSION.MINOR'"
            )
            row = cur.fetchone()
            return int(row[0]) if row else None
        except Exception:
            return None

    def _add_if_exists(paths: List[Path], value: Optional[Path]):
        if value and value.exists() and value not in paths:
            paths.append(value)

    def _collect_site_packages() -> List[Path]:
        discovered: List[Path] = []
        try:
            for path_str in site.getsitepackages():
                p = Path(path_str)
                if p.exists() and p not in discovered:
                    discovered.append(p)
        except Exception:
            pass

        try:
            user_site = site.getusersitepackages()
            if user_site:
                p = Path(user_site)
                if p.exists() and p not in discovered:
                    discovered.append(p)
        except Exception:
            pass

        try:
            for path_str in sys.path:
                p = Path(path_str)
                if p.name.lower() in {"site-packages", "dist-packages"} and p.exists() and p not in discovered:
                    discovered.append(p)
        except Exception:
            pass

        return discovered

    candidates: List[Path] = []

    # Prefer bundled proj data in the active Python environment (layout >= 5).
    for site_packages in _collect_site_packages():
        _add_if_exists(candidates, site_packages / 'rasterio' / 'proj_data')
        _add_if_exists(candidates, site_packages / 'pyogrio' / 'proj_data')

    # Then pyproj's bundled data dir if compatible.
    try:
        from pyproj import datadir as _pyproj_datadir

        pyproj_dir = _pyproj_datadir.get_data_dir()
        if pyproj_dir:
            _add_if_exists(candidates, Path(pyproj_dir))
    except Exception:
        pass

    existing = os.environ.get('PROJ_LIB')
    if existing:
        _add_if_exists(candidates, Path(existing))

    # Generic OS-level fallback if PROJ is installed system-wide.
    for system_proj in [
        Path('/usr/share/proj'),
        Path('/usr/local/share/proj'),
        Path('/opt/homebrew/share/proj'),
        Path('C:/Program Files/PROJ/share/proj'),
    ]:
        _add_if_exists(candidates, system_proj)

    for proj_dir in candidates:
        if not proj_dir.exists():
            continue
        layout_minor = _layout_minor(proj_dir)
        if layout_minor is None:
            continue
        if layout_minor >= 5:
            os.environ['PROJ_LIB'] = str(proj_dir)
            print(f"[PROJ] Set PROJ_LIB to: {proj_dir} (layout {layout_minor})")
            return

    print(f"[PROJ] WARNING: Could not find compatible proj.db (layout >= 5)")

_setup_proj_lib()

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from skimage.segmentation import slic
from skimage.filters.rank import median
from skimage.morphology import disk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

VECTOR_OVERLAY_COLOR = (255, 255, 0)


def _normalize_pseudo_mercator_crs(crs: CRS) -> CRS:
    if crs is None:
        return crs
    crs_text = str(crs)
    if crs_text.startswith("LOCAL_CS") and "Pseudo-Mercator" in crs_text:
        return CRS.from_epsg(3857)
    return crs


def _auto_tile_size(height: int, width: int, max_pixels: int) -> int:
    max_pixels = max(128 * 128, int(max_pixels))
    tile_size = int(math.sqrt(max_pixels))
    tile_size = max(128, tile_size)
    tile_size = min(tile_size, max(height, width))
    return tile_size


def _generate_tile_windows(width: int, height: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    windows: List[Tuple[int, int, int, int]] = []
    step = max(1, tile_size - max(0, overlap))
    for row in range(0, height, step):
        for col in range(0, width, step):
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            windows.append((row, col, h, w))
    return windows


def _resolve_tile_output_dir(base_path: Path, output_path: Optional[str], suffix: str) -> Path:
    if output_path:
        out_path = Path(output_path)
        if out_path.suffix:
            return out_path.with_name(out_path.stem + suffix)
        return out_path
    return base_path.with_name(base_path.stem + suffix)


def _filter_geometries_by_bounds(geoms: List, bounds: Tuple[float, float, float, float]) -> List:
    minx, miny, maxx, maxy = bounds
    return [
        geom for geom in geoms
        if geom is not None and not geom.is_empty and
        not (geom.bounds[2] < minx or geom.bounds[0] > maxx or geom.bounds[3] < miny or geom.bounds[1] > maxy)
    ]


def _web_mercator_forward(lon: float, lat: float) -> tuple:
    """Convert WGS84 (lon, lat) to Web Mercator (x, y) coordinates"""
    EARTH_RADIUS = 20037508.34  # meters
    x = lon * EARTH_RADIUS / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) * EARTH_RADIUS / math.pi
    return x, y


def _transform_geometries_to_web_mercator(gdf):
    """
    Transform GeoDataFrame from EPSG:4326 to Web Mercator projection.
    Uses manual transformation because PROJ.db may not be available.
    """
    from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
    
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
    
    # Update CRS to match the local projection
    local_crs_str = 'LOCAL_CS["WGS 84 / Pseudo-Mercator",UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
    try:
        gdf_copy = gdf_copy.set_crs(local_crs_str, allow_override=True)
    except:
        # If CRS setting fails, just force the string attribute
        gdf_copy.crs = local_crs_str
    
    return gdf_copy

from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter


def _detect_structures_mask(raster_data: np.ndarray, brightness_threshold: int = 150, ndvi_threshold: float = 0.3) -> np.ndarray:
    """
    Detect potential structures (buildings/trees) based on spectral characteristics.
    
    Args:
        raster_data: (bands, height, width) array
        brightness_threshold: Minimum average brightness for structures (0-255)
        ndvi_threshold: NDVI threshold for vegetation (trees)
    
    Returns:
        Boolean mask indicating structure pixels
    """
    height, width = raster_data.shape[1], raster_data.shape[2]
    
    # Calculate brightness as mean of RGB bands
    rgb_data = raster_data[:3] if raster_data.shape[0] >= 3 else raster_data
    brightness = np.mean(rgb_data, axis=0)
    
    # High brightness mask (buildings, bright surfaces)
    bright_mask = brightness > brightness_threshold
    
    # Vegetation mask using NDVI (trees)
    vegetation_mask = np.zeros((height, width), dtype=bool)
    if raster_data.shape[0] >= 4:
        red = raster_data[2].astype(np.float32)
        nir = raster_data[3].astype(np.float32)
        ndvi = (nir - red) / (nir + red + 1e-6)
        vegetation_mask = ndvi > ndvi_threshold
    
    # Combine: structures = bright areas + vegetation
    structure_mask = bright_mask | vegetation_mask
    
    return structure_mask


def _detect_shadows_and_infer(classification_raster: np.ndarray, original_raster: np.ndarray, dilation_radius: int = 5, brightness_threshold: int = 100) -> np.ndarray:
    """
    Detect shadows near structures and reclassify them with adjacent structure's material label.
    
    Args:
        classification_raster: (height, width) labeled classification (class IDs)
        original_raster: (bands, height, width) original raster data
        dilation_radius: How many pixels to dilate structure regions
        brightness_threshold: Max brightness to consider as shadow
    
    Returns:
        Updated classification raster with shadows reclassified
    """
    height, width = classification_raster.shape
    output_raster = classification_raster.astype(np.int32).copy()
    
    # Detect structure pixels
    structure_mask = _detect_structures_mask(original_raster)
    
    # Dilate to create a "shadow zone" around structures
    dilated_structures = binary_dilation(structure_mask, iterations=dilation_radius)
    
    # Calculate brightness
    rgb_data = original_raster[:3] if original_raster.shape[0] >= 3 else original_raster
    brightness = np.mean(rgb_data, axis=0)
    
    # Shadow pixels: low brightness within dilated structure zone
    shadow_candidates = (brightness < brightness_threshold) & dilated_structures & ~structure_mask
    
    if np.any(shadow_candidates):
        # For each shadow pixel, assign the most common adjacent structure class
        # Use maximum_filter to get the most likely surrounding class
        kernel_size = 2 * dilation_radius + 1
        
        # Find adjacent non-zero classes
        labels_float = classification_raster.astype(np.float32)
        local_max = maximum_filter(labels_float, size=kernel_size, mode='constant', cval=0)
        
        # Assign shadow pixels the value of their nearest structure
        shadow_indices = np.where(shadow_candidates)
        for y, x in zip(shadow_indices[0], shadow_indices[1]):
            # Find most common class in local neighborhood
            y_start = max(0, y - dilation_radius)
            y_end = min(height, y + dilation_radius + 1)
            x_start = max(0, x - dilation_radius)
            x_end = min(width, x + dilation_radius + 1)
            
            neighborhood = classification_raster[y_start:y_end, x_start:x_end]
            neighborhood_nonzero = neighborhood[neighborhood > 0]
            
            if len(neighborhood_nonzero) > 0:
                # Get most common class
                unique, counts = np.unique(neighborhood_nonzero, return_counts=True)
                most_common_class = unique[np.argmax(counts)]
                output_raster[y, x] = most_common_class
    
    return output_raster.astype(classification_raster.dtype)


def _classify_tile_worker(args: Tuple[str, Tuple[int, int, int, int], Dict[str, bool], np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int]], str, str, str]) -> str:
    raster_path, window_tuple, feature_flags, scaler_mean, scaler_scale, centers, color_table, smoothing, output_dir, tile_name = args
    row, col, height, width = window_tuple
    window = Window(col, row, width, height)

    with rasterio.open(raster_path) as src:
        tile_data = src.read(window=window)
        profile = src.profile.copy()
        tile_transform = window_transform(window, src.transform)
        tile_crs = _normalize_pseudo_mercator_crs(src.crs)

    profile.update(
        height=height,
        width=width,
        transform=tile_transform,
        crs=tile_crs,
        count=3,
        dtype=np.uint8,
        interleave='band'
    )

    features = _extract_pixel_features(tile_data, feature_flags, verbose=False)
    scale = np.where(scaler_scale == 0, 1.0, scaler_scale)
    features_normalized = (features - scaler_mean) / scale
    distances = cdist(features_normalized, centers, metric='euclidean')
    labels = np.argmin(distances, axis=1) + 1
    predicted_raster = labels.reshape(height, width)

    if smoothing and smoothing != "none":
        try:
            kernel_size = int(smoothing.split("_")[1]) if "_" in smoothing else 2
            predicted_raster = median(predicted_raster.astype(np.uint16), disk(kernel_size))
        except Exception:
            pass

    rgb = _apply_color_table(predicted_raster, color_table, verbose=False)

    output_path = Path(output_dir) / tile_name
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(rgb)

    return str(output_path)


def _rasterize_tile_worker(args: Tuple[str, Optional[Tuple[int, int, int, int]], List[Tuple[List, int]], str, str]) -> str:
    classification_path, window_tuple, layer_geoms, output_dir, tile_name = args

    if window_tuple is None:
        with rasterio.open(classification_path) as src:
            raster_array = src.read()
            meta = src.meta.copy()
            transform = src.transform
            height = src.height
            width = src.width
            raster_crs = _normalize_pseudo_mercator_crs(src.crs)
    else:
        row, col, height, width = window_tuple
        window = Window(col, row, width, height)
        with rasterio.open(classification_path) as src:
            raster_array = src.read(window=window)
            meta = src.meta.copy()
            transform = window_transform(window, src.transform)
            raster_crs = _normalize_pseudo_mercator_crs(src.crs)

        meta.update(height=height, width=width, transform=transform)

    meta["crs"] = raster_crs

    output_array = raster_array.copy()
    bounds = array_bounds(height, width, transform)

    for geoms, burn_value in layer_geoms:
        filtered = _filter_geometries_by_bounds(geoms, bounds)
        if not filtered:
            continue
        shapes = [(geom, burn_value) for geom in filtered]
        burned_mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            all_touched=True
        )
        if output_array.ndim == 2:
            output_array[burned_mask > 0] = burn_value
        else:
            r, g, b = VECTOR_OVERLAY_COLOR
            output_array[0][burned_mask > 0] = r
            output_array[1][burned_mask > 0] = g
            output_array[2][burned_mask > 0] = b

    output_path = Path(output_dir) / tile_name
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(output_array)

    return str(output_path)


def rasterize_vector_onto_raster(raster_path: str, gdf, burn_value: int, output_path: str, crs):
    """
    Rasterize vector GeoDataFrame onto an existing raster.
    Preserves all georeferencing from the original raster.
    
    Args:
        raster_path: Path to base raster (GeoTIFF)
        gdf: GeoDataFrame with vector data (already loaded)
        burn_value: Value to write for vector pixels
        output_path: Path to save output raster
        crs: Target CRS for vectors
    """
    print(f"\n  [RASTERIZE VECTOR] Adding vector layer to raster...")
    print(f"    Base raster: {raster_path}")
    print(f"    Burn value: {burn_value}")
    
    # Load base raster metadata
    with rasterio.open(raster_path) as src:
        raster_array = src.read()
        meta = src.meta.copy()
        transform = src.transform
        width = src.width
        height = src.height
        raster_crs = _normalize_pseudo_mercator_crs(src.crs)
        original_dtype = raster_array.dtype

    meta["crs"] = raster_crs
    
    print(f"    Base raster: {raster_path}")
    print(f"      Absolute path: {Path(raster_path).resolve()}")
    print(f"    Raster shape: {raster_array.shape}, original dtype: {original_dtype}")
    print(f"    Raster data range: min={np.min(raster_array)}, max={np.max(raster_array)}")
    print(f"    Transform: {transform}")
    print(f"    CRS: {raster_crs}")
    print(f"    Meta: {meta}")
    
    # GeoDataFrame is already loaded, just match CRS if needed
    print(f"    Vector features: {len(gdf)}")
    print(f"    Vector CRS: {gdf.crs}")
    
    # Match CRS if needed  
    if gdf.crs is None:
        gdf = gdf.set_crs(crs, allow_override=True)
        print(f"    Set vector CRS to match raster")
    elif str(gdf.crs) != str(crs):
        print(f"    [INFO] CRS mismatch - transforming geometries")
        print(f"      From: {gdf.crs}")
        print(f"      To: {crs}")
        
        try:
            gdf = gdf.to_crs(crs)
            print(f"      ✓ Transformation successful!")
            print(f"      Original bounds: {gdf.total_bounds}")
            print(f"      Transformed bounds: {gdf.to_crs(crs).total_bounds}")
        except Exception as e:
            print(f"      [ERROR] Standard transformation failed: {e}")
            print(f"      Attempting manual transformation...")
            try:
                # Fall back to manual Web Mercator if standard transformation fails
                gdf = _transform_geometries_to_web_mercator(gdf)
                print(f"      ✓ Manual transformation successful!")
            except Exception as e2:
                print(f"      [ERROR] Manual transformation also failed: {e2}")
                raise RuntimeError("CRS transformation failed; cannot rasterize with mismatched coordinates")
    
    print(f"    Vector CRS after processing: {gdf.crs}")
    
    # Create geometry-value pairs
    shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    print(f"    Valid geometries: {len(shapes)}")
    
    if not shapes:
        print(f"    WARNING: No valid geometries to rasterize")
        # Just copy the raster
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(raster_array)
        return
    
    # Debug: Check geometry bounds with detailed analysis
    print(f"    [BOUNDS ANALYSIS]")
    gdf_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Calculate raster bounds from transform
    raster_minx = transform.c
    raster_maxx = transform.c + width * transform.a  # a is pixel width (usually positive)
    raster_miny = transform.f + height * transform.e  # e is pixel height (usually negative)
    raster_maxy = transform.f
    
    print(f"    Vector bounds: X=[{gdf_bounds[0]:.6f}, {gdf_bounds[2]:.6f}], Y=[{gdf_bounds[1]:.6f}, {gdf_bounds[3]:.6f}]")
    print(f"    Raster bounds: X=[{raster_minx:.6f}, {raster_maxx:.6f}], Y=[{raster_miny:.6f}, {raster_maxy:.6f}]")
    print(f"    Raster transform: a={transform.a}, e={transform.e}, c={transform.c}, f={transform.f}")
    print(f"    Raster size: {width}x{height}")
    
    # Check if bounds overlap (component-wise)
    bounds_x_overlap = not (gdf_bounds[2] < raster_minx or gdf_bounds[0] > raster_maxx)
    bounds_y_overlap = not (gdf_bounds[3] < raster_miny or gdf_bounds[1] > raster_maxy)
    bounds_overlap = bounds_x_overlap and bounds_y_overlap
    
    print(f"    X-overlap: {bounds_x_overlap}, Y-overlap: {bounds_y_overlap}, Total overlap: {bounds_overlap}")
    
    if not bounds_overlap:
        print(f"    [CRITICAL] Geometry bounds do NOT overlap with raster bounds!")
        print(f"    This is likely why rasterization produces 0 pixels.")
    
    # Count geometries actually within raster bounds
    valid_in_bounds = sum(1 for geom in gdf.geometry 
                         if (geom.bounds[2] > raster_minx and geom.bounds[0] < raster_maxx and
                             geom.bounds[3] > raster_miny and geom.bounds[1] < raster_maxy))
    print(f"    Geometries within raster bounds: {valid_in_bounds}/{len(gdf)}")
    
    # Rasterize vector
    print(f"    Rasterizing {len(shapes)} geometries...")
    burned_mask = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True
    )
    
    pixels_burned = np.sum(burned_mask > 0)
    print(f"    Pixels rasterized: {pixels_burned}")
    
    # If no pixels were burned, try to debug and potentially fix
    if pixels_burned == 0:
        print(f"\n    [DEBUG] No pixels rasterized! Analyzing...")
        print(f"      - Number of shapes: {len(shapes)}")
        print(f"      - Raster size: {width}x{height}")
        print(f"      - Transform: {transform}")
        print(f"      - CRS: {crs}")
        print(f"      - Bounds overlap: {bounds_overlap}")
        print(f"      - Geometries in bounds: {valid_in_bounds}")
        
        # Check if GEOMETRIES themselves are valid and non-empty
        empty_count = sum(1 for geom, _ in shapes if geom.is_empty)
        print(f"      - Empty geometries in shapes list: {empty_count}")
        
        # Print first few geometry details
        if shapes:
            for i in range(min(3, len(shapes))):
                geom, val = shapes[i]
                bounds = geom.bounds
                print(f"      - Shape {i}: bounds={bounds}, area={geom.area:.2f}")
        
        print(f"    [POSSIBLE CAUSES]")
        print(f"      1. Coordinate system mismatch between raster and geometries")
        print(f"      2. All geometries are outside the raster bounds")  
        print(f"      3. Transform matrix is incorrect for the geometry coordinates")
        print(f"      4. Geometry coordinates are in a different projection than raster")
        print(f"    [SUGGESTION] Check that vector and raster are in the SAME coordinate system")
        print(f"                 and that geometry coordinates actually map to raster pixels.")
        
        # Try to verify first geometry bounds
        if shapes:
            first_geom = shapes[0][0]
            print(f"      - First geometry bounds: {first_geom.bounds}")
            print(f"      - First geometry is_valid: {first_geom.is_valid}")
        
        # POTENTIAL FIX: If bounds don't overlap, geometries might be in different CRS
        # Try to reproject geometries to match raster bounds more closely
        if not bounds_overlap:
            print(f"      [ATTEMPTING FIX] Geometries appear to be outside raster bounds")
            # Don't try to fix - just warn
            print(f"      Consider checking if vector is in the same projection as raster")
    
    # Merge with raster - write burned pixels with burn_value
    print(f"    Original raster dtype: {raster_array.dtype}, shape: {raster_array.shape}")
    print(f"    Burned mask stats: min={np.min(burned_mask)}, max={np.max(burned_mask)}, sum={np.sum(burned_mask > 0)}")
    print(f"    Burn value: {burn_value} (type: {type(burn_value).__name__})")
    
    if raster_array.ndim == 2:
        # Single band
        output_array = raster_array.copy()
        output_array[burned_mask > 0] = burn_value
    else:
        # Multi-band - paint overlay color on all bands
        output_array = raster_array.copy()
        r, g, b = VECTOR_OVERLAY_COLOR
        output_array[0][burned_mask > 0] = r
        output_array[1][burned_mask > 0] = g
        output_array[2][burned_mask > 0] = b
    
    print(f"    Output array dtype: {output_array.dtype}, shape: {output_array.shape}")
    print(f"    Output array range: min={np.min(output_array)}, max={np.max(output_array)}")
    unique_values = np.unique(output_array)
    print(f"    Unique values in output: {unique_values[:20]}..." if len(unique_values) > 20 else f"    Unique values in output: {unique_values}")
    
    # Save result
    print(f"    Saving to: {output_path}")
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(output_array)
    
    print(f"    [OK] Vector rasterized successfully")


def classify(
    raster_path: str,
    classes: List[Dict[str, str]],
    vector_layers: List[Dict[str, str]],
    smoothing: str,
    feature_flags: Dict[str, bool],
    output_path: str | None = None,
    tile_mode: bool = False,
    tile_max_pixels: int = 512 * 512,
    tile_overlap: int = 0,
    tile_output_dir: str | None = None,
    tile_workers: Optional[int] = None,
    detect_shadows: bool = False,
    max_threads: Optional[int] = None
) -> Dict[str, object]:
    """
    Complete classification pipeline: KMeans + Vector rasterization.
    This is a convenience wrapper that calls classify_and_export() then rasterize_vectors_onto_classification().
    """
    import tempfile
    
    print("\n" + "="*70)
    print("COMPLETE CLASSIFICATION PIPELINE")
    print("="*70)
    
    # === STEP 1: Classify and export ===
    print("\n>>> STEP 1: Classification & Export")
    temp_dir = Path(tempfile.gettempdir())
    temp_output = temp_dir / f"classification_temp_{np.random.randint(1000000)}.tif"
    
    result1 = classify_and_export(
        raster_path=raster_path,
        classes=classes,
        smoothing=smoothing,
        feature_flags=feature_flags,
        output_path=str(temp_output) if not tile_mode else None,
        tile_mode=tile_mode,
        tile_max_pixels=tile_max_pixels,
        tile_overlap=tile_overlap,
        tile_output_dir=tile_output_dir,
        tile_workers=tile_workers
    )
    
    if result1["status"] != "ok":
        return result1
    
    classif_file = result1["outputPath"]
    print(f"Classification saved to: {classif_file}")
    
    # === STEP 2: Rasterize vectors ===
    if vector_layers:
        print("\n>>> STEP 2: Vector Rasterization")
        result2 = rasterize_vectors_onto_classification(
            classification_path=classif_file,
            vector_layers=vector_layers,
            classes=classes,
            output_path=output_path,
            tile_mode=tile_mode,
            tile_max_pixels=tile_max_pixels,
            tile_overlap=tile_overlap,
            tile_output_dir=tile_output_dir,
            tile_workers=tile_workers,
            max_threads=max_threads
        )
        
        if result2["status"] != "ok":
            return result2
        
        final_output = result2["outputPath"]
    else:
        print("\n>>> STEP 2: No vectors provided, skipping")
        final_output = classif_file
        # Copy to final output path if provided
        if output_path and not tile_mode:
            import shutil
            shutil.copy(classif_file, output_path)
            final_output = output_path
    
    print("\n" + "="*70)
    print("[OK] COMPLETE CLASSIFICATION PIPELINE FINISHED")
    print("="*70)
    print(f"Final output: {final_output}")
    print("="*70 + "\n")
    
    return {
        "status": "ok",
        "outputPath": str(final_output)
    }


def classify_and_export(
    raster_path: str,
    classes: List[Dict[str, str]],
    smoothing: str,
    feature_flags: Dict[str, bool],
    output_path: str | None = None,
    tile_mode: bool = False,
    tile_max_pixels: int = 512 * 512,
    tile_overlap: int = 0,
    tile_output_dir: str | None = None,
    tile_workers: Optional[int] = None,
    detect_shadows: bool = False,
    max_threads: Optional[int] = None
) -> Dict[str, object]:
    """
    Step 1: KMeans classification and color export (without vectors).
    Outputs an RGB GeoTIFF with the classified clusters colored.
    
    Returns: {"status": "ok", "outputPath": "..."}
    """
    path = Path(raster_path)
    if not path.exists():
        return {"status": "error", "message": "Raster path not found"}

    n_clusters = len(classes)
    
    print("="*70)
    print("STEP 1: CLASSIFICATION & EXPORT (No Vectors)")
    print("="*70)

    # === Load raster ===
    print(f"\n[1/5] Loading raster: {path.name}")
    with rasterio.open(path) as src:
        raster_data = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = _normalize_pseudo_mercator_crs(src.crs)
        height, width = raster_data.shape[1], raster_data.shape[2]
        n_bands = raster_data.shape[0]

    profile["crs"] = crs
    
    print(f"  Dimensions: {height}x{width}, {n_bands} bands")
    print(f"  CRS: {crs}")

    # === Extract features ===
    print(f"\n[2/5] Extracting pixel-level features...")
    pixel_features = _extract_pixel_features(raster_data, feature_flags)
    print(f"  Feature vector shape: {pixel_features.shape}")

    # === KMeans clustering ===
    print(f"\n[3/5] Running KMeans clustering ({n_clusters} clusters)...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(pixel_features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(features_normalized)
    print(f"  [OK] KMeans fitted")

    if tile_mode:
        print(f"\n[4/5] Tiled classification (multiprocessing)...")
        tile_size = _auto_tile_size(height, width, tile_max_pixels)
        windows = _generate_tile_windows(width, height, tile_size, tile_overlap)
        output_dir = _resolve_tile_output_dir(path, tile_output_dir, "_classified_tiles")
        output_dir.mkdir(parents=True, exist_ok=True)

        color_table = _build_color_table(classes, n_clusters)
        scaler_mean = scaler.mean_.astype(np.float32)
        scaler_scale = scaler.scale_.astype(np.float32)
        centers = kmeans.cluster_centers_.astype(np.float32)

        # Determine max workers: use tile_workers if specified, or fill up to max_threads (if set), else all CPUs
        if tile_workers and tile_workers > 0:
            max_workers = tile_workers
        elif max_threads and max_threads > 0:
            max_workers = max_threads
        else:
            max_workers = max(1, os.cpu_count() or 1)
        
        jobs = []
        for row, col, h, w in windows:
            tile_name = f"{path.stem}_tile_r{row}_c{col}.tif"
            jobs.append((
                raster_path,
                (row, col, h, w),
                feature_flags,
                scaler_mean,
                scaler_scale,
                centers,
                color_table,
                smoothing,
                str(output_dir),
                tile_name
            ))

        tile_outputs: List[str] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_classify_tile_worker, job) for job in jobs]
            for future in as_completed(futures):
                tile_outputs.append(future.result())

        print(f"  [OK] Wrote {len(tile_outputs)} tiles to {output_dir}")
        print("\n" + "="*70)
        print("[OK] STEP 1 COMPLETE: Classification & Export (Tiles)")
        print("="*70)

        return {
            "status": "ok",
            "outputPath": str(output_dir),
            "tileOutputs": sorted(tile_outputs),
            "message": "Classification complete (tiles). Use output directory for Step 2."
        }

    # === NN assignment ===
    print(f"\n[4/5] Assigning pixels to clusters...")
    distances = cdist(features_normalized, kmeans.cluster_centers_, metric='euclidean')
    pixel_labels = np.argmin(distances, axis=1) + 1
    predicted_raster = pixel_labels.reshape(height, width)
    
    unique_classes = np.unique(predicted_raster)
    print(f"  [OK] Classes: {unique_classes}")
    
    # === Smoothing ===
    print(f"\n[5/5] Smoothing...")
    if smoothing and smoothing != "none":
        try:
            kernel_size = int(smoothing.split("_")[1]) if "_" in smoothing else 2
            print(f"  Applying median filter (kernel size: {kernel_size})...")
            predicted_raster = median(predicted_raster.astype(np.uint16), disk(kernel_size))
            print(f"  [OK] Smoothing applied")
        except Exception as e:
            print(f"  Smoothing error: {e}, skipping")
    
    print(f"  Classes after smoothing: {np.unique(predicted_raster)}")

    # === Shadow detection and inference ===
    if detect_shadows:
        print(f"\n[6/6] Detecting and inferring shadows...")
        predicted_raster = _detect_shadows_and_infer(predicted_raster, raster_data)
        print(f"  [OK] Shadow detection complete")
        step_num = 7
    else:
        step_num = 6
    
    # === Save classification ===
    if output_path:
        output_color_path = Path(output_path)
    else:
        output_color_path = path.with_name(path.stem + "_classified.tif")
    
    print(f"\n[{step_num}/{step_num}] Saving classified output...")
    print(f"  Output: {output_color_path}")
    
    # Compute colors
    color_table = _compute_all_colors(raster_data, predicted_raster, n_clusters, 0, classes)
    
    # Apply colors
    rgb = _apply_color_table(predicted_raster, color_table)
    
    # Write RGB output
    rgb_profile = profile.copy()
    rgb_profile.update(
        count=3,
        dtype=np.uint8,
        driver='GTiff',
        interleave='band'
    )
    
    if Path(output_color_path).exists():
        Path(output_color_path).unlink()
    
    with rasterio.open(output_color_path, 'w', **rgb_profile) as dst:
        dst.write(rgb)
    
    print(f"  [OK] Classification saved to {output_color_path}")
    
    print("\n" + "="*70)
    print("[OK] STEP 1 COMPLETE: Classification & Export")
    print("="*70)
    
    return {
        "status": "ok",
        "outputPath": str(output_color_path),
        "message": "Classification complete. Use output file for Step 2 (vector rasterization)."
    }


def rasterize_vectors_onto_classification(
    classification_path: str,
    vector_layers: List[Dict[str, str]],
    classes: List[Dict[str, str]],
    output_path: str | None = None,
    tile_mode: bool = False,
    tile_max_pixels: int = 512 * 512,
    tile_overlap: int = 0,
    tile_output_dir: str | None = None,
    tile_workers: Optional[int] = None,
    max_threads: Optional[int] = None
) -> Dict[str, object]:
    """
    Step 2: Rasterize vector layers onto an existing classification file.
    Takes the RGB classification from Step 1 and overlays vector geometries.
    
    Args:
        classification_path: Path to classification RGB file (from Step 1)
        vector_layers: List of vector layers with filePath
        classes: Class definitions (for reference)
        output_path: Optional output path (defaults to input with suffix)
    
    Returns: {"status": "ok", "outputPath": "..."}
    """
    classif_path = Path(classification_path)
    if not classif_path.exists():
        return {"status": "error", "message": f"Classification file not found: {classification_path}"}
    
    print("="*70)
    print("STEP 2: VECTOR RASTERIZATION")
    print("="*70)
    
    # === Load classification metadata ===
    print(f"\n[1/3] Loading classification: {classif_path.name}")
    classif_data = None
    if classif_path.is_dir():
        tile_files = sorted(classif_path.glob("*.tif"))
        if not tile_files:
            return {"status": "error", "message": f"No tiles found in: {classification_path}"}
        with rasterio.open(tile_files[0]) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = _normalize_pseudo_mercator_crs(src.crs)
            height, width = src.height, src.width
    elif tile_mode:
        with rasterio.open(classif_path) as src:
            profile = src.profile.copy()
            transform = src.transform
            crs = _normalize_pseudo_mercator_crs(src.crs)
            height, width = src.height, src.width
    else:
        with rasterio.open(classif_path) as src:
            classif_data = src.read(1)  # Read first band (classification values)
            profile = src.profile.copy()
            transform = src.transform
            crs = _normalize_pseudo_mercator_crs(src.crs)
            height, width = classif_data.shape

    profile["crs"] = crs
    
    print(f"  Shape: {height}x{width}")
    if classif_data is not None:
        print(f"  Data range: {np.min(classif_data)}-{np.max(classif_data)}")
        print(f"  Classes: {np.unique(classif_data)}")
    
    # Update profile to single-band for intermediate processing
    profile.update(count=1, dtype=np.uint16, interleave='band')
    
    # === Load and validate vectors ===
    print(f"\n[2/3] Validating {len(vector_layers)} vector layers...")
    validated_vectors = []
    
    for idx, layer in enumerate(vector_layers):
        layer_path = Path(layer["filePath"])
        print(f"\n  Layer {idx+1}/{len(vector_layers)}: {layer_path.name}")
        
        try:
            if not layer_path.exists():
                print(f"    [ERROR] File not found: {layer_path}")
                continue
                
            gdf = gpd.read_file(layer_path)
            if gdf.empty:
                print(f"    [ERROR] Empty GeoDataFrame")
                continue
            
            print(f"    Features: {len(gdf)}")
            print(f"    Current CRS: {gdf.crs}")
            
            # Match CRS if needed
            if gdf.crs is None:
                gdf = gdf.set_crs(crs, allow_override=True)
                print(f"    [OK] Set CRS to match classification")
            elif str(gdf.crs) != str(crs):
                try:
                    gdf = gdf.to_crs(crs)
                    print(f"    [OK] Transformed CRS")
                except Exception as e:
                    print(f"    [ERROR] CRS transform failed: {e}")
                    print(f"    [SKIP] Not forcing CRS; vector would be in the wrong coordinates")
                    continue
            
            validated_vectors.append((layer_path.name, gdf))
            print(f"    [OK] Validated")
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
    
    print(f"\n  Total validated: {len(validated_vectors)}/{len(vector_layers)}")
    
    if not validated_vectors:
        print(f"\n  No valid vectors to rasterize. Returning original classification.")
        return {
            "status": "ok",
            "outputPath": str(classif_path),
            "message": "No vectors to rasterize"
        }

    if tile_mode or classif_path.is_dir():
        print(f"\n[3/3] Rasterizing vectors (tiles)...")
        output_dir = _resolve_tile_output_dir(classif_path, tile_output_dir or output_path, "_with_vectors_tiles")
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_geoms: List[Tuple[List, int]] = []
        n_clusters = len(classes)
        for idx, (_, gdf) in enumerate(validated_vectors):
            burn_value = n_clusters + idx + 1
            geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
            layer_geoms.append((geoms, burn_value))

        jobs = []
        if classif_path.is_dir():
            tile_inputs = sorted(classif_path.rglob("*.tif"))
            if not tile_inputs:
                return {"status": "error", "message": f"No tiles found in: {classification_path}"}
            print(f"  Tiles found: {len(tile_inputs)}")
            for tile_path in tile_inputs:
                rel_name = tile_path.relative_to(classif_path).as_posix().replace("/", "__")
                jobs.append((str(tile_path), None, layer_geoms, str(output_dir), rel_name))
        else:
            tile_size = _auto_tile_size(height, width, tile_max_pixels)
            windows = _generate_tile_windows(width, height, tile_size, tile_overlap)
            print(f"  Tiles planned: {len(windows)} (tile_size={tile_size}, overlap={tile_overlap})")
            for row, col, h, w in windows:
                tile_name = f"{classif_path.stem}_tile_r{row}_c{col}.tif"
                jobs.append((str(classif_path), (row, col, h, w), layer_geoms, str(output_dir), tile_name))

        # Determine max workers: use tile_workers if specified, or fill up to max_threads (if set), else all CPUs
        if tile_workers and tile_workers > 0:
            max_workers = tile_workers
        elif max_threads and max_threads > 0:
            max_workers = max_threads
        else:
            max_workers = max(1, os.cpu_count() or 1)
        
        tile_outputs: List[str] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_rasterize_tile_worker, job) for job in jobs]
            for future in as_completed(futures):
                tile_outputs.append(future.result())

        print(f"\n[OK] Vector rasterization complete (tiles)")
        print(f"  Output: {output_dir}")
        print("="*70)

        return {
            "status": "ok",
            "outputPath": str(output_dir),
            "tileOutputs": sorted(tile_outputs),
            "message": "Vector rasterization complete (tiles)"
        }
    
    # === Rasterize vectors ===
    print(f"\n[3/3] Rasterizing {len(validated_vectors)} vector layers...")
    
    n_clusters = len(classes)
    working_raster = classif_path
    
    for idx, (layer_name, gdf) in enumerate(validated_vectors):
        vector_class_id = n_clusters + idx + 1
        
        print(f"\n  [Vector {idx+1}/{len(validated_vectors)}] {layer_name}")
        print(f"    Material ID: {vector_class_id}")
        
        # Determine output path
        if output_path and idx == len(validated_vectors) - 1:
            final_output = Path(output_path)
        else:
            final_output = classif_path.with_name(
                classif_path.stem + f"_with_vectors_{idx}" + classif_path.suffix
            )
        
        try:
            # Rasterize this vector
            rasterize_vector_onto_raster(
                raster_path=str(working_raster),
                gdf=gdf,
                burn_value=vector_class_id,
                output_path=str(final_output),
                crs=crs
            )
            
            print(f"    [OK] Rasterized")
            
            # Use as base for next iteration
            if idx < len(validated_vectors) - 1:
                working_raster = final_output
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Final output path
    final_path = Path(output_path) if output_path else classif_path.with_name(
        classif_path.stem + "_with_vectors" + classif_path.suffix
    )
    
    print(f"\n[OK] Vector rasterization complete")
    print(f"  Output: {final_path}")
    print("="*70)
    
    return {
        "status": "ok",
        "outputPath": str(final_path),
        "message": "Vector rasterization complete"
    }


def recommend_cluster_count(
    raster_path: str,
    feature_flags: Dict[str, bool],
    min_clusters: int = 2,
    max_clusters: int = 10
) -> int:
    """
    Analyze raster and recommend optimal number of clusters.
    Uses robust random sampling and a weighted combination of:
    - Silhouette score (higher is better)
    - Calinski-Harabasz score (higher is better)
    - Davies-Bouldin score (lower is better)
    - Inertia elbow curvature (knee preference)
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    def _normalize(values: Dict[int, float], invert: bool = False) -> Dict[int, float]:
        if not values:
            return {}
        min_v = min(values.values())
        max_v = max(values.values())
        if abs(max_v - min_v) < 1e-12:
            out = {k: 0.5 for k in values.keys()}
        else:
            out = {k: (v - min_v) / (max_v - min_v) for k, v in values.items()}
        if invert:
            out = {k: 1.0 - v for k, v in out.items()}
        return out
    
    path = Path(raster_path)
    if not path.exists():
        raise ValueError("Raster path not found")

    print("\n[Recommendation] Loading raster for analysis...")
    with rasterio.open(path) as src:
        raster_data = src.read()
    
    height, width = raster_data.shape[1], raster_data.shape[2]
    n_pixels = height * width
    
    # Extract pixel-level features
    print(f"[Recommendation] Extracting pixel features from {n_pixels} pixels...")
    features = _extract_pixel_features(raster_data, feature_flags, window_size=3)

    # Robust random sampling for better representativeness
    max_samples = 60000
    rng = np.random.default_rng(42)
    if len(features) > max_samples:
        sample_idx = rng.choice(len(features), size=max_samples, replace=False)
        sampled_features = features[sample_idx]
    else:
        sampled_features = features

    # Keep only valid finite rows
    finite_mask = np.all(np.isfinite(sampled_features), axis=1)
    sampled_features = sampled_features[finite_mask]

    if len(sampled_features) < 50:
        default_n = max(min_clusters, 3)
        print(f"[Recommendation] Not enough valid samples, returning default ({default_n})")
        return default_n

    print(f"[Recommendation] Sampled {len(sampled_features)} pixels for analysis")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(sampled_features)
    
    # Keep cluster range valid for the sampled dataset
    upper = min(max_clusters, max(min_clusters, len(features_normalized) - 1))
    lower = min_clusters
    if upper < lower:
        fallback = max(2, min(5, len(features_normalized) - 1))
        print(f"[Recommendation] Too few samples for range, returning {fallback}")
        return fallback

    print(f"[Recommendation] Testing {lower}-{upper} clusters...")

    silhouette_scores: Dict[int, float] = {}
    ch_scores: Dict[int, float] = {}
    db_scores: Dict[int, float] = {}
    inertias: Dict[int, float] = {}

    for n in range(lower, upper + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10, max_iter=300)
            labels = kmeans.fit_predict(features_normalized)

            sil_sample_size = min(8000, len(features_normalized))
            sil = silhouette_score(features_normalized, labels, sample_size=sil_sample_size, random_state=42)
            ch = calinski_harabasz_score(features_normalized, labels)
            db = davies_bouldin_score(features_normalized, labels)

            silhouette_scores[n] = float(sil)
            ch_scores[n] = float(ch)
            db_scores[n] = float(db)
            inertias[n] = float(kmeans.inertia_)

            print(f"  {n} clusters: sil={sil:.4f}, ch={ch:.2f}, db={db:.4f}, inertia={kmeans.inertia_:.2f}")
        except Exception as e:
            print(f"  {n} clusters: error - {e}")
            continue

    if not silhouette_scores:
        print("[Recommendation] No valid results, returning default (5)")
        return 5

    sil_norm = _normalize(silhouette_scores)
    ch_norm = _normalize(ch_scores)
    db_inv_norm = _normalize(db_scores, invert=True)

    # Elbow curvature on log-inertia: larger positive curvature indicates a better knee.
    curvature: Dict[int, float] = {n: 0.0 for n in inertias.keys()}
    sorted_ns = sorted(inertias.keys())
    if len(sorted_ns) >= 3:
        log_inertia = {n: math.log(max(inertias[n], 1e-9)) for n in sorted_ns}
        for idx in range(1, len(sorted_ns) - 1):
            prev_n = sorted_ns[idx - 1]
            curr_n = sorted_ns[idx]
            next_n = sorted_ns[idx + 1]
            first_drop = log_inertia[prev_n] - log_inertia[curr_n]
            second_drop = log_inertia[curr_n] - log_inertia[next_n]
            curvature[curr_n] = max(0.0, first_drop - second_drop)
    curvature_norm = _normalize(curvature)

    # Weighted consensus score + mild complexity penalty (prefer simpler model when close).
    combined: Dict[int, float] = {}
    span = max(1, upper - lower)
    for n in sorted_ns:
        complexity_penalty = 0.05 * ((n - lower) / span)
        combined[n] = (
            0.45 * sil_norm.get(n, 0.0) +
            0.25 * ch_norm.get(n, 0.0) +
            0.20 * db_inv_norm.get(n, 0.0) +
            0.10 * curvature_norm.get(n, 0.0) -
            complexity_penalty
        )

    best_score = max(combined.values())
    close = [n for n, score in combined.items() if (best_score - score) <= 0.03]
    best_n = min(close) if close else max(combined.keys(), key=lambda n: combined[n])

    print(
        f"[Recommendation] [OK] Recommended: {best_n} clusters "
        f"(combined={combined[best_n]:.4f}, sil={silhouette_scores[best_n]:.4f}, "
        f"ch={ch_scores[best_n]:.2f}, db={db_scores[best_n]:.4f})"
    )
    return best_n

def _build_class_map(classes: List[Dict[str, str]]) -> Dict[str, int]:
    return {item["id"]: idx + 1 for idx, item in enumerate(classes)}


def _extract_pixel_features(
    raster_data: np.ndarray, 
    feature_flags: Dict[str, bool],
    window_size: int = 3,
    verbose: bool = True
) -> np.ndarray:
    """
    Extract features for each pixel directly (not superpixels).
    
    Features:
    - Spectral: mean of each band in local window
    - Texture: variance in local window
    - Indices: NDVI, etc.
    
    Returns: (n_pixels, n_features) array
    """
    height, width, n_bands = raster_data.shape[1], raster_data.shape[2], raster_data.shape[0]
    n_pixels = height * width
    
    # Reshape raster to (height, width, n_bands)
    raster_hwb = np.transpose(raster_data, (1, 2, 0)).astype(np.float32)
    
    feature_list = []
    half_win = window_size // 2
    
    if verbose:
        print(f"    Extracting features for {n_pixels} pixels...")
    
    # === Spectral features (mean of each band in local window) ===
    if feature_flags.get("spectral", True):
        for band_idx in range(n_bands):
            band = raster_hwb[:, :, band_idx]
            # Use local mean for each pixel
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(band, size=window_size, mode='reflect')
            feature_list.append(local_mean.reshape(-1))
        if verbose:
            print(f"    [OK] Spectral: {n_bands} features")
    
    # === Texture features (variance in local window) ===
    if feature_flags.get("texture", True):
        gray = np.mean(raster_hwb[:, :, :3], axis=2) if n_bands >= 3 else raster_hwb[:, :, 0]
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(gray, size=window_size, mode='reflect')
        mean_sq = uniform_filter(gray ** 2, size=window_size, mode='reflect')
        variance = mean_sq - (mean ** 2)
        variance = np.maximum(variance, 0)  # Avoid negative due to numerical errors
        std_dev = np.sqrt(variance)
        feature_list.append(std_dev.reshape(-1))
        if verbose:
            print(f"    [OK] Texture: 1 feature (std dev)")
    
    # === Spectral indices (NDVI, etc.) ===
    if feature_flags.get("indices", True):
        # Check if we have NIR (band 4) and Red (band 3)
        if n_bands >= 4:
            red = raster_hwb[:, :, 2].astype(np.float32)
            nir = raster_hwb[:, :, 3].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-6)
            feature_list.append(ndvi.reshape(-1))
            if verbose:
                print(f"    [OK] Indices: 1 feature (NDVI)")
        else:
            if verbose:
                print(f"    [ERROR] Not enough bands for indices (need >= 4, got {n_bands})")
    
    if not feature_list:
        # Fallback: just use first band
        feature_list.append(raster_hwb[:, :, 0].reshape(-1))
    
    # Stack features into (n_pixels, n_features)
    features = np.stack(feature_list, axis=1)
    if verbose:
        print(f"    Final feature shape: {features.shape}")
    
    return features


def _prepare_image(raster_data: np.ndarray) -> np.ndarray:
    """Prepare image for segmentation (use first 3 bands as RGB)."""
    if raster_data.shape[0] >= 3:
        image = np.moveaxis(raster_data[:3], 0, -1)
    else:
        image = np.moveaxis(raster_data, 0, -1)
    return image.astype(np.float32)


def _compute_superpixels(image: np.ndarray) -> np.ndarray:
    """Compute superpixels using SLIC - high density, color-driven."""
    height, width = image.shape[:2]
    # Very high density - prioritize detail over speed
    target_segments = max(5000, int((height * width) / 200))
    print(f"Creating {target_segments} superpixels for {height}x{width} image...")
    segments = slic(
        image,
        n_segments=target_segments,
        compactness=0.01,  # Very low compactness - prioritize color similarity over spatial continuity
        start_label=0,
        channel_axis=-1,
        sigma=0  # No smoothing - preserve sharp boundaries
    )
    actual_segments = len(np.unique(segments))
    print(f"Created {actual_segments} actual segments")
    return segments


def _build_features(
    raster_data: np.ndarray,
    segments: np.ndarray,
    feature_flags: Dict[str, bool]
) -> np.ndarray:
    """Extract features from each segment."""
    num_segments = int(np.max(segments) + 1)
    segment_ids = segments.reshape(-1)
    counts = np.bincount(segment_ids, minlength=num_segments).astype(np.float32)

    feature_list = []
    
    # Spectral features (mean of each band)
    if feature_flags.get("spectral", False):
        for band in raster_data:
            sums = np.bincount(segment_ids, weights=band.reshape(-1), minlength=num_segments)
            feature_list.append(sums / np.maximum(counts, 1.0))

    # Texture features (variance)
    if feature_flags.get("texture", False):
        gray = np.mean(raster_data[:3], axis=0) if raster_data.shape[0] >= 3 else raster_data[0]
        gray_flat = gray.reshape(-1)
        sums = np.bincount(segment_ids, weights=gray_flat, minlength=num_segments)
        sums_sq = np.bincount(segment_ids, weights=gray_flat ** 2, minlength=num_segments)
        mean = sums / np.maximum(counts, 1.0)
        var = (sums_sq / np.maximum(counts, 1.0)) - (mean ** 2)
        feature_list.append(np.sqrt(np.maximum(var, 0.0)))

    # Spectral indices (NDVI for multispectral)
    if feature_flags.get("indices", False):
        red_index = 2  # Band 3 (0-based)
        nir_index = 3  # Band 4 (0-based)
        if 0 <= red_index < raster_data.shape[0] and 0 <= nir_index < raster_data.shape[0]:
            red = raster_data[red_index].astype(np.float32)
            nir = raster_data[nir_index].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-6)
            ndvi_mean = np.bincount(segment_ids, weights=ndvi.reshape(-1), minlength=num_segments)
            feature_list.append(ndvi_mean / np.maximum(counts, 1.0))

    if not feature_list:
        # Fallback if no features selected
        feature_list.append(np.zeros(num_segments, dtype=np.float32))

    return np.stack(feature_list, axis=1)


def _compute_all_colors(
    raster_data: np.ndarray,
    class_raster: np.ndarray,
    n_clusters: int,
    n_vectors: int,
    classes: List[Dict[str, str]] = None
) -> List[Tuple[int, int, int]]:
    """
    Compute colors for all materials:
    - Clusters 1 to n_clusters: use user-provided colors or mean RGB from original image
    - Vectors n_clusters+1 onwards: bright distinct colors
    """
    colors = []
    
    # Use first 3 bands as RGB
    rgb_bands = raster_data[:3] if raster_data.shape[0] >= 3 else raster_data
    print(f"    [Colors] Original raster shape: {raster_data.shape}, dtype: {raster_data.dtype}")
    print(f"    [Colors] RGB bands shape: {rgb_bands.shape}")
    print(f"    [Colors] RGB bands range: R={np.min(rgb_bands[0])}-{np.max(rgb_bands[0])}, G={np.min(rgb_bands[1])}-{np.max(rgb_bands[1])}, B={np.min(rgb_bands[2])}-{np.max(rgb_bands[2])}")
    print(f"    [Colors] Class raster shape: {class_raster.shape}, dtype: {class_raster.dtype}")
    print(f"    [Colors] Class raster range: min={np.min(class_raster)}, max={np.max(class_raster)}")
    
    # Colors for image clusters (from user-provided colors or mean RGB from original image)
    for cluster_id in range(1, n_clusters + 1):
        # Try to use user-provided color first
        if classes and cluster_id <= len(classes):
            hex_color = classes[cluster_id - 1].get("color", "#808080")
            try:
                r, g, b = _hex_to_rgb(hex_color)
                colors.append((r, g, b))
                print(f"    [Colors] Cluster {cluster_id}: user color {hex_color} -> ({r}, {g}, {b})")
                continue
            except:
                pass
        
        # Fallback: compute mean RGB from raster
        mask = class_raster == cluster_id
        if np.any(mask):
            # Compute mean RGB for this cluster
            r = int(np.mean(rgb_bands[0][mask]))
            g = int(np.mean(rgb_bands[1][mask])) if rgb_bands.shape[0] > 1 else r
            b = int(np.mean(rgb_bands[2][mask])) if rgb_bands.shape[0] > 2 else r
            colors.append((r, g, b))
            print(f"    [Colors] Cluster {cluster_id}: computed color ({r}, {g}, {b})")
        else:
            colors.append((128, 128, 128))  # Default gray for empty clusters
            print(f"    [Colors] Cluster {cluster_id}: NO PIXELS, using gray (128, 128, 128)")
    
    # Colors for vector layers (bright, distinct colors)
    vector_colors = [
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 255, 0),    # Yellow
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
        (0, 255, 128),    # Spring green
        (255, 0, 128),    # Rose
    ]
    
    for i in range(n_vectors):
        colors.append(vector_colors[i % len(vector_colors)])
    
    print(f"    [Colors] Color table: {n_clusters} clusters + {n_vectors} vectors = {len(colors)} total colors")
    return colors


def _build_color_table(classes: List[Dict[str, str]], class_count: int) -> List[Tuple[int, int, int]]:
    colors = []
    for item in classes[:class_count]:
        hex_color = item.get("color", "#ffffff")
        colors.append(_hex_to_rgb(hex_color))
    while len(colors) < class_count:
        colors.append((255, 255, 255))
    return colors


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    if len(value) != 6:
        return (255, 255, 255)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _apply_color_table(
    class_raster: np.ndarray,
    colors: List[Tuple[int, int, int]],
    verbose: bool = True
) -> np.ndarray:
    """
    Apply colors to classification result - pure classification output.
    Map class values to color indices (class N -> colors[N-1])
    """
    height, width = class_raster.shape
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    
    if verbose:
        print(f"    [Color Apply] Input raster shape: {class_raster.shape}, dtype: {class_raster.dtype}")
        print(f"    [Color Apply] Input raster range: min={np.min(class_raster)}, max={np.max(class_raster)}")
        print(f"    [Color Apply] Color table size: {len(colors)}")
        unique_classes = np.unique(class_raster)
        print(f"    [Color Apply] Unique class values: {unique_classes}")
        print(f"    [Color Apply] Color table: {colors}")
        print(f"    [Color Apply] Mapping: class N -> colors[N-1]")
    
    # Apply class colors
    total_pixels = height * width
    applied_count = 0
    uncolored_pixels = total_pixels
    
    for idx, (r, g, b) in enumerate(colors, start=1):
        mask = class_raster == idx
        count = np.sum(mask)
        if count > 0:
            rgb[0][mask] = r
            rgb[1][mask] = g
            rgb[2][mask] = b
            applied_count += count
            uncolored_pixels -= count
            if verbose:
                print(f"      Mapping class {idx} -> color {idx-1}: ({r}, {g}, {b}) - {count} pixels")
        else:
            if verbose:
                print(f"      Class {idx} not found in raster (would use color {idx-1}: ({r}, {g}, {b}))")
    
    # Check for uncolored pixels
    black_mask = (rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0)
    black_count = np.sum(black_mask)
    
    if verbose:
        print(f"    [Color Apply] Pixels colored: {applied_count}/{total_pixels}")
        print(f"    [Color Apply] Black (0,0,0) pixels: {black_count}/{total_pixels}")
        print(f"    [Color Apply] RGB final ranges: R={np.min(rgb[0])}-{np.max(rgb[0])}, G={np.min(rgb[1])}-{np.max(rgb[1])}, B={np.min(rgb[2])}-{np.max(rgb[2])}")
    
    # Check if all pixels are black
    if black_count == total_pixels:
        if verbose:
            print(f"    [Color Apply] ERROR: ALL PIXELS ARE BLACK!")
            print(f"    [Color Apply] Class values found: {unique_classes}")
            print(f"    [Color Apply] Expected class range: 1 to {len(colors)}")
            print(f"    [Color Apply] Check if class values match the expected range")
    
    return rgb
