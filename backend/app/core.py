from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os

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

    candidates: List[Path] = []

    # Prefer rasterio/pyogrio bundled proj data (layout >=5).
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend([
        project_root / '.venv' / 'Lib' / 'site-packages' / 'rasterio' / 'proj_data',
        project_root / '.venv' / 'Lib' / 'site-packages' / 'pyogrio' / 'proj_data',
    ])

    # Then pyproj's bundled data dir if compatible.
    try:
        from pyproj import datadir as _pyproj_datadir

        pyproj_dir = _pyproj_datadir.get_data_dir()
        if pyproj_dir:
            candidates.append(Path(pyproj_dir))
    except Exception:
        pass

    # Fallbacks: conda/system locations.
    candidates.extend([
        Path(r'C:\Users\ofeka\anaconda3\envs\py37\Library\share\proj'),
        Path(r'C:\Users\ofeka\anaconda3\Library\share\proj'),
        Path(r'C:\Program Files\PROJ\share\proj'),
    ])

    existing = os.environ.get('PROJ_LIB')
    if existing:
        candidates.append(Path(existing))

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


def _normalize_pseudo_mercator_crs(crs: CRS) -> CRS:
    if crs is None:
        return crs
    crs_text = str(crs)
    if crs_text.startswith("LOCAL_CS") and "Pseudo-Mercator" in crs_text:
        return CRS.from_epsg(3857)
    return crs


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
        # Multi-band - only modify first band
        output_array = raster_array.copy()
        # Don't change dtype - keep it as original
        output_array[0][burned_mask > 0] = burn_value
    
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
    output_path: str | None = None
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
        output_path=str(temp_output)
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
            output_path=output_path
        )
        
        if result2["status"] != "ok":
            return result2
        
        final_output = result2["outputPath"]
    else:
        print("\n>>> STEP 2: No vectors provided, skipping")
        final_output = classif_file
        # Copy to final output path if provided
        if output_path:
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
    output_path: str | None = None
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

    # === Save classification ===
    if output_path:
        output_color_path = Path(output_path)
    else:
        output_color_path = path.with_name(path.stem + "_classified.tif")
    
    print(f"\n[6/6] Saving classified output...")
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
    output_path: str | None = None
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
    
    # === Load classification ===
    print(f"\n[1/3] Loading classification: {classif_path.name}")
    with rasterio.open(classif_path) as src:
        classif_data = src.read(1)  # Read first band (classification values)
        profile = src.profile.copy()
        transform = src.transform
        crs = _normalize_pseudo_mercator_crs(src.crs)
        height, width = classif_data.shape

    profile["crs"] = crs
    
    print(f"  Shape: {height}x{width}")
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
    Analyze raster and recommend optimal number of clusters using Silhouette score.
    Uses pixel-level features with sampling for speed.
    """
    from sklearn.metrics import silhouette_score
    
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
    
    # Sample for speed (use every Nth pixel)
    sample_rate = max(1, n_pixels // 50000)  # Max 50k sampled pixels
    sampled_features = features[::sample_rate]
    print(f"[Recommendation] Sampled {len(sampled_features)} pixels for analysis")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(sampled_features)
    
    # Try different cluster counts and compute silhouette scores
    print(f"[Recommendation] Testing {min_clusters}-{max_clusters} clusters...")
    scores = {}
    for n in range(min_clusters, max_clusters + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=5, max_iter=200)
            labels = kmeans.fit_predict(features_normalized)
            
            # Silhouette score (higher is better)
            score = silhouette_score(features_normalized, labels, sample_size=min(5000, len(features_normalized)))
            scores[n] = score
            print(f"  {n} clusters: score {score:.4f}")
        except Exception as e:
            print(f"  {n} clusters: error - {e}")
            continue
    
    if not scores:
        print("[Recommendation] No valid results, returning default (5)")
        return 5
    
    # Return the cluster count with highest silhouette score
    best_n = max(scores.keys(), key=lambda n: scores[n])
    print(f"[Recommendation] [OK] Recommended: {best_n} clusters (score: {scores[best_n]:.4f})")
    return best_n

def _build_class_map(classes: List[Dict[str, str]]) -> Dict[str, int]:
    return {item["id"]: idx + 1 for idx, item in enumerate(classes)}


def _extract_pixel_features(
    raster_data: np.ndarray, 
    feature_flags: Dict[str, bool],
    window_size: int = 3
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
    
    print(f"    Extracting features for {n_pixels} pixels...")
    
    # === Spectral features (mean of each band in local window) ===
    if feature_flags.get("spectral", True):
        for band_idx in range(n_bands):
            band = raster_hwb[:, :, band_idx]
            # Use local mean for each pixel
            from scipy.ndimage import uniform_filter
            local_mean = uniform_filter(band, size=window_size, mode='reflect')
            feature_list.append(local_mean.reshape(-1))
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
        print(f"    [OK] Texture: 1 feature (std dev)")
    
    # === Spectral indices (NDVI, etc.) ===
    if feature_flags.get("indices", True):
        # Check if we have NIR (band 4) and Red (band 3)
        if n_bands >= 4:
            red = raster_hwb[:, :, 2].astype(np.float32)
            nir = raster_hwb[:, :, 3].astype(np.float32)
            ndvi = (nir - red) / (nir + red + 1e-6)
            feature_list.append(ndvi.reshape(-1))
            print(f"    [OK] Indices: 1 feature (NDVI)")
        else:
            print(f"    [ERROR] Not enough bands for indices (need >= 4, got {n_bands})")
    
    if not feature_list:
        # Fallback: just use first band
        feature_list.append(raster_hwb[:, :, 0].reshape(-1))
    
    # Stack features into (n_pixels, n_features)
    features = np.stack(feature_list, axis=1)
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
    colors: List[Tuple[int, int, int]]
) -> np.ndarray:
    """
    Apply colors to classification result - pure classification output.
    Map class values to color indices (class N -> colors[N-1])
    """
    height, width = class_raster.shape
    rgb = np.zeros((3, height, width), dtype=np.uint8)
    
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
            print(f"      Mapping class {idx} -> color {idx-1}: ({r}, {g}, {b}) - {count} pixels")
        else:
            print(f"      Class {idx} not found in raster (would use color {idx-1}: ({r}, {g}, {b}))")
    
    # Check for uncolored pixels
    black_mask = (rgb[0] == 0) & (rgb[1] == 0) & (rgb[2] == 0)
    black_count = np.sum(black_mask)
    
    print(f"    [Color Apply] Pixels colored: {applied_count}/{total_pixels}")
    print(f"    [Color Apply] Black (0,0,0) pixels: {black_count}/{total_pixels}")
    print(f"    [Color Apply] RGB final ranges: R={np.min(rgb[0])}-{np.max(rgb[0])}, G={np.min(rgb[1])}-{np.max(rgb[1])}, B={np.min(rgb[2])}-{np.max(rgb[2])}")
    
    # Check if all pixels are black
    if black_count == total_pixels:
        print(f"    [Color Apply] ERROR: ALL PIXELS ARE BLACK!")
        print(f"    [Color Apply] Class values found: {unique_classes}")
        print(f"    [Color Apply] Expected class range: 1 to {len(colors)}")
        print(f"    [Color Apply] Check if class values match the expected range")
    
    return rgb
