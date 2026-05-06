"""SAM3-first 6-material classification pipeline.

This module orchestrates the v6 flow:

  1. Acquire road and building masks — from user-uploaded shapefile if
     provided, else from SAM3 (when ``sam3_enabled``), else empty.
  2. Run KMeans classification on the 4 ``source="kmeans"`` materials only
     (BM_VEGETATION, BM_WATER, BM_SAND, BM_SOIL), reusing
     ``core.classify_and_export`` with the filtered class list.
  3. Merge the road and building masks onto the classification with hard
     override (Phase 2).  Phase 3 will replace this with soft-prior fusion
     plus per-component disagreement veto.
  4. Rewrite the Composite_Material_Table XML so it includes all 6 materials
     even though only 4 came from KMeans.

The existing ``backend/app/core.classify_and_export`` and
``backend/app/road_extraction.{extract_feature_masks,merge_feature_masks_onto_classification}``
functions are reused unchanged — this module is a thin orchestrator.
"""
from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import core, road_extraction


_MASK_FEATURE_FOR_CLASS: Dict[str, str] = {
    "BM_ASPHALT":  "roads",
    "BM_CONCRETE": "buildings",
}


def _split_classes_by_source(
    classes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kmeans_classes, mask_classes) using each entry's ``source`` field.

    Falls back to looking the name up in ``core.MEA_CLASSES`` when a request
    payload omits the ``source`` field (the network ``ClassItem`` model only
    carries id/name/color).
    """
    by_name = {c["name"]: c for c in core.MEA_CLASSES}
    kmeans_classes: List[Dict[str, Any]] = []
    mask_classes: List[Dict[str, Any]] = []
    for cls in classes:
        src = cls.get("source")
        if not src:
            ref = by_name.get(cls.get("name", ""), {})
            src = ref.get("source", "kmeans")
        (mask_classes if src == "mask" else kmeans_classes).append(cls)
    return kmeans_classes, mask_classes


def _acquire_mask(
    raster_path: str,
    feature_type: str,
    user_shapefile: Optional[str],
    sam3_enabled: bool,
    progress_callback: Optional[Callable] = None,
) -> Tuple[Optional[str], str]:
    """Return ``(mask_geotiff_path, source_label)`` for one feature.

    Priority:
      1. ``user_shapefile`` if non-empty and the file exists -> rasterize it.
      2. ``sam3_enabled`` and the pre-filter passes -> run SAM3 extraction.
      3. Else -> return (None, "skipped").
    """
    if user_shapefile:
        sp = Path(user_shapefile)
        if sp.exists():
            print(f"[pipeline] {feature_type}: rasterising user shapefile {sp.name}")
            mask = _rasterise_user_shapefile_to_mask(raster_path, str(sp))
            if mask:
                return mask, "shapefile"
            print(f"[pipeline] {feature_type}: shapefile rasterisation failed, falling through")
        else:
            print(f"[pipeline] {feature_type}: shapefile not found at {user_shapefile}")

    if not sam3_enabled:
        return None, "disabled"

    should_run, reason = road_extraction.should_extract_feature(raster_path, feature_type)
    if not should_run:
        print(f"[pipeline] {feature_type}: pre-filter says skip ({reason})")
        return None, "skipped"

    print(f"[pipeline] {feature_type}: pre-filter ok ({reason}); running SAM3")
    result = road_extraction.extract_feature_masks(
        input_path=raster_path,
        feature_type=feature_type,
        progress_callback=progress_callback,
    )
    if result.get("status") != "ok":
        print(f"[pipeline] {feature_type}: SAM3 returned {result.get('status')} — using empty mask")
        return None, "sam3_failed"

    mask_paths = result.get("maskPaths") or []
    if not mask_paths:
        return None, "sam3_empty"

    # Roads/buildings have one mask per feature (one entry in FEATURE_CONFIGS).
    return mask_paths[0], "sam3"


def _rasterise_user_shapefile_to_mask(
    raster_path: str,
    shapefile_path: str,
) -> Optional[str]:
    """Burn a vector shapefile to a binary GeoTIFF aligned with ``raster_path``.

    The mask is written to the system temp directory (not next to the input
    shapefile) so this works on read-only inputs and never collides with
    pre-existing files.  Returns the mask path on success, or ``None``.
    """
    try:
        import tempfile
        import geopandas as gpd
        import rasterio
        from rasterio.features import rasterize
    except Exception as exc:
        print(f"[pipeline] shapefile rasterise: missing geo deps ({exc})")
        return None

    try:
        with rasterio.open(raster_path) as src:
            transform = src.transform
            shape = (src.height, src.width)
            target_crs = src.crs

        gdf = gpd.read_file(shapefile_path)
        if gdf.empty:
            print(f"[pipeline] shapefile {shapefile_path} is empty")
            return None

        if gdf.crs is not None and target_crs is not None and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        mask = rasterize(
            ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty),
            out_shape=shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )

        tmp = tempfile.NamedTemporaryFile(
            prefix=f"v6_mask_{Path(shapefile_path).stem}_",
            suffix=".tif",
            delete=False,
        )
        tmp.close()
        out_path = tmp.name
        profile = {
            "driver": "GTiff", "height": shape[0], "width": shape[1],
            "count": 1, "dtype": "uint8", "crs": target_crs, "transform": transform,
            "compress": "deflate",
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask, 1)
        return out_path
    except Exception as exc:
        print(f"[pipeline] shapefile rasterise failed: {exc}")
        return None


def classify_v6(
    raster_path: str,
    classes: List[Dict[str, Any]],
    smoothing: str,
    feature_flags: Dict[str, bool],
    output_path: Optional[str] = None,
    sam3_enabled: bool = True,
    road_shapefile: Optional[str] = None,
    building_shapefile: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
    **classify_kwargs: Any,
) -> Dict[str, Any]:
    """SAM3-first 6-material orchestrator.  Returns the same dict shape as
    ``core.classify_and_export``."""
    raster = Path(raster_path)
    if not raster.exists():
        return {"status": "error", "message": f"Raster not found: {raster_path}"}

    print("=" * 70)
    print("PIPELINE v6: SAM3-first 6-material classification")
    print(f"  raster:           {raster.name}")
    print(f"  sam3_enabled:     {sam3_enabled}")
    print(f"  road_shapefile:   {road_shapefile!r}")
    print(f"  building_shapefile: {building_shapefile!r}")
    print("=" * 70)

    t_start = _time.perf_counter()

    # ── Phase 0: Mask acquisition ──────────────────────────────────────────
    def _phase_cb(label: str) -> Optional[Callable]:
        if not progress_callback:
            return None
        def _inner(phase: str, done: int, total: int) -> None:
            progress_callback(f"{label}: {phase}", done, total)
        return _inner

    road_mask_path, road_source = _acquire_mask(
        raster_path, "roads", road_shapefile, sam3_enabled,
        progress_callback=_phase_cb("Roads (SAM3)"),
    )
    bldg_mask_path, bldg_source = _acquire_mask(
        raster_path, "buildings", building_shapefile, sam3_enabled,
        progress_callback=_phase_cb("Buildings (SAM3)"),
    )
    print(f"[pipeline] road mask source:     {road_source} -> {road_mask_path}")
    print(f"[pipeline] building mask source: {bldg_source} -> {bldg_mask_path}")

    # ── Phase 1: KMeans classification on the 4 natural materials only ────
    kmeans_classes, mask_classes = _split_classes_by_source(classes)
    if not kmeans_classes:
        return {"status": "error",
                "message": "No KMeans-source classes in payload (need at least 1)"}

    print(f"[pipeline] KMeans on {len(kmeans_classes)} natural materials: "
          f"{[c['name'] for c in kmeans_classes]}")

    classify_result = core.classify_and_export(
        raster_path=raster_path,
        classes=kmeans_classes,
        smoothing=smoothing,
        feature_flags=feature_flags,
        output_path=output_path,
        progress_callback=progress_callback,
        **classify_kwargs,
    )
    if classify_result.get("status") != "ok":
        return classify_result

    classification_path = classify_result.get("outputPath")
    if not classification_path:
        return {"status": "error", "message": "classify_and_export returned no outputPath"}

    # ── Phase 2: Hard-override fusion of road and building masks ──────────
    by_name = {c["name"]: c for c in core.MEA_CLASSES}
    fusion_inputs: List[Tuple[str, Tuple[int, int, int], str]] = []
    if road_mask_path:
        rgb = core._hex_to_rgb(by_name["BM_ASPHALT"]["color"])
        fusion_inputs.append((road_mask_path, rgb, "BM_ASPHALT"))
    if bldg_mask_path:
        rgb = core._hex_to_rgb(by_name["BM_CONCRETE"]["color"])
        # Building written AFTER road -> wins where they overlap.
        fusion_inputs.append((bldg_mask_path, rgb, "BM_CONCRETE"))

    if fusion_inputs:
        mask_paths = [p for p, _, _ in fusion_inputs]
        colors = [c for _, c, _ in fusion_inputs]
        print(f"[pipeline] merging {len(mask_paths)} mask(s) onto classification")
        merge_result = road_extraction.merge_feature_masks_onto_classification(
            classification_path=classification_path,
            mask_paths=mask_paths,
            colors=colors,
            progress_callback=_phase_cb("Mask fusion"),
        )
        if merge_result.get("status") != "ok":
            print(f"[pipeline] mask merge failed: {merge_result.get('message')}")
        else:
            classification_path = merge_result.get("outputPath", classification_path)

    # ── Phase 3: Rewrite XML with full 6-material composite table ──────────
    # core.classify_and_export wrote a 4-entry XML.  Now that masks have
    # contributed BM_ASPHALT and BM_CONCRETE pixels, the companion XMLs need
    # to list all 6 classes.  In tile mode, both the original tile dir AND
    # the merged tile dir need their XMLs rewritten.
    paths_needing_xml: List[Path] = []
    pre_merge_path = Path(classify_result.get("outputPath") or "")
    final_path = Path(classification_path)

    for p in (pre_merge_path, final_path):
        if not p or str(p) == "":
            continue
        if p.is_dir():
            paths_needing_xml.extend(
                f for f in p.iterdir()
                if f.is_file() and f.suffix.lower() in (".tif", ".tiff", ".img")
            )
        elif p.exists():
            paths_needing_xml.append(p)

    # Also include any tileOutputs returned by the merge step (they may live
    # in a sibling _merged/ folder that doesn't show up in `final_path` enum).
    for tile_path in classify_result.get("tileOutputs") or []:
        tp = Path(tile_path)
        if tp.exists() and tp not in paths_needing_xml:
            paths_needing_xml.append(tp)

    seen: set = set()
    for target in paths_needing_xml:
        key = str(target.resolve())
        if key in seen:
            continue
        seen.add(key)
        core._write_composite_material_xml(target, classes)

    duration = _time.perf_counter() - t_start
    print(f"[pipeline] classify_v6 finished in {duration:.1f}s")

    classify_result["outputPath"] = str(classification_path)
    classify_result["maskSources"] = {"roads": road_source, "buildings": bldg_source}
    classify_result["meaSchemaVersion"] = 6
    return classify_result
