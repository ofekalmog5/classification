"""Phase-1 unit tests for the 6-material MEA schema.

Run from project root:
    python -m pytest backend/tests/test_mea_v6.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Allow `from app import ...` when running outside the backend/ working dir.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

# core.py pulls in heavy geo/ML deps (skimage, sklearn, rasterio).  If any are
# missing we skip the core-dependent tests but still run the lightweight
# JSON-schema and profile-migration tests below.
try:
    from app.core import (
        MEA_CLASSES,
        _MEA_MASK_MATERIALS,
        _MEA_KMEANS_MATERIALS,
        _MEA_ANCHOR_MAP,
        _MEA_COMPOSITE_NAMES,
        _MEA_LEGACY_TO_PARENT,
        _build_mea_cluster_mapping,
        _is_mea_classes,
        _mea_material_type,
    )
    _CORE_OK = True
    _CORE_SKIP_REASON = ""
except Exception as _exc:  # noqa: BLE001 — environment-specific dep absence
    _CORE_OK = False
    _CORE_SKIP_REASON = f"core.py import failed: {_exc}"

from app.mea_profile import _migrate_legacy_profile

requires_core = pytest.mark.skipif(not _CORE_OK, reason=_CORE_SKIP_REASON)


# ─── Schema tests ────────────────────────────────────────────────────────────

@requires_core
def test_mea_classes_has_exactly_6_entries():
    assert len(MEA_CLASSES) == 6


@requires_core
def test_mea_classes_required_fields():
    required = {"id", "name", "color", "composite_name", "source", "material_type", "sub_absorbs", "anchors"}
    for cls in MEA_CLASSES:
        missing = required - set(cls.keys())
        assert not missing, f"{cls.get('name')} missing keys: {missing}"


@requires_core
def test_source_split_3_mask_3_kmeans():
    assert len(_MEA_MASK_MATERIALS) == 3
    assert len(_MEA_KMEANS_MATERIALS) == 3
    assert _MEA_MASK_MATERIALS == {"BM_ASPHALT", "BM_CONCRETE", "BM_WATER"}
    assert _MEA_KMEANS_MATERIALS == {"BM_VEGETATION", "BM_SAND", "BM_SOIL"}


@requires_core
def test_anchor_map_non_empty_per_class():
    for cls in MEA_CLASSES:
        anchors = _MEA_ANCHOR_MAP[cls["name"]]
        assert anchors, f"{cls['name']} has no anchors"
        for a in anchors:
            assert len(a) == 3, f"{cls['name']} anchor {a} not RGB triple"
            assert all(0 <= c <= 255 for c in a), f"{cls['name']} anchor {a} out of range"


@requires_core
def test_composite_names_present_for_all_6():
    for cls in MEA_CLASSES:
        assert cls["name"] in _MEA_COMPOSITE_NAMES
        assert _MEA_COMPOSITE_NAMES[cls["name"]] == cls["composite_name"]


@requires_core
def test_legacy_to_parent_map_covers_all_old_13_minus_6_kept():
    # The 6 we kept by name should NOT be in the legacy->parent map.
    kept_names = {c["name"] for c in MEA_CLASSES}
    for legacy in _MEA_LEGACY_TO_PARENT:
        assert legacy not in kept_names, f"{legacy} should not be in legacy map"
    # Legacy parents must be valid 6-material names.
    for parent in _MEA_LEGACY_TO_PARENT.values():
        assert parent in kept_names, f"parent {parent} not in MEA_CLASSES"


# ─── Cluster mapping tests (multi-anchor) ────────────────────────────────────

@requires_core
def test_multi_anchor_cost_prefers_nearest_anchor():
    """A cluster RGB matching one of BM_VEGETATION's sub-anchors (e.g. the old
    BM_LAND_GRASS color) should be assigned to BM_VEGETATION."""
    # BM_LAND_GRASS legacy RGB = (124, 252, 0); it's now an anchor of BM_VEGETATION.
    cluster_rgbs = [(124, 252, 0)]
    mapping, colors = _build_mea_cluster_mapping(cluster_rgbs, MEA_CLASSES)
    assert len(mapping) == 1
    assert mapping[0]["material"] == "BM_VEGETATION"
    # Output color is the parent's primary hex, not the anchor color.
    assert mapping[0]["colorHex"] == "#228B22"


@requires_core
def test_multi_anchor_cost_assigns_metal_to_concrete():
    """Old BM_METAL_STEEL color (112, 128, 144) should now map to BM_CONCRETE."""
    cluster_rgbs = [(112, 128, 144)]
    mapping, _ = _build_mea_cluster_mapping(cluster_rgbs, MEA_CLASSES)
    assert mapping[0]["material"] == "BM_CONCRETE"


@requires_core
def test_cluster_mapping_handles_empty_inputs():
    mapping, colors = _build_mea_cluster_mapping([], MEA_CLASSES)
    assert mapping == []
    assert colors == []
    mapping, colors = _build_mea_cluster_mapping([(1, 2, 3)], [])
    assert mapping == []


@requires_core
def test_cluster_mapping_with_frequency_prior_breaks_ties():
    """When two materials are equidistant, the higher-prevalence one wins."""
    # Use a neutral mid-gray cluster equidistant-ish from concrete and asphalt.
    cluster_rgbs = [(120, 120, 120)]
    prior_concrete_wins = {"BM_CONCRETE": 0.50, "BM_ASPHALT": 0.10}
    mapping, _ = _build_mea_cluster_mapping(cluster_rgbs, MEA_CLASSES, material_prior=prior_concrete_wins)
    assert mapping[0]["material"] == "BM_CONCRETE"


# ─── _is_mea_classes ─────────────────────────────────────────────────────────

@requires_core
def test_is_mea_classes_accepts_full_v6_list():
    assert _is_mea_classes(MEA_CLASSES) is True


@requires_core
def test_is_mea_classes_rejects_empty():
    assert _is_mea_classes([]) is False
    assert _is_mea_classes(None) is False


@requires_core
def test_is_mea_classes_rejects_legacy_13_names():
    legacy = [{"id": "x", "name": "BM_FOLIAGE", "color": "#006400"}]
    assert _is_mea_classes(legacy) is False


@requires_core
def test_is_mea_classes_accepts_partial_v6_subset():
    # _is_mea_classes returns True when ALL submitted names are in MEA_CLASSES,
    # even if it's a subset. (Useful for callers that want only some materials.)
    subset = [{"id": "1", "name": "BM_VEGETATION", "color": "#228B22"}]
    assert _is_mea_classes(subset) is True


# ─── material_type helper ────────────────────────────────────────────────────

@requires_core
def test_material_type_lookup():
    assert _mea_material_type("BM_ASPHALT") == "ROAD_SURFACE"
    assert _mea_material_type("BM_CONCRETE") == "BUILDING"
    assert _mea_material_type("BM_VEGETATION") == "VEGETATION"
    assert _mea_material_type("BM_WATER") == "WATER"
    assert _mea_material_type("BM_SAND") == "SOIL_EARTH"
    assert _mea_material_type("BM_SOIL") == "SOIL_EARTH"
    assert _mea_material_type("BM_NONEXISTENT") == "OTHER"


# ─── mea_defaults.json file ──────────────────────────────────────────────────

def test_mea_defaults_json_is_v3():
    path = _PROJECT_ROOT / "shared" / "mea_defaults.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 3, f"expected version 3, got {data.get('version')}"


def test_mea_defaults_json_has_6_materials():
    path = _PROJECT_ROOT / "shared" / "mea_defaults.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides = data["material_overrides"]
    expected_names = {"BM_ASPHALT", "BM_CONCRETE", "BM_VEGETATION",
                      "BM_WATER", "BM_SAND", "BM_SOIL"}
    assert set(overrides.keys()) == expected_names


def test_mea_defaults_json_each_entry_has_required_fields():
    path = _PROJECT_ROOT / "shared" / "mea_defaults.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"reference_color", "reference_rgb", "composite_name", "source",
                "material_type", "sub_absorbs", "anchors", "tolerance_radius"}
    for name, entry in data["material_overrides"].items():
        missing = required - set(entry.keys())
        assert not missing, f"{name} missing fields in JSON: {missing}"


# ─── Legacy profile migration ────────────────────────────────────────────────

def test_migrate_v2_profile_absorbs_legacy_materials():
    v2_profile = {
        "version": 2,
        "material_overrides": {
            "BM_VEGETATION":     {"anchors": [[34, 139, 34]]},
            "BM_FOLIAGE":        {"anchors": [[10, 80, 10]]},
            "BM_LAND_GRASS":     {"anchors": [[124, 252, 0]]},
            "BM_LAND_DRY_GRASS": {"anchors": [[189, 183, 107]]},
            "BM_METAL":          {"anchors": [[169, 171, 176]]},
            "BM_ASPHALT":        {"anchors": [[44, 44, 56]]},
        },
    }
    migrated = _migrate_legacy_profile(v2_profile)
    assert migrated["version"] == 3

    veg_anchors = migrated["material_overrides"]["BM_VEGETATION"]["anchors"]
    assert [10, 80, 10] in veg_anchors
    assert [124, 252, 0] in veg_anchors
    assert [189, 183, 107] in veg_anchors

    concrete_anchors = migrated["material_overrides"]["BM_CONCRETE"]["anchors"]
    assert [169, 171, 176] in concrete_anchors

    # Legacy keys should be gone.
    assert "BM_FOLIAGE" not in migrated["material_overrides"]
    assert "BM_LAND_GRASS" not in migrated["material_overrides"]
    assert "BM_METAL" not in migrated["material_overrides"]


def test_migrate_v3_profile_is_noop():
    v3 = {"version": 3, "material_overrides": {"BM_VEGETATION": {"anchors": []}}}
    migrated = _migrate_legacy_profile(v3)
    assert migrated == v3


def test_migrate_v2_without_legacy_keys_just_bumps_version():
    v2 = {"version": 2, "material_overrides": {"BM_VEGETATION": {"anchors": []}}}
    migrated = _migrate_legacy_profile(v2)
    assert migrated["version"] == 3
    assert migrated["material_overrides"] == {"BM_VEGETATION": {"anchors": []}}


def test_migrate_v2_handles_malformed_legacy_entry():
    """A v2 profile where a legacy material override is None or non-dict
    must not crash the migration."""
    v2 = {
        "version": 2,
        "material_overrides": {
            "BM_VEGETATION": {"anchors": [[34, 139, 34]]},
            "BM_FOLIAGE": None,        # malformed: explicit null
            "BM_LAND_GRASS": 42,        # malformed: not a dict
        },
    }
    migrated = _migrate_legacy_profile(v2)
    assert migrated["version"] == 3
    assert "BM_FOLIAGE" not in migrated["material_overrides"]
    assert "BM_LAND_GRASS" not in migrated["material_overrides"]
    # Existing veg anchors preserved.
    assert migrated["material_overrides"]["BM_VEGETATION"]["anchors"] == [[34, 139, 34]]


# ─── Phase 2 — pipeline.classify_v6 splitting ────────────────────────────────

@requires_core
def test_split_classes_by_source_uses_source_field():
    """Classes carrying their source field route correctly."""
    from app.pipeline import _split_classes_by_source
    classes = list(MEA_CLASSES)  # full 6 entries with source
    kmeans, masks = _split_classes_by_source(classes)
    assert {c["name"] for c in masks} == {"BM_ASPHALT", "BM_CONCRETE", "BM_WATER"}
    assert {c["name"] for c in kmeans} == {"BM_VEGETATION", "BM_SAND", "BM_SOIL"}


@requires_core
def test_split_classes_by_source_falls_back_to_canonical_lookup():
    """When the request omits source (raw network ClassItem), look it up by name."""
    from app.pipeline import _split_classes_by_source
    bare = [
        {"id": "class-1", "name": "BM_ASPHALT",    "color": "#2D2D30"},
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
    ]
    kmeans, masks = _split_classes_by_source(bare)
    assert [c["name"] for c in masks] == ["BM_ASPHALT"]
    assert [c["name"] for c in kmeans] == ["BM_VEGETATION"]


@requires_core
def test_split_classes_by_source_unknown_name_defaults_to_kmeans():
    """A custom (non-MEA) name with no source field should default to kmeans."""
    from app.pipeline import _split_classes_by_source
    custom = [{"id": "x", "name": "CUSTOM_FOO", "color": "#abcdef"}]
    kmeans, masks = _split_classes_by_source(custom)
    assert kmeans == custom
    assert masks == []


# ─── Phase 3 — soft-prior fusion + per-component veto ────────────────────────

def _try_import_geo():
    try:
        import numpy
        import rasterio
        import scipy.ndimage  # noqa: F401
        return numpy, rasterio
    except Exception:
        return None, None


_NP_RIO = _try_import_geo()
requires_geo = pytest.mark.skipif(
    _NP_RIO[0] is None,
    reason="numpy/rasterio/scipy not available in test env",
)


def _write_synthetic_geotiff(path, rgb_array, np_mod, rio_mod):
    """Helper: write a 3-band uint8 GeoTIFF from an (H,W,3) numpy array."""
    arr = np_mod.transpose(rgb_array, (2, 0, 1)).astype(np_mod.uint8)
    profile = {
        "driver": "GTiff", "height": arr.shape[1], "width": arr.shape[2],
        "count": 3, "dtype": "uint8",
        "transform": rio_mod.transform.from_origin(0, arr.shape[1], 1, 1),
        "crs": "EPSG:32636",  # arbitrary metric CRS
    }
    with rio_mod.open(str(path), "w", **profile) as dst:
        dst.write(arr)


def _write_synthetic_mask(path, mask_array, np_mod, rio_mod, ref_geotiff):
    """Helper: write a 1-band uint8 mask aligned with ref_geotiff."""
    with rio_mod.open(str(ref_geotiff)) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8")
    with rio_mod.open(str(path), "w", **profile) as dst:
        dst.write(mask_array.astype(np_mod.uint8), 1)


@requires_core
@requires_geo
def test_fusion_compatible_component_painted(tmp_path):
    """A road-mask component overlapping mostly KMeans soil (compatible) gets
    painted with BM_ASPHALT."""
    np = _NP_RIO[0]; rio = _NP_RIO[1]
    from app.pipeline import _fuse_with_priors_and_veto

    H, W = 20, 20
    soil_rgb = (101, 67, 33)         # BM_SOIL
    asphalt_rgb = (45, 45, 48)       # BM_ASPHALT
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = soil_rgb[0]; rgb[..., 1] = soil_rgb[1]; rgb[..., 2] = soil_rgb[2]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    cls_path = tmp_path / "classification.tif"
    mask_path = tmp_path / "road_mask.tif"
    _write_synthetic_geotiff(cls_path, rgb, np, rio)
    _write_synthetic_mask(mask_path, mask, np, rio, cls_path)

    kmeans_classes = [
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
        {"id": "class-4", "name": "BM_WATER",      "color": "#1C6BA0"},
        {"id": "class-5", "name": "BM_SAND",       "color": "#EDC9AF"},
        {"id": "class-6", "name": "BM_SOIL",       "color": "#654321"},
    ]
    fusion_inputs = [(str(mask_path), asphalt_rgb, "BM_ASPHALT", "sam3")]

    result = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=fusion_inputs,
        kmeans_classes=kmeans_classes,
    )
    assert result["status"] == "ok"
    assert result["fusionStats"]["BM_ASPHALT"]["vetoed"] == 0
    assert result["fusionStats"]["BM_ASPHALT"]["total"] == 1

    with rio.open(result["outputPath"]) as out:
        out_rgb = out.read()
    # Pixels under the mask should be painted asphalt color.
    assert int(out_rgb[0, 10, 10]) == asphalt_rgb[0]
    # Pixels outside the mask should still be soil color.
    assert int(out_rgb[0, 0, 0]) == soil_rgb[0]


@requires_core
@requires_geo
def test_fusion_vegetation_component_vetoed(tmp_path):
    """A road-mask component overlapping a forest (BM_VEGETATION pixels) is
    vetoed because the KMeans evidence overwhelmingly disagrees with road."""
    np = _NP_RIO[0]; rio = _NP_RIO[1]
    from app.pipeline import _fuse_with_priors_and_veto

    H, W = 20, 20
    veg_rgb = (34, 139, 34)          # BM_VEGETATION
    asphalt_rgb = (45, 45, 48)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = veg_rgb[0]; rgb[..., 1] = veg_rgb[1]; rgb[..., 2] = veg_rgb[2]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[5:15, 5:15] = 1

    cls_path = tmp_path / "classification.tif"
    mask_path = tmp_path / "road_mask.tif"
    _write_synthetic_geotiff(cls_path, rgb, np, rio)
    _write_synthetic_mask(mask_path, mask, np, rio, cls_path)

    kmeans_classes = [
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
        {"id": "class-4", "name": "BM_WATER",      "color": "#1C6BA0"},
        {"id": "class-5", "name": "BM_SAND",       "color": "#EDC9AF"},
        {"id": "class-6", "name": "BM_SOIL",       "color": "#654321"},
    ]
    fusion_inputs = [(str(mask_path), asphalt_rgb, "BM_ASPHALT", "sam3")]

    result = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=fusion_inputs,
        kmeans_classes=kmeans_classes,
    )
    assert result["status"] == "ok"
    assert result["fusionStats"]["BM_ASPHALT"]["vetoed"] == 1
    assert result["fusionStats"]["BM_ASPHALT"]["total"] == 1

    with rio.open(result["outputPath"]) as out:
        out_rgb = out.read()
    # Vetoed: pixel under the mask should still be vegetation, not asphalt.
    assert int(out_rgb[0, 10, 10]) == veg_rgb[0]
    assert int(out_rgb[1, 10, 10]) == veg_rgb[1]


@requires_core
@requires_geo
def test_fusion_drifted_vegetation_still_vetoes_within_tolerance(tmp_path):
    """Vegetation pixels that drift slightly from the canonical anchor
    (still well within the ±17/channel tolerance band) should still be
    detected as incompatible with road."""
    np = _NP_RIO[0]; rio = _NP_RIO[1]
    from app.pipeline import _fuse_with_priors_and_veto

    H, W = 20, 20
    # Drift the green a bit: from (34,139,34) -> (40,130,40).
    # d² = 36 + 81 + 36 = 153, well within the 900 tolerance.
    drifted_veg = (40, 130, 40)
    asphalt_rgb = (45, 45, 48)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = drifted_veg[0]; rgb[..., 1] = drifted_veg[1]; rgb[..., 2] = drifted_veg[2]
    mask = np.zeros((H, W), dtype=np.uint8); mask[5:15, 5:15] = 1

    cls_path = tmp_path / "classification.tif"
    mask_path = tmp_path / "road_mask.tif"
    _write_synthetic_geotiff(cls_path, rgb, np, rio)
    _write_synthetic_mask(mask_path, mask, np, rio, cls_path)

    kmeans_classes = [
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
        {"id": "class-4", "name": "BM_WATER",      "color": "#1C6BA0"},
        {"id": "class-5", "name": "BM_SAND",       "color": "#EDC9AF"},
        {"id": "class-6", "name": "BM_SOIL",       "color": "#654321"},
    ]
    result = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=[(str(mask_path), asphalt_rgb, "BM_ASPHALT", "sam3")],
        kmeans_classes=kmeans_classes,
    )
    assert result["fusionStats"]["BM_ASPHALT"]["vetoed"] == 1, \
        "Slightly drifted vegetation should still trigger the veto"


@requires_core
@requires_geo
def test_fusion_color_outside_tolerance_does_not_trigger_veto(tmp_path):
    """A muted green that's outside the ±17/channel tolerance band should
    NOT count as 'incompatible' — the mask paints over it normally."""
    np = _NP_RIO[0]; rio = _NP_RIO[1]
    from app.pipeline import _fuse_with_priors_and_veto

    H, W = 20, 20
    # Far from (34,139,34): d² = 676 + 361 + 676 = 1713 > 900.
    out_of_band = (60, 120, 60)
    asphalt_rgb = (45, 45, 48)
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = out_of_band[0]; rgb[..., 1] = out_of_band[1]; rgb[..., 2] = out_of_band[2]
    mask = np.zeros((H, W), dtype=np.uint8); mask[5:15, 5:15] = 1

    cls_path = tmp_path / "classification.tif"
    mask_path = tmp_path / "road_mask.tif"
    _write_synthetic_geotiff(cls_path, rgb, np, rio)
    _write_synthetic_mask(mask_path, mask, np, rio, cls_path)

    kmeans_classes = [
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
        {"id": "class-4", "name": "BM_WATER",      "color": "#1C6BA0"},
        {"id": "class-5", "name": "BM_SAND",       "color": "#EDC9AF"},
        {"id": "class-6", "name": "BM_SOIL",       "color": "#654321"},
    ]
    result = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=[(str(mask_path), asphalt_rgb, "BM_ASPHALT", "sam3")],
        kmeans_classes=kmeans_classes,
    )
    assert result["fusionStats"]["BM_ASPHALT"]["vetoed"] == 0, \
        "Out-of-band color should not trigger the veto"


@requires_core
@requires_geo
def test_fusion_shapefile_threshold_higher_than_sam3(tmp_path):
    """A mostly-vegetation component is vetoed when source=sam3 (threshold
    0.55) but kept when source=shapefile (threshold 0.85)."""
    np = _NP_RIO[0]; rio = _NP_RIO[1]
    from app.pipeline import _fuse_with_priors_and_veto

    H, W = 20, 20
    veg_rgb = (34, 139, 34)
    soil_rgb = (101, 67, 33)
    asphalt_rgb = (45, 45, 48)

    # 70% vegetation, 30% soil under the mask.  This is above the SAM3
    # veto threshold (0.55) but below the shapefile one (0.85).
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = soil_rgb[0]; rgb[..., 1] = soil_rgb[1]; rgb[..., 2] = soil_rgb[2]
    rgb[5:12, 5:15, 0] = veg_rgb[0]
    rgb[5:12, 5:15, 1] = veg_rgb[1]
    rgb[5:12, 5:15, 2] = veg_rgb[2]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[5:15, 5:15] = 1   # 100 pixels: 70 veg + 30 soil

    cls_path = tmp_path / "classification.tif"
    mask_path = tmp_path / "road_mask.tif"
    _write_synthetic_geotiff(cls_path, rgb, np, rio)
    _write_synthetic_mask(mask_path, mask, np, rio, cls_path)

    kmeans_classes = [
        {"id": "class-3", "name": "BM_VEGETATION", "color": "#228B22"},
        {"id": "class-4", "name": "BM_WATER",      "color": "#1C6BA0"},
        {"id": "class-5", "name": "BM_SAND",       "color": "#EDC9AF"},
        {"id": "class-6", "name": "BM_SOIL",       "color": "#654321"},
    ]

    # SAM3 source -> vetoed (70% > 55% threshold)
    result_sam3 = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=[(str(mask_path), asphalt_rgb, "BM_ASPHALT", "sam3")],
        kmeans_classes=kmeans_classes,
    )
    assert result_sam3["fusionStats"]["BM_ASPHALT"]["vetoed"] == 1

    # Shapefile source -> kept (70% < 85% threshold)
    result_shp = _fuse_with_priors_and_veto(
        classification_path=str(cls_path),
        fusion_inputs=[(str(mask_path), asphalt_rgb, "BM_ASPHALT", "shapefile")],
        kmeans_classes=kmeans_classes,
    )
    assert result_shp["fusionStats"]["BM_ASPHALT"]["vetoed"] == 0
