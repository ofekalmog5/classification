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
def test_source_split_2_mask_4_kmeans():
    assert len(_MEA_MASK_MATERIALS) == 2
    assert len(_MEA_KMEANS_MATERIALS) == 4
    assert _MEA_MASK_MATERIALS == {"BM_ASPHALT", "BM_CONCRETE"}
    assert _MEA_KMEANS_MATERIALS == {"BM_VEGETATION", "BM_WATER", "BM_SAND", "BM_SOIL"}


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
    assert {c["name"] for c in masks} == {"BM_ASPHALT", "BM_CONCRETE"}
    assert {c["name"] for c in kmeans} == {"BM_VEGETATION", "BM_WATER", "BM_SAND", "BM_SOIL"}


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
