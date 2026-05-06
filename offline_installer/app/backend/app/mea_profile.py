"""
MEA material calibration profile — read-only consumer for the main app.

Profiles are written by the standalone MEA Calibration Tool and stored at:
  %ProgramData%\\MaterialClassification\\mea_calibration_profile.json

The main app only reads the active profile. All write operations live in the
calibration tool.  When no user profile exists, the factory defaults bundled
with the app are used (shared/mea_defaults.json).

A v2→v3 migration shim absorbs legacy 13-material profiles into the new
6-material schema in-memory on load.  The on-disk profile is left untouched
until the calibration tool saves a fresh one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

_SHARED_PROFILE_PATH = (
    Path(os.getenv("PROGRAMDATA", "C:/ProgramData"))
    / "MaterialClassification"
    / "mea_calibration_profile.json"
)

_FACTORY_DEFAULT_PATH = Path(__file__).parent.parent.parent / "shared" / "mea_defaults.json"

# Maps each legacy 13-material name to its 6-material parent.
# Mirrors core.py _MEA_LEGACY_TO_PARENT to keep this module standalone.
_LEGACY_TO_PARENT: Dict[str, str] = {
    "BM_PAINT_ASPHALT":  "BM_ASPHALT",
    "BM_ROCK":           "BM_CONCRETE",
    "BM_METAL":          "BM_CONCRETE",
    "BM_METAL_STEEL":    "BM_CONCRETE",
    "BM_FOLIAGE":        "BM_VEGETATION",
    "BM_LAND_GRASS":     "BM_VEGETATION",
    "BM_LAND_DRY_GRASS": "BM_VEGETATION",
}


def _load_factory_defaults() -> Dict[str, Any]:
    try:
        return json.loads(_FACTORY_DEFAULT_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[mea_profile] Warning: failed to read factory defaults: {e}")
        return {}


def _migrate_legacy_profile(user: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v2 (13-material) user profile to v3 (6-material) in-memory.

    Legacy materials' anchor RGBs are folded into their new parent's anchor
    list so the parent material absorbs the calibrated color samples. The
    legacy keys are then removed.  No-op if profile is already v3+.
    """
    if int(user.get("version", 1)) >= 3:
        return user

    user_mats: Dict[str, Any] = dict(user.get("material_overrides", {}))
    if not any(name in user_mats for name in _LEGACY_TO_PARENT):
        # v2 profile but no legacy keys — just bump version.
        out = dict(user)
        out["version"] = 3
        return out

    print("[mea_profile] Migrating v2 profile -> v3 (absorbing legacy materials)")
    for legacy_name, parent_name in _LEGACY_TO_PARENT.items():
        legacy_mat = user_mats.pop(legacy_name, None)
        if not isinstance(legacy_mat, dict):
            continue   # missing, malformed (None / int), or non-dict — skip
        parent_mat = user_mats.setdefault(parent_name, {"anchors": []})
        parent_anchors = list(parent_mat.get("anchors", []))
        for anchor in legacy_mat.get("anchors", []) or []:
            if anchor not in parent_anchors:
                parent_anchors.append(anchor)
        parent_mat["anchors"] = parent_anchors

    out = dict(user)
    out["material_overrides"] = user_mats
    out["version"] = 3
    return out


def load_active_profile() -> Dict[str, Any]:
    """Return the active profile, merging user overrides onto factory defaults.

    Legacy v2 user profiles are migrated to v3 in-memory before merging.
    """
    if _SHARED_PROFILE_PATH.exists():
        try:
            user = json.loads(_SHARED_PROFILE_PATH.read_text(encoding="utf-8"))
            user = _migrate_legacy_profile(user)
            factory = _load_factory_defaults()
            merged = _merge_profile(factory, user)
            merged["_source"] = "user"
            return merged
        except Exception as e:
            print(f"[mea_profile] Warning: failed to read user profile: {e}")

    profile = _load_factory_defaults()
    profile["_source"] = "factory"
    return profile


def profile_status() -> Dict[str, Any]:
    """Return a lightweight status dict for GET /mea-profile/status."""
    if _SHARED_PROFILE_PATH.exists():
        try:
            user = json.loads(_SHARED_PROFILE_PATH.read_text(encoding="utf-8"))
            user = _migrate_legacy_profile(user)
            mat_count = len(user.get("material_overrides", {}))
            return {
                "active": True,
                "source": "user",
                "name": user.get("name", ""),
                "created_at": user.get("created_at", ""),
                "raster_path": user.get("raster_path", ""),
                "material_count": mat_count,
                "is_factory_default": False,
                "profile_path": str(_SHARED_PROFILE_PATH),
            }
        except Exception:
            pass

    factory = _load_factory_defaults()
    return {
        "active": True,
        "source": "factory",
        "name": factory.get("name", "Factory Default"),
        "created_at": factory.get("created_at", ""),
        "raster_path": "",
        "material_count": len(factory.get("material_overrides", {})),
        "is_factory_default": True,
        "profile_path": str(_FACTORY_DEFAULT_PATH),
    }


def _merge_profile(factory: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user profile onto factory defaults — user values take precedence."""
    merged = dict(factory)
    merged.update({k: v for k, v in user.items() if k != "material_overrides"})

    factory_mats = factory.get("material_overrides", {})
    user_mats = user.get("material_overrides", {})
    merged_mats: Dict[str, Any] = {}
    for mat_name, factory_mat in factory_mats.items():
        if mat_name in user_mats:
            merged_mats[mat_name] = {**factory_mat, **user_mats[mat_name]}
        else:
            merged_mats[mat_name] = dict(factory_mat)
    # Any extra materials in the user profile that aren't in factory defaults
    for mat_name, user_mat in user_mats.items():
        if mat_name not in merged_mats:
            merged_mats[mat_name] = dict(user_mat)
    merged["material_overrides"] = merged_mats

    # Merge bias_overrides
    factory_bias = factory.get("bias_overrides", {})
    user_bias = user.get("bias_overrides", {})
    merged["bias_overrides"] = {**factory_bias, **user_bias}

    return merged
