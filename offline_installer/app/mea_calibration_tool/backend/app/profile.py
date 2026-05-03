"""
MEA calibration profile management for the standalone calibration tool.

Writes to the shared profile path:
  %ProgramData%\MaterialClassification\mea_calibration_profile.json

This is the only component that has write access to the shared profile.
The main app reads via mea_profile.py (read-only consumer).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_PROFILE_DIR = (
    Path(os.getenv("PROGRAMDATA", "C:/ProgramData")) / "MaterialClassification"
)
_PROFILE_PATH = _PROFILE_DIR / "mea_calibration_profile.json"

# Factory defaults bundled with the tool (two levels up from this file → repo root)
_FACTORY_DEFAULT_PATH = (
    Path(__file__).parent.parent.parent.parent / "shared" / "mea_defaults.json"
)


def profile_path() -> Path:
    return _PROFILE_PATH


def load_profile() -> Optional[Dict[str, Any]]:
    """Return the saved user profile, or None if none exists."""
    if not _PROFILE_PATH.exists():
        return None
    try:
        return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[profile] Warning: failed to read profile: {e}")
        return None


def load_factory_defaults() -> Dict[str, Any]:
    try:
        return json.loads(_FACTORY_DEFAULT_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[profile] Warning: failed to read factory defaults: {e}")
        return {}


def save_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """Persist *data* as the active calibration profile (v2 schema)."""
    profile = dict(data)
    profile["version"] = 2
    profile.setdefault("created_at", datetime.now(timezone.utc).isoformat())

    # Compute tolerance_radius for each material that has std_rgb
    for mat_name, mo in profile.get("material_overrides", {}).items():
        if "sample_std_rgb" in mo and "tolerance_radius" not in mo:
            std = mo["sample_std_rgb"]
            max_std = max(std[0], std[1], std[2]) if len(std) >= 3 else std[0]
            mo["tolerance_radius"] = max(30, int(round(2 * max_std)))

    _PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    _PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return profile


def delete_profile() -> bool:
    if _PROFILE_PATH.exists():
        _PROFILE_PATH.unlink()
        return True
    return False


def export_profile(dest_path: str) -> str:
    if not _PROFILE_PATH.exists():
        raise FileNotFoundError("No active profile to export.")
    shutil.copy2(_PROFILE_PATH, dest_path)
    return dest_path


def import_profile(src_path: str) -> Dict[str, Any]:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return save_profile(data)
