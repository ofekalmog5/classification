"""
Persistent configuration for the Classification Web App.

Settings are stored in a JSON file next to the executable (or in the project
root during development).  This allows paths like SAM3_LOCAL_DIR and
HF_CACHE_DIR to survive restarts without requiring environment variables.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Config file location
# ---------------------------------------------------------------------------

def _config_dir() -> Path:
    """Return the directory where config.json lives.

    - Frozen exe  → next to the .exe
    - Development → project root (two levels up from backend/app/)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent.parent


_CONFIG_FILE = _config_dir() / "app_config.json"

_DEFAULTS: Dict[str, Any] = {
    "sam3_local_dir": None,       # str | None  — path to sam3-main folder
    "hf_cache_dir": None,         # str | None  — custom HuggingFace cache dir
    "offline_mode": False,        # bool        — force offline (no network)
}

_cache: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load() -> Dict[str, Any]:
    """Load config from disk (cached after first read)."""
    global _cache
    if _cache is not None:
        return _cache

    cfg = dict(_DEFAULTS)
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                stored = json.load(f)
            cfg.update({k: v for k, v in stored.items() if k in _DEFAULTS})
        except Exception as e:
            print(f"[config] Warning: failed to read {_CONFIG_FILE}: {e}")

    _cache = cfg
    _apply_env(cfg)
    return cfg


def save(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge *updates* into the config, write to disk, and return the full config."""
    global _cache
    cfg = load()
    cfg.update({k: v for k, v in updates.items() if k in _DEFAULTS})
    _cache = cfg

    try:
        _CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[config] Warning: failed to write {_CONFIG_FILE}: {e}")

    _apply_env(cfg)
    return cfg


def get(key: str, default: Any = None) -> Any:
    """Get a single config value."""
    return load().get(key, default)


def config_path() -> str:
    """Return the path to the config file (for display in UI)."""
    return str(_CONFIG_FILE)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _apply_env(cfg: Dict[str, Any]):
    """Push config values into environment variables where needed."""
    if cfg.get("offline_mode"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    hf_cache = cfg.get("hf_cache_dir")
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache
