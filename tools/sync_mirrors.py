"""Sync canonical files into the offline_installer and mea_calibration_tool mirrors.

The codebase keeps three near-identical trees:
  - canonical:           backend/, shared/, web_app/
  - offline-installer:   offline_installer/app/{backend,shared,web_app,mea_calibration_tool}/
  - calibration-tool:    mea_calibration_tool/{backend,web_app}/

Drift between trees has caused bugs in the past (see git history). This script
copies canonical → mirrors verbatim and reports any pre-existing divergence.

Usage:
    python tools/sync_mirrors.py             # write mode: overwrite mirrors
    python tools/sync_mirrors.py --check     # CI mode: exit non-zero on drift
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# (canonical_relative_path, [mirror_relative_paths])
SYNC_PAIRS: List[Tuple[str, List[str]]] = [
    # Backend Python — main classification engine
    ("backend/app/core.py", [
        "offline_installer/app/backend/app/core.py",
    ]),
    ("backend/app/main.py", [
        "offline_installer/app/backend/app/main.py",
    ]),
    ("backend/app/mea_profile.py", [
        "offline_installer/app/backend/app/mea_profile.py",
    ]),
    ("backend/app/road_extraction.py", [
        "offline_installer/app/backend/app/road_extraction.py",
    ]),
    # Shared MEA defaults
    ("shared/mea_defaults.json", [
        "offline_installer/app/shared/mea_defaults.json",
    ]),
    # Frontend MEA constants — both web_app and the calibration tool's web_app
    ("web_app/src/constants/mea.ts", [
        "offline_installer/app/web_app/src/constants/mea.ts",
        "mea_calibration_tool/web_app/src/constants/mea.ts",
        "offline_installer/app/mea_calibration_tool/web_app/src/constants/mea.ts",
    ]),
    ("web_app/src/types.ts", [
        "offline_installer/app/web_app/src/types.ts",
    ]),
    # Calibration tool's types.ts has its own schema BUT mirrors to its own
    # offline copy.  We sync calibration -> calibration-offline (not from
    # canonical web_app types.ts).
    ("mea_calibration_tool/web_app/src/types.ts", [
        "offline_installer/app/mea_calibration_tool/web_app/src/types.ts",
    ]),
]


def sync_one(canonical: Path, mirror: Path, check_only: bool) -> bool:
    """Sync one canonical -> mirror pair. Returns True if any change happened
    (or would happen in --check mode)."""
    if not canonical.exists():
        print(f"  SKIP: canonical missing -> {canonical}")
        return False

    if not mirror.exists():
        print(f"  NEW : {mirror}")
        if not check_only:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, mirror)
        return True

    canonical_bytes = canonical.read_bytes()
    mirror_bytes = mirror.read_bytes()

    if canonical_bytes == mirror_bytes:
        return False

    print(f"  DIFF: {mirror}")
    if not check_only:
        shutil.copy2(canonical, mirror)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Don't write — exit non-zero on any drift.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    print(f"[sync_mirrors] root = {project_root}")
    print(f"[sync_mirrors] mode = {'check' if args.check else 'write'}")

    drifted = False
    for canonical_rel, mirror_rels in SYNC_PAIRS:
        canonical = project_root / canonical_rel
        print(f"\n• {canonical_rel}")
        for mirror_rel in mirror_rels:
            mirror = project_root / mirror_rel
            if sync_one(canonical, mirror, check_only=args.check):
                drifted = True

    print()
    if args.check:
        if drifted:
            print("[sync_mirrors] DRIFT detected. Run without --check to fix.")
            return 1
        print("[sync_mirrors] OK: all mirrors match canonical.")
        return 0
    print("[sync_mirrors] OK: write pass complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
