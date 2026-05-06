"""One-shot script: delete dead post-processing block and replace _build_mea_cluster_mapping.

Run from project root:
    python tools/_phase1_core_cleanup.py

Idempotent: detects whether the cleanup already happened and skips.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


CORE_PY = Path("backend/app/core.py")

# ── Replacement for _build_mea_cluster_mapping ────────────────────────────────
NEW_CLUSTER_MAPPING = '''def _build_mea_cluster_mapping(
    cluster_rgbs: List[Tuple[int, int, int]],
    material_classes: List[Dict[str, str]],
    cluster_counts: List[int] | None = None,
    material_prior: Dict[str, float] | None = None,
    cluster_semantics: List[Dict[str, float]] | None = None,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int, int]]]:
    """Assign each KMeans cluster to a MEA material via min-anchor RGB distance.

    Each material in MEA_CLASSES carries a list of RGB anchors that absorb
    clusters from across its semantic group (e.g. BM_VEGETATION absorbs the
    old foliage / grass / dry-grass anchors). A cluster is assigned to the
    material whose nearest anchor is closest in RGB space. Multiple clusters
    may map to the same material — this is correct for multi-anchor absorption.

    The optional ``material_prior`` adds a frequency-weighted nudge: rare
    materials get a small cost bump so common ones win ties.

    Returns ``(mapping, color_table)`` matching the legacy contract.
    """
    if not cluster_rgbs or not material_classes:
        return [], []

    n_clusters = len(cluster_rgbs)
    n_materials = len(material_classes)

    material_anchors: List[List[Tuple[int, int, int]]] = []
    for cls in material_classes:
        name = cls.get("name", "")
        if name in _MEA_ANCHOR_MAP:
            anchors = [tuple(int(c) for c in a) for a in _MEA_ANCHOR_MAP[name]]
        else:
            anchors = [_hex_to_rgb(cls.get("color", "#ffffff"))]
        material_anchors.append(anchors)

    cost = np.zeros((n_clusters, n_materials), dtype=np.float64)
    for i, crgb in enumerate(cluster_rgbs):
        for j, anchors in enumerate(material_anchors):
            min_d2 = min(
                (crgb[0] - ar) ** 2 + (crgb[1] - ag) ** 2 + (crgb[2] - ab) ** 2
                for (ar, ag, ab) in anchors
            )
            cost[i, j] = float(min_d2)

    if material_prior:
        prevalence_bump = MEA_PREVALENCE_WEIGHT * 1000.0
        for j, cls in enumerate(material_classes):
            p = float(material_prior.get(cls.get("name", ""), 0.0))
            cost[:, j] += prevalence_bump * (1.0 - p)

    cluster_to_material = np.argmin(cost, axis=1)

    mapping: List[Dict[str, object]] = []
    assigned: List[Tuple[int, int, int]] = []
    for i, j in enumerate(cluster_to_material):
        cls = material_classes[int(j)]
        rgb = _hex_to_rgb(cls.get("color", "#ffffff"))
        mapping.append({
            "cluster": i + 1,
            "material": cls.get("name", "UNKNOWN"),
            "colorHex": cls.get("color", "#ffffff"),
            "colorRGB": rgb,
        })
        assigned.append(rgb)

    return mapping, assigned
'''


def remove_dead_block(src: str) -> tuple[str, int]:
    """Remove the dead post-processing block: from `_preprocess_shadow_balance`
    through `apply_road_object_removal`, stopping at the next live function.

    Live boundary: `def _classify_tile_worker(`.
    """
    start_re = re.compile(r"\ndef _preprocess_shadow_balance\(")
    end_re = re.compile(r"\ndef _classify_tile_worker\(")

    start = start_re.search(src)
    end = end_re.search(src)
    if start is None or end is None:
        return src, 0
    if start.start() >= end.start():
        return src, 0  # already removed

    removed = src[start.start():end.start()]
    new_src = src[:start.start()] + "\n\n" + src[end.start() + 1:]
    return new_src, removed.count("\n")


def replace_cluster_mapping(src: str) -> tuple[str, bool]:
    """Replace _build_mea_cluster_mapping body. Bounded by next top-level def."""
    # Marker comment in the new implementation we use to detect already-replaced.
    if "Assign each KMeans cluster to a MEA material via min-anchor RGB distance." in src:
        return src, False

    start_re = re.compile(r"^def _build_mea_cluster_mapping\(", re.MULTILINE)
    start = start_re.search(src)
    if start is None:
        raise SystemExit("Could not find _build_mea_cluster_mapping definition")

    # End at the next top-level def (no leading whitespace).
    end_re = re.compile(r"^def [a-zA-Z_]", re.MULTILINE)
    end_match = None
    for m in end_re.finditer(src, pos=start.end()):
        end_match = m
        break
    if end_match is None:
        raise SystemExit("Could not find function-end boundary for _build_mea_cluster_mapping")

    new_src = src[:start.start()] + NEW_CLUSTER_MAPPING + "\n\n" + src[end_match.start():]
    return new_src, True


def main() -> int:
    if not CORE_PY.exists():
        print(f"FATAL: {CORE_PY} not found (run from project root)")
        return 1

    src = CORE_PY.read_text(encoding="utf-8")
    original_lines = src.count("\n")

    src, removed = remove_dead_block(src)
    print(f"Dead block removed: {removed} lines")

    src, replaced = replace_cluster_mapping(src)
    print(f"Cluster mapping replaced: {replaced}")

    new_lines = src.count("\n")
    print(f"core.py: {original_lines} -> {new_lines} lines (delta {new_lines - original_lines})")

    CORE_PY.write_text(src, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
