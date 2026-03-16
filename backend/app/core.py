from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import os
import sys
import gc
import site
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

# ── GDAL / VSI I/O tuning (must come before any rasterio import) ────────────
# Larger block-cache reduces tile re-reads; ALL_CPUS enables parallel
# compression/decompression; VSI read-cache speeds up windowed reads.
os.environ.setdefault("GDAL_CACHEMAX",      "1024")       # 1 GB GDAL block cache
os.environ.setdefault("GDAL_NUM_THREADS",   "ALL_CPUS")   # parallel compress/decompress
os.environ.setdefault("VSI_CACHE",          "TRUE")        # per-handle read-ahead cache
os.environ.setdefault("VSI_CACHE_SIZE",     "134217728")  # 128 MB per handle
os.environ.setdefault("GDAL_TIFF_INTERNAL_MASK", "YES")   # keep masks inside TIFF

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from skimage.segmentation import slic
from skimage.filters.rank import median, entropy as sk_rank_entropy
from skimage.morphology import disk
from skimage.measure import label as sk_label
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

# ─── KMeans acceleration (faiss > cuML > sklearn fallback) ────────────────────

# Persistent directory for GPU packages (shared with server_launcher.py).
# If faiss-gpu was installed previously (by EXE or dev run), it lives here.
_GPU_PKG_DIR = Path(os.environ.get('APPDATA', Path.home())) / 'ClassificationApp' / 'gpu_packages'
_pkg_str = str(_GPU_PKG_DIR)
if _GPU_PKG_DIR.exists() and _pkg_str not in sys.path:
    sys.path.insert(0, _pkg_str)


def _detect_gpu() -> tuple:
    """Detect NVIDIA GPU via pynvml or nvidia-smi.  Returns (available, info_str)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            raw = pynvml.nvmlDeviceGetName(handle)
            name = raw.decode() if isinstance(raw, bytes) else raw
            mem  = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = mem.free // (1024 * 1024)
            return True, f"{name} ({free_mb} MB free VRAM)"
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return True, r.stdout.strip().split("\n")[0].strip()
    except Exception:
        pass
    return False, "No NVIDIA GPU detected"


_GPU_AVAILABLE, _GPU_INFO = _detect_gpu()


class _FaissKMeans:
    """Sklearn-compatible wrapper around faiss.Kmeans (CPU or GPU)."""

    def __init__(self, n_clusters: int, *, max_iter: int = 80, use_gpu: bool = False):
        self.n_clusters      = n_clusters
        self.max_iter        = max_iter
        self.use_gpu         = use_gpu
        self.cluster_centers_: np.ndarray | None = None
        self.inertia_: float = 0.0
        self._km             = None

    def fit(self, X: np.ndarray):
        import faiss
        X = np.ascontiguousarray(X, dtype=np.float32)
        km = faiss.Kmeans(X.shape[1], self.n_clusters,
                          niter=self.max_iter, verbose=False,
                          gpu=self.use_gpu)
        km.train(X)
        self._km = km
        self.cluster_centers_ = km.centroids.copy()
        self.inertia_ = float(km.obj[-1]) if len(km.obj) > 0 else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        _, I = self._km.index.search(X, 1)
        return I.flatten()

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.predict(X)


def _try_faiss_gpu_probe() -> bool:
    """Return True if faiss-gpu is importable and can open GPU resources."""
    try:
        # Flush any stale faiss modules so a fresh import picks up new packages
        for mod in list(sys.modules):
            if 'faiss' in mod:
                del sys.modules[mod]
        import faiss
        res = faiss.StandardGpuResources()
        idx = faiss.GpuIndexFlatL2(res, 2)
        idx.add(np.zeros((1, 2), dtype=np.float32))
        del idx, res
        return True
    except Exception:
        return False


def _auto_install_faiss_gpu() -> bool:
    """One-time auto-install of the correct faiss-gpu wheel into _GPU_PKG_DIR.
    Returns True if faiss-gpu is working after installation."""
    import subprocess as _sp, re as _re

    # Detect CUDA major version from nvidia-smi
    cuda_major = None
    try:
        r = _sp.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        m = _re.search(r'CUDA Version:\s*(\d+)', r.stdout)
        if m:
            cuda_major = int(m.group(1))
    except Exception:
        pass

    candidates = []
    if cuda_major and cuda_major >= 12:
        candidates = ['faiss-gpu-cu12', 'faiss-gpu-cu11', 'faiss-gpu']
    elif cuda_major:
        candidates = ['faiss-gpu-cu11', 'faiss-gpu']
    else:
        candidates = ['faiss-gpu']

    _GPU_PKG_DIR.mkdir(parents=True, exist_ok=True)
    _pkg = str(_GPU_PKG_DIR)
    if _pkg not in sys.path:
        sys.path.insert(0, _pkg)

    for pkg in candidates:
        print(f"[GPU] Installing {pkg} -> {_GPU_PKG_DIR} ...")
        try:
            r = _sp.run(
                [sys.executable, '-m', 'pip', 'install',
                 '--target', str(_GPU_PKG_DIR),
                 '--quiet', '--no-deps', '--disable-pip-version-check', pkg],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0 and _try_faiss_gpu_probe():
                print(f"[GPU] {pkg} active ✓")
                return True
            if r.returncode != 0:
                print(f"[GPU] {pkg} install failed (exit {r.returncode})")
        except Exception as e:
            print(f"[GPU] {pkg} install error: {e}")
    return False


def _probe_acceleration() -> tuple:
    """Return (engine, gpu_available, gpu_info) where engine is 'faiss-gpu',
    'faiss-cpu', 'cuml', or 'sklearn'."""
    # ── faiss: try GPU first, then CPU ────────────────────────────────────────
    try:
        import faiss
        if _GPU_AVAILABLE:
            if _try_faiss_gpu_probe():
                return "faiss-gpu", True, _GPU_INFO
            # GPU present but faiss-gpu not available — try auto-install once
            print("[GPU] NVIDIA GPU detected but faiss-gpu not available, attempting auto-install...")
            if _auto_install_faiss_gpu():
                return "faiss-gpu", True, _GPU_INFO
            print("[GPU] Could not activate faiss-gpu — using faiss-cpu")
        return "faiss-cpu", _GPU_AVAILABLE, _GPU_INFO
    except ImportError:
        pass
    # ── cuML (Linux/WSL2 only) ─────────────────────────────────────────────────
    if _GPU_AVAILABLE:
        try:
            from cuml.cluster import MiniBatchKMeans as _  # noqa: F401
            return "cuml", True, _GPU_INFO
        except Exception:
            pass
    # ── sklearn CPU fallback ───────────────────────────────────────────────────
    return "sklearn", _GPU_AVAILABLE, _GPU_INFO


_ACCEL_ENGINE, _ACCEL_GPU, _ACCEL_GPU_INFO = _probe_acceleration()
print(f"[KMeans] engine={_ACCEL_ENGINE}  gpu={_ACCEL_GPU}  {_ACCEL_GPU_INFO if _ACCEL_GPU else '(CPU only)'}")


def _make_kmeans(n_clusters: int, *, mini_batch: bool = True):
    """Return the fastest available KMeans: faiss-gpu > faiss-cpu > cuML > sklearn."""
    if _ACCEL_ENGINE == "faiss-gpu":
        return _FaissKMeans(n_clusters, max_iter=80 if mini_batch else 300, use_gpu=True)
    if _ACCEL_ENGINE == "faiss-cpu":
        return _FaissKMeans(n_clusters, max_iter=80 if mini_batch else 300, use_gpu=False)
    if _ACCEL_ENGINE == "cuml":
        try:
            if mini_batch:
                from cuml.cluster import MiniBatchKMeans as _CuMBK
                return _CuMBK(n_clusters=n_clusters, random_state=42,
                               max_iter=80, batch_size=65536)
            from cuml.cluster import KMeans as _CuKM
            return _CuKM(n_clusters=n_clusters, random_state=42,
                          n_init=10, max_iter=300)
        except Exception as _e:
            print(f"[KMeans] cuML failed ({_e}), falling back to sklearn")
    # sklearn CPU
    if mini_batch:
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                               n_init=1, max_iter=80, batch_size=65536)
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)

# Max pixels sampled for KMeans training (quality is unchanged above ~50k)
MAX_TRAIN_PIXELS = 100_000
# Adaptive sampling bounds: PCA-based estimate will clamp to this range.
_MIN_TRAIN_PIXELS = 20_000
_MAX_TRAIN_PIXELS = 120_000
_PCA_PRESAMPLE    = 8_000    # tiny pre-sample for variance estimation
import math

VECTOR_OVERLAY_COLOR = (255, 255, 0)


def _pca_adaptive_n_train(
    pixel_features: np.ndarray,
    n_clusters: int,
) -> int:
    """Estimate an adequate training-sample size using a fast PCA on a tiny sub-sample.

    **Idea** - the number of training pixels KMeans needs scales with the
    *spectral complexity* of the image, not its pixel count.  A tile that is
    almost entirely one material (road, bare soil) can be clustered perfectly
    from 20 K pixels, while a mixed urban scene may need 100 K+.

    Algorithm
    ---------
    1. Draw a tiny random sub-sample (``_PCA_PRESAMPLE`` pixels, default 8 K).
    2. Standardise features (important: RGB 0-255 vs NDVI -1..1) then run PCA.
    3. Compute the **normalised Shannon entropy** of the eigenvalue spectrum:
       - *High entropy* (eigenvalues ~ equal) -> isotropic noise, one material,
         uniform scene -> **simple** -> fewer training pixels.
       - *Low entropy* (a few eigenvalues dominate) -> clear cluster structure,
         multiple distinct materials -> **complex** -> more training pixels.
    4. Combine entropy-based *complexity* score with ``n_clusters`` to produce
       the final pixel budget, linearly interpolated between
       ``_MIN_TRAIN_PIXELS`` (20 K) and ``_MAX_TRAIN_PIXELS`` (120 K).

    The PCA itself runs in < 5 ms on 8 K x 5 features - negligible overhead.

    Returns
    -------
    int
        Recommended number of training pixels, clamped to
        [``_MIN_TRAIN_PIXELS``, ``_MAX_TRAIN_PIXELS``].
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_px, n_feat = pixel_features.shape
    if n_px <= _MIN_TRAIN_PIXELS or n_feat <= 1:
        return min(n_px, MAX_TRAIN_PIXELS)

    # ---- tiny random sub-sample for speed ----------------------------------
    rng = np.random.default_rng(42)
    n_pre = min(_PCA_PRESAMPLE, n_px)
    idx = rng.choice(n_px, n_pre, replace=False)
    sample = pixel_features[idx].astype(np.float64)

    # Standardise so that RGB (0-255) and NDVI (-1..1) contribute equally
    sample = StandardScaler().fit_transform(sample)

    # ---- lightweight PCA ---------------------------------------------------
    n_comp = min(n_feat, n_pre)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(sample)

    # ---- eigenvalue entropy -> complexity -----------------------------------
    ev_ratio = pca.explained_variance_ratio_               # sums to ~1.0
    ev_ratio = np.clip(ev_ratio, 1e-12, None)              # avoid log(0)
    entropy = -float(np.sum(ev_ratio * np.log(ev_ratio)))
    max_entropy = float(np.log(n_comp))                    # uniform distrib.
    norm_entropy = entropy / max(max_entropy, 1e-9)        # 0 -> 1

    # complexity: 0 = simple (high entropy), 1 = complex (low entropy)
    complexity = 1.0 - norm_entropy

    # ---- map complexity -> pixel budget -------------------------------------
    # Weight by n_clusters: more clusters -> need more samples to separate
    cluster_weight = min(1.0, n_clusters / 15.0)
    # Final fraction: blend 60 % complexity + 40 % cluster demand
    frac = 0.6 * complexity + 0.4 * cluster_weight

    raw = _MIN_TRAIN_PIXELS + frac * (_MAX_TRAIN_PIXELS - _MIN_TRAIN_PIXELS)
    result = int(np.clip(raw, _MIN_TRAIN_PIXELS, _MAX_TRAIN_PIXELS))
    result = min(result, n_px)  # never exceed available pixels

    print(f"    [PCA] feats={n_feat}, entropy={norm_entropy:.2f}, "
          f"complexity={complexity:.2f}, clusters={n_clusters} "
          f"-> train_pixels={result:,}")
    return result


_MEA_ROAD_MATERIALS = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT", "BM_ROCK"}
_MEA_VEG_MATERIALS = {"BM_LAND_GRASS", "BM_LAND_DRY_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
_MEA_SOIL_MATERIALS = {"BM_SOIL", "BM_SAND"}
_MEA_METAL_MATERIALS = {"BM_METAL", "BM_METAL_STEEL"}


def _mea_material_type(name: str) -> str:
    if name in _MEA_ROAD_MATERIALS:
        return "ROAD_SURFACE"
    if name in _MEA_VEG_MATERIALS:
        return "VEGETATION"
    if name in _MEA_SOIL_MATERIALS:
        return "SOIL_EARTH"
    if name in _MEA_METAL_MATERIALS:
        return "METAL"
    if name == "BM_WATER":
        return "WATER"
    return "OTHER"


def _write_mea_palette_reference(output_dir: Path) -> str:
    """Write a CSV reference of MEA classes/materials/final RGB values."""
    output_dir.mkdir(parents=True, exist_ok=True)
    palette_path = output_dir / "mea_palette_reference.csv"
    lines = ["class_id,material,material_type,hex_color,final_rgb"]
    for cls in MEA_CLASSES:
        name = cls.get("name", "UNKNOWN")
        hex_color = cls.get("color", "#000000")
        rgb = _hex_to_rgb(hex_color)
        lines.append(
            f"{cls.get('id','')},{name},{_mea_material_type(name)},{hex_color},\"{rgb[0]},{rgb[1]},{rgb[2]}\""
        )
    palette_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(palette_path)

# Shared MEA material class definitions (used by GUI and CLI)
MEA_CLASSES = [
    {"id": "class-1",  "name": "BM_ASPHALT",        "color": "#2D2D30"},
    {"id": "class-2",  "name": "BM_CONCRETE",       "color": "#B4B4B4"},
    {"id": "class-3",  "name": "BM_FOLIAGE",        "color": "#006400"},
    {"id": "class-4",  "name": "BM_LAND_DRY_GRASS", "color": "#BDB76B"},
    {"id": "class-5",  "name": "BM_LAND_GRASS",     "color": "#7CFC00"},
    {"id": "class-6",  "name": "BM_METAL",          "color": "#A9ABB0"},
    {"id": "class-7",  "name": "BM_METAL_STEEL",    "color": "#708090"},
    {"id": "class-8",  "name": "BM_PAINT_ASPHALT",  "color": "#3C3F41"},
    {"id": "class-9",  "name": "BM_ROCK",           "color": "#827B73"},
    {"id": "class-10", "name": "BM_SAND",           "color": "#EDC9AF"},
    {"id": "class-11", "name": "BM_SOIL",           "color": "#654321"},
    {"id": "class-12", "name": "BM_VEGETATION",     "color": "#228B22"},
    {"id": "class-13", "name": "BM_WATER",          "color": "#1C6BA0"},
]

# Maps each MEA material's BM_ name to its Composite_Material display name used in XML exports.
_MEA_COMPOSITE_NAMES: Dict[str, str] = {
    "BM_ASPHALT":        "ASPHALT",
    "BM_CONCRETE":       "CONCRETE",
    "BM_FOLIAGE":        "FOLIAGE",
    "BM_LAND_DRY_GRASS": "DRY_GRASS",
    "BM_LAND_GRASS":     "LAND_GRASS",
    "BM_METAL":          "METAL",
    "BM_METAL_STEEL":    "METAL_STEEL",
    "BM_PAINT_ASPHALT":  "PAINT_ASPHALT",
    "BM_ROCK":           "ROCK",
    "BM_SAND":           "SAND",
    "BM_SOIL":           "SOIL",
    "BM_VEGETATION":     "GENVEGETATION",
    "BM_WATER":          "WATER",
}


def _write_composite_material_xml(
    output_path,
    classes: List[Dict[str, str]],
) -> Optional[str]:
    """Write a Composite_Material_Table XML alongside a classification output file.

    The XML is placed next to ``output_path`` with the same stem and a ``.xml``
    extension.  Each MEA class produces one ``<Composite_Material>`` entry whose
    color is the ARGB hex of the class colour (fully-opaque, lower-case).
    """
    import xml.etree.ElementTree as ET

    xml_path = Path(output_path).with_suffix(".xml")

    root = ET.Element("Composite_Material_Table")

    for idx, cls in enumerate(classes, start=1):
        bm_name   = cls.get("name", "")
        color_hex = cls.get("color", "#000000")
        composite_name = _MEA_COMPOSITE_NAMES.get(bm_name, bm_name.replace("BM_", ""))

        # Build ARGB colour: #ff + RRGGBB (lower-case, no alpha adjustment).
        if color_hex.startswith("#") and len(color_hex) == 7:
            argb_color = f"#ff{color_hex[1:].lower()}"
        else:
            argb_color = "#ff000000"

        cm = ET.SubElement(root, "Composite_Material", index=str(idx))
        ET.SubElement(cm, "Name").text = composite_name
        ET.SubElement(cm, "Color").text = argb_color
        ps  = ET.SubElement(cm, "Primary_Substrate")
        ET.SubElement(ps, "Thickness").text = "1"
        mat = ET.SubElement(ps, "Material")
        ET.SubElement(mat, "Name").text = bm_name
        ET.SubElement(mat, "Weight").text = "100"

    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    try:
        tree.write(str(xml_path), encoding="unicode", xml_declaration=False)
        print(f"  [XML] Composite material table written: {xml_path}")
        return str(xml_path)
    except Exception as exc:
        print(f"  [XML] Warning: could not write material XML: {exc}")
        return None


# Relative expected prevalence in urban aerial scenes (higher = usually more common).
MEA_MATERIAL_FREQUENCY_PRIOR_URBAN: Dict[str, float] = {
    "BM_CONCRETE": 0.18,
    "BM_ASPHALT": 0.16,
    "BM_SOIL": 0.15,
    "BM_ROCK": 0.08,
    "BM_PAINT_ASPHALT": 0.04,
    "BM_SAND": 0.04,
    "BM_WATER": 0.04,
    "BM_LAND_DRY_GRASS": 0.07,
    "BM_LAND_GRASS": 0.06,
    "BM_VEGETATION": 0.05,
    "BM_FOLIAGE": 0.03,
    "BM_METAL": 0.012,   # metal is rare in top-down urban aerial
    "BM_METAL_STEEL": 0.006,
}

# Relative expected prevalence in open/rural scenes.
MEA_MATERIAL_FREQUENCY_PRIOR_OPEN: Dict[str, float] = {
    "BM_SOIL": 0.25,
    "BM_LAND_GRASS": 0.13,
    "BM_VEGETATION": 0.12,
    "BM_LAND_DRY_GRASS": 0.11,
    "BM_SAND": 0.08,
    "BM_ROCK": 0.06,
    "BM_WATER": 0.05,
    "BM_CONCRETE": 0.05,
    "BM_ASPHALT": 0.04,
    "BM_FOLIAGE": 0.03,
    "BM_PAINT_ASPHALT": 0.003,
    "BM_METAL": 0.008,   # very rare in aerial open-land
    "BM_METAL_STEEL": 0.003,
}

# Backward-compatible default prior.
MEA_MATERIAL_FREQUENCY_PRIOR: Dict[str, float] = dict(MEA_MATERIAL_FREQUENCY_PRIOR_URBAN)

# Weight for prevalence term in assignment cost (0=color-only, 1=mostly prevalence).
MEA_PREVALENCE_WEIGHT = 0.28

# Weight for semantic term (vegetation/road cues) in assignment cost.
MEA_SEMANTIC_WEIGHT = 0.55  # raised: semantic penalties must matter over pure color

# Scene-adaptive prior sampling configuration (fast regional estimation).
MEA_SCENE_SAMPLE_RADIUS_METERS = 90.0
MEA_SCENE_MAX_SAMPLE_ZONES = 16


def _open_weight_from_rgb_arrays(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> Tuple[float, float, float, float]:
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = (maxc - minc) / (maxc + 1e-6)

    exg = (2.0 * g) - r - b
    vegetation_frac = float(np.mean((exg > 0.08) & (g > 0.22)))
    soil_frac = float(np.mean((r > g) & (g > b) & (sat > 0.07) & (maxc > 0.18)))
    gray_urban_frac = float(np.mean((sat < 0.12) & (maxc > 0.15) & (maxc < 0.85)))

    open_weight = 0.5 + 1.1 * (vegetation_frac + 0.65 * soil_frac - 0.9 * gray_urban_frac)
    open_weight = float(np.clip(open_weight, 0.0, 1.0))
    return open_weight, vegetation_frac, soil_frac, gray_urban_frac


def _scene_adaptive_mea_prior(
    raster_data: np.ndarray,
    material_classes: List[Dict[str, str]],
    transform=None,
) -> Tuple[Dict[str, float], str]:
    """
    Infer whether the scene is more urban or open from RGB color statistics,
    then blend urban/open material priors accordingly.
    """
    rgb = raster_data[:3] if raster_data.shape[0] >= 3 else raster_data
    if rgb.shape[0] < 3:
        return dict(MEA_MATERIAL_FREQUENCY_PRIOR), "default"

    r_all = rgb[0].astype(np.float32)
    g_all = rgb[1].astype(np.float32)
    b_all = rgb[2].astype(np.float32)
    height, width = r_all.shape

    px_x = float(abs(getattr(transform, "a", 1.0))) if transform is not None else 1.0
    px_y = float(abs(getattr(transform, "e", 1.0))) if transform is not None else 1.0
    meters_per_px = max(0.01, (px_x + px_y) * 0.5)
    radius_px = int(np.clip(MEA_SCENE_SAMPLE_RADIUS_METERS / meters_per_px, 20, 140))

    if height < (2 * radius_px + 1) or width < (2 * radius_px + 1):
        # Fallback: global quick estimate on small images
        stride = max(1, int(math.sqrt((r_all.size / 220_000.0))))
        r = r_all[::stride, ::stride]
        g = g_all[::stride, ::stride]
        b = b_all[::stride, ::stride]
        scale = np.percentile(np.concatenate([r.ravel(), g.ravel(), b.ravel()]), 98)
        scale = float(max(scale, 1.0))
        r = np.clip(r / scale, 0.0, 1.0)
        g = np.clip(g / scale, 0.0, 1.0)
        b = np.clip(b / scale, 0.0, 1.0)
        open_weight, vegetation_frac, soil_frac, gray_urban_frac = _open_weight_from_rgb_arrays(r, g, b)
        zone_count = 1
    else:
        zone_grid = max(2, int(math.ceil(math.sqrt(MEA_SCENE_MAX_SAMPLE_ZONES))))
        ys = np.linspace(radius_px, height - radius_px - 1, zone_grid, dtype=int)
        xs = np.linspace(radius_px, width - radius_px - 1, zone_grid, dtype=int)

        zone_weights: List[float] = []
        zone_veg: List[float] = []
        zone_soil: List[float] = []
        zone_gray: List[float] = []
        zone_count = 0

        for cy in ys:
            for cx in xs:
                if zone_count >= MEA_SCENE_MAX_SAMPLE_ZONES:
                    break
                y0, y1 = cy - radius_px, cy + radius_px + 1
                x0, x1 = cx - radius_px, cx + radius_px + 1
                zr = r_all[y0:y1, x0:x1]
                zg = g_all[y0:y1, x0:x1]
                zb = b_all[y0:y1, x0:x1]

                z_stride = max(1, int(math.sqrt((zr.size / 6000.0))))
                zr = zr[::z_stride, ::z_stride]
                zg = zg[::z_stride, ::z_stride]
                zb = zb[::z_stride, ::z_stride]

                z_scale = np.percentile(np.concatenate([zr.ravel(), zg.ravel(), zb.ravel()]), 98)
                z_scale = float(max(z_scale, 1.0))
                zr = np.clip(zr / z_scale, 0.0, 1.0)
                zg = np.clip(zg / z_scale, 0.0, 1.0)
                zb = np.clip(zb / z_scale, 0.0, 1.0)

                w, v, s, gr = _open_weight_from_rgb_arrays(zr, zg, zb)
                zone_weights.append(w)
                zone_veg.append(v)
                zone_soil.append(s)
                zone_gray.append(gr)
                zone_count += 1
            if zone_count >= MEA_SCENE_MAX_SAMPLE_ZONES:
                break

        if zone_count == 0:
            return dict(MEA_MATERIAL_FREQUENCY_PRIOR), "default"

        open_weight = float(np.mean(zone_weights))
        vegetation_frac = float(np.mean(zone_veg))
        soil_frac = float(np.mean(zone_soil))
        gray_urban_frac = float(np.mean(zone_gray))

    priors: Dict[str, float] = {}
    for cls in material_classes:
        name = cls.get("name", "")
        p_u = MEA_MATERIAL_FREQUENCY_PRIOR_URBAN.get(name, 0.05)
        p_o = MEA_MATERIAL_FREQUENCY_PRIOR_OPEN.get(name, 0.05)
        priors[name] = ((1.0 - open_weight) * p_u) + (open_weight * p_o)

    s = float(sum(priors.values()))
    if s > 0:
        for key in list(priors.keys()):
            priors[key] = priors[key] / s

    radius_m = radius_px * meters_per_px
    profile = (
        f"adaptive(open={open_weight:.2f}, veg={vegetation_frac:.2f}, soil={soil_frac:.2f}, "
        f"gray={gray_urban_frac:.2f}, zones={zone_count}, radius~{radius_m:.0f}m)"
    )
    return priors, profile

# Ordered candidate palette for vector overlays – chosen to be visually vivid
_VECTOR_CANDIDATE_COLORS: List[Tuple[int, int, int]] = [
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (255, 255, 255),  # White
    (0, 255, 128),    # Spring Green
    (255, 0, 128),    # Rose
    (0, 128, 255),    # Azure
    (128, 0, 255),    # Purple
    (200, 250, 50),   # Lime
]


def _pick_vector_overlay_colors(
    classes: List[Dict[str, str]], n_vectors: int
) -> List[Tuple[int, int, int]]:
    """Return n_vectors RGB colors that are visually distinct from every classification class color."""
    used: List[Tuple[int, int, int]] = []
    for cls in classes:
        hex_color = cls.get("color", "")
        if hex_color.startswith("#") and len(hex_color) == 7:
            try:
                used.append((
                    int(hex_color[1:3], 16),
                    int(hex_color[3:5], 16),
                    int(hex_color[5:7], 16),
                ))
            except ValueError:
                pass

    def _dist(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    THRESHOLD = 80.0
    chosen: List[Tuple[int, int, int]] = []
    for _ in range(n_vectors):
        for candidate in _VECTOR_CANDIDATE_COLORS:
            if candidate in chosen:
                continue
            if all(_dist(candidate, u) > THRESHOLD for u in used):
                chosen.append(candidate)
                break
        else:
            # Fallback: first unused candidate regardless of proximity
            for candidate in _VECTOR_CANDIDATE_COLORS:
                if candidate not in chosen:
                    chosen.append(candidate)
                    break
            else:
                chosen.append(VECTOR_OVERLAY_COLOR)
    return chosen


def _normalize_pseudo_mercator_crs(crs: CRS) -> CRS:
    if crs is None:
        return crs
    crs_text = str(crs)
    if crs_text.startswith("LOCAL_CS") and "Pseudo-Mercator" in crs_text:
        return CRS.from_epsg(3857)
    return crs


def _driver_for_path(path) -> str:
    """Return rasterio driver name based on output file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == '.img':
        return 'HFA'
    return 'GTiff'


def _profile_for_driver(profile: dict, driver: str) -> dict:
    """Return a copy of profile with the given driver, removing incompatible GTiff-only keys if needed."""
    out = dict(profile)
    out['driver'] = driver
    if driver == 'HFA':
        # HFA does not support GTiff creation options
        for key in ('compress', 'tiled', 'blockxsize', 'blockysize', 'zlevel', 'predictor', 'interleave'):
            out.pop(key, None)
    return out


def _output_tiff_profile(profile: dict, *, dtype: str | None = None) -> dict:
    """Return a write-optimised GTiff profile.

    Applies tiled layout with 512x512 blocks, DEFLATE compression at the
    fastest level (zlevel=1) and horizontal-differencing predictor=2.  For
    uint8 classification maps this typically yields 3-5x smaller files
    compared to uncompressed GTiff and avoids the in-memory write-buffer
    ceiling that rasterio hits with very large uncompressed rasters.
    """
    out = dict(profile)
    out["driver"]     = "GTiff"
    out["tiled"]      = True
    out["blockxsize"] = 512
    out["blockysize"] = 512
    out["compress"]   = "deflate"
    out["zlevel"]     = 1     # fastest deflate - good size/speed tradeoff for uint8 maps
    out["predictor"]  = 2     # horizontal differencing - effective for float/byte rasters
    out["interleave"] = "band"
    if dtype is not None:
        out["dtype"] = dtype
    return out


def _auto_tile_size(height: int, width: int, max_pixels: int) -> int:
    max_pixels = max(128 * 128, int(max_pixels))
    tile_size = int(math.sqrt(max_pixels))
    tile_size = max(128, tile_size)
    tile_size = min(tile_size, max(height, width))
    return tile_size


def _sample_raster_for_training(
    raster_path: str,
    feature_flags: Dict[str, bool],
    n_samples: int = MAX_TRAIN_PIXELS,
    grid_steps: int = 8,
    window_size: int = 512,
) -> np.ndarray:
    """Return a pixel-feature matrix sampled uniformly across the raster.

    Spreads ``grid_steps x grid_steps`` windows over the full image extent
    and concatenates their features.  Used when the raster is too large to
    load into RAM for full-image feature extraction.
    """
    rng = np.random.default_rng(42)
    parts: List[np.ndarray] = []
    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        rows_pos = np.linspace(0, max(0, H - window_size), grid_steps, dtype=int)
        cols_pos = np.linspace(0, max(0, W - window_size), grid_steps, dtype=int)
        for r in rows_pos:
            for c in cols_pos:
                win = Window(int(c), int(r), min(window_size, W - int(c)), min(window_size, H - int(r)))
                tile_data = src.read(window=win)
                parts.append(_extract_pixel_features(tile_data, feature_flags, verbose=False))
    combined = np.concatenate(parts, axis=0)
    if len(combined) > n_samples:
        idx = rng.choice(len(combined), n_samples, replace=False)
        combined = combined[idx]
    return combined


def _available_ram_bytes() -> int:
    """Return available physical RAM in bytes (best-effort, falls back to 2 GB)."""
    try:
        import ctypes
        import ctypes.wintypes
        class _MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength",               ctypes.c_ulong),
                ("dwMemoryLoad",           ctypes.c_ulong),
                ("ullTotalPhys",           ctypes.c_ulonglong),
                ("ullAvailPhys",           ctypes.c_ulonglong),
                ("ullTotalPageFile",       ctypes.c_ulonglong),
                ("ullAvailPageFile",       ctypes.c_ulonglong),
                ("ullTotalVirtual",        ctypes.c_ulonglong),
                ("ullAvailVirtual",        ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual",ctypes.c_ulonglong),
            ]
        stat = _MEMSTATUS()
        stat.dwLength = ctypes.sizeof(stat)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullAvailPhys)
    except Exception:
        pass
    try:
        # Linux / macOS fallback via /proc/meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 2 * 1024 ** 3  # conservative 2 GB fallback


def suggest_tile_size(raster_path: str, workers: int = 4) -> int:
    """Return an ideal square tile side length (pixels) for the given raster.

    Picks the largest power-of-2 size from {256, 512, 1024, 2048, 4096} such
    that processing a single tile fits within ~20 % of available RAM divided
    by the number of workers.  The result is capped so the tile is never
    larger than the image itself.
    """
    _SNAP = [4096, 2048, 1024, 512, 256]
    try:
        with rasterio.open(raster_path) as src:
            img_h, img_w = src.height, src.width
            bands = src.count
            itemsize = int(np.dtype(src.dtypes[0]).itemsize)
    except Exception:
        return 1024

    avail = _available_ram_bytes()
    # Budget: 60 % of available RAM shared across workers, x6 scratch copies.
    # GDAL + NumPy own most of the rest; the x6 scratch multiplier is conservative.
    budget_bytes = avail * 0.60 / max(1, workers)
    bytes_per_px = max(bands, 3) * max(itemsize, 4) * 6  # x6 for working copies
    max_px = int(budget_bytes / bytes_per_px)
    ideal = int(math.sqrt(max(max_px, 256 * 256)))

    for snap in _SNAP:
        if ideal >= snap and min(img_h, img_w) >= snap:
            return snap
    return 1024   # minimum useful tile size


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
    # Default: create 'output' folder next to the source file
    return base_path.parent / "output"


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

from scipy.ndimage import binary_dilation, maximum_filter, minimum_filter, uniform_filter, distance_transform_edt
import scipy.ndimage as ndi


def _nearest_center_chunked(
    X: np.ndarray,
    centers: np.ndarray,
    chunk: int = 262_144,
) -> np.ndarray:
    """Return the 0-based nearest-center index for every row in *X*.

    Uses chunked squared-L2 distances computed via BLAS ``dgemm``/``sgemm``
    (the ``@`` operator) instead of a full NxK ``cdist`` call:

        ||x − c||² = ||x||² − 2 x·cᵀ + ||c||²

    Peak extra memory is bounded by ``chunk x K x dtype_bytes``
    (~ 16 MB for chunk=131 072, K=32, float32) regardless of N, whereas
    ``cdist`` allocates the full NxK matrix at once.

    Both *X* and *centers* should be float32 for maximum throughput
    (BLAS SGEMM is typically 2x faster than DGEMM on float64).

    When the data is large enough (> 2 chunks), processing is distributed
    across multiple threads.  BLAS ``sgemm`` and NumPy ``argmin`` both
    release the GIL, so thread-parallel chunks yield near-linear speed-up
    on multi-core machines.
    """
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    n = X.shape[0]
    labels = np.empty(n, dtype=np.int32)
    c2 = (centers * centers).sum(axis=1)    # (K,)

    def _assign_chunk(start_end):
        s, e = start_end
        xc = X[s:e]                           # (B, F)
        x2 = (xc * xc).sum(axis=1)           # (B,)
        dot = xc @ centers.T                  # (B, K) - BLAS SGEMM
        sq = x2[:, None] + c2[None, :] - 2.0 * dot
        np.maximum(sq, 0.0, out=sq)
        labels[s:e] = sq.argmin(axis=1)

    ranges = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    if len(ranges) <= 2:
        # Small data - sequential is faster (no thread-spawn overhead)
        for rng in ranges:
            _assign_chunk(rng)
    else:
        _n_workers = min(len(ranges), max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            list(pool.map(_assign_chunk, ranges))

    return labels


# ---------------------------------------------------------------------------
# Shadow pre-processing: balance / brighten shadows BEFORE classification
# ---------------------------------------------------------------------------

def _preprocess_shadow_balance(
    raster_data: np.ndarray,
    local_window: int = 61,
    darkness_ratio: float = 0.62,
    blend_strength: float = 0.80,
    min_shadow_px: int = 120,
) -> np.ndarray:
    """Detect shadow regions and locally brighten them to match surrounding areas.

    This is run **before** any classification so that shadow-darkened pixels
    are spectrally closer to their true material colour.  The approach:

    1. Compute per-pixel brightness (mean of RGB bands).
    2. Build a local mean brightness map (``local_window``-sized box filter).
    3. Mark shadow candidates: pixels whose brightness < ``darkness_ratio`` x
       local mean *and* that have low colour saturation (shadows are achromatic).
    4. For each shadow pixel, compute a per-channel gain that lifts its
       brightness to the surrounding non-shadow local mean, then blend it
       back with ``blend_strength`` to avoid over-correction.

    Returns a **new** raster array of the same shape and dtype with balanced
    shadow pixels.  Non-shadow pixels are returned unchanged.
    """
    if raster_data.shape[0] < 3:
        return raster_data.copy()

    out = raster_data.astype(np.float32).copy()
    r, g, b = out[0], out[1], out[2]

    brightness = (r + g + b) / 3.0
    local_mean_bright = uniform_filter(brightness, size=local_window, mode='reflect')

    # Shadow candidate: darker than local context + low saturation
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = (maxc - minc) / (maxc + 1e-6)

    dark_mask = brightness < (darkness_ratio * local_mean_bright)
    low_sat_mask = sat < 0.25
    shadow_mask = dark_mask & low_sat_mask & (brightness > 5.0)  # ignore pure black

    # Remove tiny isolated noise blobs (vectorised - no Python loop)
    shadow_labeled, n_blobs = ndi.label(shadow_mask)
    if n_blobs > 0:
        blob_sizes = np.bincount(shadow_labeled.ravel())
        small_ids = np.where(blob_sizes < min_shadow_px)[0]
        if len(small_ids) > 0:
            shadow_mask[np.isin(shadow_labeled, small_ids)] = False

    n_shadow = int(np.sum(shadow_mask))
    if n_shadow == 0:
        return raster_data.copy()

    total_px = brightness.size
    pct = 100.0 * n_shadow / total_px
    print(f"    [shadow-balance] {n_shadow:,} shadow pixels ({pct:.1f}%) detected, balancing...")

    # Target brightness for each shadow pixel: local mean of non-shadow neighbours.
    # We compute local mean per band excluding shadow pixels via a simple trick:
    # replace shadow pixels with NaN-like 0 weighting.
    weight = np.where(shadow_mask, 0.0, 1.0).astype(np.float32)
    weight_sum = uniform_filter(weight, size=local_window, mode='reflect')
    weight_sum = np.maximum(weight_sum, 1e-6)

    def _balance_band(band_idx):
        """Correct one band - called from threads (uniform_filter releases GIL)."""
        band = out[band_idx]
        masked_band = np.where(shadow_mask, 0.0, band)
        local_mean_band = uniform_filter(masked_band, size=local_window, mode='reflect') / weight_sum
        current = np.maximum(band, 1e-2)
        gain = local_mean_band / current
        gain = np.clip(gain, 0.5, 4.0)
        corrected = band * gain
        band[shadow_mask] = (
            blend_strength * corrected[shadow_mask]
            + (1.0 - blend_strength) * band[shadow_mask]
        )

    _n_bands = min(raster_data.shape[0], 3)
    with ThreadPoolExecutor(max_workers=_n_bands) as _tp:
        list(_tp.map(_balance_band, range(_n_bands)))

    # Clip to valid range
    dtype_max = float(np.iinfo(raster_data.dtype).max) if np.issubdtype(raster_data.dtype, np.integer) else 1.0
    np.clip(out, 0.0, dtype_max, out=out)

    return out.astype(raster_data.dtype)


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


def _detect_shadows_and_infer(
    classification_raster: np.ndarray,
    original_raster: np.ndarray,
    classes: Optional[List[Dict[str, str]]] = None,
    dilation_radius: int = 5,
    brightness_threshold: int = 100,
) -> np.ndarray:
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
    
    # Detect structure pixels (buildings/trees and other high-object proxies)
    structure_mask = _detect_structures_mask(original_raster)
    
    # Parallel: dilation and brightness/local_mean are independent
    rgb_data = original_raster[:3] if original_raster.shape[0] >= 3 else original_raster
    _brightness = [None]
    _local_mean = [None]
    _dilated = [None]

    def _compute_dilation():
        _dilated[0] = binary_dilation(structure_mask, iterations=dilation_radius)

    def _compute_brightness():
        b = np.mean(rgb_data.astype(np.float32), axis=0)
        _brightness[0] = b
        _local_mean[0] = uniform_filter(b, size=11, mode='reflect')

    with ThreadPoolExecutor(max_workers=2) as _tp:
        list(_tp.map(lambda f: f(), [_compute_dilation, _compute_brightness]))

    dilated_structures = _dilated[0]
    brightness = _brightness[0]
    local_mean = _local_mean[0]
    relative_shadow = brightness < np.minimum(brightness_threshold, local_mean * 0.72)

    # Low-saturation dark areas are more likely shadows than materials
    if rgb_data.shape[0] >= 3:
        r = rgb_data[0].astype(np.float32)
        g = rgb_data[1].astype(np.float32)
        b = rgb_data[2].astype(np.float32)
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        sat = (maxc - minc) / (maxc + 1e-6)
        low_sat = sat < 0.20
    else:
        low_sat = np.ones_like(brightness, dtype=bool)
    
    # Shadow pixels: dark relative to local context, low saturation, and close to structures
    shadow_candidates = relative_shadow & low_sat & dilated_structures & ~structure_mask

    # Avoid overriding explicit water class with shadow relabeling
    if classes:
        water_ids = {
            idx + 1 for idx, c in enumerate(classes)
            if c.get("name") == "BM_WATER"
        }
        if water_ids:
            shadow_candidates &= ~np.isin(classification_raster, list(water_ids))

    # ── Refine: exclude compact dark blobs that are likely windows or road features ──
    # Small compact blobs near building / road material are usually real features
    # (windows, cars, stripes) rather than cast shadows.  Excluding them here
    # preserves the original cluster label so that _cap_contextual_features can
    # handle them more accurately with full context.
    if classes and np.any(shadow_candidates):
        _bld_names = {"BM_CONCRETE", "BM_METAL", "BM_METAL_STEEL", "BM_ROCK"}
        _road_names = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
        _bld_ids = {idx + 1 for idx, c in enumerate(classes) if c.get("name", "") in _bld_names}
        _road_ids = {idx + 1 for idx, c in enumerate(classes) if c.get("name", "") in _road_names}
        _context_ids = _bld_ids | _road_ids
        if _context_ids:
            _ctx_mask = np.isin(classification_raster, list(_context_ids))
            _near_ctx = binary_dilation(_ctx_mask, iterations=3)
            sc_labeled, sc_n = ndi.label(shadow_candidates)
            if sc_n > 0:
                sc_sizes = np.bincount(sc_labeled.ravel())
                sc_slices = ndi.find_objects(sc_labeled)
                _struct = ndi.generate_binary_structure(2, 1)
                # Pre-filter: only look at blobs in [4..160] px range
                _eligible_sc = [
                    cid for cid in range(1, sc_n + 1)
                    if sc_slices[cid - 1] is not None
                    and 4 <= sc_sizes[cid] <= 160
                ]
                _context_ids_list = list(_context_ids)
                excluded = 0
                for comp_id in _eligible_sc:
                    sl = sc_slices[comp_id - 1]
                    area = int(sc_sizes[comp_id])
                    bb_h = sl[0].stop - sl[0].start
                    bb_w_d = sl[1].stop - sl[1].start
                    if bb_h == 0 or bb_w_d == 0:
                        continue
                    aspect = max(bb_h, bb_w_d) / max(min(bb_h, bb_w_d), 1)
                    if aspect > 5.0:
                        continue
                    compactness = area / max(bb_h * bb_w_d, 1)
                    if compactness < 0.20:
                        continue
                    comp_px = (sc_labeled == comp_id)
                    if float(np.mean(_near_ctx[comp_px])) < 0.75:
                        continue
                    r0 = max(0, sl[0].start - 2); r1 = min(height, sl[0].stop + 2)
                    c0 = max(0, sl[1].start - 2); c1 = min(width, sl[1].stop + 2)
                    loc_lbl = sc_labeled[r0:r1, c0:c1]
                    loc_cls = classification_raster[r0:r1, c0:c1]
                    cmask = (loc_lbl == comp_id)
                    ring = ndi.binary_dilation(cmask, structure=_struct, iterations=2) & (~cmask) & (loc_cls > 0)
                    if not np.any(ring):
                        continue
                    ctx_frac = float(np.mean(np.isin(loc_cls[ring], _context_ids_list)))
                    if ctx_frac >= 0.50:
                        shadow_candidates[comp_px] = False
                        excluded += 1
                if excluded:
                    print(f"    [shadow] Excluded {excluded} window/road-feature blobs from shadow candidates")
    
    if np.any(shadow_candidates):
        # Build boolean mask of veg/water pixels (avoid assigning shadow to these).
        veg_water_mask = np.zeros((height, width), dtype=bool)
        if classes:
            for cls_idx, cls_item in enumerate(classes):
                cname = cls_item.get("name", "")
                if any(kw in cname for kw in ("GRASS", "VEGETATION", "FOLIAGE", "WATER")):
                    veg_water_mask |= (classification_raster == (cls_idx + 1))

        # Solid-material source pixels: non-veg/water, labelled, not themselves shadow candidates.
        solid_source = (~veg_water_mask) & (classification_raster > 0) & (~shadow_candidates)

        # Vectorised nearest-neighbour: for every shadow pixel assign the nearest
        # solid-material pixel's class in O(H*W) via distance transform.
        if np.any(solid_source):
            dist_result = distance_transform_edt(~solid_source, return_indices=True)
            if isinstance(dist_result, tuple) and len(dist_result) == 2:
                _, nearest_indices = dist_result
                iy, ix = nearest_indices[0], nearest_indices[1]
                output_raster[shadow_candidates] = classification_raster[iy, ix][shadow_candidates]
        else:
            # Fallback: nearest any-class pixel.
            any_labelled = classification_raster > 0
            if np.any(any_labelled):
                dist_result = distance_transform_edt(~any_labelled, return_indices=True)
                if isinstance(dist_result, tuple) and len(dist_result) == 2:
                    _, nearest_indices = dist_result
                    iy, ix = nearest_indices[0], nearest_indices[1]
                    output_raster[shadow_candidates] = classification_raster[iy, ix][shadow_candidates]

    return output_raster.astype(classification_raster.dtype)


def _morphological_road_cleanup(
    classification_raster: np.ndarray,
    classes: List[Dict[str, str]],
    min_opening_radius: int = 2,
) -> np.ndarray:
    """Remove thin, isolated road-class pixels via morphological opening.

    Morphological opening = erosion followed by dilation.  Thin features
    (1-2 px wide spurious road classifications) are eroded away, then
    genuine compact road regions are restored by the dilation.  Pixels
    removed by the opening are relabelled to the nearest non-road class
    using a one-step dilation of the surrounding non-road mask.

    Only targets MEA road-class labels (BM_ASPHALT, BM_CONCRETE,
    BM_PAINT_ASPHALT).  Other classes are left unchanged.
    """
    from skimage.morphology import binary_opening, disk as morph_disk
    import scipy.ndimage as _ndi

    _ROAD_NAMES = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
    road_label_ids: list[int] = []
    for idx, cls in enumerate(classes):
        if cls.get("name") in _ROAD_NAMES:
            road_label_ids.append(idx + 1)  # labels are 1-indexed

    if not road_label_ids:
        return classification_raster

    result = classification_raster.copy()

    # Build combined binary mask of all road pixels
    road_mask = np.isin(classification_raster, road_label_ids)
    total_road = int(np.sum(road_mask))
    if total_road < 10:
        return result

    selem = morph_disk(min_opening_radius)
    opened = binary_opening(road_mask, selem)

    # Pixels that were road but removed by opening
    removed = road_mask & ~opened
    n_removed = int(np.sum(removed))
    if n_removed == 0:
        return result

    # Replace removed pixels with the nearest non-road class via dilation
    non_road = ~road_mask
    # Distance transform from non-road gives us the nearest non-road pixel
    _, nearest_idx = _ndi.distance_transform_edt(road_mask, return_distances=True, return_indices=True)
    # For each removed pixel, pick the label of the nearest non-road pixel
    result[removed] = classification_raster[nearest_idx[0][removed], nearest_idx[1][removed]]

    print(f"    [morph] Removed {n_removed} thin road pixels ({100.0*n_removed/max(1,total_road):.1f}% of road)")

    return result


def _absorb_isolated_small_patches(
    classification_raster: np.ndarray,
    classes: Optional[List[Dict[str, str]]] = None,
    max_patch_pixels: int = 80,
) -> np.ndarray:
    """
    Absorb small isolated classification blobs into their dominant surrounding class.

    Uses a two-phase vectorised approach for speed:

    **Phase A - distance-transform bulk replace** (handles the vast majority of
    small components in one shot, no Python loop):
      1. Label all connected components.
      2. Build a mask of *small, non-protected* components.
      3. Use ``distance_transform_edt`` with ``return_indices`` to find the
         nearest *non-small* pixel for every small pixel -> assign its class.

    **Phase B - local ring refinement** (optional, cheap second pass):
      For any remaining small component whose class changed, verify via a
      local border ring that the replacement class makes contextual sense.
      This pass is only needed for edge cases where the nearest large body
      is not the surrounding ring majority; it runs on very few components.

    Vegetation/water classes are PROTECTED and never absorbed.
    """
    # Build set of class IDs that should never be absorbed.
    protected_ids: set = set()
    if classes:
        for cls_idx, cls_item in enumerate(classes):
            cname = cls_item.get("name", "")
            if any(kw in cname for kw in ("GRASS", "VEGETATION", "FOLIAGE", "WATER")):
                protected_ids.add(cls_idx + 1)

    output = classification_raster.copy()
    labeled, n_components = ndi.label(classification_raster > 0)
    if n_components == 0:
        return output

    component_sizes = np.bincount(labeled.ravel())

    # ── Phase A: bulk nearest-neighbour via distance transform ──
    # Identify small, non-protected component IDs.
    small_ids = set()
    for comp_id in range(1, n_components + 1):
        if component_sizes[comp_id] > max_patch_pixels:
            continue
        small_ids.add(comp_id)

    if not small_ids:
        return output

    # For each small component, get the class from a representative pixel and
    # skip protected ones.
    slices = ndi.find_objects(labeled)
    _eligible_ids = set()
    for comp_id in small_ids:
        sl = slices[comp_id - 1]
        if sl is None:
            continue
        local_lbl = labeled[sl]
        local_cls = output[sl]
        comp_vals = local_cls[local_lbl == comp_id]
        if len(comp_vals) == 0:
            continue
        comp_class = int(np.bincount(comp_vals).argmax())
        if comp_class in protected_ids:
            continue
        _eligible_ids.add(comp_id)

    if not _eligible_ids:
        return output

    # Mask of small eligible pixels (to be replaced).
    small_mask = np.isin(labeled, list(_eligible_ids))
    # Source pixels = everything that is NOT a small-eligible component and IS labelled.
    source_mask = (~small_mask) & (classification_raster > 0)

    if not np.any(source_mask):
        return output

    # Distance transform yields nearest source pixel for every small pixel.
    _, nearest_idx = distance_transform_edt(~source_mask, return_indices=True)
    iy, ix = nearest_idx[0], nearest_idx[1]
    output[small_mask] = classification_raster[iy[small_mask], ix[small_mask]]

    return output


def _cap_contextual_features(
    classification_raster: np.ndarray,
    original_raster: np.ndarray,
    classes: Optional[List[Dict[str, str]]] = None,
    max_window_px: int = 150,
    max_road_feature_px: int = 250,
    max_facade_px: int = 600,
    min_feature_px: int = 4,
) -> np.ndarray:
    """Context-aware capping of small features on buildings and roads.

    Runs **after** shadow detection and patch absorption to correct remaining
    misclassifications with high confidence based on spatial context:

    1. **Windows** - small dark compact blobs inside / abutting a building
       material zone -> forced to the surrounding building-material class.
    2. **Road stripes** - bright elongated blobs inside road surfaces -> forced
       to the surrounding road-material class.
    3. **Road objects (cars)** - small distinct-colour blobs entirely inside a
       road zone -> forced to the surrounding road-material class.
    4. **Building facades** - medium-sized dark strips adjacent to bright
       building material -> forced to the adjacent building-material class.

    Vegetation / water classes are **never overwritten**.
    """
    h, w = classification_raster.shape
    output = classification_raster.copy()
    if classes is None:
        return output

    # --- class-ID sets ---
    building_names = {"BM_CONCRETE", "BM_METAL", "BM_METAL_STEEL", "BM_ROCK"}
    road_names = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
    protected_kw = ("GRASS", "VEGETATION", "FOLIAGE", "WATER")

    building_ids: set = set()
    road_ids: set = set()
    protected_ids: set = set()
    for idx, c in enumerate(classes):
        cname = c.get("name", "")
        if cname in building_names:
            building_ids.add(idx + 1)
        if cname in road_names:
            road_ids.add(idx + 1)
        if any(kw in cname for kw in protected_kw):
            protected_ids.add(idx + 1)

    if not building_ids and not road_ids:
        return output

    # --- image features ---
    rgb = original_raster[:3].astype(np.float32)
    brightness = np.mean(rgb, axis=0)
    local_mean = uniform_filter(brightness, size=15, mode="reflect")
    struct = ndi.generate_binary_structure(2, 1)

    building_mask = np.isin(output, list(building_ids)) if building_ids else np.zeros((h, w), dtype=bool)
    road_mask = np.isin(output, list(road_ids)) if road_ids else np.zeros((h, w), dtype=bool)
    prot_mask = np.isin(output, list(protected_ids)) if protected_ids else np.zeros((h, w), dtype=bool)

    # Pre-compute building dilation with max iterations (reused for windows + facades).
    _near_building_4 = binary_dilation(building_mask, iterations=4) if np.any(building_mask) else None

    capped_window = 0
    capped_road = 0
    capped_facade = 0

    # =================================================================
    # 1. WINDOWS - dark compact blobs near building material
    # =================================================================
    if _near_building_4 is not None:
        # iterations=3 is a subset of iterations=4; using the larger buffer is
        # functionally equivalent (at most 1-pixel wider) and avoids a second dilation.
        near_building = _near_building_4
        dark_near_bld = (
            (brightness < local_mean * 0.68)
            & near_building
            & (~building_mask)
            & (~prot_mask)
        )
        lbl_w, n_w = ndi.label(dark_near_bld)
        if n_w > 0:
            sizes_w = np.bincount(lbl_w.ravel())
            slices_w = ndi.find_objects(lbl_w)
            # Pre-filter: skip components outside size range in one pass
            _eligible_w = [
                cid for cid in range(1, n_w + 1)
                if slices_w[cid - 1] is not None
                and min_feature_px <= sizes_w[cid] <= max_window_px
            ]
            _bld_ids_list = list(building_ids)
            for cid in _eligible_w:
                sl = slices_w[cid - 1]
                area = int(sizes_w[cid])
                bb_h = sl[0].stop - sl[0].start
                bb_w_dim = sl[1].stop - sl[1].start
                if bb_h == 0 or bb_w_dim == 0:
                    continue
                aspect = max(bb_h, bb_w_dim) / max(min(bb_h, bb_w_dim), 1)
                if aspect > 4.5:
                    continue
                compactness = area / max(bb_h * bb_w_dim, 1)
                if compactness < 0.25:
                    continue

                r0 = max(0, sl[0].start - 2); r1 = min(h, sl[0].stop + 2)
                c0 = max(0, sl[1].start - 2); c1 = min(w, sl[1].stop + 2)
                local_lbl = lbl_w[r0:r1, c0:c1]
                local_cls = output[r0:r1, c0:c1]
                cmask = (local_lbl == cid)
                ring = ndi.binary_dilation(cmask, structure=struct, iterations=2) & (~cmask) & (local_cls > 0)
                if not np.any(ring):
                    continue
                ring_vals = local_cls[ring]
                bld_frac = float(np.mean(np.isin(ring_vals, _bld_ids_list)))
                if bld_frac < 0.50:
                    continue

                bld_ring = ring_vals[np.isin(ring_vals, _bld_ids_list)]
                if len(bld_ring) == 0:
                    continue
                dom_cls = int(np.bincount(bld_ring).argmax())
                if dom_cls == 0:
                    continue
                output[lbl_w == cid] = dom_cls
                capped_window += 1

    # =================================================================
    # 2. ROAD FEATURES - stripes (bright elongated) + cars (small blobs)
    # =================================================================
    if np.any(road_mask):
        near_road = binary_dilation(road_mask, iterations=2)
        road_mean_bright = float(np.mean(brightness[road_mask]))

        # Bright features (white / yellow paint stripes)
        bright_on_road = (brightness > road_mean_bright * 1.30) & near_road
        # Contrasting dark features (cars, tar joints)
        contrast_on_road = (
            (np.abs(brightness - local_mean) > local_mean * 0.22)
            & near_road
            & (~road_mask)
        )
        road_candidate = (bright_on_road | contrast_on_road) & (~prot_mask) & (~building_mask)

        lbl_r, n_r = ndi.label(road_candidate)
        if n_r > 0:
            sizes_r = np.bincount(lbl_r.ravel())
            slices_r = ndi.find_objects(lbl_r)
            _eligible_r = [
                cid for cid in range(1, n_r + 1)
                if slices_r[cid - 1] is not None
                and min_feature_px <= sizes_r[cid] <= max_road_feature_px
            ]
            _road_ids_list = list(road_ids)
            for cid in _eligible_r:
                sl = slices_r[cid - 1]

                r0 = max(0, sl[0].start - 2); r1 = min(h, sl[0].stop + 2)
                c0 = max(0, sl[1].start - 2); c1 = min(w, sl[1].stop + 2)
                local_lbl = lbl_r[r0:r1, c0:c1]
                local_cls = output[r0:r1, c0:c1]
                cmask = (local_lbl == cid)
                ring = ndi.binary_dilation(cmask, structure=struct, iterations=2) & (~cmask) & (local_cls > 0)
                if not np.any(ring):
                    continue
                ring_vals = local_cls[ring]
                rd_frac = float(np.mean(np.isin(ring_vals, _road_ids_list)))
                if rd_frac < 0.55:
                    continue

                rd_ring = ring_vals[np.isin(ring_vals, _road_ids_list)]
                if len(rd_ring) == 0:
                    continue
                dom_cls = int(np.bincount(rd_ring).argmax())
                if dom_cls == 0:
                    continue
                output[lbl_r == cid] = dom_cls
                capped_road += 1

    # =================================================================
    # 3. BUILDING FACADES - medium dark strips adjacent to buildings
    # =================================================================
    if _near_building_4 is not None:
        near_building2 = _near_building_4
        facade_candidate = (
            (brightness < local_mean * 0.75)
            & near_building2
            & (~building_mask)
            & (~road_mask)
            & (~prot_mask)
        )
        lbl_f, n_f = ndi.label(facade_candidate)
        if n_f > 0:
            sizes_f = np.bincount(lbl_f.ravel())
            slices_f = ndi.find_objects(lbl_f)
            _eligible_f = [
                cid for cid in range(1, n_f + 1)
                if slices_f[cid - 1] is not None
                and max_window_px <= sizes_f[cid] <= max_facade_px
            ]
            _bld_ids_list2 = list(building_ids)
            for cid in _eligible_f:
                sl = slices_f[cid - 1]
                area = int(sizes_f[cid])
                bb_h = sl[0].stop - sl[0].start
                bb_w_dim = sl[1].stop - sl[1].start
                if bb_h == 0 or bb_w_dim == 0:
                    continue
                aspect = max(bb_h, bb_w_dim) / max(min(bb_h, bb_w_dim), 1)
                if aspect < 1.8:
                    continue

                r0 = max(0, sl[0].start - 3); r1 = min(h, sl[0].stop + 3)
                c0 = max(0, sl[1].start - 3); c1 = min(w, sl[1].stop + 3)
                local_lbl = lbl_f[r0:r1, c0:c1]
                local_cls = output[r0:r1, c0:c1]
                cmask = (local_lbl == cid)
                ring = ndi.binary_dilation(cmask, structure=struct, iterations=3) & (~cmask) & (local_cls > 0)
                if not np.any(ring):
                    continue
                ring_vals = local_cls[ring]
                bld_frac = float(np.mean(np.isin(ring_vals, _bld_ids_list2)))
                if bld_frac < 0.45:
                    continue

                bld_ring = ring_vals[np.isin(ring_vals, _bld_ids_list2)]
                if len(bld_ring) == 0:
                    continue
                dom_cls = int(np.bincount(bld_ring).argmax())
                if dom_cls == 0:
                    continue
                output[lbl_f == cid] = dom_cls
                capped_facade += 1

    print(f"    windows={capped_window}  road_features={capped_road}  facades={capped_facade}")
    return output


def _cap_enclosed_objects_to_asphalt(
    classification_raster: np.ndarray,
    classes: Optional[List[Dict[str, str]]] = None,
    pixel_size_m: float = 0.5,
    ring_iterations: int = 3,
    min_asphalt_ring_frac: float = 0.70,
) -> np.ndarray:
    """Force small enclosed objects (e.g., lane paint/cars) into asphalt.

    Thresholds are computed from ``pixel_size_m`` so that the same real-world
    sizes are targeted regardless of image resolution:

    * **Car** ~ 2 m x 4.5 m -> area ~ 9 m²
    * **Lane stripe** ~ 0.15 m x 3 m -> area ~ 0.45 m²
    * **Max target** ~ 18 m² (large SUV / van)
    * **Infill gap** ~ 1.5 m² tiny mis-classified patches

    **Pass 1 - component ring analysis**
    Searches for small non-protected components enclosed by an asphalt ring.
    The ring may come from up to 2 connected asphalt bodies (e.g., across a
    stripe) to handle paint lines on roads.

    **Pass 2 - morphological closing**
    A second sweep closes small 1–2 px gaps inside large asphalt bodies that
    were classified as a different material (concrete, paint-asphalt, etc.).
    """
    output = classification_raster.copy()
    if classes is None:
        return output

    # ── Derive pixel-count limits from real-world sizes ──
    px_area = max(pixel_size_m, 0.01) ** 2          # m² per pixel
    min_object_px  = max(2, int(0.30 / px_area))    # ~0.30 m² (tiny stripe segment)
    max_object_px  = max(20, int(18.0 / px_area))   # ~18 m² (large SUV / van)
    max_infill_px  = max(4, int(1.5 / px_area))     # ~1.5 m² (small gap)
    print(f"    [asphalt-cap] pixel_size={pixel_size_m:.3f}m  "
          f"min_obj={min_object_px}px  max_obj={max_object_px}px  max_infill={max_infill_px}px")

    asphalt_id: Optional[int] = None
    road_ids: set = set()
    protected_ids: set = set()
    for idx, cls_item in enumerate(classes):
        cname = cls_item.get("name", "")
        cid = idx + 1
        if cname == "BM_ASPHALT":
            asphalt_id = cid
        if cname in ("BM_ASPHALT", "BM_PAINT_ASPHALT", "BM_CONCRETE"):
            road_ids.add(cid)
        if any(kw in cname for kw in ("GRASS", "VEGETATION", "FOLIAGE", "WATER")):
            protected_ids.add(cid)

    if asphalt_id is None:
        return output

    h, w = output.shape
    struct = ndi.generate_binary_structure(2, 1)

    # ── Pass 1: component-level ring enclosure (parallelised across components) ──
    candidate_mask = (output > 0) & (~np.isin(output, list(road_ids | protected_ids)))
    labeled, n_components = ndi.label(candidate_mask)
    if n_components == 0:
        pass  # skip to pass 2
    else:
        component_sizes = np.bincount(labeled.ravel())
        slices = ndi.find_objects(labeled)
        # Pre-filter: only loop over components in the valid size range
        _eligible = [
            cid for cid in range(1, n_components + 1)
            if slices[cid - 1] is not None
            and min_object_px <= component_sizes[cid] <= max_object_px
        ]
        _road_ids_list = list(road_ids)

        # Each component is independent - check it in a thread.
        # Returns comp_id if it should be reassigned to asphalt, else 0.
        def _check_comp(comp_id: int) -> int:
            sl = slices[comp_id - 1]
            pad = ring_iterations + 1
            r0 = max(0, sl[0].start - pad)
            r1 = min(h, sl[0].stop + pad)
            c0 = max(0, sl[1].start - pad)
            c1 = min(w, sl[1].stop + pad)

            local_lbl = labeled[r0:r1, c0:c1]
            local_cls = output[r0:r1, c0:c1]
            comp_mask = (local_lbl == comp_id)

            ring = (
                ndi.binary_dilation(comp_mask, structure=struct, iterations=ring_iterations)
                & (~comp_mask)
                & (local_cls > 0)
            )
            if not np.any(ring):
                return 0

            ring_vals = local_cls[ring]
            asphalt_frac = float(np.mean(ring_vals == asphalt_id))
            road_frac = float(np.mean(np.isin(ring_vals, _road_ids_list)))
            if asphalt_frac < min_asphalt_ring_frac and road_frac < 0.85:
                return 0

            # Continuity gate
            local_asphalt = (local_cls == asphalt_id)
            asphalt_lbl, _ = ndi.label(local_asphalt, structure=struct)
            touching_ids = np.unique(asphalt_lbl[ring & local_asphalt])
            touching_ids = touching_ids[touching_ids > 0]
            if len(touching_ids) < 1 or len(touching_ids) > 2:
                return 0

            return comp_id

        _n_workers = min(max(1, os.cpu_count() or 1), 8)
        if _n_workers > 1 and len(_eligible) >= _n_workers * 2:
            with ThreadPoolExecutor(max_workers=_n_workers) as _ex:
                results = list(_ex.map(_check_comp, _eligible))
        else:
            results = [_check_comp(cid) for cid in _eligible]

        reassign_ids = [r for r in results if r > 0]
        if reassign_ids:
            output[np.isin(labeled, reassign_ids)] = asphalt_id
            print(f"    asphalt_enclosed_pass1={len(reassign_ids)}")

    # ── Pass 2: morphological infill of tiny gaps (vectorised) ──
    asphalt_mask = (output == asphalt_id)
    closed = ndi.binary_closing(asphalt_mask, structure=struct, iterations=2)
    fill_mask = closed & (~asphalt_mask) & (~np.isin(output, list(protected_ids)))
    fill_labeled, n_fill = ndi.label(fill_mask)
    infilled = 0
    if n_fill > 0:
        fill_sizes = np.bincount(fill_labeled.ravel())
        # Vectorised: find all small-gap IDs and fill them at once
        small_fill_ids = np.where((fill_sizes > 0) & (fill_sizes <= max_infill_px))[0]
        small_fill_ids = small_fill_ids[small_fill_ids > 0]  # exclude background
        if len(small_fill_ids) > 0:
            output[np.isin(fill_labeled, small_fill_ids)] = asphalt_id
            infilled = len(small_fill_ids)
    if infilled > 0:
        print(f"    asphalt_infill_pass2={infilled}")

    return output


def apply_road_object_removal(
    input_path: str,
    output_path: Optional[str] = None,
    pixel_size_m: Optional[float] = None,
    progress_callback=None,
) -> str:
    """Post-processing: reclassify small enclosed objects on asphalt roads.

    Loads a classified MEA RGB GeoTIFF produced by ``classify_and_export``,
    converts it back to a label raster using the known MEA palette, runs
    ``_cap_enclosed_objects_to_asphalt``, and saves the result.

    Parameters
    ----------
    input_path:
        Path to a classified RGB GeoTIFF (output of Step 1).
    output_path:
        Destination path.  If *None* the input file is overwritten in-place.
    pixel_size_m:
        Ground-sample distance in metres.  Derived from the raster's affine
        transform when not provided.
    progress_callback:
        Optional callable ``(phase: str, done: int, total: int)`` used to
        report progress to the GUI.

    Returns
    -------
    str
        Absolute path of the saved output file.
    """
    def _cb(phase: str, done: int, total: int) -> None:
        if progress_callback is not None:
            try:
                progress_callback(phase, done, total)
            except Exception:
                pass

    _cb("Loading classified raster", 0, 1)
    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        rgb_data = src.read()          # (3, H, W) uint8

    if pixel_size_m is None:
        pixel_size_m = float(max(abs(getattr(transform, 'a', 0.5)),
                                  abs(getattr(transform, 'e', 0.5))))
    _cb("Loading classified raster", 1, 1)

    # Build packed-key -> class_id lookup from MEA palette (exact pixel values).
    classes = MEA_CLASSES
    packed_to_id: Dict[int, int] = {}
    color_table: List[Tuple[int, int, int]] = []
    for idx, cls_item in enumerate(classes):
        hex_color = cls_item.get("color", "#000000").lstrip("#")
        cr = int(hex_color[0:2], 16)
        cg = int(hex_color[2:4], 16)
        cb = int(hex_color[4:6], 16)
        packed_to_id[cr * 65536 + cg * 256 + cb] = idx + 1
        color_table.append((cr, cg, cb))

    _cb("Converting to label raster", 0, 1)
    h, w = rgb_data.shape[1], rgb_data.shape[2]
    packed = (rgb_data[0].astype(np.int32) * 65536
              + rgb_data[1].astype(np.int32) * 256
              + rgb_data[2].astype(np.int32))           # (H, W) int32

    label_raster = np.zeros((h, w), dtype=np.int32)
    for packed_key, class_id in packed_to_id.items():
        label_raster[packed == packed_key] = class_id
    _cb("Converting to label raster", 1, 1)

    # Run enclosure removal on the label raster.
    _cb("Removing road objects", 0, 1)
    print(f"\n[road-objects] Running enclosed-object removal on {input_path} ...")
    result_raster = _cap_enclosed_objects_to_asphalt(
        label_raster, classes=classes, pixel_size_m=pixel_size_m
    )
    changed = int(np.sum(result_raster != label_raster))
    print(f"  [OK] {changed} pixels reclassified to asphalt")
    _cb("Removing road objects", 1, 1)

    # Convert back to RGB and write output.
    _cb("Saving result", 0, 1)
    rgb_out = _apply_color_table(result_raster, color_table, verbose=False)

    out_path = Path(output_path) if output_path else Path(input_path)
    driver = _driver_for_path(str(out_path))
    write_profile = profile.copy()
    write_profile = _profile_for_driver(write_profile, driver)
    write_profile.update(count=3, dtype="uint8", interleave="band")
    if str(driver).upper() == "GTIFF":
        write_profile.update(
            tiled=True, blockxsize=256, blockysize=256,
            compress="deflate", predictor=2,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    with rasterio.open(out_path, "w", **write_profile) as dst:
        dst.write(rgb_out)

    print(f"  [OK] Saved to {out_path}")
    _cb("Saving result", 1, 1)
    return str(out_path)


def _classify_tile_worker(args: tuple) -> str:
    """Classify a single tile using a pre-trained global model.

    *args* is a tuple whose last element may optionally be a ``dict`` with
    extra keyword options (``smooth_pad``, ``raster_h``, ``raster_w``,
    ``classes``, ``detect_shadows``).  This keeps backward compatibility
    with callers that pass the original 10-element tuple.
    """
    # Unpack; trailing extra dict is optional.
    (raster_path, window_tuple, feature_flags,
     scaler_mean, scaler_scale, centers,
     color_table, smoothing, output_dir, tile_name, *_extra) = args
    extra: Dict[str, object] = _extra[0] if _extra else {}

    row, col, height, width = window_tuple

    # ------------------------------------------------------------------
    # Expand the read window by smooth_pad pixels on every side so the
    # median filter has full neighbourhood context at tile boundaries
    # (-> no seam / discontinuity between adjacent tiles).
    # ------------------------------------------------------------------
    smooth_pad: int = int(extra.get("smooth_pad", 0))
    raster_h: int   = int(extra.get("raster_h", 0))
    raster_w: int   = int(extra.get("raster_w", 0))

    if smooth_pad > 0 and raster_h > 0 and raster_w > 0:
        row_exp = max(0, row - smooth_pad)
        col_exp = max(0, col - smooth_pad)
        h_exp   = min(raster_h - row_exp, height + (row - row_exp) + smooth_pad)
        w_exp   = min(raster_w - col_exp, width  + (col - col_exp) + smooth_pad)
        off_row = row - row_exp
        off_col = col - col_exp
    else:
        row_exp, col_exp, h_exp, w_exp = row, col, height, width
        off_row, off_col = 0, 0

    exp_window  = Window(col_exp, row_exp, w_exp, h_exp)
    orig_window = Window(col,     row,     width, height)   # for georeferencing

    with rasterio.open(raster_path) as src:
        tile_data      = src.read(window=exp_window)
        profile        = src.profile.copy()
        tile_transform = window_transform(orig_window, src.transform)
        tile_crs       = _normalize_pseudo_mercator_crs(src.crs)

    # Profile always reflects the *original* (non-padded) tile dimensions.
    profile.update(
        height=height,
        width=width,
        transform=tile_transform,
        crs=tile_crs,
        count=3,
        dtype=np.uint8,
        interleave='band',
    )

    # Feature extraction + nearest-center assignment on the expanded area.
    try:
        features = _extract_pixel_features(tile_data, feature_flags, verbose=False)
        scale    = np.where(scaler_scale == 0, 1.0, scaler_scale).astype(np.float32)
        features_norm = ((features - scaler_mean) / scale).astype(np.float32)
        del features                        # free ~20×H×W×4 bytes immediately
        gc.collect()
        labels = _nearest_center_chunked(features_norm, centers.astype(np.float32)) + 1
        del features_norm
        gc.collect()
    except MemoryError:
        bands, h, w = tile_data.shape
        mb = bands * h * w / (1024 * 1024)
        raise MemoryError(
            f"Out of memory processing tile {tile_name} ({h}×{w}, {bands} bands, ~{mb:.0f} MB). "
            f"Reduce tile size in Performance settings."
        )
    predicted_raster   = labels.reshape(h_exp, w_exp)

    # Smoothing on the expanded area -> boundary pixels get full context.
    if smoothing and smoothing != "none":
        try:
            kernel_size    = int(smoothing.split("_")[1]) if "_" in smoothing else 2
            predicted_raster = median(predicted_raster.astype(np.uint16), disk(kernel_size))
        except Exception:
            pass

    # Crop back to original tile dimensions.
    predicted_raster = predicted_raster[off_row:off_row + height, off_col:off_col + width]
    tile_data_crop   = tile_data[:, off_row:off_row + height, off_col:off_col + width]
    del tile_data                           # full (padded) tile no longer needed
    gc.collect()

    rgb = _apply_color_table(predicted_raster, color_table, verbose=False)

    output_path_obj = Path(output_dir) / tile_name
    driver = _driver_for_path(str(output_path_obj))
    if driver == "GTiff":
        write_profile = _output_tiff_profile(profile, dtype="uint8")
        write_profile["count"] = 3
    else:
        write_profile = _profile_for_driver(profile, driver)
        write_profile.update(count=3, dtype="uint8")
    with rasterio.open(output_path_obj, 'w', **write_profile) as dst:
        dst.write(rgb)
    del rgb, tile_data_crop, predicted_raster
    gc.collect()

    return str(output_path_obj)


def _rasterize_tile_worker(args: Tuple[str, Optional[Tuple[int, int, int, int]], List[Tuple[List, int, Tuple[int, int, int]]], str, str]) -> str:
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

    for geoms, burn_value, overlay_color in layer_geoms:
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
            r, g, b = overlay_color
            output_array[0][burned_mask > 0] = r
            output_array[1][burned_mask > 0] = g
            output_array[2][burned_mask > 0] = b

    output_path = Path(output_dir) / tile_name
    driver = _driver_for_path(str(output_path))
    if driver == "GTiff":
        write_meta = _output_tiff_profile(meta)
    else:
        write_meta = _profile_for_driver(meta, driver)
    with rasterio.open(output_path, 'w', **write_meta) as dst:
        dst.write(output_array)

    return str(output_path)


def rasterize_vector_onto_raster(raster_path: str, gdf, burn_value: int, output_path: str, crs,
                                  overlay_color: Tuple[int, int, int] = VECTOR_OVERLAY_COLOR):
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
    
    # GeoDataFrame is already loaded - always reproject to the RASTER CRS
    print(f"    Vector features: {len(gdf)}")
    print(f"    Vector CRS: {gdf.crs}")
    print(f"    Raster CRS: {raster_crs}")

    # Keep the original gdf (before any reprojection) so we can fall back to it
    # if the reprojected coordinates end up outside the raster extent.
    _gdf_original = gdf.copy()

    # Use the raster's own CRS (authoritative) as the target
    _target_crs = raster_crs

    if _target_crs is None:
        print(f"    [WARN] Raster has no CRS; skipping reprojection")
    elif gdf.crs is None:
        print(f"    Vector has no CRS; assigning raster CRS directly")
        gdf = gdf.set_crs(_target_crs, allow_override=True)
    else:
        # Always reproject unconditionally - eliminates false-negative CRS.equals() mismatches
        _orig_bounds = gdf.total_bounds
        print(f"    Reprojecting vector -> raster CRS")
        print(f"      Vector CRS: {gdf.crs}")
        print(f"      Raster CRS: {_target_crs}")
        try:
            gdf = gdf.to_crs(_target_crs)
            print(f"      [OK] Reprojected. Bounds: {_orig_bounds} -> {gdf.total_bounds}")
        except Exception as e:
            print(f"      [ERROR] Reprojection failed: {e}")
            print(f"      Attempting manual Web Mercator fallback...")
            try:
                gdf = _transform_geometries_to_web_mercator(gdf)
                print(f"      [OK] Manual transformation successful")
            except Exception as e2:
                raise RuntimeError(f"CRS transformation failed: {e2}")
    
    print(f"    Vector CRS after processing: {gdf.crs}")
    
    # Create geometry-value pairs
    shapes = [(geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    print(f"    Valid geometries: {len(shapes)}")
    
    if not shapes:
        print(f"    WARNING: No valid geometries to rasterize")
        # Just copy the raster
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        _fallback_driver = _driver_for_path(output_path)
        if _fallback_driver == "GTiff":
            _fallback_meta = _output_tiff_profile(meta)
        else:
            _fallback_meta = _profile_for_driver(meta, _fallback_driver)
        with rasterio.open(output_path, 'w', **_fallback_meta) as dst:
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
        
        # FALLBACK: if bounds don't overlap, try using the original (un-reprojected)
        # geometry coords directly. This handles a common case where the vector was
        # already drawn in the raster's coordinate space but carries a different CRS
        # label (e.g. QGIS project CRS differs from layer CRS).
        if not bounds_overlap:
            print(f"      [FALLBACK] Trying original (un-reprojected) geometry coordinates...")
            _orig_shapes = [(geom, burn_value) for geom in _gdf_original.geometry
                            if geom is not None and not geom.is_empty]
            if _orig_shapes:
                _orig_gdf_bounds = _gdf_original.total_bounds
                _x_ok = not (_orig_gdf_bounds[2] < raster_minx or _orig_gdf_bounds[0] > raster_maxx)
                _y_ok = not (_orig_gdf_bounds[3] < raster_miny or _orig_gdf_bounds[1] > raster_maxy)
                if _x_ok and _y_ok:
                    print(f"      [FALLBACK] Original bounds DO overlap — using un-reprojected coords")
                    burned_mask = rasterize(
                        shapes=_orig_shapes,
                        out_shape=(height, width),
                        transform=transform,
                        fill=0,
                        all_touched=True
                    )
                    pixels_burned = int(np.sum(burned_mask > 0))
                    print(f"      [FALLBACK] Pixels burned with original coords: {pixels_burned}")
                else:
                    print(f"      [FALLBACK] Original bounds also outside raster — vector does not "
                          f"spatially overlap this raster. Check that the shapefile covers the "
                          f"same area as the image.")

    # Merge with raster - write burned pixels with burn_value
    print(f"    Original raster dtype: {raster_array.dtype}, shape: {raster_array.shape}")
    print(f"    Burned mask stats: min={np.min(burned_mask)}, max={np.max(burned_mask)}, sum={np.sum(burned_mask > 0)}")
    print(f"    Burn value: {burn_value} (type: {type(burn_value).__name__})")
    
    print(f"    [COLOR DEBUG] overlay_color={overlay_color!r}, ndim={raster_array.ndim}")
    if raster_array.ndim == 2:
        # Single band
        output_array = raster_array.copy()
        output_array[burned_mask > 0] = burn_value
    else:
        # Multi-band - paint overlay color on all bands
        output_array = raster_array.copy()
        r, g, b = overlay_color
        print(f"    [COLOR DEBUG] Writing RGB=({r},{g},{b}) to {np.sum(burned_mask > 0)} pixels")
        output_array[0][burned_mask > 0] = r
        output_array[1][burned_mask > 0] = g
        output_array[2][burned_mask > 0] = b
    
    print(f"    Output array dtype: {output_array.dtype}, shape: {output_array.shape}")
    print(f"    Output array range: min={np.min(output_array)}, max={np.max(output_array)}")
    unique_values = np.unique(output_array)
    print(f"    Unique values in output: {unique_values[:20]}..." if len(unique_values) > 20 else f"    Unique values in output: {unique_values}")
    
    # Save result
    print(f"    Saving to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    driver = _driver_for_path(output_path)
    if driver == "GTiff":
        _meta = _output_tiff_profile(meta)
    else:
        _meta = _profile_for_driver(meta, driver)
    with rasterio.open(output_path, 'w', **_meta) as dst:
        dst.write(output_array)

    print(f"    [OK] Vector rasterized successfully")


# ---------------------------------------------------------------------------
# Pipeline statistics helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 0.01:
        return "<0.01s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.1f}s"


def _build_stats_table(stages: List[Tuple[str, float]], total: float) -> str:
    """Build a printable table (also suitable for GUI display)."""
    # Compute longest label for alignment
    max_label = max((len(s) for s, _ in stages), default=10)
    lines = ["\n" + "─" * (max_label + 30)]
    lines.append(f"{'Stage':<{max_label}}  {'Time':>10}  {'%':>6}")
    lines.append("─" * (max_label + 30))
    for label, dur in stages:
        pct = (dur / total * 100) if total > 0 else 0
        lines.append(f"{label:<{max_label}}  {_fmt_duration(dur):>10}  {pct:5.1f}%")
    lines.append("─" * (max_label + 30))
    lines.append(f"{'TOTAL':<{max_label}}  {_fmt_duration(total):>10}  100.0%")
    lines.append("─" * (max_label + 30))
    return "\n".join(lines)


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
    max_threads: Optional[int] = None,
    pretrained_scaler=None,
    pretrained_kmeans=None,
    pretrained_color_table=None,
    pretrained_mea_mapping=None,
    export_format: str = "tif",
    progress_callback=None,
) -> Dict[str, object]:
    """
    Complete classification pipeline: KMeans + Vector rasterization.
    This is a convenience wrapper that calls classify_and_export() then rasterize_vectors_onto_classification().
    """
    import tempfile
    _t_pipeline_start = _time.perf_counter()
    _pipeline_stages: List[Tuple[str, float]] = []
    
    # Normalise: treat empty strings the same as None.
    output_path = output_path.strip() if output_path else None
    
    print("\n" + "="*70)
    print("COMPLETE CLASSIFICATION PIPELINE")
    print(f"  output_path={output_path!r}")
    print("="*70)
    
    # === STEP 1: Classify and export ===
    print("\n>>> STEP 1: Classification & Export")
    
    # Always write Step 1 (classification) to the real output path so the
    # classified file is preserved.  When vectors are present, Step 2 output
    # goes to a separate adjacent folder ("with_vectors").
    step1_output = output_path
    if vector_layers:
        print(f"  (vectors present -> Step 1 to output_path, Step 2 to adjacent folder)")
    else:
        print(f"  (no vectors -> Step 1 writes to output_path: {step1_output!r})")
    
    _t0 = _time.perf_counter()
    # When tile_output_dir is not explicitly set, use the user's output_path
    # so tiles land in the directory the user chose (not next to the source).
    _effective_tile_dir = tile_output_dir or output_path
    result1 = classify_and_export(
        raster_path=raster_path,
        classes=classes,
        smoothing=smoothing,
        feature_flags=feature_flags,
        output_path=step1_output if not tile_mode else None,
        tile_mode=tile_mode,
        tile_max_pixels=tile_max_pixels,
        tile_overlap=tile_overlap,
        tile_output_dir=_effective_tile_dir,
        tile_workers=tile_workers,
        detect_shadows=detect_shadows,
        pretrained_scaler=pretrained_scaler,
        pretrained_kmeans=pretrained_kmeans,
        pretrained_color_table=pretrained_color_table,
        pretrained_mea_mapping=pretrained_mea_mapping,
        export_format=export_format,
        progress_callback=progress_callback,
    )
    _pipeline_stages.append(("Step 1: Classification & Export", _time.perf_counter() - _t0))
    
    if result1["status"] != "ok":
        return result1
    
    classif_file = result1["outputPath"]
    print(f"Classification saved to: {classif_file}")
    
    # Merge sub-stage stats from classify_and_export
    _step1_stats = result1.get("stats", [])
    
    # === STEP 2: Rasterize vectors ===
    _t0 = _time.perf_counter()
    result2 = None
    if vector_layers:
        print("\n>>> STEP 2: Vector Rasterization")
        # Write vectorized output to a separate adjacent folder so the
        # classified output from Step 1 is preserved.
        if tile_mode:
            # Tile mode: rasterize_vectors creates '_with_vectors_tiles' dir.
            _vectors_output = output_path
        else:
            _classif_path = Path(classif_file)
            _vec_dir = _classif_path.parent / "with_vectors"
            _vec_dir.mkdir(parents=True, exist_ok=True)
            _vectors_output = str(_vec_dir)
            print(f"  Vectorized output -> {_vec_dir}")

        result2 = rasterize_vectors_onto_classification(
            classification_path=classif_file,
            vector_layers=vector_layers,
            classes=classes,
            output_path=_vectors_output,
            raster_stem_hint=Path(raster_path).stem,
            tile_mode=tile_mode,
            tile_max_pixels=tile_max_pixels,
            tile_overlap=tile_overlap,
            tile_output_dir=_effective_tile_dir,
            tile_workers=tile_workers,
            max_threads=max_threads,
            progress_callback=progress_callback,
        )
        _pipeline_stages.append(("Step 2: Vector Rasterization", _time.perf_counter() - _t0))
        
        if result2["status"] != "ok":
            return result2
        
        final_output = result2["outputPath"]
    else:
        # No vectors -> Step 1 already wrote directly to the user's output_path
        print("\n>>> STEP 2: No vectors provided, skipping")
        _pipeline_stages.append(("Step 2: Vectors (skipped)", _time.perf_counter() - _t0))
        final_output = classif_file
    
    _total = _time.perf_counter() - _t_pipeline_start
    
    # Build combined stats: sub-stages from step1 + step2 + top-level
    combined_stats = list(_step1_stats)
    _step2_stats = (result2 or {}).get("stats", [])
    if _step2_stats:
        combined_stats.extend(_step2_stats)
    combined_stats.append(("PIPELINE TOTAL", _total))
    
    # Print summary
    summary_table = _build_stats_table(combined_stats[:-1], _total)
    print("\n" + "="*70)
    print("[OK] COMPLETE CLASSIFICATION PIPELINE FINISHED")
    print("="*70)
    print(summary_table)
    print(f"\nFinal output: {final_output}")
    print("="*70 + "\n")
    
    return {
        "status": "ok",
        "outputPath": str(final_output),
        "classifiedPath": str(classif_file),
        "tileOutputs": (result2 or result1).get("tileOutputs"),
        "meaMapping": result1.get("meaMapping"),
        "stats": combined_stats,
        "statsTable": summary_table,
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
    max_threads: Optional[int] = None,
    pretrained_scaler=None,
    pretrained_kmeans=None,
    pretrained_color_table=None,   # List[Tuple[int,int,int]] - skip per-image MEA mapping
    pretrained_mea_mapping=None,   # List[Dict]               - returned as-is in result
    progress_callback=None,        # callable(phase: str, done: int, total: int) | None
    export_format: str = "tif",
) -> Dict[str, object]:
    """
    Step 1: KMeans classification and color export (without vectors).
    Outputs an RGB GeoTIFF with the classified clusters colored.
    
    Returns: {"status": "ok", "outputPath": "..."}
    """
    path = Path(raster_path)
    if not path.exists():
        return {"status": "error", "message": "Raster path not found"}

    # Normalise: treat empty strings the same as None.
    output_path = output_path.strip() if output_path else None

    print(f"  [DEBUG] classify_and_export: output_path={output_path!r}")

    # Convenience wrapper so every progress call is a one-liner.
    def _cb(phase: str, done: int, total: int) -> None:
        if progress_callback is not None:
            try:
                progress_callback(phase, done, total)
            except Exception:
                pass

    n_clusters = len(classes)
    _t_ce_start = _time.perf_counter()
    _ce_stages: List[Tuple[str, float]] = []
    
    print("="*70)
    print("STEP 1: CLASSIFICATION & EXPORT (No Vectors)")
    print("="*70)

    # Emit acceleration engine as the very first progress event so the UI
    # shows which engine (GPU / CPU) is active from the moment the run starts.
    _accel_label = (
        f"GPU ({_ACCEL_ENGINE})" if _ACCEL_GPU
        else f"CPU ({_ACCEL_ENGINE})"
    )
    _cb(f"Engine: {_accel_label}", 0, 1)

    # === Load raster ===
    _t0 = _time.perf_counter()
    _cb("Loading raster", 0, 1)
    print(f"\n[1/5] Loading raster: {path.name}")
    with rasterio.open(path) as src:
        profile   = src.profile.copy()
        transform = src.transform
        crs       = _normalize_pseudo_mercator_crs(src.crs)
        height, width, n_bands = src.height, src.width, src.count
        _total_bytes = height * width * n_bands * int(np.dtype(src.dtypes[0]).itemsize)
        _ram_budget  = int(_available_ram_bytes() * 0.55)
        raster_data: np.ndarray | None = src.read() if _total_bytes <= _ram_budget else None

    profile["crs"] = crs
    if raster_data is None:
        print(f"  [warn] Raster size {_total_bytes/1e9:.1f} GB exceeds RAM budget "
              f"{_ram_budget/1e9:.1f} GB - using windowed sampling for training.")
    print(f"  Dimensions: {height}x{width}, {n_bands} bands")
    print(f"  CRS: {crs}")
    _cb("Loading raster", 1, 1)
    _ce_stages.append(("Load raster", _time.perf_counter() - _t0))

    # === Extract features ===
    _t0 = _time.perf_counter()
    _cb("Feature extraction", 0, 1)
    print(f"\n[2/5] Extracting pixel-level features...")
    if raster_data is not None:
        pixel_features    = _extract_pixel_features(raster_data, feature_flags)
        _full_features    = True
    else:
        print(f"  Using spatially-distributed sample (raster too large for full load)")
        pixel_features    = _sample_raster_for_training(str(path), feature_flags)
        _full_features    = False
    print(f"  Feature vector shape: {pixel_features.shape}")
    _cb("Feature extraction", 1, 1)
    _ce_stages.append(("Feature extraction", _time.perf_counter() - _t0))

    # === KMeans clustering ===
    _t0 = _time.perf_counter()
    _cb("KMeans clustering", 0, 1)
    print(f"\n[3/5] KMeans clustering ({n_clusters} clusters)...")
    if pretrained_scaler is not None and pretrained_kmeans is not None:
        scaler = pretrained_scaler
        kmeans = pretrained_kmeans
        if _full_features:
            # Fast float32 normalisation (avoids sklearn's float64 copy)
            _mean = scaler.mean_.astype(np.float32)
            _scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
            features_normalized = (pixel_features - _mean) / _scale
        else:
            features_normalized = None
        print(f"  [OK] Using pretrained model (training skipped)")
    else:
        scaler = StandardScaler()
        n_px   = len(pixel_features)
        # Adaptive sampling: use PCA to estimate scene complexity and pick
        # an appropriate training-pixel budget (fast: < 10 ms overhead).
        _adaptive_n = _pca_adaptive_n_train(pixel_features, n_clusters)
        if n_px > _adaptive_n:
            idx = np.random.default_rng(42).choice(n_px, _adaptive_n, replace=False)
            train_px = pixel_features[idx]
            print(f"  Subsampled {_adaptive_n:,} / {n_px:,} pixels for training (PCA-adaptive)")
        else:
            train_px = pixel_features
        # Fit scaler on SUBSAMPLE - sklearn's fit() internally converts to
        # float64 which is very expensive on the full pixel array.  The
        # mean / std from a ~100 K subsample is statistically identical to
        # the full-data estimate (law of large numbers).
        scaler.fit(train_px)
        _mean = scaler.mean_.astype(np.float32)
        _scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
        if _full_features:
            features_normalized = (pixel_features - _mean) / _scale
        else:
            features_normalized = None
        train_norm = scaler.transform(train_px)
        kmeans = _make_kmeans(n_clusters)
        kmeans.fit(train_norm)
        del train_norm, train_px
        gc.collect()
        print(f"  [OK] KMeans fitted [{_ACCEL_ENGINE}] on {len(pixel_features):,} pixels")

    mea_mapping: List[Dict[str, object]] | None = None
    scene_mea_prior: Dict[str, float] | None = None
    scene_profile: str | None = None
    if _is_mea_classes(classes) and raster_data is not None:
        scene_mea_prior, scene_profile = _scene_adaptive_mea_prior(
            raster_data,
            classes[:n_clusters],
            transform=transform,
        )
        print(f"  [OK] Scene profile: {scene_profile}")
    elif _is_mea_classes(classes):
        print(f"  [warn] Skipping scene-adaptive prior (large raster, windowed mode)")
    _cb("KMeans clustering", 1, 1)
    _ce_stages.append(("KMeans clustering", _time.perf_counter() - _t0))

    if tile_mode:
        _t0 = _time.perf_counter()
        print(f"\n[4/5] Tiled classification (multiprocessing)...")
        tile_size = _auto_tile_size(height, width, tile_max_pixels)
        windows = _generate_tile_windows(width, height, tile_size, tile_overlap)
        output_dir = _resolve_tile_output_dir(path, tile_output_dir, "_classified_tiles")
        output_dir.mkdir(parents=True, exist_ok=True)

        if pretrained_color_table is not None:
            # Shared model: reuse the pre-built color table - every image / tile gets
            # identical cluster ⇒ material assignments (batch / tiling consistency).
            color_table = pretrained_color_table
            mea_mapping = pretrained_mea_mapping
            print("  [OK] Using shared pre-built color table (batch consistency mode)")
        else:
            _EMPTY_SEM = {"veg": 0.0, "road": 0.0, "water": 0.0, "asphalt": 0.0,
                          "line": 0.0, "water_conf": 0.0, "dry": 0.0, "sand": 0.0,
                          "grass": 0.0, "gray_frac": 0.0, "dark_gray_frac": 0.0, "warm_frac": 0.0,
                          "size_frac": 0.0, "blue_dom_frac": 0.0}
            if _is_mea_classes(classes):
                cluster_rgbs = _cluster_rgb_from_kmeans_centers(scaler, kmeans.cluster_centers_.astype(np.float32))
                if features_normalized is not None:
                    tile_labels       = kmeans.predict(features_normalized)
                    cluster_counts    = np.bincount(tile_labels, minlength=n_clusters).astype(int).tolist()
                    tile_class_raster = tile_labels.reshape(height, width) + 1
                    try:
                        cluster_semantics = _cluster_semantic_scores(raster_data, tile_class_raster, n_clusters)
                    except Exception as sem_err:
                        print(f"  [warn] Semantic scoring failed: {sem_err}")
                        cluster_semantics = [dict(_EMPTY_SEM) for _ in range(n_clusters)]
                else:
                    print(f"  [warn] Cluster semantics skipped (large raster / windowed training)")
                    cluster_counts    = None
                    cluster_semantics = [dict(_EMPTY_SEM) for _ in range(n_clusters)]
                mea_mapping, color_table = _build_mea_cluster_mapping(
                    cluster_rgbs,
                    classes[:n_clusters],
                    cluster_counts=cluster_counts,
                    material_prior=scene_mea_prior,
                    cluster_semantics=cluster_semantics,
                )
                print("  [OK] MEA nearest-color mapping applied (tile mode)")
                for m in mea_mapping:
                    print(f"    Cluster {m['cluster']} -> {m['material']} ({m['colorHex']}, RGB{m['colorRGB']})")
            else:
                color_table = _build_color_table(classes, n_clusters)
        scaler_mean  = scaler.mean_.astype(np.float32)
        scaler_scale = scaler.scale_.astype(np.float32)
        centers      = kmeans.cluster_centers_.astype(np.float32)

        # Determine max workers: use tile_workers if specified, or fill up to max_threads (if set), else all CPUs
        if tile_workers and tile_workers > 0:
            max_workers = tile_workers
        elif max_threads and max_threads > 0:
            max_workers = max_threads
        else:
            max_workers = max(1, os.cpu_count() or 1)
        
        # Compute the padding each tile worker needs to read beyond its boundary
        # so the smoothing filter has full context (eliminates tile-boundary seams).
        _smooth_pad = 0
        if smoothing and smoothing != "none":
            try:
                _smooth_pad = int(smoothing.split("_")[1]) if "_" in smoothing else 2
            except Exception:
                _smooth_pad = 2

        _tile_extra: Dict[str, object] = {
            "smooth_pad":    _smooth_pad,
            "raster_h":      height,
            "raster_w":      width,
            "classes":       classes,     # enables per-tile post-processing
            "detect_shadows": detect_shadows,
        }

        # Determine output extension: prefer explicit suffix from output_path, else use export_format
        if output_path and Path(output_path).suffix:
            out_ext = Path(output_path).suffix
        else:
            out_ext = f".{export_format}" if export_format and export_format != "tif" else ".tif"
        jobs = []
        for row, col, h, w in windows:
            tile_name = f"{path.stem}_tile_r{row}_c{col}{out_ext}"
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
                tile_name,
                _tile_extra,   # <-- new: seamless smoothing + post-processing
            ))

        tile_outputs: List[str] = []
        _n_jobs = len(jobs)
        # Use chunksize > 1 when many tiles to reduce IPC overhead
        _chunksize = max(1, _n_jobs // (max_workers * 4))
        _cb("Classifying tiles", 0, _n_jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_classify_tile_worker, job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                tile_outputs.append(future.result())
                _cb("Classifying tiles", len(tile_outputs), _n_jobs)
                if len(tile_outputs) % max(1, _n_jobs // 10) == 0 or len(tile_outputs) == _n_jobs:
                    print(f"    tiles: {len(tile_outputs)}/{_n_jobs} done")

        print(f"  [OK] Wrote {len(tile_outputs)} tiles to {output_dir}")
        _ce_stages.append(("Tiled classification", _time.perf_counter() - _t0))
        _ce_total = _time.perf_counter() - _t_ce_start
        _ce_table = _build_stats_table(_ce_stages, _ce_total)
        print("\n" + "="*70)
        print("[OK] STEP 1 COMPLETE: Classification & Export (Tiles)")
        print(_ce_table)
        print("="*70)

        # Write companion XML alongside the first tile (same dir as all tiles).
        _tile_xml: Optional[str] = None
        if _is_mea_classes(classes) and tile_outputs:
            _tile_xml = _write_composite_material_xml(tile_outputs[0], classes)

        return {
            "status": "ok",
            "outputPath": str(output_dir),
            "tileOutputs": sorted(tile_outputs),
            "message": "Classification complete (tiles). Use output directory for Step 2.",
            "meaMapping": mea_mapping,
            "stats": _ce_stages,
            "statsTable": _ce_table,
            "xmlPath": _tile_xml,
        }

    # === NN assignment ===
    _t0 = _time.perf_counter()
    _cb("Pixel assignment", 0, 1)
    print(f"\n[4/5] Assigning pixels to clusters...")
    pixel_labels = _nearest_center_chunked(
        features_normalized.astype(np.float32),
        kmeans.cluster_centers_.astype(np.float32),
    ) + 1
    predicted_raster = pixel_labels.reshape(height, width)
    
    unique_classes = np.unique(predicted_raster)
    print(f"  [OK] Classes: {unique_classes}")
    _ce_stages.append(("Pixel assignment (NN)", _time.perf_counter() - _t0))
    _cb("Pixel assignment", 1, 1)

    # Free large feature arrays — no longer needed after pixel assignment.
    del pixel_features, features_normalized, pixel_labels
    gc.collect()

    # === Smoothing ===
    _t0 = _time.perf_counter()
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
    _ce_stages.append(("Smoothing", _time.perf_counter() - _t0))

    step_num = 6
    _t0 = _time.perf_counter()
    _cb("Saving output", 0, 1)
    
    # Determine output extension from export_format (default: .tif)
    _out_ext = f".{export_format}" if export_format and export_format != "tif" else ".tif"
    
    # === Save classification ===
    if output_path:
        output_color_path = Path(output_path)
        # If the user supplied a directory, place the classified file inside it.
        if output_color_path.is_dir() or (not output_color_path.suffix and not output_color_path.exists()):
            output_color_path.mkdir(parents=True, exist_ok=True)
            output_color_path = output_color_path / (path.stem + "_classified" + _out_ext)
        elif not output_color_path.suffix:
            # No extension on the file path - append the chosen format
            output_color_path = output_color_path.with_suffix(_out_ext)
        print(f"  [OUTPUT] Using user-specified path: {output_color_path}")
    else:
        # No output path provided - place the classified file in an 'output'
        # subfolder next to the source raster.
        _default_out_dir = path.parent / "output"
        _default_out_dir.mkdir(parents=True, exist_ok=True)
        output_color_path = _default_out_dir / (path.stem + "_classified" + _out_ext)
        print(f"  [OUTPUT] No output path set - fallback: {output_color_path}")
    
    print(f"\n[{step_num}/{step_num}] Saving classified output...")
    print(f"  Output: {output_color_path}")
    
    # Compute colors
    if pretrained_color_table is not None:
        # Shared model: skip per-image MEA mapping entirely.
        color_table = pretrained_color_table
        mea_mapping = pretrained_mea_mapping
        print("  [OK] Using shared pre-built color table (batch consistency mode)")
    elif _is_mea_classes(classes):
        cluster_rgbs = _cluster_rgb_means_from_class_raster(raster_data, predicted_raster, n_clusters)
        cluster_counts = np.bincount((predicted_raster.ravel() - 1).astype(int), minlength=n_clusters).astype(int).tolist()
        try:
            cluster_semantics = _cluster_semantic_scores(raster_data, predicted_raster, n_clusters)
        except Exception as sem_err:
            print(f"  [warn] Semantic scoring failed: {sem_err}")
            cluster_semantics = [{"veg": 0.0, "road": 0.0, "water": 0.0, "asphalt": 0.0, "line": 0.0, "water_conf": 0.0, "dry": 0.0, "sand": 0.0, "grass": 0.0, "gray_frac": 0.0, "dark_gray_frac": 0.0, "achro_frac": 0.0, "warm_frac": 0.0, "size_frac": 0.0, "blue_dom_frac": 0.0} for _ in range(n_clusters)]
        mea_mapping, color_table = _build_mea_cluster_mapping(
            cluster_rgbs,
            classes[:n_clusters],
            cluster_counts=cluster_counts,
            material_prior=scene_mea_prior,
            cluster_semantics=cluster_semantics,
        )
        print("  [OK] MEA nearest-color mapping applied")
        for m in mea_mapping:
            print(f"    Cluster {m['cluster']} -> {m['material']} ({m['colorHex']}, RGB{m['colorRGB']})")
    else:
        color_table = _compute_all_colors(raster_data, predicted_raster, n_clusters, 0, classes)
    
    # Apply colors
    rgb = _apply_color_table(predicted_raster, color_table)

    # Free large arrays before writing — only rgb is needed from here.
    del predicted_raster, raster_data
    gc.collect()

    # Write RGB output
    rgb_profile = profile.copy()
    driver = _driver_for_path(output_color_path)
    rgb_profile = _profile_for_driver(rgb_profile, driver)
    rgb_profile.update(
        count=3,
        dtype=np.uint8,
        interleave='band'
    )
    # Use tiled writing for GeoTIFF so GDAL writes block-by-block, avoiding
    # the ~480 MB in-memory buffer limit on large rasters.
    if str(driver).upper() == "GTIFF":
        rgb_profile.update(
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress='deflate',
            zlevel=1,
            predictor=2,
        )
    
    output_color_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(output_color_path).exists():
        Path(output_color_path).unlink()
    
    with rasterio.open(output_color_path, 'w', **rgb_profile) as dst:
        dst.write(rgb)

    print(f"  [OK] Classification saved to {output_color_path}")
    _cb("Saving output", 1, 1)
    _ce_stages.append(("Save output", _time.perf_counter() - _t0))
    _ce_total = _time.perf_counter() - _t_ce_start
    _ce_table = _build_stats_table(_ce_stages, _ce_total)

    print("\n" + "="*70)
    print("[OK] STEP 1 COMPLETE: Classification & Export")
    print(_ce_table)
    print("="*70)

    # Write companion Composite_Material_Table XML for MEA classifications.
    xml_out: Optional[str] = None
    if _is_mea_classes(classes):
        xml_out = _write_composite_material_xml(output_color_path, classes)

    return {
        "status": "ok",
        "outputPath": str(output_color_path),
        "message": "Classification complete. Use output file for Step 2 (vector rasterization).",
        "meaMapping": mea_mapping,
        "stats": _ce_stages,
        "statsTable": _ce_table,
        "xmlPath": xml_out,
    }


# ---------------------------------------------------------------------------
# Smart representative-raster selection for large tile sets
# ---------------------------------------------------------------------------

def _select_representative_rasters(
    raster_paths: List[str],
    max_rasters: int | None = None,
) -> Tuple[List[str], List[str]]:
    """Select a spatially diverse subset of rasters for training / colour-table.

    When a large area is captured as many small tiles, neighbouring tiles share
    almost identical spectral distributions.  Training on *all* of them wastes
    time without improving model quality.  This function picks a representative
    subset using lightweight metadata reads (no pixel I/O).

    **Algorithm**
    1. If ``len(raster_paths) <= max_rasters`` -> return all.
    2. Read each raster's bounding-box centroid (only metadata - fast).
    3. Overlay an adaptive grid on the centroid cloud and pick one raster per
       cell (the one closest to the cell centre) -> spatially stratified set.
    4. If the grid yields fewer than *max_rasters*, pad with random picks from
       the remaining rasters (spectral diversity insurance).

    Parameters
    ----------
    raster_paths : list of str
        All raster file paths.
    max_rasters : int or None
        Budget.  ``None`` -> ``max(8, ceil(sqrt(N)))``.

    Returns
    -------
    (selected, skipped) : (List[str], List[str])
        ``selected`` - paths chosen for training.
        ``skipped``  - paths *not* chosen (still need classification).
    """
    import math

    N = len(raster_paths)
    if max_rasters is None:
        max_rasters = max(8, math.ceil(math.sqrt(N)))
    max_rasters = min(max_rasters, N)

    if N <= max_rasters:
        return list(raster_paths), []

    # ── Step 1: read centroids (metadata only - very cheap) ──────────────
    centroids: List[Tuple[float, float]] = []  # (cx, cy) per raster
    valid_indices: List[int] = []
    for i, rp in enumerate(raster_paths):
        try:
            with rasterio.open(rp) as src:
                b = src.bounds  # left, bottom, right, top
                cx = (b.left + b.right) / 2.0
                cy = (b.bottom + b.top) / 2.0
                centroids.append((cx, cy))
                valid_indices.append(i)
        except Exception:
            pass  # skip unreadable files

    # If we couldn't read any metadata, fall back to random selection
    if not valid_indices:
        rng = np.random.default_rng(42)
        sel = rng.choice(N, max_rasters, replace=False).tolist()
        sel_set = set(sel)
        return (
            [raster_paths[i] for i in sel],
            [raster_paths[i] for i in range(N) if i not in sel_set],
        )

    xs = np.array([c[0] for c in centroids], dtype=np.float64)
    ys = np.array([c[1] for c in centroids], dtype=np.float64)

    # ── Step 2: adaptive grid ────────────────────────────────────────────
    # We want roughly max_rasters cells.  Use a grid whose #cells ~ budget.
    n_cells_target = max_rasters
    x_range = xs.max() - xs.min() if xs.max() != xs.min() else 1.0
    y_range = ys.max() - ys.min() if ys.max() != ys.min() else 1.0
    aspect  = x_range / y_range if y_range > 0 else 1.0
    # nx * ny ~ n_cells_target,  nx/ny ~ aspect
    ny = max(1, int(round(math.sqrt(n_cells_target / aspect))))
    nx = max(1, int(round(n_cells_target / ny)))

    cell_w = x_range / nx if nx > 0 else x_range
    cell_h = y_range / ny if ny > 0 else y_range
    # Guard against zero-size cells (all rasters have same centroid)
    if cell_w == 0:
        cell_w = 1.0
    if cell_h == 0:
        cell_h = 1.0

    # Assign each centroid to a grid cell
    x_min, y_min = xs.min(), ys.min()
    cell_map: Dict[Tuple[int, int], List[int]] = {}  # (ix, iy) -> [index in valid_indices]
    for k, (cx, cy) in enumerate(centroids):
        ix = min(int((cx - x_min) / cell_w), nx - 1)
        iy = min(int((cy - y_min) / cell_h), ny - 1)
        cell_map.setdefault((ix, iy), []).append(k)

    # Pick the raster closest to cell centre from each non-empty cell
    selected_set: set = set()
    for (ix, iy), members in cell_map.items():
        ccx = x_min + (ix + 0.5) * cell_w
        ccy = y_min + (iy + 0.5) * cell_h
        best_k = min(members, key=lambda k: (centroids[k][0] - ccx) ** 2 + (centroids[k][1] - ccy) ** 2)
        selected_set.add(valid_indices[best_k])

    # ── Step 3: pad with random picks if under budget ────────────────────
    rng = np.random.default_rng(42)
    remaining = [i for i in valid_indices if i not in selected_set]
    n_extra   = max_rasters - len(selected_set)
    if n_extra > 0 and remaining:
        extra = rng.choice(remaining, min(n_extra, len(remaining)), replace=False).tolist()
        selected_set.update(extra)

    selected = sorted(selected_set)
    skipped  = sorted(set(range(N)) - selected_set)

    print(f"[SmartSelect] {len(selected)}/{N} rasters selected for training "
          f"(grid {nx}x{ny}, budget {max_rasters})")

    return (
        [raster_paths[i] for i in selected],
        [raster_paths[i] for i in skipped],
    )


def train_kmeans_model(
    raster_paths,   # str  OR  List[str]
    classes: List[Dict[str, str]],
    feature_flags: Dict[str, bool],
    detect_shadows: bool = False,
    max_train_pixels: int = MAX_TRAIN_PIXELS,
):
    """Train ONE shared MiniBatchKMeans model on one or more rasters.

    When given a list of rasters, samples proportionally from every file so
    the resulting model has seen the full spectral diversity of the dataset.
    This ensures that cluster ⇒ material assignments are identical across all
    images and tiles when the same model is reused.

    Returns: (scaler, kmeans)
    """
    if isinstance(raster_paths, (str, Path)):
        raster_paths = [str(raster_paths)]
    raster_paths = [str(p) for p in raster_paths]

    n_clusters  = len(classes)
    n_per       = max(4096, max_train_pixels // max(1, len(raster_paths)))
    parts: List[np.ndarray] = []
    for p in raster_paths:
        try:
            print(f"[Training] Sampling {Path(p).name} ...")
            parts.append(_sample_raster_for_training(p, feature_flags, n_samples=n_per))
        except Exception as e:
            print(f"[Training][warn] Could not sample {p}: {e}")
    if not parts:
        raise RuntimeError("[Training] No pixel features could be sampled from any raster.")

    combined = np.concatenate(parts, axis=0)
    rng = np.random.default_rng(42)
    if len(combined) > max_train_pixels:
        combined = combined[rng.choice(len(combined), max_train_pixels, replace=False)]

    scaler     = StandardScaler()
    train_norm = scaler.fit_transform(combined)
    kmeans     = _make_kmeans(n_clusters)
    kmeans.fit(train_norm)
    print(f"[Training] Shared model ready  -  "
          f"{len(combined):,} px from {len(raster_paths)} raster(s), "
          f"{n_clusters} clusters")
    return scaler, kmeans


def build_shared_color_table(
    raster_paths: List[str],
    scaler,
    kmeans,
    classes: List[Dict[str, str]],
    feature_flags: Dict[str, bool],
    max_color_rasters: int | None = None,
) -> Tuple[Optional[List[Dict[str, object]]], List[Tuple[int, int, int]]]:
    """Derive ONE shared color_table from averaged cluster semantics across all rasters.

    Only a spatially-representative **subset** is fully loaded for semantics
    (controlled by ``max_color_rasters``).  This keeps the cost manageable when
    there are hundreds of tiles covering a large area.

    Returns (mea_mapping, color_table).  Both are built from the SAME global
    model so every image and every tile receives identical cluster ⇒ material
    assignments.
    """
    # Smart subset - avoid loading all tiles when most are spectrally redundant
    color_paths, _skipped = _select_representative_rasters(raster_paths, max_color_rasters)

    n_clusters = len(classes)
    _EMPTY_SEM = {"veg": 0.0, "road": 0.0, "water": 0.0, "asphalt": 0.0,
                  "line": 0.0, "water_conf": 0.0, "dry": 0.0, "sand": 0.0,
                  "grass": 0.0, "gray_frac": 0.0, "dark_gray_frac": 0.0, "warm_frac": 0.0,
                  "size_frac": 0.0, "blue_dom_frac": 0.0}

    # Accumulate pixel-weighted semantic scores from every raster we can fit in RAM.
    accum_sem    = [{k: 0.0 for k in _EMPTY_SEM} for _ in range(n_clusters)]
    accum_counts = np.zeros(n_clusters, dtype=np.float64)
    n_sampled    = 0

    for rp in color_paths:
        try:
            with rasterio.open(rp) as src:
                H, W   = src.height, src.width
                nbytes = H * W * src.count * int(np.dtype(src.dtypes[0]).itemsize)
                if nbytes > int(_available_ram_bytes() * 0.40):
                    print(f"[SharedTable][warn] {Path(rp).name} too large for semantics, skipping")
                    continue
                rd = src.read()
        except Exception as e:
            print(f"[SharedTable][warn] Cannot open {rp}: {e}")
            continue

        feat   = _extract_pixel_features(rd, feature_flags, verbose=False)
        scale  = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
        fnorm  = ((feat - scaler.mean_) / scale).astype(np.float32)
        labels = _nearest_center_chunked(fnorm, kmeans.cluster_centers_.astype(np.float32)) + 1
        cr     = labels.reshape(H, W)
        counts = np.bincount(labels - 1, minlength=n_clusters).astype(np.float64)

        try:
            sems = _cluster_semantic_scores(rd, cr, n_clusters)
            for j, sem in enumerate(sems):
                w = counts[j]
                for k, v in sem.items():
                    accum_sem[j][k] = accum_sem[j].get(k, 0.0) + v * w
            accum_counts += counts
            n_sampled += 1
        except Exception as e:
            print(f"[SharedTable][warn] Semantics failed for {Path(rp).name}: {e}")

    # Compute weighted average.
    avg_semantics: List[Dict[str, float]] = []
    for j in range(n_clusters):
        w = accum_counts[j]
        if w > 0:
            avg_semantics.append({k: v / w for k, v in accum_sem[j].items()})
        else:
            avg_semantics.append(dict(_EMPTY_SEM))

    cluster_rgbs      = _cluster_rgb_from_kmeans_centers(scaler, kmeans.cluster_centers_.astype(np.float32))
    cluster_counts_l  = accum_counts.astype(int).tolist()

    if _is_mea_classes(classes):
        mea_mapping, color_table = _build_mea_cluster_mapping(
            cluster_rgbs, classes[:n_clusters],
            cluster_counts=cluster_counts_l,
            material_prior=None,
            cluster_semantics=avg_semantics,
        )
        print(f"[SharedTable] Color table built from {n_sampled}/{len(color_paths)} raster(s) "
              f"(of {len(raster_paths)} total)")
        for m in mea_mapping:
            print(f"  Cluster {m['cluster']:>2} -> {m['material']:<22} {m['colorHex']}")
        return mea_mapping, color_table
    else:
        return None, _build_color_table(classes, n_clusters)


def rasterize_vectors_onto_classification(
    classification_path: str,
    vector_layers: List[Dict[str, str]],
    classes: List[Dict[str, str]],
    output_path: str | None = None,
    raster_stem_hint: str | None = None,
    tile_mode: bool = False,
    tile_max_pixels: int = 512 * 512,
    tile_overlap: int = 0,
    tile_output_dir: str | None = None,
    tile_workers: Optional[int] = None,
    max_threads: Optional[int] = None,
    progress_callback=None,
) -> Dict[str, object]:
    """
    Step 2: Rasterize vector layers onto an existing classification file.
    Takes the RGB classification from Step 1 and overlays vector geometries.
    
    Args:
        classification_path: Path to classification RGB file (from Step 1)
        vector_layers: List of vector layers with filePath
        classes: Class definitions (for reference)
        output_path: Optional output path (defaults to input with suffix)
        raster_stem_hint: When classification_path is a temp file, use this
            stem for naming the output file (e.g. original raster name).
    
    Returns: {"status": "ok", "outputPath": "..."}
    """
    classif_path = Path(classification_path)
    # Normalise: treat empty strings the same as None.
    output_path = output_path.strip() if output_path else None
    # Derive a human-friendly stem for naming output files.  When the
    # classification was written to a temp file (e.g. "classification_temp_123456"),
    # we prefer the original raster name supplied via raster_stem_hint.
    _friendly_stem = raster_stem_hint or classif_path.stem
    if not classif_path.exists():
        return {"status": "error", "message": f"Classification file not found: {classification_path}"}
    
    _t_rv_start = _time.perf_counter()
    _rv_stages: List[Tuple[str, float]] = []
    
    print("="*70)
    print("STEP 2: VECTOR RASTERIZATION")
    print(f"  [DEBUG] output_path={output_path!r}")
    print(f"  [DEBUG] classes count={len(classes)}, vector_layers count={len(vector_layers)}")
    for _dbg_i, _dbg_cls in enumerate(classes[:3]):
        print(f"  [DEBUG] class[{_dbg_i}]: id={_dbg_cls.get('id')}, color={_dbg_cls.get('color')}")
    for _dbg_i, _dbg_vl in enumerate(vector_layers):
        print(f"  [DEBUG] vector[{_dbg_i}]: classId={_dbg_vl.get('classId')}, overrideColor={_dbg_vl.get('overrideColor')}")
    print("="*70)
    
    # === Load classification metadata ===
    _t0 = _time.perf_counter()
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
            classif_data = src.read()  # Read ALL bands (could be 1-band class or 3-band RGB)
            profile = src.profile.copy()
            transform = src.transform
            crs = _normalize_pseudo_mercator_crs(src.crs)
            if classif_data.ndim == 3:
                bands, height, width = classif_data.shape
            else:
                height, width = classif_data.shape

    profile["crs"] = crs
    
    print(f"  Shape: {height}x{width}")
    if classif_data is not None:
        print(f"  Data range: {np.min(classif_data)}-{np.max(classif_data)}")
        print(f"  Bands/ndim: {classif_data.ndim} (shape: {classif_data.shape})")
    
    # Keep the original band count from the source profile, but ensure dtype is at least uint8
    if "dtype" not in profile or profile["dtype"] == "bool":
        profile.update(dtype=np.uint8)
    profile.update(interleave='band')
    _rv_stages.append(("Load classification", _time.perf_counter() - _t0))
    
    # === Resolve classId -> overrideColor from class definitions ===
    # Build two lookups: classId -> RGB  AND  material-name -> RGB
    _class_color_map: Dict[str, Tuple[int, int, int]] = {}
    _name_color_map: Dict[str, Tuple[int, int, int]] = {}
    for cls in classes:
        _cid  = cls.get("id", "")
        _name = cls.get("name", "")
        _hex  = cls.get("color", "")
        if _hex.startswith("#") and len(_hex) == 7:
            try:
                _rgb = (int(_hex[1:3], 16), int(_hex[3:5], 16), int(_hex[5:7], 16))
                if _cid:
                    _class_color_map[_cid] = _rgb
                if _name:
                    _name_color_map[_name] = _rgb
            except ValueError:
                pass
    print(f"  Class color map (by id):   {_class_color_map}")
    print(f"  Class color map (by name): {_name_color_map}")

    # Also build a fallback from MEA_CLASSES so color resolution works even
    # when the caller passes only a classId that matches the built-in MEA palette.
    _mea_id_map:   Dict[str, Tuple[int, int, int]] = {}
    _mea_name_map: Dict[str, Tuple[int, int, int]] = {}
    for _mcls in MEA_CLASSES:
        _mhex = _mcls.get("color", "")
        if _mhex.startswith("#") and len(_mhex) == 7:
            try:
                _mrgb = (int(_mhex[1:3], 16), int(_mhex[3:5], 16), int(_mhex[5:7], 16))
                _mea_id_map[_mcls.get("id", "")]     = _mrgb
                _mea_name_map[_mcls.get("name", "")] = _mrgb
            except ValueError:
                pass

    # Always resolve color from classId / material name (material assigned in 'attach to vector').
    # Priority: classId in caller classes > material name in caller classes >
    #           classId in MEA palette > material name in MEA palette > keep existing > auto-color.
    for layer in vector_layers:
        _cid  = layer.get("classId", "")
        _lname = layer.get("name", "")   # sometimes the layer name IS the material name
        _resolved: Optional[Tuple[int, int, int]] = None

        if _cid and _cid in _class_color_map:
            _resolved = _class_color_map[_cid]
            print(f"  Resolved by classId={_cid!r} (caller classes) -> {_resolved}")
        elif _lname and _lname in _name_color_map:
            _resolved = _name_color_map[_lname]
            print(f"  Resolved by layer name={_lname!r} (caller classes) -> {_resolved}")
        elif _cid and _cid in _mea_id_map:
            _resolved = _mea_id_map[_cid]
            print(f"  Resolved by classId={_cid!r} (MEA palette) -> {_resolved}")
        elif _lname and _lname in _mea_name_map:
            _resolved = _mea_name_map[_lname]
            print(f"  Resolved by layer name={_lname!r} (MEA palette) -> {_resolved}")
        else:
            print(f"  WARNING: classId={_cid!r} / name={_lname!r} not found in any color map; "
                  f"existing overrideColor={layer.get('overrideColor')!r} will be used (or auto-color)")

        if _resolved is not None:
            layer["overrideColor"] = list(_resolved)

    # === Load and validate vectors ===
    _t0 = _time.perf_counter()
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
            print(f"    Vector CRS: {gdf.crs} (EPSG:{gdf.crs.to_epsg() if gdf.crs else None})")
            print(f"    Raster CRS: {crs} (EPSG:{CRS(crs).to_epsg() if crs else None})")
            
            # Always reproject vector to match the classification raster CRS.
            # Unconditional to_crs() avoids false-negative CRS.equals() mismatches
            # (e.g. LOCAL_CS vs EPSG:3857).
            if crs is None:
                print(f"    [WARN] Raster has no CRS; cannot reproject - assuming coords already match")
            elif gdf.crs is None:
                gdf = gdf.set_crs(crs, allow_override=True)
                print(f"    [OK] Vector had no CRS; assigned raster CRS directly")
            else:
                _orig_bounds = gdf.total_bounds
                _vec_epsg  = gdf.crs.to_epsg() if gdf.crs else None
                _rast_epsg = CRS(crs).to_epsg() if crs else None
                print(f"    [CRS] Vector EPSG:{_vec_epsg} -> Raster EPSG:{_rast_epsg} - reprojecting")
                try:
                    gdf = gdf.to_crs(crs)
                    print(f"    [OK] Reprojected. Bounds: {_orig_bounds} -> {gdf.total_bounds}")
                except Exception as e:
                    print(f"    [ERROR] CRS reproject failed: {e}")
                    # Try EPSG code fallback in case the CRS object is non-serialisable
                    try:
                        _epsg = CRS(crs).to_epsg()
                        if _epsg:
                            gdf = gdf.to_crs(f"EPSG:{_epsg}")
                            print(f"    [OK] Reprojected via EPSG:{_epsg} fallback. Bounds: {_orig_bounds} -> {gdf.total_bounds}")
                        else:
                            raise ValueError("No EPSG code available")
                    except Exception as e2:
                        print(f"    [ERROR] EPSG fallback also failed: {e2}")
                        print(f"    [SKIP] Vector skipped due to CRS reproject failure")
                        continue

            # Bounds sanity-check: warn if vector doesn't appear to overlap raster.
            # The rasterize step has its own fallback, but log it here for visibility.
            if crs is not None:
                _rminx = transform.c
                _rmaxx = transform.c + width * transform.a
                _rminy = transform.f + height * transform.e
                _rmaxy = transform.f
                _vb = gdf.total_bounds  # [minx, miny, maxx, maxy]
                _xok = not (_vb[2] < _rminx or _vb[0] > _rmaxx)
                _yok = not (_vb[3] < _rminy or _vb[1] > _rmaxy)
                if not (_xok and _yok):
                    print(f"    [WARN] Reprojected vector bounds {list(_vb)} do NOT overlap "
                          f"raster bounds [{_rminx:.2f},{_rminy:.2f},{_rmaxx:.2f},{_rmaxy:.2f}]. "
                          f"The rasterize step will attempt a coordinate fallback automatically.")

            validated_vectors.append((layer_path.name, gdf, layer.get("overrideColor")))
            print(f"    [OK] Validated")
        except Exception as e:
            print(f"    [ERROR] {e}")
            continue
    
    print(f"\n  Total validated: {len(validated_vectors)}/{len(vector_layers)}")
    _rv_stages.append(("Validate vectors", _time.perf_counter() - _t0))
    
    if not validated_vectors:
        print(f"\n  No valid vectors to rasterize. Returning original classification.")
        return {
            "status": "ok",
            "outputPath": str(classif_path),
            "message": "No vectors to rasterize"
        }

    if tile_mode or classif_path.is_dir():
        _t0 = _time.perf_counter()
        print(f"\n[3/3] Rasterizing vectors (tiles)...")
        output_dir = _resolve_tile_output_dir(classif_path, tile_output_dir or output_path, "_with_vectors_tiles")
        output_dir.mkdir(parents=True, exist_ok=True)

        layer_geoms: List[Tuple[List, int, Tuple[int, int, int]]] = []
        n_clusters = len(classes)
        _auto_colors = _pick_vector_overlay_colors(classes, len(validated_vectors))
        overlay_colors = [
            tuple(int(x) for x in oc) if isinstance(oc, (list, tuple)) and len(oc) == 3
            else _auto_colors[i]
            for i, (_, _, oc) in enumerate(validated_vectors)
        ]
        for idx, (_, gdf, _oc) in enumerate(validated_vectors):
            burn_value = n_clusters + idx + 1
            geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
            layer_geoms.append((geoms, burn_value, overlay_colors[idx]))

        jobs = []
        out_ext = Path(output_path).suffix if output_path else '.tif'
        if classif_path.is_dir():
            tile_inputs = sorted(
                p for p in classif_path.rglob("*")
                if p.is_file() and p.suffix.lower() in ('.tif', '.tiff', '.img')
            )
            if not tile_inputs:
                return {"status": "error", "message": f"No tiles found in: {classification_path}"}
            print(f"  Tiles found: {len(tile_inputs)}")
            for tile_path in tile_inputs:
                out_name = tile_path.stem + out_ext
                jobs.append((str(tile_path), None, layer_geoms, str(output_dir), out_name))
        else:
            tile_size = _auto_tile_size(height, width, tile_max_pixels)
            windows = _generate_tile_windows(width, height, tile_size, tile_overlap)
            print(f"  Tiles planned: {len(windows)} (tile_size={tile_size}, overlap={tile_overlap})")
            for row, col, h, w in windows:
                tile_name = f"{classif_path.stem}_tile_r{row}_c{col}{out_ext}"
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
        _rv_stages.append(("Rasterize vectors (tiles)", _time.perf_counter() - _t0))
        _rv_total = _time.perf_counter() - _t_rv_start
        _rv_table = _build_stats_table(_rv_stages, _rv_total)
        print(_rv_table)
        print("="*70)

        return {
            "status": "ok",
            "outputPath": str(output_dir),
            "tileOutputs": sorted(tile_outputs),
            "message": "Vector rasterization complete (tiles)",
            "stats": _rv_stages,
            "statsTable": _rv_table,
        }
    
    # === Rasterize vectors ===
    _t0 = _time.perf_counter()
    print(f"\n[3/3] Rasterizing {len(validated_vectors)} vector layers...")
    
    n_clusters = len(classes)
    working_raster = classif_path
    _auto_colors = _pick_vector_overlay_colors(classes, len(validated_vectors))
    overlay_colors = [
        tuple(int(x) for x in oc) if isinstance(oc, (list, tuple)) and len(oc) == 3
        else _auto_colors[i]
        for i, (_, _, oc) in enumerate(validated_vectors)
    ]
    print(f"  [COLOR DEBUG] auto_colors={_auto_colors}")
    print(f"  [COLOR DEBUG] overlay_colors={overlay_colors}")
    for _vi, (_, _, _voc) in enumerate(validated_vectors):
        print(f"  [COLOR DEBUG] validated_vectors[{_vi}] overrideColor={_voc!r}")

    for idx, (layer_name, gdf, _oc) in enumerate(validated_vectors):
        vector_class_id = n_clusters + idx + 1
        
        print(f"\n  [Vector {idx+1}/{len(validated_vectors)}] {layer_name}")
        print(f"    Material ID: {vector_class_id}")
        print(f"    Overlay color: {overlay_colors[idx]}")
        
        # Determine output path
        if output_path and idx == len(validated_vectors) - 1:
            _op = Path(output_path)
            # If user specified a directory, place the file inside it
            if _op.is_dir() or (not _op.suffix and not _op.exists()):
                _op.mkdir(parents=True, exist_ok=True)
                final_output = _op / (_friendly_stem + "_with_vectors" + classif_path.suffix)
            else:
                final_output = _op
                final_output.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Intermediate iteration or no user path - place in 'output' folder
            _default_out_dir = classif_path.parent / "output"
            _default_out_dir.mkdir(parents=True, exist_ok=True)
            final_output = _default_out_dir / (
                _friendly_stem + f"_with_vectors_{idx}" + classif_path.suffix
            )
        
        try:
            # Rasterize this vector
            rasterize_vector_onto_raster(
                raster_path=str(working_raster),
                gdf=gdf,
                burn_value=vector_class_id,
                output_path=str(final_output),
                crs=crs,
                overlay_color=overlay_colors[idx]
            )
            
            print(f"    [OK] Rasterized")
            
            # Use as base for next iteration
            if idx < len(validated_vectors) - 1:
                working_raster = final_output
        except Exception as e:
            print(f"    [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Final output path - use the resolved path from the last iteration
    if output_path:
        _op = Path(output_path)
        if _op.is_dir() or (not _op.suffix and not _op.exists()):
            final_path = _op / (_friendly_stem + "_with_vectors" + classif_path.suffix)
        else:
            final_path = _op
    else:
        # No user path - place in 'output' folder
        _default_out_dir = classif_path.parent / "output"
        _default_out_dir.mkdir(parents=True, exist_ok=True)
        final_path = _default_out_dir / (
            _friendly_stem + "_with_vectors" + classif_path.suffix
        )
    
    print(f"\n[OK] Vector rasterization complete")
    print(f"  Output: {final_path}")
    _rv_stages.append(("Rasterize vectors", _time.perf_counter() - _t0))
    _rv_total = _time.perf_counter() - _t_rv_start
    _rv_table = _build_stats_table(_rv_stages, _rv_total)
    print(_rv_table)
    print("="*70)
    
    return {
        "status": "ok",
        "outputPath": str(final_path),
        "message": "Vector rasterization complete",
        "stats": _rv_stages,
        "statsTable": _rv_table,
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
            kmeans = _make_kmeans(n, mini_batch=False)
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

    All independent ``uniform_filter`` calls (spectral bands + texture) are
    dispatched in a SINGLE thread pool for maximum parallelism.  The
    expensive ``np.ascontiguousarray(transpose)`` is avoided - we work with
    per-band views of the original (bands, H, W) array converted to float32
    only once.
    """
    n_bands, height, width = raster_data.shape[0], raster_data.shape[1], raster_data.shape[2]
    n_pixels = height * width

    # Convert bands to float32 views - one contiguous array per band.
    # This replaces the former full transpose + ascontiguousarray which
    # allocated an entire (H, W, bands) copy.
    _bands_f32: list[np.ndarray] = []
    for bi in range(n_bands):
        _bands_f32.append(np.ascontiguousarray(raster_data[bi], dtype=np.float32))

    feature_list: list[np.ndarray] = []

    if verbose:
        print(f"    Extracting features for {n_pixels:,} pixels ({n_bands} bands)...")

    # ---- Collect all uniform_filter jobs and run them in one thread pool ----
    _do_spectral = feature_flags.get("spectral", True)
    _do_texture = feature_flags.get("texture", True)

    _spectral_out: list[np.ndarray] = []
    _m1: np.ndarray | None = None
    _m2: np.ndarray | None = None
    _gray: np.ndarray | None = None

    # Pre-allocate all output buffers
    if _do_spectral:
        _spectral_out = [np.empty((height, width), dtype=np.float32) for _ in range(n_bands)]
    if _do_texture:
        if n_bands >= 3:
            _gray = (_bands_f32[0] + _bands_f32[1] + _bands_f32[2]) * np.float32(1.0 / 3.0)
        else:
            _gray = _bands_f32[0].copy()
        _gray_sq = np.empty_like(_gray)
        np.multiply(_gray, _gray, out=_gray_sq)
        _m1 = np.empty((height, width), dtype=np.float32)
        _m2 = np.empty((height, width), dtype=np.float32)

    # Build a flat list of (input, output) filter jobs
    _filter_jobs: list[tuple[np.ndarray, np.ndarray]] = []
    if _do_spectral:
        for bi in range(n_bands):
            _filter_jobs.append((_bands_f32[bi], _spectral_out[bi]))
    if _do_texture:
        _filter_jobs.append((_gray, _m1))
        _filter_jobs.append((_gray_sq, _m2))

    if _filter_jobs:
        def _run_filter(job):
            inp, out = job
            uniform_filter(inp, size=window_size, mode='reflect', output=out)

        _n_workers = min(len(_filter_jobs), max(1, (os.cpu_count() or 4)))
        if _n_workers >= 2 and len(_filter_jobs) >= 2:
            with ThreadPoolExecutor(max_workers=_n_workers) as _tp:
                list(_tp.map(_run_filter, _filter_jobs))
        else:
            for job in _filter_jobs:
                _run_filter(job)

    # Collect spectral features
    if _do_spectral:
        for bi in range(n_bands):
            feature_list.append(_spectral_out[bi].reshape(-1))
        if verbose:
            print(f"    [OK] Spectral: {n_bands} features")

    # Collect texture feature
    if _do_texture:
        if _do_texture:
            del _gray_sq  # free early
        std_dev = np.sqrt(np.maximum(_m2 - _m1 * _m1, 0.0, dtype=np.float32))
        feature_list.append(std_dev.reshape(-1))
        if verbose:
            print(f"    [OK] Texture: 1 feature (std dev)")

    # === Spectral indices (NDVI when NIR band present) ===
    if feature_flags.get("indices", True):
        if n_bands >= 4:
            ndvi = (_bands_f32[3] - _bands_f32[2]) / (_bands_f32[3] + _bands_f32[2] + 1e-6)
            feature_list.append(ndvi.reshape(-1).astype(np.float32))
            if verbose:
                print(f"    [OK] Indices: 1 feature (NDVI)")
        else:
            if verbose:
                print(f"    [INFO] No NIR band for NDVI (need >= 4, got {n_bands})")

    # === VARI + HSV Saturation (vegetation / colour indices) ===
    if feature_flags.get("colorIndices", True) and n_bands >= 3:
        _r, _g, _b = _bands_f32[0], _bands_f32[1], _bands_f32[2]

        # VARI: Visible Atmospherically Resistant Index  (G-R)/(G+R-B+eps)
        _denom = _g + _r - _b
        _denom = np.where(np.abs(_denom) < 1e-6, 1e-6, _denom)
        vari_feat = np.clip((_g - _r) / _denom, -1.0, 1.0).astype(np.float32)
        feature_list.append(vari_feat.reshape(-1))

        # HSV Saturation: (max-min)/(max+eps)
        _maxc = np.maximum(np.maximum(_r, _g), _b)
        _minc = np.minimum(np.minimum(_r, _g), _b)
        sat_feat = ((_maxc - _minc) / (_maxc + 1e-6)).astype(np.float32)
        feature_list.append(sat_feat.reshape(-1))

        if verbose:
            print(f"    [OK] Colour indices: VARI + HSV Saturation (2 features)")

    # === Local entropy (texture complexity) ===
    if feature_flags.get("entropy", False):
        # Re-use gray computed earlier when possible
        _gray_ref = _gray if _gray is not None else (
            (_bands_f32[0] + _bands_f32[1] + _bands_f32[2]) * np.float32(1.0 / 3.0)
            if n_bands >= 3 else _bands_f32[0]
        )
        _gmin, _gmax = float(np.min(_gray_ref)), float(np.max(_gray_ref))
        if _gmax > _gmin:
            _gray_u8 = (((_gray_ref - _gmin) / (_gmax - _gmin)) * 255.0).astype(np.uint8)
        else:
            _gray_u8 = np.zeros((height, width), dtype=np.uint8)
        _ENT_DOWNSAMPLE_THRESHOLD = 1_500_000
        if n_pixels > _ENT_DOWNSAMPLE_THRESHOLD:
            _ds = 2
            _gray_u8_ds = _gray_u8[::_ds, ::_ds]
            _ent_ds = sk_rank_entropy(_gray_u8_ds, disk(2)).astype(np.float32)
            _ent = np.repeat(np.repeat(_ent_ds, _ds, axis=0), _ds, axis=1)[:height, :width]
        else:
            _ent = sk_rank_entropy(_gray_u8, disk(2)).astype(np.float32)
        feature_list.append(_ent.reshape(-1))
        if verbose:
            _ds_tag = " (2x downsampled)" if n_pixels > _ENT_DOWNSAMPLE_THRESHOLD else ""
            print(f"    [OK] Entropy: 1 feature (local entropy, disk-2{_ds_tag})")

    if not feature_list:
        feature_list.append(_bands_f32[0].reshape(-1))

    # Pre-allocate output (n_pixels, n_features) in float32 and fill column-by-column
    n_features = len(feature_list)
    features = np.empty((n_pixels, n_features), dtype=np.float32)
    for col_idx, col_data in enumerate(feature_list):
        features[:, col_idx] = col_data

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


def _build_color_table(
    classes: List[Dict[str, str]],
    class_count: int,
    cluster_rgbs: List[Tuple[int, int, int]] | None = None,
) -> List[Tuple[int, int, int]]:
    colors = []
    for item in classes[:class_count]:
        hex_color = item.get("color", "#ffffff")
        colors.append(_hex_to_rgb(hex_color))
    while len(colors) < class_count:
        colors.append((255, 255, 255))

    if _is_mea_classes(classes) and cluster_rgbs and len(cluster_rgbs) == class_count:
        _, assigned_colors = _build_mea_cluster_mapping(cluster_rgbs, classes[:class_count])
        return assigned_colors

    return colors


def _is_mea_classes(classes: List[Dict[str, str]] | None) -> bool:
    if not classes:
        return False
    mea_names = {item["name"] for item in MEA_CLASSES}
    return all((c.get("name") in mea_names) for c in classes)


def _elongated_continuity_score(mask: np.ndarray) -> float:
    """Estimate road-like continuity from connected elongated components."""
    total = int(np.sum(mask))
    if total < 80:
        return 0.0

    lbl_obj = sk_label(mask, connectivity=1)
    if isinstance(lbl_obj, tuple):
        lbl = np.asarray(lbl_obj[0], dtype=np.int32)
    else:
        lbl = np.asarray(lbl_obj, dtype=np.int32)
    n_comp = int(np.max(lbl))
    if n_comp <= 0:
        return 0.0

    elongated_area = 0
    for comp_id, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        comp = (lbl[sl] == comp_id)
        area = int(np.sum(comp))
        if area < 28:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        thin_ratio = max(h, w) / max(1, min(h, w))
        fill_ratio = area / float(max(1, h * w))
        if thin_ratio >= 3.0 and fill_ratio <= 0.78:
            elongated_area += area

    score = elongated_area / float(total)
    return float(np.clip(score * 1.4, 0.0, 1.0))


def _waterbody_continuity_score(mask: np.ndarray) -> float:
    """Estimate water-like morphology: large, compact, continuous regions."""
    total = int(np.sum(mask))
    if total < 120:
        return 0.0

    lbl_obj = sk_label(mask, connectivity=1)
    if isinstance(lbl_obj, tuple):
        lbl = np.asarray(lbl_obj[0], dtype=np.int32)
    else:
        lbl = np.asarray(lbl_obj, dtype=np.int32)
    n_comp = int(np.max(lbl))
    if n_comp <= 0:
        return 0.0

    main_area = 0
    compact_area = 0
    for comp_id, sl in enumerate(ndi.find_objects(lbl), start=1):
        if sl is None:
            continue
        comp = (lbl[sl] == comp_id)
        area = int(np.sum(comp))
        if area < 40:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        fill_ratio = area / float(max(1, h * w))
        aspect = max(h, w) / max(1, min(h, w))
        main_area = max(main_area, area)
        if fill_ratio >= 0.38 and aspect <= 3.5:
            compact_area += area

    dominance = main_area / float(total)
    compact_frac = compact_area / float(total)
    score = (0.55 * dominance) + (0.45 * compact_frac)
    return float(np.clip(score, 0.0, 1.0))


def _cluster_semantic_scores(
    raster_data: np.ndarray,
    class_raster: np.ndarray,
    n_clusters: int,
) -> List[Dict[str, float]]:
    """
    Per-cluster semantic cues for MEA remapping:
    - vegetation score from ExG / green dominance
    - road score from grayness + elongated continuity
    - dry/barren score to separate desert/soil from green vegetation
    Uses sampling to remain fast on large rasters.
    """
    rgb = raster_data[:3] if raster_data.shape[0] >= 3 else raster_data
    if rgb.shape[0] < 3:
        return [{"veg": 0.0, "road": 0.0, "water": 0.0, "asphalt": 0.0, "line": 0.0, "water_conf": 0.0, "dry": 0.0, "sand": 0.0, "grass": 0.0, "gray_frac": 0.0, "dark_gray_frac": 0.0, "achro_frac": 0.0, "warm_frac": 0.0, "size_frac": 0.0, "blue_dom_frac": 0.0} for _ in range(n_clusters)]

    h, w = class_raster.shape
    stride = max(1, int(math.sqrt((h * w) / 420_000.0)))

    cls = class_raster[::stride, ::stride]
    r = rgb[0][::stride, ::stride].astype(np.float32)
    g = rgb[1][::stride, ::stride].astype(np.float32)
    b = rgb[2][::stride, ::stride].astype(np.float32)

    scale = np.percentile(np.concatenate([r.ravel(), g.ravel(), b.ravel()]), 98)
    scale = float(max(scale, 1.0))
    r = np.clip(r / scale, 0.0, 1.0)
    g = np.clip(g / scale, 0.0, 1.0)
    b = np.clip(b / scale, 0.0, 1.0)

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    sat = (maxc - minc) / (maxc + 1e-6)

    exg = (2.0 * g) - r - b
    vari = (g - r) / (g + r - b + 1e-6)

    # Gray exclusion mask: saturation < 0.15 is effectively achromatic
    # (concrete, asphalt, plaster) and must NOT be called vegetation.
    # Widened from 0.12 -> 0.15 to catch near-gray asphalt with a micro
    # green tint from atmospheric scattering in aerial imagery.
    _achromatic = (sat < 0.15)
    # Dark road exclusion: dark-to-medium brightness + low saturation.
    # Aerial photos of asphalt often show maxc 0.25-0.55, sat 0.10-0.18
    # with a slight green dominant channel that fools hue-based detectors.
    _dark_road = (sat < 0.20) & (maxc > 0.10) & (maxc < 0.60) & (exg < 0.06)

    # Path 1: strong greenness index (classic vegetation signal)
    strong_veg = (
        (exg > 0.06)
        & (g > (r * 1.05))
        & (g > (b * 1.03))
        & (sat > 0.10)
        & (maxc > 0.10)
        & ~_achromatic
        & ~_dark_road
    )
    # Path 2: hue-based green detection - g is the dominant channel with clear saturation
    hue_green = (
        (g >= maxc - 0.005)          # green is the maximum channel
        & (sat > 0.16)               # needs real chromaticity, NOT gray (was 0.12)
        & (maxc > 0.08)              # not too dark/black
        & (exg > 0.03)               # meaningful green excess (was 0.01)
        & (g > b)                    # not cyan/teal
        & ~_achromatic
    )
    # Path 3: VARI-based green detection - catches olive / muted vegetation that
    # ExG and hue-based paths miss (VARI is more sensitive to low-chroma greens).
    vari_green = (
        (vari > 0.08)                # was 0.05 - tighter to avoid gray roads
        & (sat > 0.14)               # raised from 0.10 - must have clear colour
        & (maxc > 0.10)
        & (g >= r)
        & ~_achromatic
        & ~_dark_road
    )
    veg_pixels = strong_veg | hue_green | vari_green
    # Dry/barren soil: warm-toned (r dominates b), not strongly green, moderate brightness.
    # Require r > b * 1.10 to prevent neutral gray concrete/plaster from triggering.
    dry_barren_pixels = (
        (r > (g * 1.04))
        & (r > (b * 1.10))   # warm tone required - concrete/asphalt has r ~ b
        & (b < (g * 1.12))
        & (sat < 0.30)
        & (maxc > 0.20)
        & (exg < 0.04)
    )
    # Sand – Path 1: warm hue, red clearly dominant over green and blue
    warm_sand_pixels = (
        (r > (g * 1.05))
        & (r > (b * 1.18))
        & (g >= b)
        & (sat < 0.36)
        & (maxc > 0.32)
        & (exg < 0.01)
    )
    # Sand – Path 2: pale / bleached desert sand (nearly white, slight warm tint).
    # Requires clearly negative ExG (r+b > 2g) to avoid matching warm-gray
    # concrete / plaster building surfaces which typically have ExG ~ 0.
    pale_sand_pixels = (
        (r >= g)
        & (r > (b * 1.12))
        & (g >= b)
        & (sat < 0.15)
        & (maxc > 0.52)
        & (exg < -0.01)
    )
    sand_pixels = warm_sand_pixels | pale_sand_pixels
    # Dry grass: yellowish-green, r ~ g both dominate b, moderate saturation, tiny ExG
    dry_grass_pixels = (
        (r > (b * 1.08))
        & (g > (b * 1.05))
        & (r < (g * 1.18))
        & (sat > 0.07)
        & (maxc > 0.18)
        & (exg > -0.05)
        & (exg < 0.12)
    )
    # Lush grass: green-dominant, saturated, clearly not sand or dry
    lush_grass_pixels = (
        (g >= maxc - 0.005)
        & (exg > 0.08)
        & (sat > 0.14)
        & (maxc > 0.18)
        & (r < (g * 1.05))
    )
    gray_pixels = (sat < 0.16) & (maxc > 0.12) & (maxc < 0.86)
    dark_gray_pixels = gray_pixels & (maxc < 0.46)
    # Achromatic road pixels: sat < 0.15 catches pure gray asphalt/concrete.
    # Also include _dark_road pixels (sat < 0.20, dark-medium, low exg).
    achromatic_road_pixels = (_achromatic | _dark_road) & (maxc > 0.10) & (maxc < 0.80)
    # Warm-toned pixels: red channel clearly warmer than blue.
    # This is the primary discriminator: soil/sand have r >> b,
    # while concrete/asphalt have r ~ g ~ b (neutral/cool).
    warm_pixels = (
        (r > (b * 1.12))   # red clearly exceeds blue (warm tint)
        & (r >= g * 0.90)  # not strongly green (that would be vegetation)
        & (sat > 0.04)     # exclude near-white / very pale neutrals
        & (maxc > 0.15)    # not too dark
        & (exg < 0.06)     # suppress green-dominant pixels
    )

    luma = (0.299 * r) + (0.587 * g) + (0.114 * b)
    gy, gx = np.gradient(luma)
    grad_mag = np.sqrt((gx * gx) + (gy * gy))
    grad_scale = float(max(np.percentile(grad_mag, 95), 1e-6))
    grad_norm = np.clip(grad_mag / grad_scale, 0.0, 1.0)
    edge_pixels = grad_norm > 0.30

    if raster_data.shape[0] >= 4:
        nir = raster_data[3][::stride, ::stride].astype(np.float32)
        nir = np.clip(nir / scale, 0.0, 1.0)
        ndwi = (g - nir) / (g + nir + 1e-6)
        water_pixels = (ndwi > 0.05) & (sat < 0.30) & (maxc > 0.06)
    else:
        blue_excess = b - np.maximum(r, g)
        # Path 1: classic blue-dominant water (rivers, pools)
        blue_water = (blue_excess > 0.03) & (sat < 0.35) & (b > 0.10) & (r < 0.55)
        # Path 2: dark + neutral (deep water / shadowed water surface)
        dark_water = (maxc < 0.28) & (sat < 0.22) & (b >= r) & (b >= g)
        # Path 3: blue is dominant channel (b == maxc)
        blue_dominant = (b >= maxc - 0.005) & (sat > 0.06) & (maxc > 0.10) & (r < 0.55)
        water_pixels = blue_water | dark_water | blue_dominant

    # Blue-dominant pixels: blue clearly exceeds both red and green channels.
    # Water bodies and dark shadows tend to have b > r and b > g, while vegetation
    # and roads do not. Used as an additional water-vs-vegetation discriminator.
    blue_dom_pixels = (b > (g + 0.03)) & (b > (r + 0.03)) & (maxc > 0.08)
    # Total sampled pixels with a valid cluster label (used for size_frac).
    total_px: int = int(np.sum(cls > 0))

    # ── Pre-compute continuity scores for ALL clusters in parallel ──
    # _elongated_continuity_score and _waterbody_continuity_score each run
    # sk_label + ndi.find_objects which release the GIL, so threading is
    # effective.  This avoids the sequential per-cluster bottleneck.
    _cluster_masks = [(cls == cid) for cid in range(1, n_clusters + 1)]

    def _cont_pair(mask):
        return (_elongated_continuity_score(mask), _waterbody_continuity_score(mask))

    if n_clusters >= 4:
        _n_cont_workers = min(n_clusters, max(1, (os.cpu_count() or 4)))
        with ThreadPoolExecutor(max_workers=_n_cont_workers) as _tp:
            _cont_results = list(_tp.map(_cont_pair, _cluster_masks))
    else:
        _cont_results = [_cont_pair(m) for m in _cluster_masks]

    out: List[Dict[str, float]] = []
    for cluster_id in range(1, n_clusters + 1):
        m = (cls == cluster_id)
        px = int(np.sum(m))
        if px == 0:
            out.append({"veg": 0.0, "road": 0.0, "water": 0.0, "asphalt": 0.0, "line": 0.0, "water_conf": 0.0, "dry": 0.0, "sand": 0.0, "grass": 0.0, "gray_frac": 0.0, "dark_gray_frac": 0.0, "achro_frac": 0.0, "warm_frac": 0.0, "size_frac": 0.0, "blue_dom_frac": 0.0})
            continue

        veg_frac = float(np.mean(veg_pixels[m]))
        dry_frac = float(np.mean(dry_barren_pixels[m]))
        sand_frac = float(np.mean(sand_pixels[m]))
        dry_grass_frac = float(np.mean(dry_grass_pixels[m]))
        lush_grass_frac = float(np.mean(lush_grass_pixels[m]))
        gray_frac = float(np.mean(gray_pixels[m]))
        dark_gray_frac = float(np.mean(dark_gray_pixels[m]))
        achro_frac = float(np.mean(achromatic_road_pixels[m]))
        warm_frac = float(np.mean(warm_pixels[m]))
        blue_dom_frac = float(np.mean(blue_dom_pixels[m]))
        size_frac = float(px) / float(max(1, total_px))  # relative cluster size
        water_frac = float(np.mean(water_pixels[m]))
        edge_frac = float(np.mean(edge_pixels[m]))
        # Continuity scores are pre-computed in parallel (see below)
        continuity = _cont_results[cluster_id - 1][0]
        water_cont = _cont_results[cluster_id - 1][1]
        # Gray signal alone is sufficient to call a cluster "road-like" when strong.
        # Achromatic fraction (sat<0.12) is the strongest road indicator.
        road_score = float(np.clip(
            max(
                (0.30 * gray_frac) + (0.30 * achro_frac) + (0.40 * continuity),
                0.65 * gray_frac if gray_frac >= 0.55 else 0.0,  # very gray -> road even without elongation
                0.80 * achro_frac if achro_frac >= 0.40 else 0.0,  # strong achromatic -> road
            ),
            0.0, 1.0,
        ))
        # Asphalt score: when the cluster is strongly gray/achromatic we allow some
        # green contamination (road-side vegetation / shadow edges can bleed into a
        # road cluster).  The net vegetation fraction is reduced by gray+achro so that
        # a road cluster with mixed green at its edges still scores high.
        _net_veg = max(0.0, veg_frac - 0.60 * gray_frac - 0.40 * achro_frac)
        asphalt_score = float(np.clip(
            ((0.25 * gray_frac) + (0.25 * dark_gray_frac) + (0.25 * achro_frac) + (0.25 * road_score))
            * (1.0 - 1.40 * _net_veg)      # heavy green suppression (but attenuated by gray)
            * (1.0 - 0.60 * sand_frac),    # warm sand suppression
            0.0, 1.0,
        ))
        line_score = float(np.clip((0.65 * edge_frac) + (0.35 * continuity), 0.0, 1.0))
        water_score = float(np.clip((0.45 * water_frac) + (0.55 * water_cont), 0.0, 1.0))
        # dry_score: warm_frac replaces gray_frac - gray ≠ dry/barren (that's concrete).
        dry_score = float(np.clip(
            (0.70 * dry_frac)
            + (0.30 * warm_frac * (1.0 - water_score))
            - (0.40 * gray_frac),   # subtract gray - gray cluster is NOT dry soil
            0.0, 1.0,
        ))
        # Sand: warm/pale sand signal, suppressed by green/water/road; also strongly penalised
        # when lush_grass_frac exceeds sand_frac (cluster is genuinely grassy)
        sand_score = float(np.clip(
            (0.65 * sand_frac + 0.35 * dry_frac)
            * (1.0 - 0.90 * lush_grass_frac)         # lush green kills sand confidence
            * (1.0 - 0.75 * veg_frac)
            * (1.0 - 0.75 * water_score)
            * (1.0 - 0.40 * road_score)
            * (1.0 - 0.45 * max(0.0, dry_grass_frac - sand_frac)),
            0.0,
            1.0,
        ))
        # Grass: pure green indication - used to gate vegetation assignments away from sand
        grass_score = float(np.clip(
            lush_grass_frac
            * (1.0 - 0.70 * sand_frac)
            * (1.0 - 0.60 * dry_frac),
            0.0,
            1.0,
        ))
        # High confidence water when color OR shape agrees, and road/asphalt cues are weak.
        # Warm-tone suppression: warm pixels (r >> b) are soil/earth, not water.
        _warm_suppress = max(0.0, 1.0 - 1.8 * warm_frac) if warm_frac >= 0.20 else 1.0
        water_conf = float(np.clip(
            min(1.0, water_score * 1.20)
            * (1.0 if (water_frac >= 0.22 or water_cont >= 0.40) else 0.45)
            * (1.0 - 0.65 * road_score)
            * (1.0 - 0.60 * asphalt_score)
            * (1.0 - 0.45 * line_score)
            * _warm_suppress,
            0.0,
            1.0,
        ))
        # Roads/asphalt should not be interpreted as vegetation.
        # gray_frac directly suppresses veg interpretation: an achromatic cluster
        # cannot be vegetation regardless of any mixed green pixels at its edges.
        # dark_gray_frac provides additional suppression for dark asphalt tones.
        # achro_frac (sat < 0.12) is the strongest: truly neutral pixels = road.
        _gray_suppress = min(1.0, 0.90 * gray_frac + 0.50 * dark_gray_frac + 1.20 * achro_frac)
        veg_score = float(np.clip(
            veg_frac
            * (1.0 - 0.40 * road_score)
            * (1.0 - 0.65 * water_score)
            * (1.0 - 0.50 * dry_score)
            * (1.0 - _gray_suppress),
            0.0,
            1.0,
        ))

        out.append({
            "veg": veg_score,
            "road": road_score,
            "water": water_score,
            "asphalt": asphalt_score,
            "line": line_score,
            "water_conf": water_conf,
            "dry": dry_score,
            "sand": sand_score,
            "grass": grass_score,
            "gray_frac": gray_frac,
            "dark_gray_frac": dark_gray_frac,
            "achro_frac": achro_frac,
            "warm_frac": warm_frac,
            "size_frac": size_frac,
            "blue_dom_frac": blue_dom_frac,
        })

    return out


def _build_mea_cluster_mapping(
    cluster_rgbs: List[Tuple[int, int, int]],
    material_classes: List[Dict[str, str]],
    cluster_counts: List[int] | None = None,
    material_prior: Dict[str, float] | None = None,
    cluster_semantics: List[Dict[str, float]] | None = None,
) -> Tuple[List[Dict[str, object]], List[Tuple[int, int, int]]]:
    material_colors = [_hex_to_rgb(item.get("color", "#ffffff")) for item in material_classes]
    material_names = [item.get("name", "UNKNOWN") for item in material_classes]
    material_hex = [item.get("color", "#ffffff") for item in material_classes]

    assigned: List[Tuple[int, int, int]] = [(255, 255, 255)] * len(cluster_rgbs)
    mapping: List[Dict[str, object]] = [
        {
            "cluster": i + 1,
            "material": "UNKNOWN",
            "colorHex": "#ffffff",
            "colorRGB": (255, 255, 255),
        }
        for i in range(len(cluster_rgbs))
    ]

    if not cluster_rgbs or not material_colors:
        return mapping, assigned

    cluster_freq_arr: np.ndarray | None = None

    # Build RGB squared-distance cost matrix: rows=clusters, cols=materials
    _VEG_NAMES = {"BM_LAND_GRASS", "BM_LAND_DRY_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
    _WATER_NAMES = {"BM_WATER"}
    _METAL_NAMES = {"BM_METAL", "BM_METAL_STEEL"}
    _SOIL_NAMES = {"BM_SOIL", "BM_SAND"}
    _ROAD_NAMES = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
    _ROADLIKE_NAMES = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT", "BM_ROCK"}
    _ASPHALT_ANCHORS = [
        (44, 44, 56),
        (52, 55, 72),
        (91, 91, 101),
        (91, 90, 96),
    ]
    # Primary asphalt RGB for direct distance check
    _BM_ASPHALT_RGB = (45, 45, 48)
    _BM_VEGETATION_RGB = (34, 139, 34)
    _BM_LAND_GRASS_RGB = (124, 252, 0)
    _BM_LAND_DRY_GRASS_RGB = (189, 183, 107)
    # At 0.5m/px resolution color is already a strong signal.
    # Base biases are kept modest; semantics provide the main disambiguation.
    VEG_BASE_BIAS   =  2000.0   # small nudge: easily overridden by green color + bonus
    WATER_BASE_BIAS =  8000.0   # small nudge: easily overridden by blue color + bonus
    METAL_BASE_BIAS = 30000.0   # metal is genuinely rare in top-down aerial

    cost = np.zeros((len(cluster_rgbs), len(material_colors)), dtype=np.float64)
    for i, rgb in enumerate(cluster_rgbs):
        _max_c = max(rgb[0], rgb[1], rgb[2])
        _min_c = min(rgb[0], rgb[1], rgb[2])
        _csat = (_max_c - _min_c) / max(_max_c, 1)
        _min_anchor_d2 = min(
            ((rgb[0] - ar) ** 2 + (rgb[1] - ag) ** 2 + (rgb[2] - ab) ** 2)
            for (ar, ag, ab) in _ASPHALT_ANCHORS
        )
        _near_asphalt_anchor = _min_anchor_d2 <= (34 * 34)
        # Direct distance to BM_ASPHALT vs all vegetation RGB values
        _d2_asphalt = ((rgb[0] - _BM_ASPHALT_RGB[0]) ** 2 +
                       (rgb[1] - _BM_ASPHALT_RGB[1]) ** 2 +
                       (rgb[2] - _BM_ASPHALT_RGB[2]) ** 2)
        _d2_veg = ((rgb[0] - _BM_VEGETATION_RGB[0]) ** 2 +
                   (rgb[1] - _BM_VEGETATION_RGB[1]) ** 2 +
                   (rgb[2] - _BM_VEGETATION_RGB[2]) ** 2)
        _d2_land_grass = ((rgb[0] - _BM_LAND_GRASS_RGB[0]) ** 2 +
                          (rgb[1] - _BM_LAND_GRASS_RGB[1]) ** 2 +
                          (rgb[2] - _BM_LAND_GRASS_RGB[2]) ** 2)
        _d2_dry_grass = ((rgb[0] - _BM_LAND_DRY_GRASS_RGB[0]) ** 2 +
                         (rgb[1] - _BM_LAND_DRY_GRASS_RGB[1]) ** 2 +
                         (rgb[2] - _BM_LAND_DRY_GRASS_RGB[2]) ** 2)
        _closer_to_asphalt = _d2_asphalt < _d2_veg
        _closer_to_asphalt_vs_land_grass = _d2_asphalt < _d2_land_grass
        _closer_to_asphalt_vs_dry_grass = _d2_asphalt < _d2_dry_grass
        # How close is this cluster to the nearest vegetation anchor?
        _min_d2_veg = min(_d2_veg, _d2_land_grass, _d2_dry_grass)
        # Green-proximity score: 1.0 when cluster sits right on a veg colour,
        # fading to 0.0 at squared-distance 25000 (~158 RGB units away).
        _green_proximity = max(0.0, 1.0 - _min_d2_veg / 25000.0)
        # Continuous neutrality score: 1.0 = pure gray, 0.0 = sat >= 0.28.
        # Used to scale vegetation penalty and asphalt reward smoothly so that
        # even slightly warm gray roads (worn asphalt with dust) still bias toward roads.
        _neutrality = max(0.0, 1.0 - _csat / 0.28)
        _gray_veg_extra  = _neutrality * _neutrality * 18000.0   # up to +18K for pure gray  (was 25K)
        _gray_road_bonus = _neutrality * _neutrality * 12000.0   # up to -12K for pure gray
        # Green-channel dominance: G clearly higher than R and B -> tree / vegetation.
        # This catches dark foliage (olive, dark-green) even when _green_proximity is moderate.
        _green_dominant = (rgb[1] > rgb[0] * 1.12) and (rgb[1] > rgb[2] * 1.15) and (rgb[1] > 30)
        # Binary helpers for soil/warm guards (unchanged logic)
        _is_gray_cluster = (_csat < 0.18) and (20 <= _max_c <= 210)
        _is_warm_cluster = (rgb[0] > rgb[2] * 1.18) and (_max_c > 40)
        for j, mat_rgb in enumerate(material_colors):
            base = (
                (rgb[0] - mat_rgb[0]) ** 2
                + (rgb[1] - mat_rgb[1]) ** 2
                + (rgb[2] - mat_rgb[2]) ** 2
            )
            if material_names[j] in _VEG_NAMES:
                # Gray penalty: achromatic clusters are penalised against vegetation.
                # Non-gray clusters only get the small base bias.
                base += VEG_BASE_BIAS + _gray_veg_extra
                # GREEN PROXIMITY BONUS: clusters whose mean RGB is close to any
                # vegetation anchor get a discount that cancels the base bias and
                # pulls them toward the matching vegetation class.  This ensures
                # genuinely green clusters are assigned to vegetation.
                if _green_proximity > 0.0:
                    base = max(0.0, base - _green_proximity * 30000.0)   # was 25K
                # GREEN-CHANNEL DOMINANCE BONUS: trees often have G > R and G > B
                # even when not super-saturated.  Discount vegetation classes.
                if _green_dominant:
                    base = max(0.0, base - 35000.0)   # was 30K
                # Gray road-anchors: penalty only for genuinely achromatic clusters.
                if _near_asphalt_anchor and _neutrality >= 0.45:
                    base += 50000.0
                # Closer-to-asphalt penalty: only fire for gray/neutral clusters
                # to avoid penalising warm/brown clusters that happen to be
                # numerically closer to (45,45,48) than to vegetation greens.
                if _closer_to_asphalt and _neutrality >= 0.40:
                    base += 60000.0
                # Extra block for BM_VEGETATION and BM_FOLIAGE when clearly gray.
                if material_names[j] in {"BM_VEGETATION", "BM_FOLIAGE"} and _neutrality >= 0.55:   # was 0.50
                    base += 50000.0   # was 60K
                # Grass classes when cluster is both closer-to-asphalt AND gray.
                if material_names[j] == "BM_LAND_GRASS" and _closer_to_asphalt_vs_land_grass and _neutrality >= 0.45:   # was 0.40
                    base += 40000.0   # was 50K
                if material_names[j] == "BM_LAND_DRY_GRASS" and _closer_to_asphalt_vs_dry_grass and _neutrality >= 0.45:   # was 0.40
                    base += 40000.0   # was 50K
            elif material_names[j] in _WATER_NAMES:
                base += WATER_BASE_BIAS
            elif material_names[j] in _METAL_NAMES:
                base += METAL_BASE_BIAS
            # Gray cluster -> reward asphalt/concrete/paint_asphalt directly.
            # Subtracting from the base cost makes road materials more likely to win
            # over any vegetation class whose color happens to be numerically closer.
            if material_names[j] in _ROADLIKE_NAMES:
                base = max(0.0, base - _gray_road_bonus)
                # Road-anchor bonus only for achromatic clusters.
                if _near_asphalt_anchor and _neutrality >= 0.45:
                    base = max(0.0, base - 20000.0)
                # BM_ASPHALT bonus when cluster is gray AND closer to asphalt.
                if material_names[j] == "BM_ASPHALT" and _closer_to_asphalt and _neutrality >= 0.35:
                    base = max(0.0, base - 25000.0)
                # Penalise road for warm clusters (earthy surface, not pavement).
                if _is_warm_cluster:
                    base += 15000.0
                # Penalise road for green-proximate clusters.
                # A cluster near vegetation RGB is unlikely to be a road surface.
                if _green_proximity >= 0.15:
                    base += _green_proximity * 18000.0
                # Green-channel dominance: tree canopy -> strongly penalise all road.
                if _green_dominant:
                    base += 25000.0
            # Neutral/cool cluster -> penalise soil/sand materials (they are warm).
            if _is_gray_cluster and material_names[j] in _SOIL_NAMES:
                base += 20000.0
            # Extra block: bright grass class should not win for clearly gray anchors.
            if _near_asphalt_anchor and material_names[j] == "BM_LAND_GRASS" and _neutrality >= 0.50:
                base += 40000.0
            cost[i, j] = base

    # Optional prevalence-aware bias:
    # match large clusters to materials expected to be common in the scene.
    if cluster_counts is not None and len(cluster_counts) == len(cluster_rgbs):
        total_count = float(sum(cluster_counts))
        if total_count > 0:
            cluster_freq_arr = np.array(cluster_counts, dtype=np.float64) / total_count
            prior_source = material_prior if material_prior else MEA_MATERIAL_FREQUENCY_PRIOR
            prior_raw = np.array(
                [prior_source.get(name, 0.05) for name in material_names],
                dtype=np.float64,
            )
            prior_sum = float(np.sum(prior_raw))
            if prior_sum > 0:
                material_prior_arr = prior_raw / prior_sum
                color_norm = cost / 195075.0  # max squared RGB distance: 255^2 * 3
                prevalence_penalty = np.abs(cluster_freq_arr[:, None] - material_prior_arr[None, :])
                cost = ((1.0 - MEA_PREVALENCE_WEIGHT) * color_norm) + (MEA_PREVALENCE_WEIGHT * prevalence_penalty)

    if cluster_semantics is not None and len(cluster_semantics) == len(cluster_rgbs):
        # Always normalize cost to [0-1] before mixing with semantic penalties (same scale).
        _cost_max = float(np.max(cost)) if np.max(cost) > 0.0 else 1.0
        cost = cost / _cost_max

        vegetation_materials = {"BM_LAND_GRASS", "BM_LAND_DRY_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
        lush_vegetation_materials = {"BM_LAND_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
        dry_vegetation_materials = {"BM_LAND_DRY_GRASS"}
        road_materials = {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
        asphalt_materials = {"BM_ASPHALT", "BM_PAINT_ASPHALT"}
        paint_materials = {"BM_PAINT_ASPHALT"}
        water_materials = {"BM_WATER"}
        soil_materials = {"BM_SOIL"}

        sem_penalty = np.zeros_like(cost, dtype=np.float64)
        for i, sem in enumerate(cluster_semantics):
            veg_score = float(sem.get("veg", 0.0))
            road_score = float(sem.get("road", 0.0))
            water_score = float(sem.get("water", 0.0))
            asphalt_score = float(sem.get("asphalt", 0.0))
            line_score = float(sem.get("line", 0.0))
            water_conf = float(sem.get("water_conf", 0.0))
            dry_score = float(sem.get("dry", 0.0))
            sand_score = float(sem.get("sand", 0.0))
            grass_score = float(sem.get("grass", 0.0))
            gray_frac = float(sem.get("gray_frac", 0.0))
            dark_gray_frac = float(sem.get("dark_gray_frac", 0.0))
            achro_frac = float(sem.get("achro_frac", 0.0))
            warm_frac = float(sem.get("warm_frac", 0.0))
            size_frac = float(sem.get("size_frac", 0.0))
            blue_dom_frac = float(sem.get("blue_dom_frac", 0.0))
            for j, name in enumerate(material_names):
                target_veg = 1.0 if name in vegetation_materials else 0.0
                target_road = 1.0 if name in road_materials else 0.0
                target_asphalt = 1.0 if name in asphalt_materials else 0.0
                target_line = 1.0 if name in paint_materials else (0.30 if name == "BM_ASPHALT" else 0.15 if name == "BM_CONCRETE" else 0.0)
                target_water = 1.0 if name in water_materials else 0.0
                target_dry = 1.0 if (name in dry_vegetation_materials or name in soil_materials or name == "BM_ROCK") else 0.0
                target_sand = 1.0 if name == "BM_SAND" else 0.0
                pen = (
                    (0.70 * abs(veg_score - target_veg))
                    + (1.05 * abs(road_score - target_road))
                    + (0.95 * abs(asphalt_score - target_asphalt))
                    + (0.75 * abs(line_score - target_line))
                    + (1.35 * abs(water_score - target_water))
                    + (0.95 * abs(dry_score - target_dry))
                    + (0.90 * abs(sand_score - target_sand))
                )

                # Water gating: road-like clusters should not become water.
                if name == "BM_WATER":
                    cluster_freq_val = float(cluster_freq_arr[i]) if cluster_freq_arr is not None else 0.0
                    if road_score >= 0.40:
                        pen += 2.5
                    elif road_score >= 0.25:
                        pen += 1.2
                    if water_score < 0.25:
                        pen += 2.0
                    if cluster_freq_val < 0.02:
                        pen += 1.0
                    # Gate: require at least some water confidence.
                    if water_conf < 0.20:
                        pen += 2.5
                    if water_conf < 0.12:
                        pen += 3.5
                    # Blue-channel gate: water bodies have blue-dominant pixels.
                    # Low blue_dom + low confidence usually means shadow or dark-vegetated
                    # area rather than an actual water body.
                    if blue_dom_frac < 0.08 and water_conf < 0.40:
                        pen += 3.0
                    elif blue_dom_frac < 0.15 and water_conf < 0.30:
                        pen += 1.5
                    # Cluster-size gate: very small fragments are unlikely water bodies.
                    if size_frac < 0.005 and water_conf < 0.45:
                        pen += 2.0
                    elif size_frac < 0.015 and water_conf < 0.35:
                        pen += 1.0
                    # Warm-tone guard: warm-toned clusters (r >> b) are soil/earth,
                    # not water bodies.  Brown earth fields form large compact blobs
                    # that inflate water_cont, so we need explicit warm-tone exclusion.
                    if warm_frac >= 0.40:
                        pen += 5.0   # strongly warm -> definitely not water
                    elif warm_frac >= 0.28:
                        pen += 3.0
                    elif warm_frac >= 0.18:
                        pen += 1.5
                    # Dry-score guard: dry/barren clusters are earth/soil, not water.
                    if dry_score >= 0.35:
                        pen += 3.5
                    elif dry_score >= 0.20:
                        pen += 1.8
                    # Compound: warm + dry = classic soil, never water.
                    if warm_frac >= 0.20 and dry_score >= 0.15:
                        pen += 2.5
                    # Sand-score guard: sandy clusters should not become water.
                    if sand_score >= 0.30:
                        pen += 2.0
                    # Ambiguity tie-breaker: when vegetation signal is present alongside
                    # a moderate water signal, prefer vegetation over water.
                    # Misclassifying vegetation as water is worse than the reverse.
                    if veg_score >= 0.15 and water_score < 0.50:
                        pen += 1.5
                    if veg_score >= 0.25 and water_score < 0.65:
                        pen += 1.5   # stacks: +3.0 for moderate veg + non-dominant water

                # Vegetation preferred over water when signals are ambiguous.
                if name in vegetation_materials and water_score >= 0.20 and veg_score >= 0.10:
                    pen -= 0.8   # slight bias: vegetation is more common than water bodies
                if name in vegetation_materials:
                    if veg_score < 0.04:
                        pen += 2.0   # was 2.5 at < 0.06
                    elif veg_score < 0.10:
                        pen += 0.4   # was 0.6 at < 0.15
                    if road_score >= 0.45 and veg_score < 0.30:
                        pen += 2.0   # was 2.5 at road >= 0.40
                    if asphalt_score >= 0.40 and veg_score < 0.30:
                        pen += 2.0   # was 2.5 at asphalt >= 0.35
                    if road_score >= 0.35 and gray_frac >= 0.25 and veg_score < 0.20:
                        pen += 2.5   # was 3.5 at road >= 0.30, gray >= 0.18, veg < 0.25
                    if asphalt_score >= 0.30 and gray_frac >= 0.25 and veg_score < 0.20:
                        pen += 2.5   # was 3.5 at asphalt >= 0.25, gray >= 0.18, veg < 0.25
                    # Achromatic fraction: strong indicator of road/building, not vegetation
                    if achro_frac >= 0.45 and veg_score < 0.20:
                        pen += 4.0
                    elif achro_frac >= 0.30 and veg_score < 0.15:
                        pen += 2.5
                    elif achro_frac >= 0.18 and veg_score < 0.10:
                        pen += 1.5
                    if water_score >= 0.50:
                        pen += 2.0
                    if dry_score >= 0.50 and name in lush_vegetation_materials:
                        pen += 1.5
                    # Gray-toned clusters: tiered gate - achromatic -> not vegetation.
                    # BUT: dampen when veg_score is moderate-to-high (tree clusters
                    # often have shadow pixels that inflate gray_frac).
                    _gray_dampen = 1.0
                    if veg_score >= 0.35:
                        _gray_dampen = 0.10   # strong veg signal -> almost ignore gray_frac (was 0.15 at 0.40)
                    elif veg_score >= 0.20:
                        _gray_dampen = 0.30   # moderate veg -> strongly reduce (was 0.40 at 0.25)
                    elif veg_score >= 0.12:
                        _gray_dampen = 0.55   # weak veg -> reduce (was 0.70 at 0.15)
                    if gray_frac >= 0.55:
                        pen += 4.0 * _gray_dampen   # was 5.0 at 0.50
                    elif gray_frac >= 0.40:
                        pen += 2.5 * _gray_dampen   # was 3.5 at 0.35
                    elif gray_frac >= 0.25:
                        pen += 1.5 * _gray_dampen   # was 2.0 at 0.20
                    elif gray_frac >= 0.15:
                        pen += 0.5 * _gray_dampen   # was 0.8 at 0.12
                    # Dark gray confirms road/asphalt, not vegetation.
                    if dark_gray_frac >= 0.45:
                        pen += 5.0 * _gray_dampen   # was 6.0 at 0.40
                    elif dark_gray_frac >= 0.30:
                        pen += 3.0 * _gray_dampen   # was 4.0 at 0.25
                    elif dark_gray_frac >= 0.15:
                        pen += 1.0 * _gray_dampen   # was 1.5 at 0.12
                    # Dark gray + asphalt signal -> block vegetation
                    if dark_gray_frac >= 0.25 and asphalt_score >= 0.20 and veg_score < 0.20:
                        pen += 3.0   # was 4.0 at dark_gray >= 0.20, asphalt >= 0.15, veg < 0.25
                    # Combined: gray + asphalt signal -> push away from vegetation.
                    if gray_frac >= 0.30 and asphalt_score >= 0.20 and veg_score < 0.20:
                        pen += 2.0   # was 2.5 at gray >= 0.25, asphalt >= 0.15, veg < 0.25
                    # Gray cluster + low veg -> block vegetation.
                    if gray_frac >= 0.25 and veg_score < 0.10 and grass_score < 0.08:
                        pen += 2.5   # was 3.0 at gray >= 0.20, veg < 0.15, grass < 0.10
                    if gray_frac >= 0.40 and veg_score < 0.08:
                        pen += 2.5   # was 3.0 at gray >= 0.35, veg < 0.10
                    if gray_frac >= 0.22 and (road_score >= 0.35 or asphalt_score >= 0.28) and veg_score < 0.15:
                        pen += 2.0   # was 2.5 at gray >= 0.18, road >= 0.28 or asphalt >= 0.22, veg < 0.20
                    # BM_LAND_GRASS: penalise for gray clusters (only when veg_score is low).
                    if name == "BM_LAND_GRASS" and (gray_frac >= 0.25 or dark_gray_frac >= 0.15) and veg_score < 0.20:
                        pen += 3.0   # was 4.0 at gray >= 0.20 or dark_gray >= 0.12, veg < 0.25
                    if name == "BM_LAND_GRASS" and (road_score >= 0.35 or asphalt_score >= 0.30) and veg_score < 0.15:
                        pen += 2.5   # was 3.5 at road >= 0.30 or asphalt >= 0.25, veg < 0.20
                    # BM_LAND_DRY_GRASS: penalise for gray clusters (only when veg_score is low).
                    if name == "BM_LAND_DRY_GRASS" and (gray_frac >= 0.25 or dark_gray_frac >= 0.15) and veg_score < 0.20:
                        pen += 2.5   # was 3.5 at gray >= 0.20 or dark_gray >= 0.12, veg < 0.25
                    if name == "BM_LAND_DRY_GRASS" and (road_score >= 0.35 or asphalt_score >= 0.30) and veg_score < 0.15:
                        pen += 2.0   # was 3.0 at road >= 0.30 or asphalt >= 0.25, veg < 0.20
                    # BM_VEGETATION / BM_FOLIAGE: penalise for gray/road clusters
                    # ONLY when veg evidence is weak.
                    if name in {"BM_VEGETATION", "BM_FOLIAGE"}:
                        if (gray_frac >= 0.22 or dark_gray_frac >= 0.15) and veg_score < 0.15:
                            pen += 3.5   # was 4.5 at gray >= 0.18 or dark_gray >= 0.10, veg < 0.20
                        if (road_score >= 0.30 or asphalt_score >= 0.25) and veg_score < 0.20:
                            pen += 3.0   # was 4.0 at road >= 0.25 or asphalt >= 0.20, veg < 0.25
                        if gray_frac >= 0.30 and (road_score >= 0.28 or asphalt_score >= 0.22) and veg_score < 0.15:
                            pen += 3.0   # was 4.0 at gray >= 0.25, road >= 0.22 or asphalt >= 0.18, veg < 0.20
                if name in dry_vegetation_materials and veg_score >= 0.62 and dry_score < 0.20:
                    pen += 0.8
                # Vegetation bonus: awarded when cluster is genuinely green.
                # Tiered: stronger bonus for higher veg confidence.
                # Boosted from prior values to counter-balance reduced penalties.
                if name in vegetation_materials and veg_score >= 0.08 and gray_frac < 0.40:
                    pen -= 3.0    # NEW lower tier (was 2.5 at 0.12 / gray < 0.35)
                if name in vegetation_materials and veg_score >= 0.15 and gray_frac < 0.35:
                    pen -= 4.5    # was 4.0 at 0.20 / gray < 0.30
                if name in vegetation_materials and veg_score >= 0.30 and gray_frac < 0.30:
                    pen -= 5.0    # was 4.0 at 0.35 / gray < 0.25
                if name in vegetation_materials and veg_score >= 0.45 and gray_frac < 0.25:
                    pen -= 4.5    # was 3.5 at 0.50 / gray < 0.20
                if name in vegetation_materials and veg_score >= 0.60 and gray_frac < 0.20:
                    pen -= 4.0    # was 3.0 at 0.65 / gray < 0.15
                # stacks: up to -21.0 for very green, non-gray clusters (was -17.0)
                # Grass bonus: strong lush-grass signal confirms grass assignment.
                if name in vegetation_materials and grass_score >= 0.06 and gray_frac < 0.30:
                    pen -= 3.5    # was 3.0 at 0.08 / gray < 0.25
                if name in vegetation_materials and grass_score >= 0.15 and gray_frac < 0.25:
                    pen -= 3.5    # was 3.0 at 0.20 / gray < 0.20
                if name in vegetation_materials and grass_score >= 0.30 and gray_frac < 0.20:
                    pen -= 3.0    # was 2.5 at 0.35 / gray < 0.15
                # stacks: up to -10.0 for strong grass signal (was -8.5)

                # ── Asphalt guard: penalise asphalt when cluster has green signal ──
                if name in asphalt_materials:
                    if veg_score >= 0.35:
                        pen += 4.0   # clearly vegetation-green -> not asphalt
                    elif veg_score >= 0.22:
                        pen += 2.5
                    elif veg_score >= 0.12:
                        pen += 1.0   # even low veg signal -> mild asphalt doubt
                    if grass_score >= 0.20:
                        pen += 2.5   # grass pixels in cluster -> not asphalt
                    if grass_score >= 0.35:
                        pen += 2.5   # stacks: +5.0 for very grassy cluster
                    # Gray cluster: tiered reward for asphalt assignment.
                    if gray_frac >= 0.50:
                        pen -= 2.5
                    elif gray_frac >= 0.35:
                        pen -= 1.8
                    elif gray_frac >= 0.20:
                        pen -= 1.2
                    elif gray_frac >= 0.12:
                        pen -= 0.5
                    # Dark gray confirms asphalt.
                    if dark_gray_frac >= 0.40:
                        pen -= 4.0   # very strong dark gray -> almost certainly asphalt
                    elif dark_gray_frac >= 0.20:
                        pen -= 2.5
                    elif dark_gray_frac >= 0.10:
                        pen -= 1.0
                    # Dark gray + low warm: asphalt signal.
                    if dark_gray_frac >= 0.25 and warm_frac < 0.15:
                        pen -= 2.5
                    # Gray cluster with weak vegetation -> prefer asphalt.
                    # GUARD: only reward when veg signal is truly weak.
                    if gray_frac >= 0.20 and veg_score < 0.15 and grass_score < 0.10:
                        pen -= 2.0
                    if gray_frac >= 0.35 and veg_score < 0.10 and grass_score < 0.08:
                        pen -= 2.0
                    if gray_frac >= 0.18 and (road_score >= 0.28 or asphalt_score >= 0.22) and veg_score < 0.18:
                        pen -= 1.8
                    if name == "BM_ASPHALT" and (gray_frac >= 0.20 or dark_gray_frac >= 0.12) and veg_score < 0.18:
                        pen -= 2.0
                    if name == "BM_ASPHALT" and (road_score >= 0.28 or asphalt_score >= 0.22) and veg_score < 0.20:
                        pen -= 1.8
                    # Achromatic fraction: strong asphalt/concrete signal
                    if achro_frac >= 0.45 and veg_score < 0.20:
                        pen -= 3.5
                    elif achro_frac >= 0.30 and veg_score < 0.15:
                        pen -= 2.5
                    elif achro_frac >= 0.18 and veg_score < 0.12:
                        pen -= 1.5

                # ── Sand-vs-asphalt guard: penalise asphalt if cluster looks warm/sandy ──
                if name in asphalt_materials and sand_score >= 0.35:
                    pen += 1.8
                if name in asphalt_materials and sand_score >= 0.55:
                    pen += 1.5   # stacks: +3.3 for very sandy clusters
                # ── Warm-tone guard: warm clusters -> penalise asphalt/concrete ──
                # Concrete and asphalt are neutral/cool gray; warm-toned clusters are soil/sand.
                if name in asphalt_materials or name == "BM_CONCRETE":
                    if warm_frac >= 0.55:
                        pen += 5.0   # strongly warm -> definitely not road
                    elif warm_frac >= 0.40:
                        pen += 3.5
                    elif warm_frac >= 0.25:
                        pen += 2.0
                    elif warm_frac >= 0.12:
                        pen += 0.8
                    # Compound: warm AND low gray -> earth, not asphalt
                    if warm_frac >= 0.20 and gray_frac < 0.20:
                        pen += 2.5

                # ── Sand guard: penalise sand when cluster is dark/achromatic (-> asphalt) ──
                if name == "BM_SAND" and dark_gray_frac >= 0.30:
                    pen += 2.5   # dark gray cluster is asphalt, not sand
                if name == "BM_SAND" and asphalt_score >= 0.40:
                    pen += 1.8
                # Sand requires warm tone; neutral/cool gray cluster is concrete, not sand.
                if name == "BM_SAND":
                    if gray_frac >= 0.40 and warm_frac < 0.20:
                        pen += 4.0   # achromatic cluster -> not sand
                    elif gray_frac >= 0.25 and warm_frac < 0.20:
                        pen += 2.0
                    if warm_frac >= 0.35:
                        pen -= 1.0   # bonus: warm cluster confirms sand possibility

                # ── Soil/Earthen guard: warm required, penalise when neutral/gray ──
                # Soil and earthen are inherently warm-brown; a gray-dominant cluster is
                # concrete/asphalt, never soil.
                if name in soil_materials:
                    if road_score >= 0.52:
                        pen += 1.4
                    if asphalt_score >= 0.48:
                        pen += 1.3
                    if line_score >= 0.45:
                        pen += 0.8
                    # Neutral/gray cluster -> strongly penalise soil/earthen.
                    if gray_frac >= 0.45 and warm_frac < 0.15:
                        pen += 5.0
                    elif gray_frac >= 0.30 and warm_frac < 0.20:
                        pen += 3.0
                    elif gray_frac >= 0.20 and warm_frac < 0.20:
                        pen += 1.5
                    # Dark gray directly confirms asphalt/concrete, NEVER soil.
                    # BM_SOIL (RGB 101,67,33) is warm-brown;
                    # a dark-gray cluster (e.g. worn asphalt) has none of those warm tones.
                    if dark_gray_frac >= 0.35:
                        pen += 5.0
                    elif dark_gray_frac >= 0.20:
                        pen += 3.0
                    elif dark_gray_frac >= 0.10:
                        pen += 1.5
                    # Dark gray + low warm = road surface; strongest combined soil exclusion.
                    if dark_gray_frac >= 0.20 and warm_frac < 0.15:
                        pen += 3.0
                    # Achromatic fraction: soil is warm, not achromatic
                    if achro_frac >= 0.35 and warm_frac < 0.15:
                        pen += 4.0
                    elif achro_frac >= 0.20 and warm_frac < 0.20:
                        pen += 2.0
                    # Compound: high gray + low dry -> road surface, not earth.
                    if gray_frac >= 0.25 and dry_score < 0.15:
                        pen += 2.5
                    # Warm cluster with dry signal -> reward.
                    if warm_frac >= 0.30 and dry_score >= 0.20:
                        pen -= 1.5
                    elif warm_frac >= 0.20:
                        pen -= 0.6

                # Vegetation penalized for clear water-like signatures.
                if name in vegetation_materials and water_score >= 0.55:
                    pen += 1.5
                if name in lush_vegetation_materials and water_conf >= 0.40:
                    pen += 1.5

                # Strong preference bonuses for solid materials when semantics agree.
                if name in asphalt_materials and road_score >= 0.30:
                    pen -= 1.4
                if name in asphalt_materials and asphalt_score >= 0.30:
                    pen -= 1.2
                if name == "BM_CONCRETE" and road_score >= 0.30:
                    pen -= 1.1
                if name == "BM_CONCRETE" and road_score >= 0.50:
                    pen -= 1.0   # stacks: -2.1 for strong road signal
                if name == "BM_CONCRETE" and gray_frac >= 0.50 and veg_score < 0.15:
                    pen -= 2.0
                elif name == "BM_CONCRETE" and gray_frac >= 0.35 and veg_score < 0.18:
                    pen -= 1.5
                elif name == "BM_CONCRETE" and gray_frac >= 0.20 and veg_score < 0.20:
                    pen -= 1.0   # concrete roads are light-gray
                # Concrete also benefits from dark_gray (shaded concrete).
                if name == "BM_CONCRETE" and dark_gray_frac >= 0.15 and veg_score < 0.15:
                    pen -= 0.6
                # Gray cluster with weak vegetation -> prefer concrete.
                # GUARD: only reward when veg/grass signals are truly absent.
                if name == "BM_CONCRETE" and gray_frac >= 0.20 and veg_score < 0.15 and grass_score < 0.10:
                    pen -= 2.0
                if name == "BM_CONCRETE" and gray_frac >= 0.35 and veg_score < 0.10 and grass_score < 0.08:
                    pen -= 2.0
                # Achromatic fraction: concrete bonus
                if name == "BM_CONCRETE" and achro_frac >= 0.35 and veg_score < 0.15:
                    pen -= 2.0
                elif name == "BM_CONCRETE" and achro_frac >= 0.20 and veg_score < 0.12:
                    pen -= 1.2
                # Concrete guard: penalise when cluster has green/veg signal
                if name == "BM_CONCRETE" and veg_score >= 0.25:
                    pen += 3.0
                if name == "BM_CONCRETE" and veg_score >= 0.40:
                    pen += 2.5   # stacks: +5.5 for clearly green cluster -> not concrete
                if name == "BM_CONCRETE" and grass_score >= 0.15:
                    pen += 2.0
                if name == "BM_CONCRETE" and grass_score >= 0.30:
                    pen += 2.0   # stacks: +4.0 for grassy cluster -> not concrete
                if name in paint_materials and line_score >= 0.30:
                    pen -= 1.0
                if name in soil_materials and dry_score >= 0.30:
                    pen -= 1.2
                if name in soil_materials and warm_frac >= 0.30 and dry_score >= 0.20:
                    pen -= 0.8   # extra bonus: warm + dry = classic soil signal
                if name == "BM_ROCK" and dry_score >= 0.30:
                    pen -= 1.0
                # BM_ROCK (130,123,115): use as neutral road-like fallback for gray asphalt
                # shades that are not strongly saturated/green.
                if name == "BM_ROCK" and gray_frac >= 0.18:
                    pen -= 2.0
                if name == "BM_ROCK" and dark_gray_frac >= 0.12:
                    pen -= 1.4
                if name == "BM_ROCK" and road_score >= 0.28:
                    pen -= 1.6
                if name == "BM_ROCK" and asphalt_score >= 0.22:
                    pen -= 1.4
                if name == "BM_ROCK" and gray_frac >= 0.18 and veg_score < 0.35:
                    pen -= 2.2
                if name == "BM_ROCK" and veg_score >= 0.45 and gray_frac < 0.15:
                    pen += 2.0
                # Water bonus when there is any water confidence.
                if name in water_materials and water_conf >= 0.22:
                    pen -= 1.5
                if name in water_materials and water_conf >= 0.45:
                    pen -= 1.5  # stacks: -3.0 for clearly water clusters
                # Metal penalty.
                if name in {"BM_METAL", "BM_METAL_STEEL"}:
                    pen += 2.5

                # Sand vs vegetation discrimination.
                if name == "BM_SAND":
                    if sand_score >= 0.35:
                        pen -= 1.2
                    if sand_score >= 0.50:
                        pen -= 1.2   # stacks: -2.4
                    if sand_score >= 0.65:
                        pen -= 0.8   # stacks: -3.2 for clearly sandy clusters
                    if grass_score >= 0.20:
                        pen += 2.5   # green grass cluster should never be sand
                    elif veg_score >= 0.25:
                        pen += 2.0
                if name in {"BM_LAND_DRY_GRASS", "BM_LAND_GRASS"}:
                    # Both grass classes should be penalised when the cluster is sandy
                    if sand_score >= 0.45 and sand_score > dry_score:
                        pen += 1.8
                    elif sand_score >= 0.35:
                        pen += 0.9
                    # Grass classes need real green signal to be assigned
                    if name == "BM_LAND_GRASS" and grass_score < 0.08:
                        pen += 1.5   # very little lush green -> unlikely to be live grass
                    if name == "BM_LAND_DRY_GRASS" and grass_score < 0.04 and sand_score >= 0.30:
                        pen += 1.0   # dry + sandy -> push toward BM_SAND
                    # Achromatic (gray/white) surface = building / road, definitely not grass.
                    # gray_frac > 0.30 means more than 30 % of pixels are low-saturation.
                    if gray_frac >= 0.55 and veg_score < 0.15:
                        pen += 4.0   # very achromatic -> structural surface, not grass
                    elif gray_frac >= 0.35 and veg_score < 0.15:
                        pen += 2.5
                    elif gray_frac >= 0.20 and veg_score < 0.10:
                        pen += 1.5
                    # Achromatic fraction: grass classes need color, not gray
                    if achro_frac >= 0.40 and veg_score < 0.15:
                        pen += 3.5
                    elif achro_frac >= 0.25 and veg_score < 0.10:
                        pen += 2.0

                # Cross-guard: strong water should not be road/vegetation.
                if water_score >= 0.55 and (name in vegetation_materials or name in road_materials):
                    pen += 1.5

                sem_penalty[i, j] = pen

        cost = ((1.0 - MEA_SEMANTIC_WEIGHT) * cost) + (MEA_SEMANTIC_WEIGHT * sem_penalty)

    # Many-to-one assignment: each cluster picks its lowest-cost material
    # independently.  This allows multiple clusters to share the same
    # material (e.g. two road clusters can both get BM_ASPHALT).
    col_ind = np.argmin(cost, axis=1)  # shape (n_clusters,)
    for i, j in enumerate(col_ind):
        assigned_rgb = material_colors[int(j)]
        assigned[i] = assigned_rgb
        mapping[i] = {
            "cluster": i + 1,
            "material": material_names[int(j)],
            "colorHex": material_hex[int(j)],
            "colorRGB": assigned_rgb,
        }

    # Final semantic guardrails to reduce severe confusions.
    if cluster_semantics is not None and len(cluster_semantics) == len(cluster_rgbs):
        vegetation_materials = {"BM_LAND_GRASS", "BM_LAND_DRY_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
        lush_vegetation_materials = {"BM_LAND_GRASS", "BM_VEGETATION", "BM_FOLIAGE"}
        soil_materials = {"BM_SOIL"}
        water_idx = next((idx for idx, n in enumerate(material_names) if n == "BM_WATER"), None)
        road_candidate_idxs = [
            idx for idx, n in enumerate(material_names)
            if n in {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT"}
        ]
        dry_ground_candidate_idxs = [
            idx for idx, n in enumerate(material_names)
            if n in {"BM_LAND_DRY_GRASS", "BM_SOIL", "BM_ROCK"}
        ]

        for i, sem in enumerate(cluster_semantics):
            veg_score = float(sem.get("veg", 0.0))
            road_score = float(sem.get("road", 0.0))
            water_score = float(sem.get("water", 0.0))
            asphalt_score = float(sem.get("asphalt", 0.0))
            line_score = float(sem.get("line", 0.0))
            water_conf = float(sem.get("water_conf", 0.0))
            dry_score = float(sem.get("dry", 0.0))
            sand_score = float(sem.get("sand", 0.0))
            dark_gray_frac = float(sem.get("dark_gray_frac", 0.0))
            current_mat = str(mapping[i].get("material", "UNKNOWN"))
            road_like = max(road_score, asphalt_score, line_score)
            water_like = (water_conf >= 0.28) or (
                water_score >= 0.45 and road_like < 0.35 and veg_score < 0.45
            )

            # Strong road/asphalt cues should never end up as water.
            if current_mat == "BM_WATER" and (road_score >= 0.40 or asphalt_score >= 0.38):
                if road_candidate_idxs:
                    rgb = cluster_rgbs[i]
                    best_idx = min(
                        road_candidate_idxs,
                        key=lambda idx: (
                            (rgb[0] - material_colors[idx][0]) ** 2
                            + (rgb[1] - material_colors[idx][1]) ** 2
                            + (rgb[2] - material_colors[idx][2]) ** 2
                        ),
                    )
                    assigned[i] = material_colors[best_idx]
                    mapping[i] = {
                        "cluster": i + 1,
                        "material": material_names[best_idx],
                        "colorHex": material_hex[best_idx],
                        "colorRGB": material_colors[best_idx],
                    }
                    current_mat = str(mapping[i].get("material", "UNKNOWN"))
                    continue

            # Water cues should not end up as vegetation.
            if current_mat in (vegetation_materials | soil_materials) and water_like and water_idx is not None:
                assigned[i] = material_colors[water_idx]
                mapping[i] = {
                    "cluster": i + 1,
                    "material": material_names[water_idx],
                    "colorHex": material_hex[water_idx],
                    "colorRGB": material_colors[water_idx],
                }
                current_mat = str(mapping[i].get("material", "UNKNOWN"))

            # Dry/barren clusters should not remain lush vegetation - only when evidence is very clear.
            if current_mat in lush_vegetation_materials and dry_score >= 0.60 and veg_score < 0.22:
                if dry_ground_candidate_idxs:
                    rgb = cluster_rgbs[i]
                    best_idx = min(
                        dry_ground_candidate_idxs,
                        key=lambda idx: (
                            (rgb[0] - material_colors[idx][0]) ** 2
                            + (rgb[1] - material_colors[idx][1]) ** 2
                            + (rgb[2] - material_colors[idx][2]) ** 2
                        ),
                    )
                    assigned[i] = material_colors[best_idx]
                    mapping[i] = {
                        "cluster": i + 1,
                        "material": material_names[best_idx],
                        "colorHex": material_hex[best_idx],
                        "colorRGB": material_colors[best_idx],
                    }
                    current_mat = str(mapping[i].get("material", "UNKNOWN"))

            # Sand ↔ grass/dry-grass post-assignment refinement.
            sand_idx_local = next((idx for idx, n in enumerate(material_names) if n == "BM_SAND"), None)
            dry_grass_idx_local = next((idx for idx, n in enumerate(material_names) if n == "BM_LAND_DRY_GRASS"), None)
            lush_grass_idx_local = next((idx for idx, n in enumerate(material_names) if n == "BM_LAND_GRASS"), None)
            grass_score = float(sem.get("grass", 0.0))

            def _remap_cluster_to(target_idx: int) -> None:
                """Reassign cluster i to the material at target_idx in-place."""
                assigned[i] = material_colors[target_idx]
                mapping[i] = {
                    "cluster": i + 1,
                    "material": material_names[target_idx],
                    "colorHex": material_hex[target_idx],
                    "colorRGB": material_colors[target_idx],
                }

            # Cluster mapped to dry grass but sand signal is clearly stronger -> remap to sand.
            if current_mat == "BM_LAND_DRY_GRASS" and sand_score >= 0.50 and sand_score > (dry_score + 0.12) and veg_score < 0.22 and grass_score < 0.15:
                if sand_idx_local is not None:
                    _remap_cluster_to(sand_idx_local)
                    current_mat = "BM_SAND"

            # Cluster mapped to lush grass but the cluster has almost no green signal
            # and sand is clearly dominant -> remap to sand (desert misclassified as grass).
            if current_mat == "BM_LAND_GRASS" and sand_score >= 0.45 and grass_score < 0.12 and veg_score < 0.20:
                if sand_idx_local is not None:
                    _remap_cluster_to(sand_idx_local)
                    current_mat = "BM_SAND"

            # Cluster mapped to sand but green (grass) signal is significant -> remap to grass.
            if current_mat == "BM_SAND":
                if grass_score >= 0.25 and veg_score >= 0.20:
                    # Strong lush grass evidence - prefer BM_LAND_GRASS
                    target_idx = lush_grass_idx_local if lush_grass_idx_local is not None else dry_grass_idx_local
                    if target_idx is not None:
                        _remap_cluster_to(target_idx)
                        current_mat = material_names[target_idx]
                elif veg_score >= 0.30 and dry_score > sand_score:
                    # Moderate veg + more dry signal - prefer BM_LAND_DRY_GRASS
                    if dry_grass_idx_local is not None:
                        _remap_cluster_to(dry_grass_idx_local)
                        current_mat = "BM_LAND_DRY_GRASS"

            # Hard reject uncertain water: remap water to the best non-water road/ground-like material.
            if current_mat == "BM_WATER" and not water_like:
                non_water_idxs = [idx for idx, n in enumerate(material_names) if n != "BM_WATER"]
                if non_water_idxs:
                    rgb = cluster_rgbs[i]
                    # Prefer asphalt/road/soil families for uncertain water.
                    preferred = [
                        idx for idx, n in enumerate(material_names)
                        if n in {"BM_ASPHALT", "BM_CONCRETE", "BM_PAINT_ASPHALT", "BM_SOIL", "BM_ROCK"}
                    ]
                    candidate_idxs = preferred if preferred else non_water_idxs
                    best_idx = min(
                        candidate_idxs,
                        key=lambda idx: (
                            (rgb[0] - material_colors[idx][0]) ** 2
                            + (rgb[1] - material_colors[idx][1]) ** 2
                            + (rgb[2] - material_colors[idx][2]) ** 2
                        ),
                    )
                    assigned[i] = material_colors[best_idx]
                    mapping[i] = {
                        "cluster": i + 1,
                        "material": material_names[best_idx],
                        "colorHex": material_hex[best_idx],
                        "colorRGB": material_colors[best_idx],
                    }
                    current_mat = str(mapping[i].get("material", "UNKNOWN"))

            # Road/asphalt/line cues should not remain vegetation (needs strong road signal at 0.5m).
            if current_mat in (vegetation_materials | soil_materials) and road_like >= 0.55:
                if road_candidate_idxs:
                    rgb = cluster_rgbs[i]
                    best_idx = min(
                        road_candidate_idxs,
                        key=lambda idx: (
                            (rgb[0] - material_colors[idx][0]) ** 2
                            + (rgb[1] - material_colors[idx][1]) ** 2
                            + (rgb[2] - material_colors[idx][2]) ** 2
                        ),
                    )
                    assigned[i] = material_colors[best_idx]
                    mapping[i] = {
                        "cluster": i + 1,
                        "material": material_names[best_idx],
                        "colorHex": material_hex[best_idx],
                        "colorRGB": material_colors[best_idx],
                    }
                    current_mat = str(mapping[i].get("material", "UNKNOWN"))

            # Asphalt mapped but strong green signal -> remap to vegetation.
            asphalt_idx_local = next((idx for idx, n in enumerate(material_names) if n == "BM_ASPHALT"), None)
            if current_mat in {"BM_ASPHALT", "BM_PAINT_ASPHALT"} and veg_score >= 0.35 and grass_score >= 0.12:
                veg_candidates = [idx for idx, n in enumerate(material_names) if n in vegetation_materials]
                if veg_candidates:
                    rgb = cluster_rgbs[i]
                    best_idx = min(
                        veg_candidates,
                        key=lambda idx: (
                            (rgb[0] - material_colors[idx][0]) ** 2
                            + (rgb[1] - material_colors[idx][1]) ** 2
                            + (rgb[2] - material_colors[idx][2]) ** 2
                        ),
                    )
                    _remap_cluster_to(best_idx)
                    current_mat = material_names[best_idx]

            # Sand -> asphalt guardrail: cluster mapped to sand but is dark-achromatic.
            if current_mat == "BM_SAND" and asphalt_score >= 0.35 and dark_gray_frac >= 0.25 and sand_score < 0.40:
                if asphalt_idx_local is not None:
                    _remap_cluster_to(asphalt_idx_local)
                    current_mat = "BM_ASPHALT"

            # Asphalt -> sand guardrail: cluster mapped to asphalt but is warm/sandy.
            if current_mat == "BM_ASPHALT" and sand_score >= 0.50 and asphalt_score < 0.25:
                if sand_idx_local is not None:
                    _remap_cluster_to(sand_idx_local)
                    current_mat = "BM_SAND"

            # If mapped to vegetation but water evidence is strong, force water.
            if current_mat in vegetation_materials and (water_like or water_conf >= 0.30) and water_idx is not None:
                assigned[i] = material_colors[water_idx]
                mapping[i] = {
                    "cluster": i + 1,
                    "material": material_names[water_idx],
                    "colorHex": material_hex[water_idx],
                    "colorRGB": material_colors[water_idx],
                }

    return mapping, assigned


def _assign_nearest_material_colors(
    cluster_rgbs: List[Tuple[int, int, int]],
    material_colors: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, int]]:
    """
    Assign each cluster to the closest material color (RGB Euclidean distance).
    Prefer one-to-one assignment while material colors are still available.
    """
    if not cluster_rgbs or not material_colors:
        return material_colors

    assigned: List[Tuple[int, int, int]] = []
    available = set(range(len(material_colors)))

    for rgb in cluster_rgbs:
        if available:
            best_idx = min(
                available,
                key=lambda j: (
                    (rgb[0] - material_colors[j][0]) ** 2
                    + (rgb[1] - material_colors[j][1]) ** 2
                    + (rgb[2] - material_colors[j][2]) ** 2
                ),
            )
            available.remove(best_idx)
        else:
            best_idx = min(
                range(len(material_colors)),
                key=lambda j: (
                    (rgb[0] - material_colors[j][0]) ** 2
                    + (rgb[1] - material_colors[j][1]) ** 2
                    + (rgb[2] - material_colors[j][2]) ** 2
                ),
            )
        assigned.append(material_colors[best_idx])

    return assigned


def _cluster_rgb_means_from_class_raster(
    raster_data: np.ndarray,
    class_raster: np.ndarray,
    n_clusters: int,
) -> List[Tuple[int, int, int]]:
    rgb_bands = raster_data[:3] if raster_data.shape[0] >= 3 else raster_data
    cluster_rgbs: List[Tuple[int, int, int]] = []
    for cluster_id in range(1, n_clusters + 1):
        mask = class_raster == cluster_id
        if np.any(mask):
            r = int(np.mean(rgb_bands[0][mask]))
            g = int(np.mean(rgb_bands[1][mask])) if rgb_bands.shape[0] > 1 else r
            b = int(np.mean(rgb_bands[2][mask])) if rgb_bands.shape[0] > 2 else r
            cluster_rgbs.append((r, g, b))
        else:
            cluster_rgbs.append((128, 128, 128))
    return cluster_rgbs


def _cluster_rgb_from_kmeans_centers(
    scaler: StandardScaler,
    centers: np.ndarray,
) -> List[Tuple[int, int, int]]:
    if centers.size == 0:
        return []
    original_space = scaler.inverse_transform(centers)
    out: List[Tuple[int, int, int]] = []
    for row in original_space:
        r = int(np.clip(round(float(row[0])), 0, 255))
        g = int(np.clip(round(float(row[1] if row.shape[0] > 1 else row[0])), 0, 255))
        b = int(np.clip(round(float(row[2] if row.shape[0] > 2 else row[0])), 0, 255))
        out.append((r, g, b))
    return out


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
    Apply colors to classification result using a vectorised LUT lookup.
    Map class values to color indices (class N -> colors[N-1]).
    """
    height, width = class_raster.shape
    n_colors = len(colors)

    if verbose:
        print(f"    [Color Apply] Input raster shape: {class_raster.shape}, dtype: {class_raster.dtype}")
        print(f"    [Color Apply] Input raster range: min={np.min(class_raster)}, max={np.max(class_raster)}")
        print(f"    [Color Apply] Color table size: {n_colors}")
        unique_classes = np.unique(class_raster)
        print(f"    [Color Apply] Unique class values: {unique_classes}")

    # Build LUT: index 0 = background (black), 1..n_colors = class colours
    lut = np.zeros((n_colors + 1, 3), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(colors, start=1):
        if idx <= n_colors:
            lut[idx] = (r, g, b)

    # Clip class raster to valid LUT range, then fancy-index
    flat = class_raster.ravel().astype(np.intp)
    np.clip(flat, 0, n_colors, out=flat)
    mapped = lut[flat]  # (H*W, 3)
    rgb = mapped.reshape(height, width, 3).transpose(2, 0, 1).copy()  # (3, H, W)

    if verbose:
        applied_count = int(np.sum(class_raster > 0))
        total_pixels = height * width
        black_count = total_pixels - applied_count
        print(f"    [Color Apply] Pixels colored: {applied_count}/{total_pixels}")
        print(f"    [Color Apply] Black (0,0,0) pixels: {black_count}/{total_pixels}")
        print(f"    [Color Apply] RGB final ranges: R={np.min(rgb[0])}-{np.max(rgb[0])}, "
              f"G={np.min(rgb[1])}-{np.max(rgb[1])}, B={np.min(rgb[2])}-{np.max(rgb[2])}")

        if applied_count == 0:
            print(f"    [Color Apply] ERROR: ALL PIXELS ARE BLACK!")
            print(f"    [Color Apply] Class values found: {unique_classes}")
            print(f"    [Color Apply] Expected class range: 1 to {n_colors}")

    return rgb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test vector overlay color resolution.")
    parser.add_argument("--test-vector-color", action="store_true", help="Test vector overlay color resolution")
    parser.add_argument("--input-folder", type=str, help="Input folder (classification raster)")
    parser.add_argument("--vector", type=str, help="Vector file path (shp)")
    parser.add_argument("--class-id", type=str, help="Class ID to assign to vector")
    parser.add_argument("--class-color", type=str, help="Class color hex (e.g. #1C6BA0)")
    args = parser.parse_args()

    if args.test_vector_color:
        # Simulate classes and vector_layers as in API
        classes = [
            {"id": args.class_id, "name": "BM_WATER", "color": args.class_color}
        ]
        vector_layers = [
            {"id": "v1", "name": Path(args.vector).name, "filePath": args.vector, "classId": args.class_id}
        ]
        print("[CLI TEST] classes:", classes)
        print("[CLI TEST] vector_layers:", vector_layers)

        # Color resolution logic (copied from rasterize_vectors_onto_classification)
        _class_color_map = {}
        for cls in classes:
            _cid = cls.get("id", "")
            _hex = cls.get("color", "")
            if _cid and _hex.startswith("#") and len(_hex) == 7:
                try:
                    _class_color_map[_cid] = (
                        int(_hex[1:3], 16), int(_hex[3:5], 16), int(_hex[5:7], 16)
                    )
                except ValueError:
                    pass
        print(f"[COLOR DEBUG] class_color_map: {_class_color_map}")

        for layer in vector_layers:
            if not layer.get("overrideColor"):
                _cid = layer.get("classId", "")
                if _cid in _class_color_map:
                    layer["overrideColor"] = list(_class_color_map[_cid])
                    print(f"[COLOR DEBUG] Resolved classId={_cid!r} -> overrideColor={layer['overrideColor']}")

        validated_vectors = [(Path(layer["filePath"]).name, None, layer.get("overrideColor")) for layer in vector_layers]
        _auto_colors = [(255,255,0)]
        overlay_colors = [
            tuple(int(x) for x in oc) if isinstance(oc, (list, tuple)) and len(oc) == 3
            else _auto_colors[i]
            for i, (_, _, oc) in enumerate(validated_vectors)
        ]
        print(f"[COLOR DEBUG] overlay_colors: {overlay_colors}")
        if overlay_colors[0] == (28, 107, 160):
            print("[RESULT] PASS: Overlay color is correct blue (28, 107, 160)")
        else:
            print(f"[RESULT] FAIL: Overlay color is {overlay_colors[0]}, expected (28, 107, 160)")