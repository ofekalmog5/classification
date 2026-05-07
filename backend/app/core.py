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


class _CupyKMeans:
    """Sklearn-compatible KMeans on GPU via CuPy.

    Install: pip install cupy-cuda12x   (CUDA 12.x drivers, works on Windows)
             pip install cupy-cuda11x   (CUDA 11.x drivers)

    Uses batched pairwise-distance assignment to stay within VRAM limits and
    a vectorised scatter-add for center updates.
    """

    def __init__(self, n_clusters: int, *, n_init: int = 1, max_iter: int = 300,
                 tol: float = 1e-4, random_state: int | None = 42):
        self.n_clusters       = n_clusters
        self.n_init           = n_init
        self.max_iter         = max_iter
        self.tol              = tol
        self.random_state     = random_state
        self.cluster_centers_: np.ndarray | None = None
        self.labels_:          np.ndarray | None = None
        self.inertia_: float  = 0.0

    # ── public sklearn-compatible API ─────────────────────────────────────────

    def fit(self, X: np.ndarray):
        self.fit_predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import cupy as cp
        try:
            centers = cp.asarray(self.cluster_centers_)
            return self._assign(cp.asarray(np.ascontiguousarray(X, dtype=np.float32)),
                                centers).get()
        except (cp.cuda.memory.OutOfMemoryError, MemoryError):
            # GPU VRAM exhausted — fall back to CPU nearest-center.
            print("  [gpu] predict OOM, falling back to CPU")
            cp.get_default_memory_pool().free_all_blocks()
            return _nearest_center_chunked(
                np.ascontiguousarray(X, dtype=np.float32),
                self.cluster_centers_.astype(np.float32),
            )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        import warnings as _w
        _w.filterwarnings("ignore", message="CUDA path could not be detected")
        import cupy as cp
        X   = np.ascontiguousarray(X, dtype=np.float32)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)

        best_inertia    = np.inf
        best_centers_np = None
        best_labels_np  = None

        try:
            for _attempt in range(max(1, self.n_init)):
                # Random initialisation (one pass, fast)
                init_idx = rng.choice(n, self.n_clusters, replace=False)
                centers  = cp.asarray(X[init_idx], dtype=cp.float32)   # (k, d)
                X_gpu    = cp.asarray(X)                                # (n, d)

                for _it in range(self.max_iter):
                    labels  = self._assign(X_gpu, centers)
                    new_c   = self._update(X_gpu, labels, centers, self.n_clusters, d)
                    shift   = float(cp.linalg.norm(new_c - centers))
                    centers = new_c
                    if shift < self.tol:
                        break

                labels   = self._assign(X_gpu, centers)
                inertia  = float(cp.sum(self._sq_dists_assigned(X_gpu, centers, labels)))

                if inertia < best_inertia:
                    best_inertia    = inertia
                    best_centers_np = centers.get()
                    best_labels_np  = labels.get().astype(np.int32)

                del X_gpu, centers, labels
                cp.get_default_memory_pool().free_all_blocks()

        except (cp.cuda.memory.OutOfMemoryError, MemoryError, Exception) as gpu_err:
            # GPU ran out of VRAM — fall back to faiss-cpu (fast) or sklearn.
            print(f"  [gpu] CuPy OOM ({gpu_err}), falling back to CPU…")
            try:
                del X_gpu, centers, labels
            except Exception:
                pass
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            try:
                # Prefer faiss-cpu: 3-8x faster than sklearn
                _cpu_km = _FaissKMeans(self.n_clusters, max_iter=self.max_iter, use_gpu=False)
                best_labels_np  = _cpu_km.fit_predict(X).astype(np.int32)
                best_centers_np = _cpu_km.cluster_centers_
                best_inertia    = 0.0
                print(f"  [gpu] faiss-cpu fallback complete.")
            except Exception:
                # Last resort: sklearn MiniBatchKMeans
                from sklearn.cluster import MiniBatchKMeans as _MBKM
                _cpu_km = _MBKM(
                    n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    random_state=self.random_state or 42,
                    batch_size=min(4096, n),
                )
                best_labels_np  = _cpu_km.fit_predict(X).astype(np.int32)
                best_centers_np = _cpu_km.cluster_centers_
                best_inertia    = float(_cpu_km.inertia_)
                print(f"  [gpu] sklearn fallback complete.")

        self.cluster_centers_ = best_centers_np
        self.labels_           = best_labels_np
        self.inertia_          = best_inertia
        return best_labels_np

    # ── internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _assign(X_gpu, centers, batch: int = 65_536):
        """Nearest-center assignment, chunked so VRAM usage stays bounded."""
        import cupy as cp
        n      = X_gpu.shape[0]
        labels = cp.empty(n, dtype=cp.int32)
        sq_c   = cp.sum(centers ** 2, axis=1)                     # (k,)
        for start in range(0, n, batch):
            Xb   = X_gpu[start:start + batch]                     # (b, d)
            sq_x = cp.sum(Xb ** 2, axis=1)                        # (b,)
            # ||x - c||^2 = ||x||^2 - 2 x·cT + ||c||^2
            dists = sq_x[:, None] - 2.0 * (Xb @ centers.T) + sq_c[None, :]
            labels[start:start + batch] = cp.argmin(dists, axis=1)
        return labels

    @staticmethod
    def _update(X_gpu, labels, old_centers, k: int, d: int):
        """Vectorised center update; keeps old center for any empty cluster."""
        import cupy as cp
        sums   = cp.zeros((k, d), dtype=cp.float32)
        counts = cp.zeros(k,      dtype=cp.float32)
        cp.add.at(sums,   labels, X_gpu)
        cp.add.at(counts, labels, cp.float32(1.0))
        new_c         = old_centers.copy()
        valid         = counts > 0
        new_c[valid]  = sums[valid] / counts[valid, None]
        return new_c

    @staticmethod
    def _sq_dists_assigned(X_gpu, centers, labels):
        """Per-point squared distance to its assigned center (for inertia)."""
        diff = X_gpu - centers[labels]
        return (diff * diff).sum(axis=1)


def _probe_acceleration() -> tuple:
    """Return (engine, gpu_available, gpu_info) where engine is one of:
    'faiss-gpu' | 'cupy' | 'faiss-cpu' | 'cuml' | 'sklearn'.

    Priority chain (GPU always beats CPU when available):
      faiss-gpu  — conda install faiss-gpu (fastest; not on pip)
      cupy       — pip install cupy-cuda12x (GPU, Windows-compatible, no conda)
      faiss-cpu  — pip install faiss-cpu (fast CPU, 3-8x sklearn)
      cuml       — conda/WSL2 only
      sklearn    — always available, pure CPU fallback
    """
    # ── 1. faiss-gpu (conda only, fastest) ────────────────────────────────────
    if _GPU_AVAILABLE:
        try:
            import faiss
            res = faiss.StandardGpuResources()
            idx = faiss.GpuIndexFlatL2(res, 2)
            idx.add(np.zeros((1, 2), dtype=np.float32))
            del idx, res
            return "faiss-gpu", True, _GPU_INFO
        except Exception:
            pass  # faiss-gpu not available, try next

    # ── 2. CuPy GPU KMeans (pip install cupy-cuda12x, works on Windows) ───────
    if _GPU_AVAILABLE:
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message="CUDA path could not be detected")
                import cupy as cp
                a = cp.array([1.0, 2.0], dtype=cp.float32)
                _ = float(a @ a)
                del a
                cp.get_default_memory_pool().free_all_blocks()
            return "cupy", True, _GPU_INFO
        except Exception as _e:
            print(f"[GPU] CuPy probe failed ({_e}) — falling back to CPU")

    # ── 3. faiss-cpu (fast CPU KMeans, no GPU needed) ─────────────────────────
    try:
        import faiss  # noqa: F401
        return "faiss-cpu", _GPU_AVAILABLE, _GPU_INFO
    except ImportError:
        pass

    # ── 4. cuML (Linux/WSL2 only) ─────────────────────────────────────────────
    if _GPU_AVAILABLE:
        try:
            from cuml.cluster import MiniBatchKMeans as _  # noqa: F401
            return "cuml", True, _GPU_INFO
        except Exception:
            pass

    # ── 5. sklearn CPU fallback ───────────────────────────────────────────────
    return "sklearn", _GPU_AVAILABLE, _GPU_INFO


_ACCEL_ENGINE, _ACCEL_GPU, _ACCEL_GPU_INFO = _probe_acceleration()
print(f"[KMeans] engine={_ACCEL_ENGINE}  gpu={_ACCEL_GPU}  {_ACCEL_GPU_INFO if _ACCEL_GPU else '(CPU only)'}")


def _make_kmeans(n_clusters: int, *, mini_batch: bool = True):
    """Return the fastest available KMeans: faiss-gpu > faiss-cpu > cupy > cuML > sklearn."""
    if _ACCEL_ENGINE == "faiss-gpu":
        return _FaissKMeans(n_clusters, max_iter=80 if mini_batch else 300, use_gpu=True)
    if _ACCEL_ENGINE == "faiss-cpu":
        return _FaissKMeans(n_clusters, max_iter=80 if mini_batch else 300, use_gpu=False)
    if _ACCEL_ENGINE == "cupy":
        return _CupyKMeans(n_clusters, n_init=1,
                           max_iter=80 if mini_batch else 300)
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


# ─── MEA 6-Material Schema ───────────────────────────────────────────────────
# Each entry carries metadata previously scattered across multiple maps:
#   - source:         "mask" (SAM3/shapefile-driven) or "kmeans" (RGB-cluster)
#   - composite_name: display name in the Composite_Material_Table XML
#   - material_type:  high-level grouping (ROAD_SURFACE, VEGETATION, ...)
#   - sub_absorbs:    legacy BM_* names absorbed into this class on profile migration
#   - anchors:        multi-anchor RGB list used by the cluster->material cost function
MEA_CLASSES = [
    {
        "id": "class-1", "name": "BM_ASPHALT", "color": "#2D2D30",
        "composite_name": "ASPHALT", "source": "mask", "material_type": "ROAD_SURFACE",
        "sub_absorbs": ["BM_PAINT_ASPHALT"],
        "anchors": [[44, 44, 56], [52, 55, 72], [91, 91, 101], [91, 90, 96], [60, 63, 65]],
    },
    {
        "id": "class-2", "name": "BM_CONCRETE", "color": "#B4B4B4",
        "composite_name": "CONCRETE", "source": "mask", "material_type": "BUILDING",
        "sub_absorbs": ["BM_ROCK", "BM_METAL", "BM_METAL_STEEL"],
        "anchors": [[180, 180, 180], [130, 123, 115], [169, 171, 176], [112, 128, 144]],
    },
    {
        "id": "class-3", "name": "BM_VEGETATION", "color": "#228B22",
        "composite_name": "GENVEGETATION", "source": "kmeans", "material_type": "VEGETATION",
        "sub_absorbs": ["BM_FOLIAGE", "BM_LAND_GRASS", "BM_LAND_DRY_GRASS"],
        # Anchors are deliberately constrained to clearly green-dominant RGBs
        # (G > R AND G > B by a comfortable margin).  Earlier revisions included
        # tan/khaki anchors like [189,183,107] and [150,160,90]; with strict 1:1
        # cluster→material assignment those are no longer needed and they
        # caused soil/sand pixels to be pulled into vegetation.
        "anchors": [
            [34, 139, 34],   # forest green
            [0, 100, 0],     # dark forest
            [124, 252, 0],   # lime / lush grass
            [80, 100, 55],   # olive / dry grass — G dominant
            [60, 90, 50],    # shadowed grass
            [110, 130, 80],  # sage / dry meadow
            [70, 110, 70],   # mid green
        ],
    },
    {
        "id": "class-4", "name": "BM_WATER", "color": "#1C6BA0",
        "composite_name": "WATER", "source": "mask", "material_type": "WATER",
        "sub_absorbs": [],
        "anchors": [[28, 107, 160]],
    },
    {
        "id": "class-5", "name": "BM_SAND", "color": "#EDC9AF",
        "composite_name": "SAND", "source": "kmeans", "material_type": "SOIL_EARTH",
        "sub_absorbs": [],
        "anchors": [[237, 201, 175]],
    },
    {
        "id": "class-6", "name": "BM_SOIL", "color": "#654321",
        "composite_name": "SOIL", "source": "kmeans", "material_type": "SOIL_EARTH",
        "sub_absorbs": [],
        # Pure-brown anchors only — deliberately avoid the greenish-brown
        # band so dry grass and shadowed vegetation do not get pulled into
        # soil.  (Greenish-brown is covered by BM_VEGETATION's olive anchors.)
        "anchors": [
            [101, 67, 33],   # earth brown
            [139, 90, 43],   # saddle brown
            [85, 55, 30],    # dark brown
            [160, 110, 70],  # tan/brown
            [120, 80, 50],   # rich brown
        ],
    },
]

# Derived lookup tables — single source of truth is MEA_CLASSES above.
_MEA_MASK_MATERIALS   = {c["name"] for c in MEA_CLASSES if c["source"] == "mask"}
_MEA_KMEANS_MATERIALS = {c["name"] for c in MEA_CLASSES if c["source"] == "kmeans"}
_MEA_ANCHOR_MAP: Dict[str, List[List[int]]] = {c["name"]: c["anchors"] for c in MEA_CLASSES}
_MEA_COMPOSITE_NAMES: Dict[str, str] = {c["name"]: c["composite_name"] for c in MEA_CLASSES}


def _resolve_active_anchor_map() -> Dict[str, List[List[int]]]:
    """Return the anchor map merged with the user's active calibration profile.

    Reads ``mea_profile.load_active_profile()`` (which already merges the user's
    calibrated overrides onto factory defaults) and overlays its per-material
    ``anchors`` arrays onto the hardcoded ``_MEA_ANCHOR_MAP``.  When the user
    has not run the calibration tool, this returns the hardcoded defaults
    unchanged.  Errors fall back to defaults so a malformed profile never
    breaks classification.
    """
    base = {k: list(v) for k, v in _MEA_ANCHOR_MAP.items()}
    try:
        from . import mea_profile  # local import — avoids hard import cycle
        profile = mea_profile.load_active_profile() or {}
        overrides = profile.get("material_overrides", {}) or {}
        for name, mat in overrides.items():
            if not isinstance(mat, dict):
                continue
            anchors = mat.get("anchors")
            if isinstance(anchors, list) and anchors:
                base[name] = [list(a) for a in anchors]
    except Exception as exc:
        print(f"[core] anchor resolver: profile read failed ({exc}); using hardcoded defaults")
    return base

# Reverse map: legacy 13-material name -> new 6-material parent (for profile migration).
_MEA_LEGACY_TO_PARENT: Dict[str, str] = {
    legacy: parent["name"]
    for parent in MEA_CLASSES
    for legacy in parent["sub_absorbs"]
}


def _mea_material_type(name: str) -> str:
    """Return the high-level material-type grouping for a BM_* name."""
    for cls in MEA_CLASSES:
        if cls["name"] == name:
            return cls["material_type"]
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


def _bounds_wgs84(
    transform,
    crs,
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Return (left, bottom, right, top) in WGS-84 / EPSG:4326.

    If the raster is already in EPSG:4326, or if reprojection fails for any
    reason, the native bounds are returned unchanged.
    """
    import rasterio.warp as _warp

    left   = transform.c
    top    = transform.f
    right  = left + width  * transform.a
    bottom = top  + height * transform.e   # e is negative → bottom < top

    if crs is None:
        return left, bottom, right, top
    try:
        if crs.to_epsg() == 4326:
            return left, bottom, right, top
        west, south, east, north = _warp.transform_bounds(
            crs, "EPSG:4326", left, bottom, right, top
        )
        return west, south, east, north
    except Exception:
        return left, bottom, right, top


def _next_pow2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def _pad_array_to_pow2(arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Zero-pad *arr* so its last two dimensions (H, W) are powers of 2.

    Returns ``(padded_array, new_height, new_width)``.  Padding is added on
    the right and bottom edges only, leaving the top-left origin unchanged so
    the raster transform stays valid.  If both dimensions are already powers of
    2 the original array is returned unchanged (no copy).
    """
    h, w = (arr.shape[-2], arr.shape[-1]) if arr.ndim >= 2 else (arr.shape[0], 1)
    new_h = _next_pow2(h)
    new_w = _next_pow2(w)
    if new_h == h and new_w == w:
        return arr, h, w          # nothing to do
    if arr.ndim == 2:
        padded = np.zeros((new_h, new_w), dtype=arr.dtype)
        padded[:h, :w] = arr
    else:
        padded = np.zeros((*arr.shape[:-2], new_h, new_w), dtype=arr.dtype)
        padded[..., :h, :w] = arr
    return padded, new_h, new_w


def _reproject_to_wgs84(
    arr: np.ndarray,
    src_transform,
    src_crs,
    width: int,
    height: int,
) -> Tuple[np.ndarray, object, int, int, object]:
    """Reproject *arr* to EPSG:4326 (WGS-84 geographic).

    Returns ``(reprojected_array, new_transform, new_height, new_width, new_crs)``.
    If the source CRS is already EPSG:4326, or reprojection fails, the inputs
    are returned unchanged so callers can always unpack the 5-tuple safely.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    dst_crs = CRS.from_epsg(4326)

    if src_crs is None:
        return arr, src_transform, height, width, src_crs
    try:
        if src_crs.to_epsg() == 4326:
            return arr, src_transform, height, width, src_crs
    except Exception:
        pass

    try:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src_crs, dst_crs, width, height, transform=src_transform,
        )
        if arr.ndim == 2:
            dst_arr = np.zeros((dst_h, dst_w), dtype=arr.dtype)
            reproject(
                source=arr, destination=dst_arr,
                src_transform=src_transform, src_crs=src_crs,
                dst_transform=dst_transform, dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
        else:
            bands = arr.shape[0]
            dst_arr = np.zeros((bands, dst_h, dst_w), dtype=arr.dtype)
            for i in range(bands):
                reproject(
                    source=arr[i], destination=dst_arr[i],
                    src_transform=src_transform, src_crs=src_crs,
                    dst_transform=dst_transform, dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
        return dst_arr, dst_transform, dst_h, dst_w, dst_crs
    except Exception as _exc:
        print(f"  [REPROJECT] Warning: could not reproject to EPSG:4326: {_exc}")
        return arr, src_transform, height, width, src_crs


def _write_txr_file(
    output_path,
    transform,
    crs,
    width: int,
    height: int,
) -> None:
    """Write a .txr sidecar file next to *output_path*.

    The .txr contains the WGS-84 bounding box in the format expected by the
    GIS toolchain (csm / EndMapTokens format), one token per line.
    """
    try:
        left, bottom, right, top = _bounds_wgs84(transform, crs, width, height)
        txr_path = Path(output_path).with_suffix(".txr")
        lines = [
            "csm=Geographic",
            "datumId=4326",
            "dcType=NONE",
            "dcSelectorId=0",
            "EndMapTokens",
            f"Top: {top:.12f}",
            f"Bottom: {bottom:.12f}",
            f"Left: {left:.12f}",
            f"Right: {right:.12f}",
        ]
        txr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"  [TXR] Warning: could not write .txr sidecar for {output_path}: {exc}")


def _write_txs_file(
    txs_path,
    image_infos: List[Tuple],
) -> None:
    """Write the all_imgs.txs batch-load script.

    *image_infos* is a list of 7-tuples:
        (image_path, left_wgs84, bottom_wgs84, right_wgs84, top_wgs84, width_px, height_px)
    """
    try:
        lib = "Source Data Library;Geospecific Imagery;Year-Round"
        lines: List[str] = []
        for img_path, left, bottom, right, top, img_w, img_h in image_infos:
            p = str(img_path)
            p_bak = f"{p} (backup)"
            # Texel counts must be power-of-2 (GeoSpecific engine requirement).
            texels_x = _next_pow2(img_w)
            texels_y = _next_pow2(img_h)
            lines += [
                f'Delete "{lib};{p_bak}"',
                f'Rename "{lib};{p}" "{p_bak}"',
                f'Create "GeoSpecific" "{lib}" "{p}"',
                f'SetEnv thisRecord "{lib};{p}"',
                f'Set $thisRecord "Zero Y" ""',
                f'Set $thisRecord "Zero X" ""',
                f'Set $thisRecord "Use 4th Channel as Alpha" "no"',
                f'Set $thisRecord "Transparency" ""',
                f'Set $thisRecord "Timestamp" ""',
                f'Set $thisRecord "Origin Y" ""',
                f'Set $thisRecord "Origin X" ""',
                f'Set $thisRecord "Number of Texels Y" {texels_y}',
                f'Set $thisRecord "Number of Texels X" {texels_x}',
                f'Set $thisRecord "Notes" ""',
                f'Set $thisRecord "Misc3" ""',
                f'Set $thisRecord "Misc2" ""',
                f'Set $thisRecord "Misc1" ""',
                f'Set $thisRecord "Meters per Texels Y" ""',
                f'Set $thisRecord "Meters per Texels X" ""',
                f'Set $thisRecord "MaterialClassification" ""',
                f'Set $thisRecord "Map Selection" ""',
                f'Set $thisRecord "Map Model" "Geographic"',
                f'Set $thisRecord "Map Datum" "WGS84"',
                f'Set $thisRecord "Local Top" {top:.12f}',
                f'Set $thisRecord "Local Right" {right:.12f}',
                f'Set $thisRecord "Local Left" {left:.12f}',
                f'Set $thisRecord "Local Bottom" {bottom:.12f}',
                f'Set $thisRecord "Gamma" 1.800000000000',
                f'Set $thisRecord "File Type" ""',
                f'Set $thisRecord "File Name" "{p}"',
                f'Set $thisRecord "Enter Date" ""',
                f'Set $thisRecord "Data Interpretation" "GeoSpecific Format"',
                f'Set $thisRecord "Custom Definition 2" ""',
                f'Set $thisRecord "Custom Definition" ""',
                f'Set $thisRecord "Conv_ Usage" "use default"',
                f'Set $thisRecord "Conv_ Selection" ""',
                f'Set $thisRecord "Conv_ Family" ""',
                f'Set $thisRecord "Auto Load" ""',
            ]
        Path(txs_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  [TXS] Batch-load script written: {txs_path}")
    except Exception as exc:
        print(f"  [TXS] Warning: could not write all_imgs.txs: {exc}")


def _write_composite_material_xml(
    output_path,
    classes: List[Dict[str, str]],
) -> Optional[str]:
    """Write a Composite_Material_Table XML alongside a classification output file.

    The XML is placed next to ``output_path`` with the same stem and a ``.xml``
    extension.  Each MEA class produces one ``<Composite_Material>`` entry whose
    color is the ARGB hex of the class colour (fully-opaque, lower-case).

    Output format (2-space indent, no XML declaration):

        <Composite_Material_Table>
          <Composite_Material index="1">
            <Name>GENVEGETATION</Name>
            <Color>#ff004600</Color>
            <Primary_Substrate>
              <Thickness>1</Thickness>
              <Material>
                <Name>BM_VEGETATION</Name>
                <Weight>100</Weight>
              </Material>
            </Primary_Substrate>
          </Composite_Material>
          ...
        </Composite_Material_Table>
    """
    xml_path = Path(output_path).with_suffix(".xml")

    lines: List[str] = ["<Composite_Material_Table>"]

    for idx, cls in enumerate(classes, start=1):
        bm_name        = cls.get("name", "")
        color_hex      = cls.get("color", "#000000")
        composite_name = _MEA_COMPOSITE_NAMES.get(bm_name, bm_name.replace("BM_", ""))

        # Build ARGB colour: #ff + RRGGBB (lower-case, no alpha adjustment).
        if color_hex.startswith("#") and len(color_hex) == 7:
            argb_color = f"#ff{color_hex[1:].lower()}"
        else:
            argb_color = "#ff000000"

        lines += [
            f'  <Composite_Material index="{idx}">',
            f'    <Name>{composite_name}</Name>',
            f'    <Color>{argb_color}</Color>',
            f'    <Primary_Substrate>',
            f'      <Thickness>1</Thickness>',
            f'      <Material>',
            f'        <Name>{bm_name}</Name>',
            f'        <Weight>100</Weight>',
            f'      </Material>',
            f'    </Primary_Substrate>',
            f'  </Composite_Material>',
        ]

    lines.append("</Composite_Material_Table>")

    try:
        xml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  [XML] Composite material table written: {xml_path}")
        return str(xml_path)
    except Exception as exc:
        print(f"  [XML] Warning: could not write material XML: {exc}")
        return None


# Relative expected prevalence in urban aerial scenes (6-material schema).
# Sub-material absorption rolled in: BM_CONCRETE absorbs ROCK + METAL + METAL_STEEL,
# BM_ASPHALT absorbs PAINT_ASPHALT, BM_VEGETATION absorbs FOLIAGE + LAND_GRASS + LAND_DRY_GRASS.
MEA_MATERIAL_FREQUENCY_PRIOR_URBAN: Dict[str, float] = {
    "BM_CONCRETE":   0.28,
    "BM_VEGETATION": 0.21,
    "BM_ASPHALT":    0.20,
    "BM_SOIL":       0.15,
    "BM_SAND":       0.04,
    "BM_WATER":      0.04,
}

# Relative expected prevalence in open/rural scenes (6-material schema).
MEA_MATERIAL_FREQUENCY_PRIOR_OPEN: Dict[str, float] = {
    "BM_VEGETATION": 0.39,
    "BM_SOIL":       0.25,
    "BM_CONCRETE":   0.12,
    "BM_SAND":       0.08,
    "BM_WATER":      0.05,
    "BM_ASPHALT":    0.04,
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
    n_samples: int = 500_000,
    grid_steps: int = 16,
    window_size: int = 256,
    n_random_extra: int = 64,
) -> np.ndarray:
    """Return a pixel-feature matrix sampled broadly across the raster.

    Uses a two-stage strategy to ensure good material coverage:

    1. **Grid sampling** — ``grid_steps × grid_steps`` evenly-spaced windows.
    2. **Random windows** — ``n_random_extra`` additional random positions.

    Each window is spatially sub-sampled to ``pixels_per_window`` pixels
    *before* feature extraction — so we never extract features for more than
    ``n_samples`` pixels total.  This avoids the old bottleneck of extracting
    83 M features and then discarding 99 % of them.
    """
    rng = np.random.default_rng()
    n_windows = grid_steps * grid_steps + n_random_extra
    pixels_per_window = max(32, n_samples // n_windows)

    parts: List[np.ndarray] = []

    def _sample_window(tile_data: np.ndarray) -> np.ndarray:
        """Extract features from a window, sub-sampling pixels first."""
        n_bands, h, w = tile_data.shape
        n_px = h * w
        if n_px <= pixels_per_window:
            return _extract_pixel_features(tile_data, feature_flags, verbose=False)
        # Randomly pick pixel indices; reshape tile to (bands, n_px) for indexing
        idx = rng.choice(n_px, pixels_per_window, replace=False)
        flat = tile_data.reshape(n_bands, n_px)[:, idx]          # (bands, k)
        sampled_tile = flat.reshape(n_bands, 1, pixels_per_window)  # fake 1×k tile
        return _extract_pixel_features(sampled_tile, feature_flags, verbose=False)

    with rasterio.open(raster_path) as src:
        H, W = src.height, src.width
        actual_win = min(window_size, H, W)

        # --- Stage 1: uniform grid ---
        rows_pos = np.linspace(0, max(0, H - actual_win), grid_steps, dtype=int)
        cols_pos = np.linspace(0, max(0, W - actual_win), grid_steps, dtype=int)
        for r in rows_pos:
            for c in cols_pos:
                win = Window(int(c), int(r),
                             min(actual_win, W - int(c)),
                             min(actual_win, H - int(r)))
                parts.append(_sample_window(src.read(window=win)))

        # --- Stage 2: random extra windows ---
        for _ in range(n_random_extra):
            r = int(rng.integers(0, max(1, H - actual_win)))
            c = int(rng.integers(0, max(1, W - actual_win)))
            win = Window(c, r,
                         min(actual_win, W - c),
                         min(actual_win, H - r))
            parts.append(_sample_window(src.read(window=win)))

    combined = np.concatenate(parts, axis=0)
    n_total = len(combined)
    print(f"  [SAMPLE] {n_total:,} px from {grid_steps}×{grid_steps} grid "
          f"+ {n_random_extra} random windows (~{pixels_per_window} px/window)")
    # Final trim to exact budget (may be slightly over due to small windows)
    if n_total > n_samples:
        combined = combined[rng.choice(n_total, n_samples, replace=False)]
        print(f"  [SAMPLE] Trimmed to {n_samples:,} training pixels")
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


# ---------------------------------------------------------------------------
# Resource management — headroom constants
# ---------------------------------------------------------------------------
# Always reserve a fraction of RAM and CPU so the OS, desktop, and other apps
# stay responsive.  These caps prevent the common OOM-crash scenario where
# every core is busy with a tile and each tile has its own scratch buffers.

_RAM_HEADROOM_BYTES: int = 2 * 1024 ** 3        # always keep ≥2 GB free
_RAM_USAGE_FRAC: float   = 0.45                 # never consume >45% of *available* RAM
_CPU_RESERVE_CORES: int  = 2                     # always keep ≥2 cores idle
_CPU_MAX_USAGE_FRAC: float = 0.50                # never use >50% of cores
_WORKER_THREAD_CAP: int  = 2                     # max threads inside each subprocess


def _usable_ram_bytes() -> int:
    """Available RAM minus headroom for system stability."""
    avail = _available_ram_bytes()
    usable = min(avail - _RAM_HEADROOM_BYTES,
                 int(avail * _RAM_USAGE_FRAC))
    return max(usable, 256 * 1024 * 1024)  # floor at 256 MB


def _safe_worker_count(requested: Optional[int] = None,
                       fallback: Optional[int] = None) -> int:
    """Return a worker count that leaves CPU headroom for the OS.

    Priority: *requested* > *fallback* > auto (half of cores, minus reserve).
    """
    cpus = os.cpu_count() or 4
    cap = max(1, min(int(cpus * _CPU_MAX_USAGE_FRAC),
                     cpus - _CPU_RESERVE_CORES))

    if requested and requested > 0:
        return max(1, min(requested, cap))
    if fallback and fallback > 0:
        return max(1, min(fallback, cap))
    return cap


def _ram_ok_for_next_tile(per_tile_bytes: int) -> bool:
    """Return True if there is enough free RAM to launch another tile worker."""
    avail = _available_ram_bytes()
    return avail - per_tile_bytes > _RAM_HEADROOM_BYTES


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

    avail = _usable_ram_bytes()
    workers = _safe_worker_count(workers)
    # Budget: usable RAM (with headroom) shared across workers, x8 scratch copies.
    # x8 accounts for GDAL read + feature extraction + normalisation + labels + colour.
    budget_bytes = avail / max(1, workers)
    bytes_per_px = max(bands, 3) * max(itemsize, 4) * 8  # x8 for working copies
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
        # Cap threads to _WORKER_THREAD_CAP so parallel tile processes don't
        # collectively saturate the CPU (each process would otherwise spawn
        # os.cpu_count() threads).
        _n_workers = min(len(ranges), _WORKER_THREAD_CAP)
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            list(pool.map(_assign_chunk, ranges))

    return labels


# ---------------------------------------------------------------------------
# Shadow pre-processing: balance / brighten shadows BEFORE classification
# ---------------------------------------------------------------------------


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
    # On MemoryError: free everything, gc, and retry once before giving up.
    for _mem_attempt in range(2):
        try:
            features = _extract_pixel_features(tile_data, feature_flags, verbose=False)
            scale    = np.where(scaler_scale == 0, 1.0, scaler_scale).astype(np.float32)
            features_norm = ((features - scaler_mean) / scale).astype(np.float32)
            del features                        # free ~20×H×W×4 bytes immediately
            gc.collect()
            # Use smaller chunks on retry to reduce peak memory
            _chunk = 131_072 if _mem_attempt == 0 else 32_768
            labels = _nearest_center_chunked(
                features_norm, centers.astype(np.float32), chunk=_chunk,
            ) + 1
            del features_norm
            gc.collect()
            break  # success
        except MemoryError:
            # Clean up any partial allocations before retry / error
            features = features_norm = labels = None
            gc.collect()
            if _mem_attempt == 0:
                print(f"  [mem] Tile {tile_name}: MemoryError on attempt 1, retrying with smaller chunks…")
                _time.sleep(0.5)  # brief pause for other workers to release
                continue
            bands_t, h_t, w_t = tile_data.shape
            mb = bands_t * h_t * w_t / (1024 * 1024)
            raise MemoryError(
                f"Out of memory processing tile {tile_name} ({h_t}×{w_t}, {bands_t} bands, ~{mb:.0f} MB) "
                f"after 2 attempts. Reduce tile size in Performance settings."
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

    # Reproject to EPSG:4326 — GeoSpecific engine requirement.
    rgb, tile_transform, height, width, tile_crs = _reproject_to_wgs84(
        rgb, tile_transform, tile_crs, width, height,
    )
    # Propagate updated georeferencing into the profile so write_profile inherits it.
    profile.update(transform=tile_transform, crs=tile_crs, height=height, width=width)

    # Pad to power-of-2 dimensions — GeoSpecific engine requirement.
    rgb, pad_h, pad_w = _pad_array_to_pow2(rgb)

    output_path_obj = Path(output_dir) / tile_name
    driver = _driver_for_path(str(output_path_obj))
    if driver == "GTiff":
        write_profile = _output_tiff_profile(profile, dtype="uint8")
        write_profile["count"] = 3
    else:
        write_profile = _profile_for_driver(profile, driver)
        write_profile.update(count=3, dtype="uint8")
    write_profile.update(height=pad_h, width=pad_w)
    with rasterio.open(output_path_obj, 'w', **write_profile) as dst:
        dst.write(rgb)
    del rgb, tile_data_crop, predicted_raster
    gc.collect()

    # Write .txr sidecar using the padded dimensions for correct geographic bounds.
    _write_txr_file(output_path_obj, tile_transform, tile_crs, pad_w, pad_h)
    _t_left, _t_bottom, _t_right, _t_top = _bounds_wgs84(tile_transform, tile_crs, pad_w, pad_h)

    # Write companion XML for every classified tile.
    _tile_classes = extra.get("classes", [])
    if _tile_classes:
        _write_composite_material_xml(output_path_obj, list(_tile_classes))

    return (str(output_path_obj), _t_left, _t_bottom, _t_right, _t_top, pad_w, pad_h)


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

    # Reproject to EPSG:4326 — GeoSpecific engine requirement.
    output_array, transform, height, width, raster_crs = _reproject_to_wgs84(
        output_array, transform, raster_crs, width, height,
    )
    meta.update(transform=transform, crs=raster_crs, height=height, width=width)

    # Pad to power-of-2 dimensions — GeoSpecific engine requirement.
    output_array, pad_h, pad_w = _pad_array_to_pow2(output_array)

    output_path = Path(output_dir) / tile_name
    driver = _driver_for_path(str(output_path))
    if driver == "GTiff":
        write_meta = _output_tiff_profile(meta)
    else:
        write_meta = _profile_for_driver(meta, driver)
    write_meta.update(height=pad_h, width=pad_w)
    with rasterio.open(output_path, 'w', **write_meta) as dst:
        dst.write(output_array)

    # Write .txr sidecar using the padded dimensions for correct geographic bounds.
    _write_txr_file(output_path, transform, raster_crs, pad_w, pad_h)
    _t_left, _t_bottom, _t_right, _t_top = _bounds_wgs84(transform, raster_crs, pad_w, pad_h)

    return (str(output_path), _t_left, _t_bottom, _t_right, _t_top, pad_w, pad_h)


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
    
    # Reproject to EPSG:4326 — GeoSpecific engine requirement.
    output_array, transform, height, width, raster_crs = _reproject_to_wgs84(
        output_array, transform, raster_crs, width, height,
    )
    meta.update(transform=transform, crs=raster_crs, height=height, width=width)

    # Pad to power-of-2 dimensions — GeoSpecific engine requirement.
    output_array, _pad_h, _pad_w = _pad_array_to_pow2(output_array)
    meta.update(height=_pad_h, width=_pad_w)

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

    print(f"    [OK] Vector rasterized successfully (padded to {_pad_w}×{_pad_h})")


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

    if detect_shadows:
        print("  [WARN] detect_shadows=True has no effect in the 6-material schema "
              "(shadow detection was removed). Parameter is deprecated.")

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
        _ram_budget  = int(_usable_ram_bytes() * 0.70)  # 70% of usable (already has headroom)
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
    print(f"\n[2/5] Extracting pixel-level features...  [tile_mode={tile_mode}]")

    # In tile mode, we only need a subsample for KMeans training — each tile
    # worker will extract its own features later.  Attempting full-image feature
    # extraction on a huge raster wastes RAM and can OOM even though the raster
    # data itself fits.  So: always use sampled training when tile_mode is on.
    if tile_mode:
        # In tile mode each worker extracts features for its own small window,
        # so we only need a lightweight sample for KMeans training — never the
        # full-image feature matrix.  This avoids OOM on large rasters where
        # the raw data fits in RAM but the feature array (n_pixels × n_features)
        # does not.
        print(f"  [TILE] Using sampled training (tile workers will extract per-tile features)")
        # Free the full raster early — tile workers read from disk.
        if raster_data is not None:
            _freed_gb = raster_data.nbytes / 1e9
            del raster_data
            raster_data = None
            gc.collect()
            print(f"  [TILE] Freed {_freed_gb:.1f} GB raster array")
        _n_samples = int(np.clip(height * width * 0.001, 100_000, 2_000_000))
        print(f"  [TILE] Adaptive sample budget: {_n_samples:,} px "
              f"(image {width}×{height} = {height*width/1e6:.1f} MP)")
        pixel_features = _sample_raster_for_training(str(path), feature_flags, n_samples=_n_samples)
        _full_features = False
    elif raster_data is not None:
        try:
            pixel_features = _extract_pixel_features(raster_data, feature_flags)
            _full_features = True
        except MemoryError as mem_err:
            print(f"\n  [AUTO] {mem_err}")
            print(f"  [AUTO] Automatically switching to tile mode...")
            _cb("Auto-switching to tile mode", 0, 1)
            return classify_and_export(
                raster_path=raster_path,
                classes=classes,
                smoothing=smoothing,
                feature_flags=feature_flags,
                output_path=output_path,
                tile_mode=True,
                tile_max_pixels=tile_max_pixels,
                tile_overlap=tile_overlap,
                tile_output_dir=tile_output_dir or output_path,
                tile_workers=tile_workers,
                detect_shadows=detect_shadows,
                max_threads=max_threads,
                pretrained_scaler=pretrained_scaler,
                pretrained_kmeans=pretrained_kmeans,
                pretrained_color_table=pretrained_color_table,
                pretrained_mea_mapping=pretrained_mea_mapping,
                progress_callback=progress_callback,
                export_format=export_format,
            )
    else:
        print(f"  Using spatially-distributed sample (raster too large for full load)")
        _n_samples = int(np.clip(height * width * 0.001, 100_000, 2_000_000))
        pixel_features = _sample_raster_for_training(str(path), feature_flags, n_samples=_n_samples)
        _full_features = False
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

        # Determine max workers — always leave headroom for the OS / desktop.
        max_workers = _safe_worker_count(tile_workers, max_threads)
        print(f"  Tile workers: {max_workers} (of {os.cpu_count()} cores)")

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
        _tile_txs_infos: List[Tuple] = []
        _n_jobs = len(jobs)
        _cb("Classifying tiles", 0, _n_jobs)

        # Estimate per-tile RAM (bands × tile_pixels × 8 scratch copies × 4 bytes)
        _tile_px = tile_size * tile_size
        _est_tile_bytes = max(n_bands, 3) * _tile_px * 8 * 4

        # Submit tiles in controlled waves — never more than max_workers in
        # flight at once, and pause if RAM pressure is high.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending: dict = {}
            job_iter = iter(enumerate(jobs))
            _done_count = 0

            _failed_tiles: List[str] = []

            def _drain_one():
                """Wait for one future to complete, collect its result."""
                nonlocal _done_count
                done_set = set()
                for f in as_completed(pending, timeout=None):
                    done_set.add(f)
                    break
                for f in done_set:
                    tile_idx = pending.pop(f)
                    _done_count += 1
                    try:
                        _res = f.result()
                        tile_outputs.append(_res[0])
                        _tile_txs_infos.append(_res)
                    except Exception as _tile_err:
                        import traceback as _tb
                        _tname = jobs[tile_idx][-2] if len(jobs[tile_idx]) > 1 else f"tile_{tile_idx}"
                        print(f"\n  [ERROR] Tile {_tname} failed — skipping and continuing:")
                        print(f"          {type(_tile_err).__name__}: {_tile_err}")
                        _tb.print_exc()
                        _failed_tiles.append(_tname)
                    _cb("Classifying tiles", _done_count, _n_jobs)
                    if _done_count % max(1, _n_jobs // 10) == 0 or _done_count == _n_jobs:
                        print(f"    tiles: {_done_count}/{_n_jobs} done"
                              + (f" ({len(_failed_tiles)} failed)" if _failed_tiles else ""))

            for idx, job in job_iter:
                # If we already have max_workers in flight, wait for one
                while len(pending) >= max_workers:
                    _drain_one()
                # Also wait if RAM is getting tight
                while len(pending) > 0 and not _ram_ok_for_next_tile(_est_tile_bytes):
                    print(f"    [mem] waiting for RAM before tile {idx+1}/{_n_jobs}…")
                    _drain_one()
                pending[executor.submit(_classify_tile_worker, job)] = idx

            # Drain remaining
            while pending:
                _drain_one()

        if _failed_tiles:
            print(f"\n  [WARN] {len(_failed_tiles)}/{_n_jobs} tiles failed:")
            for _fn in _failed_tiles:
                print(f"    - {_fn}")
        print(f"  [OK] Wrote {len(tile_outputs)}/{_n_jobs} tiles to {output_dir}")

        # Write all_imgs.txs for the complete tile set.
        if _tile_txs_infos:
            _write_txs_file(Path(output_dir) / "all_imgs.txs", _tile_txs_infos)

        _ce_stages.append(("Tiled classification", _time.perf_counter() - _t0))
        _ce_total = _time.perf_counter() - _t_ce_start
        _ce_table = _build_stats_table(_ce_stages, _ce_total)
        print("\n" + "="*70)
        print("[OK] STEP 1 COMPLETE: Classification & Export (Tiles)")
        print(_ce_table)
        print("="*70)

        return {
            "status": "ok",
            "outputPath": str(output_dir),
            "tileOutputs": sorted(tile_outputs),
            "message": "Classification complete (tiles). Use output directory for Step 2.",
            "meaMapping": mea_mapping,
            "stats": _ce_stages,
            "statsTable": _ce_table,
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

    # Reproject to EPSG:4326 — GeoSpecific engine requirement.
    _rp_transform = profile.get("transform")
    _rp_crs = profile.get("crs")
    rgb, _rp_transform, _rp_h, _rp_w, _rp_crs = _reproject_to_wgs84(
        rgb, _rp_transform, _rp_crs, rgb.shape[2], rgb.shape[1],
    )
    profile.update(transform=_rp_transform, crs=_rp_crs, height=_rp_h, width=_rp_w)

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
    
    # Pad to power-of-2 dimensions — GeoSpecific engine requirement.
    rgb, out_h, out_w = _pad_array_to_pow2(rgb)
    rgb_profile.update(height=out_h, width=out_w)

    output_color_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(output_color_path).exists():
        Path(output_color_path).unlink()

    with rasterio.open(output_color_path, 'w', **rgb_profile) as dst:
        dst.write(rgb)

    # Write .txr sidecar and all_imgs.txs using padded dimensions.
    _out_transform = profile.get("transform")
    _out_crs = profile.get("crs")
    _write_txr_file(output_color_path, _out_transform, _out_crs, out_w, out_h)
    _o_left, _o_bottom, _o_right, _o_top = _bounds_wgs84(_out_transform, _out_crs, out_w, out_h)
    _write_txs_file(
        Path(output_color_path).parent / "all_imgs.txs",
        [(str(output_color_path), _o_left, _o_bottom, _o_right, _o_top, out_w, out_h)],
    )

    print(f"  [OK] Classification saved to {output_color_path}")
    _cb("Saving output", 1, 1)
    _ce_stages.append(("Save output", _time.perf_counter() - _t0))
    _ce_total = _time.perf_counter() - _t_ce_start
    _ce_table = _build_stats_table(_ce_stages, _ce_total)

    print("\n" + "="*70)
    print("[OK] STEP 1 COMPLETE: Classification & Export")
    print(_ce_table)
    print("="*70)

    # Write companion Composite_Material_Table XML for every classification output.
    xml_out: Optional[str] = None
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
                # Also estimate feature array: ~15 features × 4 bytes (float32) per pixel
                _feat_est = H * W * 15 * 4
                _ram_avail = _usable_ram_bytes()
                if nbytes + _feat_est > int(_ram_avail * 0.70):
                    print(f"[SharedTable][warn] {Path(rp).name} too large for semantics "
                          f"(raster {nbytes/1e9:.1f}GB + features ~{_feat_est/1e9:.1f}GB "
                          f"> {_ram_avail*0.70/1e9:.1f}GB budget), skipping")
                    continue
                rd = src.read()
        except Exception as e:
            print(f"[SharedTable][warn] Cannot open {rp}: {e}")
            continue

        feat   = _extract_pixel_features(rd, feature_flags, verbose=False)
        _mean  = scaler.mean_.astype(np.float32)
        scale  = np.where(scaler.scale_ == 0, 1.0, scaler.scale_).astype(np.float32)
        fnorm  = ((feat - _mean) / scale)  # stays float32, no float64 intermediate
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

        # Determine max workers — always leave headroom for the OS / desktop.
        max_workers = _safe_worker_count(tile_workers, max_threads)
        print(f"  Rasterize workers: {max_workers} (of {os.cpu_count()} cores)")

        tile_outputs: List[str] = []
        _rv_txs_infos: List[Tuple] = []
        _rv_failed: List[str] = []
        _n_rast_jobs = len(jobs)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            pending_r: dict = {}
            for idx_r, job in enumerate(jobs):
                # Throttle: keep at most max_workers futures in flight
                while len(pending_r) >= max_workers:
                    for f in as_completed(pending_r, timeout=None):
                        _ridx = pending_r.pop(f)
                        try:
                            _rv_res = f.result()
                            tile_outputs.append(_rv_res[0])
                            _rv_txs_infos.append(_rv_res)
                        except Exception as _rv_err:
                            import traceback as _tb
                            _rtname = jobs[_ridx][-1] if len(jobs[_ridx]) > 1 else f"tile_{_ridx}"
                            print(f"\n  [ERROR] Rasterize tile {_rtname} failed — skipping:")
                            print(f"          {type(_rv_err).__name__}: {_rv_err}")
                            _tb.print_exc()
                            _rv_failed.append(_rtname)
                        break
                pending_r[executor.submit(_rasterize_tile_worker, job)] = idx_r
            # Drain remaining
            for f in as_completed(pending_r):
                _ridx = pending_r[f]
                try:
                    _rv_res = f.result()
                    tile_outputs.append(_rv_res[0])
                    _rv_txs_infos.append(_rv_res)
                except Exception as _rv_err:
                    import traceback as _tb
                    _rtname = jobs[_ridx][-1] if len(jobs[_ridx]) > 1 else f"tile_{_ridx}"
                    print(f"\n  [ERROR] Rasterize tile {_rtname} failed — skipping:")
                    print(f"          {type(_rv_err).__name__}: {_rv_err}")
                    _tb.print_exc()
                    _rv_failed.append(_rtname)

        if _rv_failed:
            print(f"\n  [WARN] {len(_rv_failed)}/{_n_rast_jobs} rasterize tiles failed:")
            for _fn in _rv_failed:
                print(f"    - {_fn}")

        # Write all_imgs.txs for the complete vector-rasterized tile set.
        if _rv_txs_infos:
            _write_txs_file(output_dir / "all_imgs.txs", _rv_txs_infos)

        # Write companion XML for every vector-rasterized tile.
        for _rv_tile_out in tile_outputs:
            _write_composite_material_xml(_rv_tile_out, classes)

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
    
    # Write .txr sidecar + all_imgs.txs for the final vector-rasterized output.
    try:
        with rasterio.open(final_path) as _rv_src:
            _rv_t  = _rv_src.transform
            _rv_c  = _normalize_pseudo_mercator_crs(_rv_src.crs)
            _rv_w  = _rv_src.width
            _rv_h  = _rv_src.height
        _write_txr_file(final_path, _rv_t, _rv_c, _rv_w, _rv_h)
        _rv_left, _rv_bottom, _rv_right, _rv_top = _bounds_wgs84(_rv_t, _rv_c, _rv_w, _rv_h)
        _write_txs_file(
            Path(final_path).parent / "all_imgs.txs",
            [(str(final_path), _rv_left, _rv_bottom, _rv_right, _rv_top, _rv_w, _rv_h)],
        )
    except Exception as _rv_exc:
        print(f"  [TXR/TXS] Warning: could not write sidecar files: {_rv_exc}")

    # Write companion XML for the final vector-rasterized output.
    _write_composite_material_xml(final_path, classes)

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

        _n_workers = min(len(_filter_jobs), _WORKER_THREAD_CAP)
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
    _feat_bytes = n_pixels * n_features * 4  # float32 = 4 bytes
    _avail = _available_ram_bytes()
    if _feat_bytes > _avail:
        raise MemoryError(
            f"Image too large for non-tiled processing: feature array would need "
            f"{_feat_bytes / 1e9:.1f} GB but only {_avail / 1e9:.1f} GB is available. "
            f"Auto-switching to tile mode."
        )
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
        _n_cont_workers = min(n_clusters, _WORKER_THREAD_CAP)
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
    """Assign each KMeans cluster to a MEA material via Hungarian 1:1 mapping.

    Each material in MEA_CLASSES carries a list of RGB anchors.  The cost from
    cluster *i* to material *j* is the squared RGB distance from the cluster's
    mean RGB to its **nearest** anchor in material *j*.  Assignment is then
    solved as a linear sum assignment problem (Hungarian) so that **each
    cluster maps to a distinct material** — preventing the previous failure
    mode where 3 KMeans clusters all collapsed onto BM_VEGETATION because
    vegetation's anchor set was wider than soil's or sand's.

    When ``n_clusters > n_materials`` (more clusters than materials), Hungarian
    assigns ``n_materials`` of them 1:1 and the leftover clusters fall back to
    nearest-material (duplicates allowed), preserving the legacy contract.

    The optional ``material_prior`` adds a frequency-weighted nudge: rare
    materials get a small cost bump so common ones win ties.

    The legacy parameters ``cluster_counts`` and ``cluster_semantics`` are
    accepted for caller compatibility but unused — the v6 SAM3-first pipeline
    delegates road/building disambiguation to mask sources.

    Returns ``(mapping, color_table)`` matching the legacy contract.
    """
    if not cluster_rgbs or not material_classes:
        return [], []
    _ = (cluster_counts, cluster_semantics)   # reserved for future re-introduction

    n_clusters = len(cluster_rgbs)
    n_materials = len(material_classes)

    # Resolve anchors against the active user profile once per call.  Profile
    # overrides (from the MEA Calibration Tool) take precedence over the
    # hardcoded MEA_CLASSES anchors.
    active_anchors = _resolve_active_anchor_map()

    material_anchors: List[List[Tuple[int, int, int]]] = []
    for cls in material_classes:
        name = cls.get("name", "")
        if name in active_anchors:
            anchors = [tuple(int(c) for c in a) for a in active_anchors[name]]
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

    # Prevalence bump: ``MEA_PREVALENCE_WEIGHT * 1000`` (≈280 cost units) is
    # tuned so rare-material penalties roughly equal a 17-unit per-channel RGB
    # gap.  Strong-color clusters (>50 RGB units from runner-up anchor) are
    # never flipped by this term; only mid-ambiguous ones are.
    if material_prior:
        prevalence_bump = MEA_PREVALENCE_WEIGHT * 1000.0
        for j, cls in enumerate(material_classes):
            p = float(material_prior.get(cls.get("name", ""), 0.0))
            cost[:, j] += prevalence_bump * (1.0 - p)

    # Hue-aware shape penalty.  Pure-RGB distance can't tell shadow-vegetation
    # (RGB ~50,60,35) from dark soil (RGB ~85,55,30) — they're closer in
    # Euclidean distance than the truly-green anchors.  This causes Hungarian
    # to invert the vegetation/soil assignment on dry farmland imagery: the
    # tan/khaki bare-ground cluster goes to vegetation while the truly-green
    # shadow cluster goes to soil.  Encode the canonical color shape of each
    # material as a soft penalty so Hungarian respects "G > R = green-class,
    # R > G = brown-class" semantics that RGB distance alone can't see.
    SHAPE_PENALTY = 30000.0   # ~ a synthetic 100-unit-per-channel RGB gap
    material_names = [cls.get("name", "") for cls in material_classes]
    for i, (r, g, b) in enumerate(cluster_rgbs):
        g_dominant = g > r              # vegetation requires G > R
        bright     = (r + g + b) > 420  # sand requires avg ≳ 140 per channel
        for j, name in enumerate(material_names):
            if name == "BM_VEGETATION" and not g_dominant:
                cost[i, j] += SHAPE_PENALTY
            elif name == "BM_SOIL" and g_dominant:
                cost[i, j] += SHAPE_PENALTY
            elif name == "BM_SAND" and not bright:
                cost[i, j] += SHAPE_PENALTY

    # Hungarian (linear_sum_assignment) finds the globally optimal 1:1
    # assignment between clusters and materials.  When n_clusters == n_materials
    # this is a perfect bijection.  When they differ, scipy assigns
    # min(n_clusters, n_materials) pairs and we resolve the remainder with
    # plain nearest-material so the function still returns one entry per
    # cluster.
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    cluster_to_material = np.full(n_clusters, -1, dtype=int)
    cluster_to_material[row_ind] = col_ind
    if (cluster_to_material < 0).any():
        leftover = np.where(cluster_to_material < 0)[0]
        cluster_to_material[leftover] = np.argmin(cost[leftover], axis=1)

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