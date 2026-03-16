"""
Runtime hook: make bundled CUDA DLLs (nvidia/*/bin/*.dll) discoverable
inside the frozen exe before CuPy's _softlink loads them.

Problem: cuda-pathfinder normally finds CUDA DLLs via importlib.metadata
(looking up the nvidia-* packages).  In a PyInstaller single-file exe the
dist-info metadata is stripped, so cuda-pathfinder always fails and CuPy
falls back to faiss-cpu even when a GPU is present.

Fix: before any cupy import happens (this hook runs first):
  1. Add every nvidia/*/bin/ dir in _MEIPASS to os.add_dll_directory()
     so the Windows DLL loader finds them when loading *.pyd extensions.
  2. Prepend those dirs to PATH so ctypes.util.find_library("cublas64_12")
     etc. resolve to the bundled DLLs.
  3. Monkey-patch cuda-pathfinder's loader to search _MEIPASS first,
     bypassing the broken importlib.metadata lookup entirely.
"""
import os
import sys
import glob
import ctypes

if sys.platform != "win32" or not getattr(sys, "frozen", False):
    # Only needed in the frozen exe on Windows
    pass
else:
    _base = sys._MEIPASS
    _nvidia_root = os.path.join(_base, "nvidia")

    # ── 1 & 2: add every nvidia/*/bin/ to DLL search paths ───────────────────
    _dll_dirs = []
    if os.path.isdir(_nvidia_root):
        for _pkg in os.listdir(_nvidia_root):
            _bin = os.path.join(_nvidia_root, _pkg, "bin")
            if os.path.isdir(_bin):
                _dll_dirs.append(_bin)
                try:
                    os.add_dll_directory(_bin)
                except Exception:
                    pass

    if _dll_dirs:
        os.environ["PATH"] = (
            os.pathsep.join(_dll_dirs) + os.pathsep + os.environ.get("PATH", "")
        )

    # ── 3: patch cuda-pathfinder to look in _MEIPASS/nvidia/*/bin/ ────────────
    # Build a fast name→path map of every DLL we bundled.
    _dll_map: dict[str, str] = {}
    for _dll in glob.glob(os.path.join(_nvidia_root, "**", "*.dll"), recursive=True):
        _dll_map[os.path.basename(_dll).lower()] = _dll

    def _meipass_load(libname: str):
        """Try the bundled CUDA DLLs first, then fall back to the real loader."""
        # libname may be e.g. "curand" or "curand*.dll" or "curand64_10"
        _stem = libname.rstrip("*").lower()
        # Exact match
        if _stem in _dll_map:
            return ctypes.CDLL(_dll_map[_stem])
        # Prefix match (libname = "curand" → curand64_10.dll)
        for _name, _path in _dll_map.items():
            if _name.startswith(_stem):
                return ctypes.CDLL(_path)
        # Nothing found — let the original loader raise a proper error
        raise OSError(f"[rthook] CUDA DLL not found in bundle: {libname!r}")

    try:
        from cuda.pathfinder._dynamic_libs import load_nvidia_dynamic_lib as _lndl_mod
        _orig = _lndl_mod.load_nvidia_dynamic_lib

        def _patched(libname: str):
            try:
                return _meipass_load(libname)
            except OSError:
                return _orig(libname)   # original as fallback

        _lndl_mod.load_nvidia_dynamic_lib = _patched
    except Exception:
        pass  # cuda-pathfinder not present or API changed — PATH/add_dll_directory suffice
