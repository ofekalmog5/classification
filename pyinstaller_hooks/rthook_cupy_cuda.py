"""
Runtime hook: pre-load bundled CUDA DLLs before CuPy imports them.

WHY THIS IS NEEDED
------------------
cuda-pathfinder (which CuPy uses to locate CUDA DLLs) searches for them
via importlib.metadata / package directories.  In a PyInstaller frozen
exe those lookup paths don't exist, so every CUDA DLL load fails and
CuPy silently falls back to faiss-cpu even on a machine with a GPU.

HOW THIS FIX WORKS
------------------
1. Every nvidia/*/bin/ dir in _MEIPASS is added to os.add_dll_directory()
   and to PATH so the Windows DLL loader can resolve transitive dependencies.

2. All CUDA DLLs are pre-loaded using ctypes.CDLL(full_absolute_path).
   After that, any later ctypes.CDLL("cublas64_12.dll") (name-only call
   inside CuPy) finds the DLL already resident in the process and
   returns immediately — no filesystem search required.

Load order: cudart first (others depend on it), then nvrtc, cublas,
curand, then everything else.
"""
import os
import sys
import glob
import ctypes

if sys.platform != "win32" or not getattr(sys, "frozen", False):
    pass  # dev mode — no action needed, cuda-pathfinder works normally
else:
    _base       = sys._MEIPASS
    _nvidia_root = os.path.join(_base, "nvidia")

    if os.path.isdir(_nvidia_root):

        # ── Step 1: add every nvidia/*/bin/ to DLL search paths ──────────────
        _dll_dirs = []
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
                os.pathsep.join(_dll_dirs)
                + os.pathsep
                + os.environ.get("PATH", "")
            )

        # ── Step 2: pre-load DLLs by full path (cudart first) ─────────────────
        _LOAD_ORDER = ["cudart", "nvrtc", "cublas", "curand"]

        def _sort_key(p: str) -> int:
            n = os.path.basename(p).lower()
            for i, prefix in enumerate(_LOAD_ORDER):
                if n.startswith(prefix):
                    return i
            return len(_LOAD_ORDER)

        _all_dlls = glob.glob(
            os.path.join(_nvidia_root, "**", "*.dll"), recursive=True
        )
        for _dll_path in sorted(_all_dlls, key=_sort_key):
            try:
                ctypes.CDLL(_dll_path)
            except Exception:
                pass  # skip optional libs (cuTENSOR etc.) gracefully
