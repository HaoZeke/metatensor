"""
Locate and load ``libmetatensor_jax.so``. Mirrors
``python/metatensor_torch/metatensor_torch/_c_lib.py``: the cmake_ext
build drops the library under the package install path, and the loader
walks a small set of search paths so a fresh editable install, an
in-tree cargo build, and an installed wheel all resolve the same way.
"""

from __future__ import annotations

import ctypes
import os
import sys


_HERE = os.path.realpath(os.path.dirname(__file__))


def _candidate_paths() -> list[str]:
    """Search order for libmetatensor_jax.so."""
    if sys.platform.startswith("darwin"):
        lib_name = "libmetatensor_jax.dylib"
    elif sys.platform.startswith("win"):
        lib_name = "metatensor_jax.dll"
    else:
        lib_name = "libmetatensor_jax.so"

    paths: list[str] = []

    # 1. cmake_ext install (next to the python package)
    paths.append(os.path.join(_HERE, "lib", lib_name))
    paths.append(os.path.join(_HERE, lib_name))

    # 2. importlib.metadata-resolved install location: when pytest runs from
    # the source tree, _HERE points at the in-tree copy that has no .so
    # next to it; the installed copy is under site-packages. Walk
    # importlib.metadata so a `pip install .` followed by pytest from
    # python/metatensor-jax/ still finds the library.
    try:
        import importlib.metadata as md
        for f in md.files("metatensor-jax") or []:
            if f.name == lib_name:
                paths.append(str(f.locate()))
    except Exception:  # noqa: BLE001
        pass

    # 3. cargo workspace build outputs
    repo_root = os.path.realpath(
        os.path.join(_HERE, "..", "..", "..", "..")
    )
    for build_type in ("release", "debug"):
        paths.append(os.path.join(repo_root, "target", build_type, lib_name))

    # 4. cmake build dir (used during `pip install -e`)
    setup_build = os.path.realpath(
        os.path.join(_HERE, "..", "..", "build", "cmake-build")
    )
    paths.append(os.path.join(setup_build, lib_name))

    return paths


def _load_library() -> ctypes.CDLL:
    """Find and dlopen the metatensor-jax shared library."""
    # Load metatensor-core first so the dynamic linker resolves the mts_*
    # symbols that metatensor-jax references.
    try:
        import metatensor as _metatensor
        _metatensor._c_lib._get_library()
    except (ImportError, AttributeError):
        pass

    for candidate in _candidate_paths():
        if os.path.isfile(candidate):
            return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)

    # Last resort: rely on the dynamic linker (LD_LIBRARY_PATH or rpath).
    return ctypes.CDLL("libmetatensor_jax.so", mode=ctypes.RTLD_GLOBAL)
