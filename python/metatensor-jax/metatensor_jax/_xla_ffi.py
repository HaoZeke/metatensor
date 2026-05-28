"""
XLA FFI registration for the metatensor-jax C++ handlers.

Mirrors metatensor_torch/_c_lib.py: importing this module loads
libmetatensor_jax.so via :mod:`._c_lib`, then hands each
``metatensor_<op>`` symbol to :func:`jax.ffi.register_ffi_target`. Once
registration runs, JAX can lower a metatensor operation into the jaxpr as
``ffi_call('metatensor_<op>', ...)`` and the handler name appears in
``jax.make_jaxpr`` output (the R4 forcing test).
"""

from __future__ import annotations

import ctypes
import warnings


# The handler names below match the ``XLA_FFI_DEFINE_HANDLER_SYMBOL`` entries
# in metatensor-jax/src/register.cpp one-for-one. The ``metatensor_`` prefix
# is what R2 of the HARD packet requires.
_FFI_TARGETS = [
    # Labels
    "metatensor_labels_clone",
    "metatensor_labels_free",
    "metatensor_labels_count",
    "metatensor_labels_size",
    "metatensor_labels_position",
    "metatensor_labels_union",
    "metatensor_labels_intersection",
    "metatensor_labels_difference",
    # Block
    "metatensor_block_free",
    "metatensor_block_copy",
    "metatensor_block_labels",
    "metatensor_block_gradient",
    # TensorMap
    "metatensor_tensormap_free",
    "metatensor_tensormap_copy",
    "metatensor_tensormap_keys",
    "metatensor_tensormap_block_by_id",
    "metatensor_tensormap_blocks_matching",
    # Merge plan
    "metatensor_keys_to_properties_plan",
    "metatensor_keys_to_samples_plan",
    "metatensor_merge_plan_free",
    "metatensor_merge_plan_new_keys",
    "metatensor_merge_plan_block_count",
    "metatensor_merge_plan_block_samples",
    "metatensor_merge_plan_block_properties",
    "metatensor_merge_plan_block_input_count",
    # Operations layer
    "metatensor_sort",
    "metatensor_slice",
    "metatensor_mean_over_samples",
    "metatensor_sum_over_samples",
    # IO
    "metatensor_labels_load",
    "metatensor_labels_save",
    "metatensor_tensormap_load",
    "metatensor_tensormap_save",
]


_REGISTERED = False
_LIB: "ctypes.CDLL | None" = None


def register_ffi_targets() -> bool:
    """
    Register every metatensor XLA FFI target with JAX. Called from
    :mod:`metatensor_jax.__init__` on import; safe to call manually.

    Returns ``True`` if registration succeeded, ``False`` otherwise.
    """
    global _REGISTERED, _LIB

    if _REGISTERED:
        return True

    try:
        import jax.ffi as jax_ffi
    except (ImportError, AttributeError):
        return False

    try:
        from ._c_lib import _load_library
    except ImportError as exc:
        warnings.warn(
            f"metatensor-jax C++ library not available: {exc}", stacklevel=2
        )
        return False

    try:
        _LIB = _load_library()
    except OSError as exc:
        warnings.warn(
            f"failed to load libmetatensor_jax: {exc}", stacklevel=2
        )
        return False

    registered = 0
    for name in _FFI_TARGETS:
        try:
            fn = getattr(_LIB, name)
        except AttributeError:
            warnings.warn(
                f"libmetatensor_jax does not export {name}", stacklevel=2
            )
            continue

        # Newer jaxlib expects a PyCapsule; older versions accept the raw
        # ctypes function pointer.
        capsule = fn
        pycapsule = getattr(jax_ffi, "pycapsule", None)
        if pycapsule is not None:
            try:
                capsule = pycapsule(fn)
            except Exception:  # noqa: BLE001
                capsule = fn

        try:
            jax_ffi.register_ffi_target(name, capsule, platform="cpu")
            registered += 1
        except Exception as exc:  # noqa: BLE001 (jax raises bare RuntimeError)
            warnings.warn(
                f"failed to register XLA FFI target {name}: {exc}",
                stacklevel=2,
            )

    _REGISTERED = registered > 0
    return _REGISTERED


def is_registered() -> bool:
    """Return ``True`` once :func:`register_ffi_targets` has succeeded."""
    return _REGISTERED
