"""
XLA FFI registration for metatensor C API functions.

Registers metatensor-xla-ffi shared library functions as XLA custom call
targets via jax.ffi.register_ffi_target. This enables compiled JAX programs
to call metatensor Rust code without going through Python callbacks.

Usage:
    import metatensor_jax
    # Registration happens automatically on import if jax.ffi is available
    # and libmetatensor_xla_ffi.so is found.

See PJRT_XLA_FFI_Integration.org for design details.
"""

import ctypes
import os
import warnings

_REGISTERED = False
_LIB = None

# List of all XLA FFI targets to register
_FFI_TARGETS = [
    # Labels
    "mts_xla_labels_clone",
    "mts_xla_labels_free",
    "mts_xla_labels_count",
    "mts_xla_labels_size",
    "mts_xla_labels_position",
    "mts_xla_labels_union",
    "mts_xla_labels_intersection",
    "mts_xla_labels_difference",
    # Blocks
    "mts_xla_block_free",
    "mts_xla_block_copy",
    "mts_xla_block_labels",
    "mts_xla_block_data",
    # TensorMap
    "mts_xla_tensormap_free",
    "mts_xla_tensormap_copy",
    "mts_xla_tensormap_keys",
    "mts_xla_tensormap_block_by_id",
    "mts_xla_tensormap_blocks_matching",
    # Merge Plan
    "mts_xla_keys_to_properties_plan",
    "mts_xla_keys_to_samples_plan",
    "mts_xla_merge_plan_free",
    "mts_xla_merge_plan_new_keys",
    "mts_xla_merge_plan_block_count",
    "mts_xla_merge_plan_block_samples",
    "mts_xla_merge_plan_block_properties",
    "mts_xla_merge_plan_block_input_count",
    # IO
    "mts_xla_labels_load",
    "mts_xla_labels_save",
    "mts_xla_tensormap_load",
    "mts_xla_tensormap_save",
]


def _find_xla_ffi_lib():
    """Find the metatensor-xla-ffi shared library."""
    # Look in common locations
    search_paths = [
        # Next to metatensor-core lib
        os.path.dirname(os.path.abspath(__file__)),
        # Cargo build output
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "target", "release",
        ),
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "target", "debug",
        ),
    ]

    lib_names = ["libmetatensor_xla_ffi.so", "libmetatensor_xla_ffi.dylib"]

    for path in search_paths:
        for name in lib_names:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                return full_path

    # Try LD_LIBRARY_PATH
    for name in lib_names:
        try:
            ctypes.CDLL(name)
            return name
        except OSError:
            pass

    return None


def register_ffi_targets():
    """
    Register all metatensor XLA FFI targets with JAX.

    This is called automatically on import of metatensor_jax if jax.ffi
    is available. It can also be called manually.

    Returns True if registration succeeded, False otherwise.
    """
    global _REGISTERED, _LIB

    if _REGISTERED:
        return True

    try:
        import jax.ffi
    except (ImportError, AttributeError):
        # jax.ffi not available (old JAX version or no JAX)
        return False

    lib_path = _find_xla_ffi_lib()
    if lib_path is None:
        warnings.warn(
            "metatensor-xla-ffi shared library not found. "
            "XLA custom call targets will not be available. "
            "Build with: cargo build -p metatensor-xla-ffi --release",
            stacklevel=2,
        )
        return False

    try:
        _LIB = ctypes.CDLL(lib_path)
    except OSError as e:
        warnings.warn(
            f"Failed to load metatensor-xla-ffi: {e}",
            stacklevel=2,
        )
        return False

    registered_count = 0
    for target_name in _FFI_TARGETS:
        try:
            fn_ptr = getattr(_LIB, target_name)
            jax.ffi.register_ffi_target(
                target_name, fn_ptr, platform="cpu"
            )
            registered_count += 1
        except (AttributeError, Exception) as e:
            warnings.warn(
                f"Failed to register XLA FFI target {target_name}: {e}",
                stacklevel=2,
            )

    _REGISTERED = registered_count > 0
    return _REGISTERED


def is_registered():
    """Check if XLA FFI targets have been registered."""
    return _REGISTERED
