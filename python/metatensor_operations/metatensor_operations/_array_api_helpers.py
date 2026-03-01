"""
Helper functions for array API operations that are not part of the standard.

Follows the pattern of scikit-learn's ``sklearn/utils/_array_api.py``. These
functions provide uniform interfaces for operations that require
backend-specific handling but are used throughout metatensor operations.
"""

from ._vendored.array_api_compat import (
    array_namespace,
    device as array_device,
    is_jax_array,
    is_numpy_array,
    is_torch_array,
)


def get_namespace(*arrays):
    """Get the array namespace for one or more arrays.

    Thin wrapper around ``array_api_compat.array_namespace``.
    """
    return array_namespace(*arrays)


def is_writeable(array):
    """Check if the array supports in-place mutation.

    numpy and torch arrays are writeable; jax arrays are not.
    """
    if is_jax_array(array):
        return False
    return True


def get_device_str(array) -> str:
    """Get a string representation of the array's device.

    Normalizes device representations across backends to a common string.
    """
    if is_numpy_array(array):
        return "cpu"
    device = array_device(array)
    return str(device)


def arrays_share_backend(*arrays) -> bool:
    """Check if all arrays use the same backend."""
    if len(arrays) < 2:
        return True
    xp0 = array_namespace(arrays[0])
    return all(array_namespace(a) is xp0 for a in arrays[1:])
