"""
Backend-agnostic array dispatch for metatensor operations.

Uses the Python Array API Standard via vendored array-api-compat to provide
uniform dispatch across numpy, torch, and jax arrays. TorchScript compilation
is preserved via ``torch_jit_is_scripting()`` guards that dead-code-eliminate
the array API path during script compilation.

Functions are organized into three categories:

Category A: Direct Array API Standard -- use ``xp = array_namespace(array)``
Category B: Framework-specific adapters -- thin wrappers for in-place ops,
            gradient tracking, and other backend-specific semantics
Category C: TorchScript-only -- sort helpers for TorchScript compilation
"""

import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ._backend import torch_jit_is_scripting, torch_jit_script
from ._vendored.array_api_compat import array_namespace, device as _array_device


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


try:
    import torch
    from torch import Tensor as TorchTensor

    torch_dtype = torch.dtype
    torch_device = torch.device

except ImportError:

    class TorchTensor:
        pass

    class torch_dtype:
        pass

    class torch_device:
        pass


UNKNOWN_ARRAY_TYPE = (
    "unknown array type in metatensor dispatch: expected an array type "
    "supported by array-api-compat (numpy, torch, jax, cupy, ...)"
)


def _check_all_torch_tensor(arrays: List[TorchTensor]):
    for array in arrays:
        if not isinstance(array, TorchTensor):
            raise TypeError(
                f"expected argument to be a torch.Tensor, but got {type(array)}"
            )


def _check_all_np_ndarray(arrays):
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"expected argument to be a np.ndarray, but got {type(array)}"
            )


def _maybe_requires_grad(result, requires_grad: bool):
    """Apply requires_grad if the array supports it (torch only)."""
    if requires_grad and hasattr(result, "requires_grad_"):
        return result.requires_grad_(True)
    return result


def _get_device(array):
    """Get array device using array-api-compat."""
    return _array_device(array)


# ============================================================================ #
# Category A: Direct Array API Standard
#
# These functions delegate to xp.func() from the array namespace.
# TorchScript path preserved via torch_jit_is_scripting() guard.
# ============================================================================ #


def sum(array, axis: Optional[int] = None):
    """Returns the sum of the elements in the array at the axis."""
    if torch_jit_is_scripting():
        return torch.sum(array, dim=axis)
    xp = array_namespace(array)
    return xp.sum(array, axis=axis)


def abs(array):
    """Returns the absolute value of the elements in the array."""
    if torch_jit_is_scripting():
        return torch.abs(array)
    xp = array_namespace(array)
    return xp.abs(array)


def all(a, axis: Optional[int] = None):
    """Test whether all array elements along a given axis evaluate to True."""
    if torch_jit_is_scripting():
        if axis is None:
            return torch.all(a)
        else:
            return torch.all(a, dim=axis)
    xp = array_namespace(a)
    return xp.all(a, axis=axis)


def concatenate(arrays: List[TorchTensor], axis: int):
    """Concatenate a group of arrays along a given axis."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor(arrays)
        return torch.cat(arrays, axis)
    xp = array_namespace(arrays[0])
    return xp.concat(arrays, axis=axis)


def stack(arrays: List[TorchTensor], axis: int):
    """Stack a group of arrays along a new axis."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor(arrays)
        return torch.stack(arrays, axis)
    xp = array_namespace(arrays[0])
    return xp.stack(arrays, axis=axis)


def sqrt(array):
    """Compute the square root of the input array."""
    if torch_jit_is_scripting():
        return torch.sqrt(array)
    xp = array_namespace(array)
    return xp.sqrt(array)


def sign(array):
    """Returns an indication of the sign of the elements in the array."""
    if torch_jit_is_scripting():
        return torch.sgn(array)
    xp = array_namespace(array)
    return xp.sign(array)


def unique(array, axis: Optional[int] = None):
    """Find the unique elements of an array."""
    if torch_jit_is_scripting():
        return torch.unique(array, dim=axis)
    # array API standard has unique_values (no axis) and unique_all.
    # Fall back to backend-specific unique for axis support.
    if isinstance(array, TorchTensor):
        return torch.unique(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, axis=axis)
    else:
        # jax and others: try xp.unique_values if no axis, else backend unique
        xp = array_namespace(array)
        if axis is None and hasattr(xp, "unique_values"):
            return xp.unique_values(array)
        # jax.numpy has unique with axis parameter
        if hasattr(xp, "unique"):
            return xp.unique(array, axis=axis)
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def unique_with_inverse(array, axis: Optional[int] = None):
    """Return the unique entries of ``array``, along with inverse indices."""
    if torch_jit_is_scripting():
        return torch.unique(array, return_inverse=True, dim=axis)
    if isinstance(array, TorchTensor):
        return torch.unique(array, return_inverse=True, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, return_inverse=True, axis=axis)
    else:
        xp = array_namespace(array)
        if axis is None and hasattr(xp, "unique_inverse"):
            result = xp.unique_inverse(array)
            return result.values, result.inverse_indices
        if hasattr(xp, "unique"):
            return xp.unique(array, return_inverse=True, axis=axis)
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def where(array):
    """Return the indices where ``array`` is True.

    Returns a tuple of arrays, one per dimension.
    """
    if torch_jit_is_scripting():
        return torch.where(array)
    if isinstance(array, TorchTensor):
        return torch.where(array)
    elif isinstance(array, np.ndarray):
        return np.where(array)
    else:
        xp = array_namespace(array)
        return xp.nonzero(array)


def solve(X, Y):
    """Computes the solution of a square system of linear equations."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor([Y])
        return torch.linalg.solve(X, Y)
    xp = array_namespace(X)
    return xp.linalg.solve(X, Y)


def norm(array, axis=None):
    """Compute the 2-norm (Frobenius norm for matrices) of the input array."""
    if torch_jit_is_scripting():
        return torch.linalg.norm(array, dim=axis)
    xp = array_namespace(array)
    if hasattr(xp, "linalg") and hasattr(xp.linalg, "norm"):
        return xp.linalg.norm(array, axis=axis)
    # Fallback for array API standard: vector_norm for 1D
    if hasattr(xp.linalg, "vector_norm"):
        return xp.linalg.vector_norm(array, axis=axis)
    raise TypeError(UNKNOWN_ARRAY_TYPE)


def allclose(
    a: TorchTensor,
    b: TorchTensor,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
):
    """Compare two arrays using ``allclose``."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor([b])
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if isinstance(a, TorchTensor):
        _check_all_torch_tensor([b])
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif isinstance(a, np.ndarray):
        _check_all_np_ndarray([b])
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        # jax.numpy has allclose
        xp = array_namespace(a)
        if hasattr(xp, "allclose"):
            return xp.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
        # Fallback: manual implementation using array API
        diff = xp.abs(a - b)
        limit = atol + rtol * xp.abs(b)
        return bool(xp.all(diff <= limit))


def nan_to_num(
    X,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
):
    """Replace NaN and Inf values."""
    if torch_jit_is_scripting():
        return torch.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
    if isinstance(X, TorchTensor):
        return torch.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
    elif isinstance(X, np.ndarray):
        return np.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
    else:
        xp = array_namespace(X)
        if hasattr(xp, "nan_to_num"):
            return xp.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
        # Fallback: manual with array API primitives
        result = xp.where(xp.isnan(X), xp.asarray(nan, dtype=X.dtype), X)
        if posinf is not None:
            result = xp.where(
                xp.isinf(result) & (result > 0),
                xp.asarray(posinf, dtype=X.dtype),
                result,
            )
        if neginf is not None:
            result = xp.where(
                xp.isinf(result) & (result < 0),
                xp.asarray(neginf, dtype=X.dtype),
                result,
            )
        return result


# ============================================================================ #
# Category B: Framework-specific adapters
#
# These functions have inherently backend-specific semantics (in-place ops,
# gradient tracking, device handling, etc.) and use thin adapters.
# ============================================================================ #


def copy(array):
    """Returns a copy of ``array``. The new data is not shared with the original."""
    if torch_jit_is_scripting():
        return array.clone()
    if isinstance(array, TorchTensor):
        return array.clone()
    elif isinstance(array, np.ndarray):
        return array.copy()
    else:
        xp = array_namespace(array)
        # array API: asarray with copy=True
        if hasattr(xp, "asarray"):
            try:
                return xp.asarray(array, copy=True)
            except TypeError:
                pass
        # jax: jnp.array creates a copy
        return xp.array(array)


def detach(array):
    """Returns a new array, detached from the computational graph, if any."""
    if torch_jit_is_scripting():
        return array.detach()
    if hasattr(array, "detach"):
        return array.detach()
    # numpy and jax arrays have no computation graph
    return array


def requires_grad(array, value: bool):
    """Set ``requires_grad`` on ``array``. No-op for numpy/jax arrays."""
    if torch_jit_is_scripting():
        if value and array.requires_grad:
            warnings.warn(
                "setting `requires_grad=True` again on a Tensor will detach "
                "the Tensor",
                stacklevel=1,
            )
        return array.detach().requires_grad_(value)
    if hasattr(array, "requires_grad_"):
        if value and getattr(array, "requires_grad", False):
            warnings.warn(
                "setting `requires_grad=True` again on a Tensor will detach "
                "the Tensor",
                stacklevel=1,
            )
        return array.detach().requires_grad_(value)
    elif isinstance(array, np.ndarray):
        if value:
            warnings.warn(
                "`requires_grad=True` does nothing for numpy arrays",
                stacklevel=1,
            )
        return array
    else:
        # jax and others: requires_grad is not applicable
        return array


def empty_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """Create an uninitialized array with the given ``shape`` and same dtype/device."""
    if torch_jit_is_scripting():
        if shape is None:
            shape = array.size()
        return torch.empty(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    xp = array_namespace(array)
    if shape is None:
        shape = list(array.shape)
    device = _get_device(array)
    result = xp.empty(shape, dtype=array.dtype, device=device)
    return _maybe_requires_grad(result, requires_grad)


def zeros_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """Create an array filled with zeros."""
    if torch_jit_is_scripting():
        if shape is None:
            shape = array.size()
        return torch.zeros(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    xp = array_namespace(array)
    if shape is None:
        shape = list(array.shape)
    device = _get_device(array)
    result = xp.zeros(shape, dtype=array.dtype, device=device)
    return _maybe_requires_grad(result, requires_grad)


def ones_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """Create an array filled with ones."""
    if torch_jit_is_scripting():
        if shape is None:
            shape = array.size()
        return torch.ones(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    xp = array_namespace(array)
    if shape is None:
        shape = list(array.shape)
    device = _get_device(array)
    result = xp.ones(shape, dtype=array.dtype, device=device)
    return _maybe_requires_grad(result, requires_grad)


def rand_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """Create an array with random uniform values in [0, 1)."""
    if torch_jit_is_scripting():
        if shape is None:
            shape = array.shape
        return torch.rand(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    # Random generation is not in the array API standard -- backend-specific
    if isinstance(array, TorchTensor):
        if shape is None:
            shape = list(array.shape)
        result = torch.rand(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        )
        return _maybe_requires_grad(result, requires_grad)
    elif isinstance(array, np.ndarray):
        if shape is None:
            shape = list(array.shape)
        return np.random.rand(*shape).astype(array.dtype)
    else:
        # jax
        xp = array_namespace(array)
        if shape is None:
            shape = list(array.shape)
        try:
            import jax

            key = jax.random.PRNGKey(0)
            return jax.random.uniform(key, shape=shape, dtype=array.dtype)
        except ImportError:
            raise TypeError(
                "random array generation requires numpy, torch, or jax"
            )


def eye_like(array, size: int):
    """Create an identity matrix with the given ``size``."""
    if torch_jit_is_scripting():
        return torch.eye(size).to(array.dtype).to(array.device)
    xp = array_namespace(array)
    device = _get_device(array)
    return xp.eye(size, dtype=array.dtype, device=device)


def is_contiguous(array):
    """Check if a given array is C-contiguous."""
    if torch_jit_is_scripting():
        return array.is_contiguous()
    if hasattr(array, "is_contiguous"):
        return array.is_contiguous()
    elif isinstance(array, np.ndarray):
        return array.flags["C_CONTIGUOUS"]
    else:
        # jax arrays are always C-contiguous
        return True


def make_contiguous(array):
    """Returns a C-contiguous array."""
    if torch_jit_is_scripting():
        if array.is_contiguous():
            return array
        return array.contiguous()
    if hasattr(array, "contiguous"):
        if hasattr(array, "is_contiguous") and array.is_contiguous():
            return array
        return array.contiguous()
    elif isinstance(array, np.ndarray):
        if array.flags["C_CONTIGUOUS"]:
            return array
        return np.ascontiguousarray(array)
    else:
        # jax: arrays are always contiguous
        return array


def rows_add(output_array, input_array, index):
    """Scatter-add rows: ``output_array[index[i]] += input_array[i]``.

    Always returns the (potentially new) output array. Callers must use the
    return value to support JAX, where arrays are immutable.
    """
    _index_array_checks(index)
    if torch_jit_is_scripting():
        if not isinstance(index, TorchTensor):
            index = torch.tensor(index).to(device=input_array.device)
        _check_all_torch_tensor([output_array, input_array, index])
        output_array.index_add_(0, index, input_array)
        return output_array
    if isinstance(input_array, TorchTensor):
        if not isinstance(index, TorchTensor):
            index = torch.tensor(index).to(device=input_array.device)
        _check_all_torch_tensor([output_array, input_array, index])
        output_array.index_add_(0, index, input_array)
        return output_array
    elif isinstance(input_array, np.ndarray):
        _check_all_np_ndarray([output_array, input_array, index])
        np.add.at(output_array, index, input_array)
        return output_array
    elif hasattr(output_array, "at"):
        # jax: functional scatter via .at[].add()
        return output_array.at[index].add(input_array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def columns_add(output_array, input_array, index):
    """Scatter-add columns: ``output_array[..., index[i]] += input_array[..., i]``.

    Always returns the (potentially new) output array. Callers must use the
    return value to support JAX, where arrays are immutable.
    """
    _index_array_checks(index)
    if torch_jit_is_scripting():
        if not isinstance(index, TorchTensor):
            index = torch.tensor(index).to(device=input_array.device)
        _check_all_torch_tensor([output_array, input_array, index])
        output_array.index_add_(-1, index, input_array)
        return output_array
    if isinstance(input_array, TorchTensor):
        if not isinstance(index, TorchTensor):
            index = torch.tensor(index).to(device=input_array.device)
        _check_all_torch_tensor([output_array, input_array, index])
        output_array.index_add_(-1, index, input_array)
        return output_array
    elif isinstance(input_array, np.ndarray):
        _check_all_np_ndarray([output_array, input_array, index])
        np.add.at(output_array, (..., index), input_array)
        return output_array
    elif hasattr(output_array, "at"):
        # jax: functional scatter
        return output_array.at[..., index].add(input_array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def scatter_last_dim(array, index: int, value):
    """Equivalent to ``array[..., index] = value``."""
    if torch_jit_is_scripting():
        index = torch.tensor([index]).to(device=array.device)
        dim = array.ndim - 1
        size = [s for s in array.shape[:-1]] + [-1]
        return array.scatter(dim, index.unsqueeze(0).expand(size), value)
    if isinstance(array, TorchTensor):
        idx = torch.tensor([index]).to(device=array.device)
        dim = array.ndim - 1
        size = [s for s in array.shape[:-1]] + [-1]
        return array.scatter(dim, idx.unsqueeze(0).expand(size), value)
    elif isinstance(array, np.ndarray):
        array[..., index] = value
        return array
    elif hasattr(array, "at"):
        # jax: functional update
        return array.at[..., index].set(value)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def take(array, indices, axis: int):
    """Select elements from ``array`` along ``axis`` using ``indices``."""
    if torch_jit_is_scripting():
        return torch.index_select(array, dim=axis, index=indices)
    if isinstance(array, TorchTensor):
        return torch.index_select(array, dim=axis, index=indices)
    elif isinstance(array, np.ndarray):
        return np.take(array, indices, axis=axis)
    else:
        xp = array_namespace(array)
        return xp.take(array, indices, axis=axis)


def mask(array, axis: int, mask):
    """Apply a boolean mask along the specified axis."""
    indices = where(mask)[0]
    if torch_jit_is_scripting():
        return torch.index_select(array, dim=axis, index=indices)
    if isinstance(array, TorchTensor):
        return torch.index_select(array, dim=axis, index=indices)
    elif isinstance(array, np.ndarray):
        if isinstance(indices, TorchTensor):
            indices = indices.detach().cpu().numpy()
        return np.take(array, indices, axis=axis)
    else:
        xp = array_namespace(array)
        return xp.take(array, indices, axis=axis)


def slice_last_dim(array, index: Union[int, TorchTensor]):
    """Equivalent to ``array[..., index]``."""
    if torch_jit_is_scripting():
        if isinstance(index, int):
            index = torch.tensor([index]).to(device=array.device)
        return array.index_select(-1, torch.as_tensor(index))
    if isinstance(array, TorchTensor):
        if isinstance(index, int):
            index = torch.tensor([index]).to(device=array.device)
        return array.index_select(-1, torch.as_tensor(index))
    elif isinstance(array, np.ndarray):
        return np.take(array, index, axis=-1)
    else:
        xp = array_namespace(array)
        if isinstance(index, int):
            index = xp.asarray([index])
        return xp.take(array, index, axis=-1)


def dot(A, B):
    """Compute dot product: ``A @ B.T``. Assumes ``B`` is 2-dimensional."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor([B])
        assert len(B.shape) == 2
        return A @ B.T
    assert len(B.shape) == 2
    xp = array_namespace(A)
    B_T = xp.matrix_transpose(B) if hasattr(xp, "matrix_transpose") else B.T
    if len(A.shape) == 2:
        return A @ B_T
    elif isinstance(A, np.ndarray):
        return np.dot(A, B_T)
    else:
        return A @ B_T


def lstsq(X, Y, rcond: Optional[float], driver: Optional[str] = None):
    """Computes least squares solution: ``X @ result ~= Y``."""
    if torch_jit_is_scripting():
        _check_all_torch_tensor([Y])
        return torch.linalg.lstsq(X, Y, rcond=rcond, driver=driver)[0]
    if isinstance(X, TorchTensor):
        _check_all_torch_tensor([Y])
        return torch.linalg.lstsq(X, Y, rcond=rcond, driver=driver)[0]
    elif isinstance(X, np.ndarray):
        _check_all_np_ndarray([Y])
        return np.linalg.lstsq(X, Y, rcond=rcond)[0]
    else:
        xp = array_namespace(X)
        if hasattr(xp, "linalg") and hasattr(xp.linalg, "lstsq"):
            return xp.linalg.lstsq(X, Y, rcond=rcond)[0]
        raise TypeError(
            "lstsq is not available for this array type"
        )


def get_device(array):
    """Returns the device of the array."""
    if torch_jit_is_scripting():
        return array.device
    if isinstance(array, TorchTensor):
        return array.device
    elif isinstance(array, np.ndarray):
        return "cpu"
    else:
        return _get_device(array)


def int_array_like(int_list: List[int], like):
    """Create an int64 array from a list, matching ``like``'s backend/device."""
    if torch_jit_is_scripting():
        if like.device.type == "meta":
            device = torch.device("cpu")
        else:
            device = like.device
        return torch.tensor(int_list, dtype=torch.int64, device=device)
    if isinstance(like, TorchTensor):
        if like.device.type == "meta":
            device = torch.device("cpu")
        else:
            device = like.device
        return torch.tensor(int_list, dtype=torch.int64, device=device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        xp = array_namespace(like)
        device = _get_device(like)
        return xp.asarray(int_list, dtype=xp.int64, device=device)


def bool_array_like(bool_list: List[bool], like):
    """Create a bool array from a list, matching ``like``'s backend/device."""
    if torch_jit_is_scripting():
        return torch.tensor(bool_list, dtype=torch.bool, device=like.device)
    if isinstance(like, TorchTensor):
        return torch.tensor(bool_list, dtype=torch.bool, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(bool_list).astype(bool)
    else:
        xp = array_namespace(like)
        device = _get_device(like)
        return xp.asarray(bool_list, dtype=xp.bool, device=device)


def indices_like(shape: List[int], like):
    """Create an enumerated index grid array matching ``like``'s backend/device."""
    if torch_jit_is_scripting():
        indices = torch.meshgrid(
            [torch.arange(s, dtype=torch.int64, device=like.device) for s in shape],
            indexing="ij",
        )
        return torch.stack(indices, dim=-1).reshape(-1, len(shape))
    if isinstance(like, TorchTensor):
        indices = torch.meshgrid(
            [torch.arange(s, dtype=torch.int64, device=like.device) for s in shape],
            indexing="ij",
        )
        return torch.stack(indices, dim=-1).reshape(-1, len(shape))
    elif isinstance(like, np.ndarray):
        return np.indices(shape).reshape(len(shape), -1).T.astype(np.int64)
    else:
        xp = array_namespace(like)
        device = _get_device(like)
        grids = xp.meshgrid(
            *[xp.arange(s, dtype=xp.int64, device=device) for s in shape],
            indexing="ij",
        )
        return xp.stack(grids, axis=-1).reshape(-1, len(shape))


def bincount(input, weights: Optional[TorchTensor] = None, minlength: int = 0):
    """Count occurrences of each value in an array of non-negative ints."""
    if torch_jit_is_scripting():
        if weights is not None:
            _check_all_torch_tensor([weights])
        return torch.bincount(input, weights=weights, minlength=minlength)
    if isinstance(input, TorchTensor):
        if weights is not None:
            _check_all_torch_tensor([weights])
        return torch.bincount(input, weights=weights, minlength=minlength)
    elif isinstance(input, np.ndarray):
        if weights is not None:
            _check_all_np_ndarray([weights])
        return np.bincount(input, weights=weights, minlength=minlength)
    else:
        xp = array_namespace(input)
        if hasattr(xp, "bincount"):
            return xp.bincount(input, weights=weights, minlength=minlength)
        raise TypeError("bincount is not available for this array type")


def make_like(array, like):
    """Transform ``array`` to use the same backend/dtype/device as ``like``."""
    if torch_jit_is_scripting():
        return to(array, backend="torch", dtype=like.dtype, device=like.device)
    if isinstance(like, TorchTensor):
        return to(array, backend="torch", dtype=like.dtype, device=like.device)
    elif isinstance(like, np.ndarray):
        return to(array, backend="numpy", dtype=like.dtype, device="cpu")
    else:
        # For jax/other: convert via to()
        return to(array, dtype=like.dtype)


def to(
    array,
    backend: Optional[str] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[Union[str, torch_device]] = None,
):
    """Convert the array to the specified backend/dtype/device."""
    if torch_jit_is_scripting():
        if backend is None:
            backend = "torch"
        if dtype is None:
            dtype = array.dtype
        if device is None:
            device = array.device
        if isinstance(device, str):
            device = torch.device(device)
        if backend == "torch":
            return array.to(dtype=dtype).to(device=device)
        elif backend == "numpy":
            raise ValueError("cannot call numpy conversion when torch-scripting")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # torch Tensor
    if isinstance(array, TorchTensor):
        if backend is None:
            backend = "torch"
        if dtype is None:
            dtype = array.dtype
        if device is None:
            device = array.device
        if isinstance(device, str):
            device = torch.device(device)

        if backend == "torch":
            return array.to(dtype=dtype).to(device=device)
        elif backend == "numpy":
            return array.detach().cpu().numpy()
        elif backend == "jax":
            try:
                import jax.numpy as jnp

                np_array = array.detach().cpu().numpy()
                return jnp.array(np_array)
            except ImportError:
                raise ValueError("jax is not installed")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # numpy array
    elif isinstance(array, np.ndarray):
        if backend is None:
            backend = "numpy"
        if backend == "numpy":
            return np.array(array, dtype=dtype)
        elif backend == "torch":
            return torch.tensor(array, dtype=dtype, device=device)
        elif backend == "jax":
            try:
                import jax.numpy as jnp

                return jnp.array(array, dtype=dtype)
            except ImportError:
                raise ValueError("jax is not installed")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    else:
        # jax or other array API arrays
        xp = array_namespace(array)
        if backend is None or backend == "jax":
            if dtype is not None:
                return xp.astype(array, dtype)
            return array
        elif backend == "numpy":
            return np.asarray(array)
        elif backend == "torch":
            np_array = np.asarray(array)
            return torch.tensor(np_array, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}")


def _index_array_checks(array):
    if len(array.shape) != 1:
        raise ValueError("Index arrays must be 1D")

    if isinstance(array, TorchTensor):
        if torch.is_floating_point(array):
            raise ValueError("Index arrays must be integers")
    elif isinstance(array, np.ndarray):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Index arrays must be integers")
    # For other backends (jax), trust the caller


# ============================================================================ #
# Category C: TorchScript-only sort helpers
#
# These functions exist solely for TorchScript compilation of
# argsort_labels_values. They are no-ops in Python mode (the numpy path is
# used instead). Do not modify without understanding TorchScript constraints.
# ============================================================================ #


def argsort_labels_values(labels_values, reverse: bool = False):
    """Sort rows as aggregated tuples, return indices.

    In TorchScript mode, uses sort_list_N helpers.
    In Python mode, uses numpy (Labels values are always CPU).
    """
    if isinstance(labels_values, TorchTensor):
        if labels_values.shape[1] == 1:
            data = [(int(row[0]), i) for i, row in enumerate(labels_values)]
            return torch.tensor(
                [i[-1] for i in sort_list_2(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 2:
            data = [
                (int(row[0]), int(row[1]), i) for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_3(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 3:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_4(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 4:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_5(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 5:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_6(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 6:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_7(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 7:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_8(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 8:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_9(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 9:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                    int(row[8]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_10(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        else:
            raise Exception("labels_values.shape[1]> 9 is not supported")
    elif isinstance(labels_values, np.ndarray):
        list_tuples: List[List[int]] = labels_values.tolist()
        for i in range(len(labels_values)):
            list_tuples[i].append(i)
        list_tuples.sort(reverse=reverse)
        return np.array(list_tuples)[:, -1]
    else:
        # For jax/other: convert to numpy for label sorting (labels are CPU)
        np_values = np.asarray(labels_values)
        list_tuples = np_values.tolist()
        for i in range(len(np_values)):
            list_tuples[i].append(i)
        list_tuples.sort(reverse=reverse)
        return np.array(list_tuples)[:, -1]


@torch_jit_script
def sort_list_2(
    data: List[Tuple[int, int]], reverse: bool = False
) -> List[Tuple[int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_3(
    data: List[Tuple[int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_4(
    data: List[Tuple[int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_5(
    data: List[Tuple[int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_6(
    data: List[Tuple[int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_7(
    data: List[Tuple[int, int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_8(
    data: List[Tuple[int, int, int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_9(
    data: List[Tuple[int, int, int, int, int, int, int, int, int]],
    reverse: bool = False,
) -> List[Tuple[int, int, int, int, int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_10(
    data: List[Tuple[int, int, int, int, int, int, int, int, int, int]],
    reverse: bool = False,
) -> List[Tuple[int, int, int, int, int, int, int, int, int, int]]:
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))
