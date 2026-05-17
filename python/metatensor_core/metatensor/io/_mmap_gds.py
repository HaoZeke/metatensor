"""
NVIDIA GPU Direct Storage (GDS) loader.

metatensor parses the NPY header for every array via mmap and hands
``(shape, dtype, file_offset)`` to a file-array callback. The callback issues
a cuFile ``pread`` (via ``kvikio``) into a pre-allocated CuPy GPU array.

cuFile dispatches between two paths automatically:

1. **Direct DMA path** (real GDS): when the GPU exposes ``pci_p2pdma``
   AND the filesystem / NVMe controller is registered with
   ``nvidia-fs``, the bytes go straight from the storage device to
   GPU memory via PCIe peer-to-peer. No host CPU, no page cache, no
   bounce buffer.
2. **Compat mode**: when any of those preconditions is missing,
   libcufile silently falls back to ``pread`` into a registered host
   bounce buffer followed by ``cudaMemcpyHtoD`` into the GPU buffer.
   API and result are identical to (1); only the data path differs.

**Hardware caveat**: NVIDIA gates the direct DMA path to *datacenter*
SKUs (A/H/L-series, RTX A-series and RTX 6000 Ada workstation cards).
Consumer GeForce GPUs (RTX 4070/4080/4090) do not expose the
``pci_p2pdma`` device attribute and therefore *cannot* take the direct
path; they run compat mode only. The metatensor code is identical on
both paths -- run the same loader on a supported GPU and direct DMA
lights up automatically with no code changes.

Requirements:
- ``cupy`` (cuda12x or cuda13x)
- ``kvikio`` (``kvikio-cu12``)
- ``torch`` (cuda build) -- DLPack-bridge cupy arrays into
  ``metatensor.create_mts_array`` which only handles numpy/torch.
- For the direct DMA path additionally: a supported GPU (see above),
  ``nvidia-fs`` kernel module loaded, and a GDS-registered filesystem
  (BeeGFS/WekaFS/GPFS/Lustre/NFS-over-RDMA/NVMe-oF-RDMA, or local
  ext4/XFS on a supported NVMe controller without mdraid).
"""

import contextlib
import ctypes
import pathlib
from typing import Optional, Union

try:
    import cupy as _cp
    import kvikio as _kvikio
    import torch as _torch  # used for DLPack interop to register the GPU array
    from kvikio.defaults import is_compat_mode_preferred as _is_compat
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "metatensor.io._mmap_gds requires cupy + kvikio + torch (cuda). "
        "Install with: uv pip install cupy-cuda12x kvikio-cu12 torch. "
        "For accelerated GDS, also load the nvidia-fs kernel module."
    ) from err

from .._block import TensorBlock
from .._c_api import (
    DLDataTypeCode,
    mts_create_file_array_callback_t,
    mts_create_partial_file_array_callback_t,
    mts_labels_t,
)
from .._c_lib import _get_library
from .._data._array import create_mts_array
from .._labels import Labels
from .._status import catch_exceptions
from .._tensor import TensorMap


@contextlib.contextmanager
def _cufile_open(path: str):
    cf = _kvikio.CuFile(path, "r")
    try:
        yield cf
    finally:
        cf.close()


_DLPACK_TO_CUPY = {
    (DLDataTypeCode.kDLFloat, 16): _cp.float16,
    (DLDataTypeCode.kDLFloat, 32): _cp.float32,
    (DLDataTypeCode.kDLFloat, 64): _cp.float64,
    (DLDataTypeCode.kDLInt, 8): _cp.int8,
    (DLDataTypeCode.kDLInt, 16): _cp.int16,
    (DLDataTypeCode.kDLInt, 32): _cp.int32,
    (DLDataTypeCode.kDLInt, 64): _cp.int64,
    (DLDataTypeCode.kDLUInt, 8): _cp.uint8,
    (DLDataTypeCode.kDLUInt, 16): _cp.uint16,
    (DLDataTypeCode.kDLUInt, 32): _cp.uint32,
    (DLDataTypeCode.kDLUInt, 64): _cp.uint64,
    (DLDataTypeCode.kDLBool, 8): _cp.bool_,
    (DLDataTypeCode.kDLComplex, 64): _cp.complex64,
    (DLDataTypeCode.kDLComplex, 128): _cp.complex128,
}


def _dlpack_to_cupy_dtype(dtype):
    if dtype.lanes != 1:
        raise ValueError(
            f"unsupported DLDataType for GDS: lanes={dtype.lanes} (expected 1)"
        )
    out = _DLPACK_TO_CUPY.get((dtype.code, dtype.bits))
    if out is None:
        raise ValueError(
            f"unsupported DLDataType for GDS: code={dtype.code} bits={dtype.bits}"
        )
    return out


def is_using_real_gds() -> bool:
    """
    Returns True if kvikio is using the GDS direct-DMA path, False if it
    has fallen back to compat mode (cuFile API + POSIX reads + host
    bounce buffer).
    """
    return not _is_compat()


def _make_gds_callback(cufile_handle):
    """
    Build a `mts_create_file_array_callback_t` that issues a cuFile
    pread of `product(shape) * dtype_bytes` bytes from `file_offset`
    directly into a freshly-allocated CuPy GPU array.
    """

    @catch_exceptions
    def callback(_user_data, shape_ptr, shape_count, dtype, file_offset, array_out):
        shape_list = [int(shape_ptr[i]) for i in range(shape_count)]
        cp_dtype = _dlpack_to_cupy_dtype(dtype)
        gpu_buf = _cp.empty(shape_list, dtype=cp_dtype)
        n_elems = int(gpu_buf.size)
        if n_elems > 0:
            n = cufile_handle.pread(gpu_buf, file_offset=int(file_offset)).get()
            if n != gpu_buf.nbytes:
                raise IOError(
                    f"cuFile short read at offset {file_offset}: "
                    f"expected {gpu_buf.nbytes} bytes, got {n}"
                )
        # metatensor's create_mts_array only knows numpy and torch arrays.
        # We zero-copy view the cupy buffer through DLPack as a torch CUDA
        # tensor, which keeps the data on the GPU; the torch tensor's
        # __dlpack__ keeps a reference to the underlying cupy memory.
        torch_view = _torch.from_dlpack(gpu_buf)
        array_out[0] = create_mts_array(torch_view)

    return callback


def load_mmap_gds(path: Union[str, pathlib.Path]) -> TensorMap:
    """
    Load a :py:class:`TensorMap` from ``path`` via NVIDIA GPU Direct
    Storage. Every value / gradient array is a CuPy GPU array filled by
    a cuFile ``pread`` directly from the file (no host-side copy on the
    GDS path; one host bounce in compat mode).

    The file must use ``STORED`` (uncompressed) ZIP entries and native
    byte order, same as :py:func:`load_mmap`.

    :param path: path of the file to load
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    encoded = path.encode("utf8")

    lib = _get_library()
    with _cufile_open(path) as cf:
        callback = _make_gds_callback(cf)
        ptr = lib.mts_tensormap_load_mmap(
            encoded,
            mts_create_file_array_callback_t(callback),
            None,
        )
    return TensorMap._from_ptr(ptr)


def _make_gds_partial_callback(cufile_handle):
    """
    Multi-region GDS callback: receives the list of `(file_offset,
    region_len)` byte runs and issues one cuFile pread per region into
    the corresponding slice of a freshly-allocated GPU array. The
    resulting array is the logical concatenation of the regions, in
    order.
    """

    @catch_exceptions
    def callback(
        _user_data,
        shape_ptr, shape_count,
        dtype,
        region_count,
        offsets_ptr, lens_ptr,
        array_out,
    ):
        shape_list = [int(shape_ptr[i]) for i in range(shape_count)]
        cp_dtype = _dlpack_to_cupy_dtype(dtype)
        gpu_buf = _cp.empty(shape_list, dtype=cp_dtype)
        flat = gpu_buf.view().reshape(-1)
        elem_bytes = flat.dtype.itemsize
        futures = []
        written_elems = 0
        for r in range(region_count):
            file_off = int(offsets_ptr[r])
            n_bytes = int(lens_ptr[r])
            if n_bytes == 0:
                continue
            if n_bytes % elem_bytes != 0:
                raise ValueError(
                    f"region length {n_bytes} is not a multiple of dtype size {elem_bytes}"
                )
            n_elems = n_bytes // elem_bytes
            dst = flat[written_elems : written_elems + n_elems]
            futures.append(
                (cufile_handle.pread(dst, file_offset=file_off), n_bytes, file_off)
            )
            written_elems += n_elems
        for fut, expected_bytes, file_off in futures:
            got = fut.get()
            if got != expected_bytes:
                raise IOError(
                    f"cuFile short read at offset {file_off}: "
                    f"expected {expected_bytes} bytes, got {got}"
                )
        torch_view = _torch.from_dlpack(gpu_buf)
        array_out[0] = create_mts_array(torch_view)

    return callback


def _labels_arg(labels: Optional[Labels]):
    """NULL pointer means 'select all on this dimension' to the C core."""
    if labels is None:
        return ctypes.POINTER(mts_labels_t)()
    return labels._as_mts_labels_t()


def load_partial_mmap_gds(
    path: Union[str, pathlib.Path],
    keys: Optional[Labels] = None,
    samples: Optional[Labels] = None,
) -> TensorMap:
    """
    Multi-region GDS partial-load: combines block / sample selection
    (mts_tensormap_load_partial_mmap) with one cuFile pread per kept
    sample row into a GPU buffer. Property selection is not supported
    here -- use :py:func:`metatensor.io.load_partial` for that.

    :param path: path of the file to load
    :param keys: optional key selector
    :param samples: optional sample selector
    """
    if isinstance(path, pathlib.Path):
        path = str(path)
    encoded = path.encode("utf8")

    lib = _get_library()
    with _cufile_open(path) as cf:
        callback = _make_gds_partial_callback(cf)
        ptr = lib.mts_tensormap_load_partial_mmap(
            encoded,
            _labels_arg(keys),
            _labels_arg(samples),
            mts_create_partial_file_array_callback_t(callback),
            None,
        )
    return TensorMap._from_ptr(ptr)


def load_block_mmap_gds(path: Union[str, pathlib.Path]) -> TensorBlock:
    """Block equivalent of :py:func:`load_mmap_gds`."""
    if isinstance(path, pathlib.Path):
        path = str(path)
    encoded = path.encode("utf8")

    lib = _get_library()
    with _cufile_open(path) as cf:
        callback = _make_gds_callback(cf)
        ptr = lib.mts_block_load_mmap(
            encoded,
            mts_create_file_array_callback_t(callback),
            None,
        )
    return TensorBlock._from_ptr(ptr, parent=None)
