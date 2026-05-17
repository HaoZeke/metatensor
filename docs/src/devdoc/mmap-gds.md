# Memory-mapped + GDS loading

This document covers two related, optional loading paths in
metatensor-core that live on top of the
`mts_create_file_array_callback_t` surface introduced for `load_mmap`.

## 1. mmap-backed partial loading (multi-region callback)

`mts_tensormap_load_partial` materialises owned
arrays via the standard `mts_create_array_callback_t`. For applications
that want zero-copy partial loads (e.g. select 10% of samples, keep all
properties, never copy the underlying bytes), the multi-region callback is:

```c
typedef mts_status_t (*mts_create_partial_file_array_callback_t)(
    void *user_data,
    const uintptr_t *shape,
    uintptr_t shape_count,
    DLDataType dtype,
    uintptr_t region_count,
    const uintptr_t *file_offsets,  // length region_count
    const uintptr_t *region_lens,   // length region_count, in bytes
    struct mts_array_t *array
);
```

The callback receives `region_count` contiguous byte runs in the file;
it constructs an `mts_array_t` whose data is the logical concatenation
of those regions. For full-row selection, `region_count` is the number
of coalesced byte runs. Contiguous selected rows are merged, so
`select_all` selection on every dimension degenerates to `region_count = 1`
and the callback is equivalent to the single-region
`mts_create_file_array_callback_t`.

The entry point is
`mts_tensormap_load_partial_mmap(path, keys, samples, create_array,
user_data)`. The "list of regions" shape mirrors
`mts_data_movement_t` (see `metatensor-core/src/data.rs`) so users
familiar with the move-data API will recognise it.

Property filtering stays on `load_partial`: selecting individual
properties produces many tiny regions and is a poor fit for mmap/GDS
callbacks. Use `load_partial_mmap` for key and sample selection, and
`load_partial` when property selection is required.

## 2. GPU Direct Storage loader

The Python module `metatensor.io._mmap_gds` uses the same
`mts_create_file_array_callback_t` is sufficient to host a path that
loads arrays directly to GPU memory through NVIDIA cuFile (via the
`kvikio` package). The C API does **not** change; the callback uses
`void *user_data` (currently `NULL`) and the `file_offset` parameter to
issue `pread` calls into pre-allocated CuPy arrays.

### Install

GDS requires:

- NVIDIA driver with `nvidia-fs` kernel module loaded
- `libcufile` (part of CUDA Toolkit >= 12.0)
- `cupy` matching your CUDA version
- `kvikio` (`pip install kvikio-cu12` for CUDA 12.x)

### Use

```python
from metatensor.io._mmap_gds import load_mmap_gds

tensor = load_mmap_gds("path/to/data.mts")
# every block.values is a cupy GPU array
```

### Limitations

- File must use the `STORED` ZIP format that `mts_*_save` writes.
- Numeric arrays must use native byte order.
- The loader opens one cuFile handle per load call; the
  handle is kept alive by the callback closure for the duration of the
  load.
- Not exercised in default CI; depends on system cuFile + cupy +
  kvikio. The CI smoke test in
  `python/metatensor_core/tests/serialization.py::test_load_mmap_gds_values_equal`
  is `importorskip`'d when those packages are missing.
