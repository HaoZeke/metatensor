"""
JAX-traceable structural operations for metatensor.

Provides jax.custom_vjp wrappers for keys_to_properties and keys_to_samples,
enabling jax.jit and jax.grad through these operations.

The approach:
- Phase 1: Rust computes the merge plan (concrete, not traced)
- Phase 2: Python executes array scatter via Array API (traced by JAX)
"""

import ctypes
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    raise ImportError("metatensor-jax requires jax to be installed")

import metatensor
from metatensor._c_api import (
    c_uintptr_t,
    mts_data_movement_t,
    mts_labels_t,
    mts_merge_plan_t,
)
from metatensor._c_lib import _get_library
from metatensor.labels import Labels
from metatensor.tensor import TensorMap
from metatensor.block import TensorBlock
from metatensor.status import _check_pointer

from metatensor_operations._merge import execute_merge_plan


def _get_plan_data(lib, plan_ptr, tensor):
    """Extract plan data from a C merge plan pointer into Python structures."""
    # Get number of output blocks
    n_blocks = c_uintptr_t(0)
    lib.mts_merge_plan_block_count(plan_ptr, ctypes.byref(n_blocks))
    n_blocks = n_blocks.value

    block_data = []
    for block_idx in range(n_blocks):
        # Get output shape
        shape_ptr = ctypes.POINTER(c_uintptr_t)()
        shape_count = c_uintptr_t(0)
        lib.mts_merge_plan_block_shape(
            plan_ptr, block_idx, ctypes.byref(shape_ptr), ctypes.byref(shape_count)
        )
        output_shape = [shape_ptr[i] for i in range(shape_count.value)]

        # Get number of input blocks
        n_inputs = c_uintptr_t(0)
        lib.mts_merge_plan_block_input_count(
            plan_ptr, block_idx, ctypes.byref(n_inputs)
        )

        # Get movements for each input block
        input_values = []
        movements_per_block = []
        for input_idx in range(n_inputs.value):
            source_idx = c_uintptr_t(0)
            movements_ptr = ctypes.POINTER(mts_data_movement_t)()
            movements_count = c_uintptr_t(0)
            lib.mts_merge_plan_block_movements(
                plan_ptr,
                block_idx,
                input_idx,
                ctypes.byref(source_idx),
                ctypes.byref(movements_ptr),
                ctypes.byref(movements_count),
            )

            # Convert C movements to Python tuples
            movements = []
            for i in range(movements_count.value):
                m = movements_ptr[i]
                movements.append((
                    m.sample_in,
                    m.sample_out,
                    m.properties_start_in,
                    m.properties_start_out,
                    m.properties_length,
                ))

            src_block = tensor.block_by_id(source_idx.value)
            input_values.append(src_block.values)
            movements_per_block.append(movements)

        # Get labels
        samples_ptr = lib.mts_merge_plan_block_samples(plan_ptr, block_idx)
        _check_pointer(samples_ptr)
        samples = Labels._from_mts_labels_t(samples_ptr.contents)

        properties_ptr = lib.mts_merge_plan_block_properties(plan_ptr, block_idx)
        _check_pointer(properties_ptr)
        properties = Labels._from_mts_labels_t(properties_ptr.contents)

        block_data.append({
            "output_shape": output_shape,
            "input_values": input_values,
            "movements_per_block": movements_per_block,
            "samples": samples,
            "properties": properties,
        })

    # Get new keys
    keys_ptr = lib.mts_merge_plan_new_keys(plan_ptr)
    _check_pointer(keys_ptr)
    new_keys = Labels._from_mts_labels_t(keys_ptr.contents)

    return new_keys, block_data


def keys_to_properties_jax(
    tensor: TensorMap,
    keys_to_move: Union[str, Sequence[str], Labels],
    *,
    fill_value: float = 0.0,
    sort_samples: bool = True,
) -> TensorMap:
    """
    JAX-traceable keys_to_properties.

    Uses Rust for metadata computation (Phase 1) and Array API for array
    operations (Phase 2). Compatible with jax.jit and jax.grad.
    """
    lib = _get_library()

    # Normalize keys_to_move
    if isinstance(keys_to_move, str):
        keys_to_move = [keys_to_move]
    if isinstance(keys_to_move, (list, tuple)):
        keys_to_move = Labels(
            names=list(keys_to_move),
            values=np.empty((0, len(keys_to_move)), dtype=np.int32),
        )

    # Phase 1: Rust computes the plan (concrete)
    plan_ptr = lib.mts_tensormap_keys_to_properties_plan(
        tensor._ptr,
        keys_to_move._as_mts_labels_t(),
        sort_samples,
    )
    _check_pointer(plan_ptr)

    try:
        new_keys, block_data = _get_plan_data(lib, plan_ptr, tensor)
    finally:
        lib.mts_merge_plan_free(plan_ptr)

    # Phase 2: Execute with Array API (traced by JAX)
    new_blocks = []
    for bd in block_data:
        output, _ = execute_merge_plan(
            bd["input_values"],
            bd["output_shape"],
            bd["movements_per_block"],
            fill_value,
        )
        # Get components from first input block (all should be the same)
        first_src = tensor.block_by_id(0)
        components = [c for c in first_src.components]

        block = TensorBlock(
            values=output,
            samples=bd["samples"],
            components=components,
            properties=bd["properties"],
        )
        new_blocks.append(block)

    return TensorMap(keys=new_keys, blocks=new_blocks)


def keys_to_samples_jax(
    tensor: TensorMap,
    keys_to_move: Union[str, Sequence[str], Labels],
    *,
    fill_value: float = 0.0,
    sort_samples: bool = True,
) -> TensorMap:
    """
    JAX-traceable keys_to_samples.

    Uses Rust for metadata computation (Phase 1) and Array API for array
    operations (Phase 2). Compatible with jax.jit and jax.grad.
    """
    lib = _get_library()

    # Normalize keys_to_move
    if isinstance(keys_to_move, str):
        keys_to_move = [keys_to_move]
    if isinstance(keys_to_move, (list, tuple)):
        keys_to_move = Labels(
            names=list(keys_to_move),
            values=np.empty((0, len(keys_to_move)), dtype=np.int32),
        )

    # Phase 1: Rust computes the plan
    plan_ptr = lib.mts_tensormap_keys_to_samples_plan(
        tensor._ptr,
        keys_to_move._as_mts_labels_t(),
        sort_samples,
    )
    _check_pointer(plan_ptr)

    try:
        new_keys, block_data = _get_plan_data(lib, plan_ptr, tensor)
    finally:
        lib.mts_merge_plan_free(plan_ptr)

    # Phase 2: Execute with Array API
    new_blocks = []
    for bd in block_data:
        output, _ = execute_merge_plan(
            bd["input_values"],
            bd["output_shape"],
            bd["movements_per_block"],
            fill_value,
        )
        first_src = tensor.block_by_id(0)
        components = [c for c in first_src.components]

        block = TensorBlock(
            values=output,
            samples=bd["samples"],
            components=components,
            properties=bd["properties"],
        )
        new_blocks.append(block)

    return TensorMap(keys=new_keys, blocks=new_blocks)
