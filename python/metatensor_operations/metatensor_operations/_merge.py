"""
Array API-based merge execution for keys_to_properties and keys_to_samples.

This module implements Phase 2 of the split approach for JAX traceability:
Rust computes the merge plan (Phase 1), Python executes the array operations
using the Array API (Phase 2). JAX can trace Phase 2.

The merge plan is a set of "movement instructions" that describe which slices
of input block arrays should be scattered into the output array.
"""

from typing import List, Optional, Tuple

from ._dispatch import array_namespace


def execute_merge_plan(
    block_values,
    output_shape: List[int],
    movements_per_block: List[List[Tuple[int, int, int, int, int]]],
    fill_value,
    gradient_info: Optional[List[dict]] = None,
):
    """
    Execute a merge plan using Array API operations.

    This function creates an output array and scatters input block data into it
    according to the movement instructions. All operations use the Array API,
    making them traceable by JAX (jax.jit, jax.grad).

    Parameters
    ----------
    block_values : list of arrays
        Values arrays from the input blocks (in order of the plan's block_plans).
    output_shape : list of int
        Shape of the output array.
    movements_per_block : list of list of (sample_in, sample_out, prop_start_in,
        prop_start_out, prop_length) tuples
        Movement instructions for each input block.
    fill_value : scalar
        Value to fill the output array with (typically 0.0).
    gradient_info : optional list of dicts
        If provided, each dict has 'block_gradients' (list of arrays),
        'output_shape' (list of int), and 'movements_per_block' (same format).

    Returns
    -------
    output : array
        The merged output array.
    gradient_outputs : list of arrays or None
        Merged gradient arrays, if gradient_info was provided.
    """
    if not block_values:
        raise ValueError("block_values must not be empty")

    xp = array_namespace(block_values[0])

    # Create output array filled with fill_value
    output = xp.full(output_shape, fill_value, dtype=block_values[0].dtype)

    # Scatter data from each input block
    for block_val, movements in zip(block_values, movements_per_block):
        if not movements:
            continue
        output = _scatter_movements(xp, output, block_val, movements)

    # Handle gradients if present
    gradient_outputs = None
    if gradient_info is not None:
        gradient_outputs = []
        for ginfo in gradient_info:
            grad_output = xp.full(
                ginfo["output_shape"], fill_value, dtype=block_values[0].dtype
            )
            for grad_val, movements in zip(
                ginfo["block_gradients"], ginfo["movements_per_block"]
            ):
                if not movements:
                    continue
                grad_output = _scatter_movements(xp, grad_output, grad_val, movements)
            gradient_outputs.append(grad_output)

    return output, gradient_outputs


def _scatter_movements(xp, output, input_array, movements):
    """
    Scatter data from input_array into output according to movement instructions.

    Each movement is (sample_in, sample_out, prop_start_in, prop_start_out, prop_length).

    For JAX, this uses functional updates (.at[].set()).
    For numpy/torch, this uses in-place indexing.
    """
    # Check if we can use the optimized batch path:
    # all movements have the same property structure
    if len(movements) > 1:
        first = movements[0]
        all_same_props = all(
            m[2] == first[2] and m[3] == first[3] and m[4] == first[4]
            for m in movements
        )
    else:
        all_same_props = True

    if all_same_props and movements:
        # Optimized: batch all samples at once
        prop_start_in = movements[0][2]
        prop_start_out = movements[0][3]
        prop_len = movements[0][4]

        samples_in = [m[0] for m in movements]
        samples_out = [m[1] for m in movements]

        src_slice = input_array[samples_in, ..., prop_start_in:prop_start_in + prop_len]

        # Use JAX-compatible functional update if available
        if hasattr(output, "at"):
            # JAX path: functional scatter
            output = output.at[samples_out, ..., prop_start_out:prop_start_out + prop_len].set(
                src_slice
            )
        else:
            # numpy/torch path: in-place assignment
            output[samples_out, ..., prop_start_out:prop_start_out + prop_len] = src_slice
    else:
        # Fallback: one movement at a time
        for sample_in, sample_out, prop_start_in, prop_start_out, prop_len in movements:
            src = input_array[sample_in, ..., prop_start_in:prop_start_in + prop_len]
            if hasattr(output, "at"):
                output = output.at[sample_out, ..., prop_start_out:prop_start_out + prop_len].set(src)
            else:
                output[sample_out, ..., prop_start_out:prop_start_out + prop_len] = src

    return output
