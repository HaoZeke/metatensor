"""
JAX bindings for metatensor.

This package registers metatensor types (TensorMap, TensorBlock, Labels) as
JAX PyTrees, enabling ``jax.jit``, ``jax.grad``, and ``jax.vmap`` to trace
through metatensor operations.

The unified ``metatensor.TensorMap`` is the single type for all backends.
JAX arrays inside blocks are the dynamic leaves; Labels and structural
metadata are static auxiliary data.
"""

import metatensor

from ._jax_array import register_jax_array_callbacks  # noqa: F401
from ._pytree import register_pytrees
from ._xla_ffi import register_ffi_targets

# Re-export unified types for convenience
Labels = metatensor.Labels
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap

# Register PyTree nodes, JAX array callbacks, and XLA FFI handlers on import
register_pytrees()
register_jax_array_callbacks()
register_ffi_targets()

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
