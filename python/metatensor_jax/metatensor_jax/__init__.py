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

# Re-export unified types for convenience
Labels = metatensor.Labels
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap

# Register PyTree nodes and JAX array callbacks on import
register_pytrees()
register_jax_array_callbacks()

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
