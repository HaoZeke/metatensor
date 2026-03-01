"""
.. _jax-tutorial-basics:

JAX with metatensor
===================

This tutorial shows how to use metatensor with JAX arrays. The same
``TensorMap`` type works with numpy, torch, and JAX -- the operations
dispatch to the correct backend automatically.

We cover creating TensorMaps with JAX arrays, running operations under
``jax.jit``, and computing gradients with ``jax.grad``.

.. py:currentmodule:: metatensor
"""

# %%
#
# Setup: import JAX and register metatensor's PyTree nodes.

import jax
import jax.numpy as jnp
import numpy as np

import metatensor
import metatensor_jax  # noqa: F401  -- registers PyTree nodes

from metatensor import Labels, TensorBlock, TensorMap

# %%
#
# Creating a TensorMap with JAX arrays
# -------------------------------------
#
# Block values use JAX arrays. Labels (metadata) stay as numpy -- they
# describe the block-sparse structure and are never differentiated.

block = TensorBlock(
    values=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [1, 0]])),
    components=[],
    properties=Labels(["n"], np.array([[0], [1]])),
)

keys = Labels(["angular_channel"], np.array([[0]]))
tm = TensorMap(keys, [block])

print(f"Block values type: {type(tm.block_by_id(0).values)}")
print(f"Values:\n{tm.block_by_id(0).values}")

# %%
#
# The unified type
# ----------------
#
# The same ``metatensor.TensorMap`` class holds any array type. There is
# no separate ``metatensor.jax.TensorMap``.

tm_np = TensorMap(
    keys,
    [TensorBlock(
        values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [1, 0]])),
        components=[],
        properties=Labels(["n"], np.array([[0], [1]])),
    )],
)

print(f"numpy TensorMap type: {type(tm_np)}")
print(f"JAX TensorMap type:   {type(tm)}")
print(f"Same class: {type(tm_np) is type(tm)}")

# %%
#
# Operations work the same way regardless of backend:

result_np = metatensor.add(tm_np, tm_np)
result_jax = metatensor.add(tm, tm)

print(f"numpy result:\n{result_np.block_by_id(0).values}")
print(f"JAX result:\n{result_jax.block_by_id(0).values}")

# %%
#
# JIT compilation
# ---------------
#
# Wrap any function with ``@jax.jit``. JAX traces the block values
# (dynamic leaves) and treats Labels as static compile-time constants.


@jax.jit
def double_values(t):
    return metatensor.add(t, t)


result = double_values(tm)
print(f"Doubled:\n{result.block_by_id(0).values}")

# %%
#
# Chaining operations works too:


@jax.jit
def double_and_square(t):
    doubled = metatensor.add(t, t)
    return metatensor.multiply(doubled, doubled)


result = double_and_square(tm)

# (2x)^2 = 4x^2
expected = tm.block_by_id(0).values ** 2 * 4
print(f"Result:\n{result.block_by_id(0).values}")
print(f"Expected (4x^2):\n{expected}")
print(f"Match: {jnp.allclose(result.block_by_id(0).values, expected)}")

# %%
#
# Automatic differentiation
# -------------------------
#
# ``jax.grad`` differentiates through metatensor operations. The gradient
# is a ``TensorMap`` with the same structure.


def loss_fn(t):
    """Sum of squared values across all blocks."""
    squared = metatensor.multiply(t, t)
    total = 0.0
    for i in range(len(squared)):
        total = total + jnp.sum(squared.block_by_id(i).values)
    return total


grad_tm = jax.grad(loss_fn)(tm)

# d/dx sum(x^2) = 2x
print(f"Values:\n{tm.block_by_id(0).values}")
print(f"Gradient (2x):\n{grad_tm.block_by_id(0).values}")
print(f"Match: {jnp.allclose(grad_tm.block_by_id(0).values, 2 * tm.block_by_id(0).values)}")

# %%
#
# JIT + grad compose:

loss_val, grad_tm = jax.jit(jax.value_and_grad(loss_fn))(tm)
print(f"Loss: {loss_val}")
print(f"Gradient shape: {grad_tm.block_by_id(0).values.shape}")

# %%
#
# Structural operations under JIT
# --------------------------------
#
# Operations that modify the block-sparse structure (sum_over_samples, join)
# work under JIT because the metadata phase runs concretely and only array
# math is traced.

reduced = metatensor.sum_over_samples(tm, sample_names="atom")
print(f"Original samples: {tm.block_by_id(0).samples}")
print(f"Reduced samples: {reduced.block_by_id(0).samples}")
print(f"Reduced values:\n{reduced.block_by_id(0).values}")

# %%
#
# Under JIT:


@jax.jit
def reduce_and_double(t):
    reduced = metatensor.sum_over_samples(t, sample_names="atom")
    return metatensor.add(reduced, reduced)


result = reduce_and_double(tm)
print(f"Result:\n{result.block_by_id(0).values}")
