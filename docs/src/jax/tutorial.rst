.. _jax-tutorial:

Tutorial: JAX with metatensor
==============================

This tutorial walks through using metatensor with JAX, covering the core
workflow: creating data, running operations under ``jax.jit``, computing
gradients, and applying ``jax.vmap``.

Prerequisites: basic familiarity with metatensor's core classes (``TensorMap``,
``TensorBlock``, ``Labels``) and with JAX's functional programming model.

Setup
-----

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    import metatensor
    import metatensor_jax  # registers PyTree nodes on import

    from metatensor import Labels, TensorBlock, TensorMap

The ``import metatensor_jax`` line registers ``TensorMap`` and ``TensorBlock``
as JAX PyTree nodes. This is all the setup needed; no configuration, no
wrapper classes.

Creating a TensorMap with JAX arrays
-------------------------------------

Block values are JAX arrays. Labels (metadata) stay as numpy, since they
describe the block-sparse structure and are not differentiated through.

.. code-block:: python

    # Two blocks, keyed by angular channel
    block_l0 = TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [1, 0]])),
        components=[],
        properties=Labels(["n"], np.array([[0], [1]])),
    )

    block_l1 = TensorBlock(
        values=jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        samples=Labels(["system", "atom"], np.array([[0, 0], [0, 1], [1, 0]])),
        components=[],
        properties=Labels(["n"], np.array([[0], [1]])),
    )

    keys = Labels(["angular_channel"], np.array([[0], [1]]))
    tm = TensorMap(keys, [block_l0, block_l1])

    print(f"Number of blocks: {len(tm)}")
    print(f"Block 0 values type: {type(tm.block_by_id(0).values)}")
    # -> <class 'jaxlib.xla_extension.ArrayImpl'>

JIT compilation
---------------

Wrap any function that takes and returns ``TensorMap`` with ``@jax.jit``.
JAX traces through the block values (dynamic leaves) while treating Labels
and structure as compile-time constants (static auxiliary data).

.. code-block:: python

    @jax.jit
    def double_and_square(t):
        doubled = metatensor.add(t, t)
        return metatensor.multiply(doubled, doubled)

    result = double_and_square(tm)

    # Verify: (2x)^2 = 4x^2
    expected = tm.block_by_id(0).values ** 2 * 4
    assert jnp.allclose(result.block_by_id(0).values, expected)

The first call traces and compiles. Subsequent calls with the same structure
reuse the compiled XLA computation.

.. note::

    If you pass a ``TensorMap`` with a different number of blocks or different
    Labels, JAX will retrace and recompile. This is the same behavior as
    passing a list of different length to any JIT-compiled function.

Automatic differentiation
-------------------------

``jax.grad`` differentiates through metatensor operations. The loss function
must return a scalar.

.. code-block:: python

    def loss_fn(t):
        """Sum of squared block values across all blocks."""
        squared = metatensor.multiply(t, t)
        total = 0.0
        for i in range(len(squared)):
            total = total + jnp.sum(squared.block_by_id(i).values)
        return total

    grad_tm = jax.grad(loss_fn)(tm)

    # grad_tm has the same structure as tm, with gradients as block values
    # d/dx (sum x^2) = 2x
    assert jnp.allclose(
        grad_tm.block_by_id(0).values,
        2 * tm.block_by_id(0).values,
    )

``jax.value_and_grad`` returns both the loss value and the gradient:

.. code-block:: python

    loss_val, grad_tm = jax.value_and_grad(loss_fn)(tm)
    print(f"Loss: {loss_val:.4f}")

Combining JIT and grad
^^^^^^^^^^^^^^^^^^^^^^

JIT and grad compose:

.. code-block:: python

    @jax.jit
    def train_step(t):
        loss, grad = jax.value_and_grad(loss_fn)(t)
        # Simple gradient descent: t - lr * grad
        updated = metatensor.subtract(t, metatensor.multiply(grad, 0.01))
        return updated, loss

    tm_current = tm
    for step in range(5):
        tm_current, loss = train_step(tm_current)
        if step % 2 == 0:
            print(f"Step {step}: loss = {float(loss):.4f}")

Vectorized computation with vmap
--------------------------------

``jax.vmap`` maps a function over a batch dimension. Since metatensor blocks
already carry explicit sample indices, the typical use case for ``vmap`` is
applying per-sample transformations or parallel processing of independent
systems.

.. code-block:: python

    def per_sample_norm(values):
        """L2 norm of each sample (row)."""
        return jnp.sqrt(jnp.sum(values ** 2, axis=-1, keepdims=True))

    # Apply to block values
    block = tm.block_by_id(0)
    norms = jax.vmap(lambda row: jnp.sqrt(jnp.sum(row ** 2)))(block.values)
    print(f"Per-sample norms: {norms}")

Structural operations
---------------------

Operations that manipulate the block-sparse structure (joining, slicing,
reducing) work under JIT because their metadata phase executes concretely
while the array phase traces.

.. code-block:: python

    # Sum over the atom dimension, keeping system
    reduced = metatensor.sum_over_samples(tm, sample_names="atom")
    print(f"Reduced samples: {reduced.block_by_id(0).samples}")

    # Join two TensorMaps along samples
    @jax.jit
    def join_and_reduce(t1, t2):
        joined = metatensor.join([t1, t2], axis="samples")
        return metatensor.sum_over_samples(joined, sample_names="atom")

    result = join_and_reduce(tm, tm)

Working with gradients
----------------------

Metatensor blocks can store gradients alongside values. These propagate
correctly through JAX's tracing:

.. code-block:: python

    # Create a block with an attached gradient
    block = TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["system"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["n"], np.array([[0], [1]])),
    )

    gradient = TensorBlock(
        values=jnp.array([[0.1, 0.2], [0.3, 0.4]]),
        samples=Labels(["sample", "parameter"], np.array([[0, 0], [1, 0]])),
        components=[],
        properties=Labels(["n"], np.array([[0], [1]])),
    )
    block.add_gradient("parameter_a", gradient)

    tm_with_grad = TensorMap(Labels.single(), [block])

    # JIT works with gradients -- they are part of the PyTree leaves
    @jax.jit
    def process(t):
        return metatensor.add(t, t)

    result = process(tm_with_grad)
    grad_block = result.block_by_id(0).gradient("parameter_a")
    assert jnp.allclose(grad_block.values, 2 * gradient.values)

Cross-backend interoperability
------------------------------

The unified ``metatensor.TensorMap`` type works identically across backends.
Code written for numpy works with JAX arrays:

.. code-block:: python

    # Create the same TensorMap with numpy
    block_np = TensorBlock(
        values=np.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["s"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    tm_np = TensorMap(Labels.single(), [block_np])

    # Create with JAX
    block_jax = TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["s"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    tm_jax = TensorMap(Labels.single(), [block_jax])

    # Same type, same operations
    assert type(tm_np) is type(tm_jax)

    result_np = metatensor.add(tm_np, tm_np)
    result_jax = metatensor.add(tm_jax, tm_jax)

    # Numerically equivalent
    np.testing.assert_allclose(
        np.array(result_np.block_by_id(0).values),
        np.array(result_jax.block_by_id(0).values),
    )

Summary
-------

=========================================  ============================================
Feature                                    How it works
=========================================  ============================================
``jax.jit``                                Block values traced; Labels are static
``jax.grad``                               Differentiates through block values
``jax.vmap``                               Maps over array dimensions
Structural ops (join, slice, reduce)       Metadata runs concretely, arrays trace
Metatensor gradients                       Included as PyTree leaves
Cross-backend                              Same ``TensorMap`` type everywhere
=========================================  ============================================
