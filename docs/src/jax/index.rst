.. _metatensor-jax:

JAX backend
===========

.. toctree::
    :maxdepth: 2
    :hidden:

    tutorial
    ../examples/jax/index

The ``metatensor-jax`` package registers metatensor types as `JAX PyTrees
<https://docs.jax.dev/en/latest/pytrees.html>`_, enabling ``jax.jit``,
``jax.grad``, and ``jax.vmap`` to work with metatensor data structures.

Installation
^^^^^^^^^^^^

.. code-block:: bash

    pip install metatensor-jax

This requires ``metatensor-core``, ``jax``, and ``jaxlib``.

Quick start
^^^^^^^^^^^

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    import metatensor
    import metatensor_jax  # registers PyTree nodes on import

    from metatensor import Labels, TensorBlock, TensorMap

    # Create a TensorMap with JAX arrays as block values
    block = TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["sample"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["feature"], np.array([[0], [1]])),
    )
    tm = TensorMap(Labels.single(), [block])

    # JIT-compile metatensor operations
    @jax.jit
    def compute(t):
        return metatensor.add(t, t)

    result = compute(tm)

How it works
^^^^^^^^^^^^

When you ``import metatensor_jax``, it registers ``TensorMap`` and
``TensorBlock`` as JAX PyTree nodes. This tells JAX how to decompose these
structures for tracing:

**Dynamic leaves** (traced by JAX):

- Block values arrays (``block.values``)
- Gradient values arrays (``gradient.values``)

**Static auxiliary data** (not traced, used for reconstruction):

- All Labels (keys, samples, components, properties)
- Gradient metadata (parameter names, gradient Labels)
- TensorMap info dictionary

This separation means that JAX traces only through the numerical arrays
while preserving the block-sparse structure and metadata as compile-time
constants. Operations that manipulate Labels (like ``keys_to_samples``)
execute concretely, while array math traces into XLA/StableHLO.

Unified type model
^^^^^^^^^^^^^^^^^^

``metatensor-jax`` uses the same ``metatensor.TensorMap`` type as all other
backends. There is no separate ``metatensor.jax.TensorMap``. This means code
written for numpy or torch works with JAX arrays without modification:

.. code-block:: python

    # Same operation, any backend
    result_np = metatensor.add(tm_numpy, tm_numpy)
    result_torch = metatensor.add(tm_torch, tm_torch)
    result_jax = metatensor.add(tm_jax, tm_jax)

    # All return the same metatensor.TensorMap type
    assert type(result_np) is type(result_jax) is TensorMap

Automatic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^

JAX automatic differentiation works through metatensor operations:

.. code-block:: python

    def loss_fn(tm):
        doubled = metatensor.add(tm, tm)
        return jnp.sum(doubled.block_by_id(0).values)

    # Compute gradient with respect to block values
    grad_tm = jax.grad(loss_fn)(tm)

    # grad_tm is a TensorMap with the same structure
    # but gradient arrays in block values
    print(grad_tm.block_by_id(0).values)  # d(loss)/d(values)

Limitations
^^^^^^^^^^^

- **No in-place mutations**: JAX arrays are immutable. All metatensor operations
  return new arrays (this is handled internally by the dispatch layer).

- **Static structure**: The block-sparse structure (number of blocks, Labels)
  must be the same across JIT calls. Different structures trigger recompilation.

- **Random operations**: ``random_like`` uses a fixed PRNG key (``PRNGKey(0)``).
  For reproducible randomness with proper key management, create arrays
  with ``jax.random`` directly.
