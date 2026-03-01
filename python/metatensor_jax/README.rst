metatensor-jax
==============

JAX bindings for `metatensor <https://docs.metatensor.org>`_.

This package registers metatensor types (``TensorMap``, ``TensorBlock``,
``Labels``) as JAX PyTrees, enabling ``jax.jit``, ``jax.grad``, and
``jax.vmap`` to trace through metatensor operations.

The unified ``metatensor.TensorMap`` is the single type for all backends.
JAX arrays inside blocks are the dynamic leaves; Labels and structural
metadata are static auxiliary data.

Installation
------------

.. code-block:: bash

    pip install metatensor-jax

Usage
-----

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    import metatensor
    import metatensor_jax  # registers PyTree nodes on import

    from metatensor import Labels, TensorBlock, TensorMap

    # Create a TensorMap with JAX arrays
    block = TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        samples=Labels(["s"], np.array([[0], [1]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    tm = TensorMap(Labels.single(), [block])

    # JIT-compile metatensor operations
    @jax.jit
    def compute(t):
        return metatensor.add(t, t)

    result = compute(tm)  # traced and compiled by XLA

    # Differentiate through metatensor operations
    def loss(t):
        doubled = metatensor.add(t, t)
        return jnp.sum(doubled.block_by_id(0).values)

    grad_tm = jax.grad(loss)(tm)
