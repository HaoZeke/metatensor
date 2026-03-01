.. _devdoc-benchmarks:

Benchmarks
==========

Metatensor includes Python benchmarks to verify that the Array API Standard
dispatch layer does not regress performance. The benchmarks live in
``python/benchmarks/``.

Running benchmarks
------------------

.. code-block:: bash

    # numpy only (baseline)
    python python/benchmarks/bench_operations.py

    # all available backends
    python python/benchmarks/bench_operations.py --all

    # Labels-specific benchmarks
    python python/benchmarks/bench_labels.py --all

What is measured
----------------

**Operations benchmarks** (``bench_operations.py``):

- **Dispatch overhead**: Uses tiny (1x2x2) TensorMaps to isolate the Python-layer
  cost of ``array_namespace()`` lookup and dispatch from actual array computation.
  This is the primary metric for verifying no regression.

- **Element-wise operations**: ``add``, ``multiply``, ``subtract`` at small (4x50x20),
  medium (10x200x50), and large (20x500x100) sizes.

- **Structural operations**: ``zeros_like``, ``ones_like``.

- **Reductions**: ``sum_over_samples``, ``mean_over_samples``.

- **Join**: Concatenating multiple TensorMaps along the samples axis.

- **JAX JIT** (with ``--jax``): Compilation time (cold start) and cached execution
  for JIT-compiled metatensor operations.

- **JAX grad** (with ``--jax``): Gradient computation throughput through
  metatensor operations.

**Labels benchmarks** (``bench_labels.py``):

- **Creation**: Constructing Labels from numpy, torch, and JAX arrays.
  Compares the cost of the array-primary path (``mts_labels_create_from_array``)
  against the legacy numpy-only path.

- **Lookup**: ``Labels.position()`` at different entry positions (first, middle, last).

- **Values access**: Cost of the ``.values`` property, which may involve lazy
  CPU materialization from the backing ``mts_array_t``.

- **Equality**: Comparison between Labels instances.

Interpreting results
--------------------

For **dispatch overhead** (tiny arrays):

- The Array API dispatch via ``array_namespace()`` adds a small constant cost
  per call compared to direct ``isinstance`` checks. This should be under 10
  microseconds per operation.

- For numpy, expect the overhead to be similar to the old dispatch (both end up
  calling the same numpy functions).

- For torch, the ``array_namespace()`` call is slightly more expensive than a
  direct ``isinstance(x, torch.Tensor)`` check, but the difference is negligible
  for any non-trivial array size.

For **larger arrays**, the dispatch overhead is amortized by computation time
and should be <1% of total time. If dispatch overhead dominates at large sizes,
something is wrong.

For **JAX JIT**, the cold start compilation cost is expected to be 10-100x
higher than cached execution. The cached execution time should be comparable
to or faster than eager numpy for the same array sizes, since XLA compiles
to optimized machine code.

Rust benchmarks
---------------

The Rust-level Labels benchmarks live in ``rust/metatensor/benches/labels.rs``
and use Criterion:

.. code-block:: bash

    cargo bench -p metatensor --features bench
