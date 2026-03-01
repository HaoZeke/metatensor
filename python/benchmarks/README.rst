Metatensor Python Benchmarks
============================

Performance benchmarks for metatensor's Python operations layer. These verify
that the Array API Standard dispatch does not regress performance compared to
the previous isinstance-based dispatch.

Running
-------

.. code-block:: bash

    # numpy only (baseline)
    python python/benchmarks/bench_operations.py

    # numpy + torch
    python python/benchmarks/bench_operations.py --torch

    # numpy + jax
    python python/benchmarks/bench_operations.py --jax

    # all backends
    python python/benchmarks/bench_operations.py --all

    # Labels-specific benchmarks
    python python/benchmarks/bench_labels.py --all

Available benchmarks
--------------------

``bench_operations.py``
    Measures operation throughput and dispatch overhead across backends:

    - **Dispatch overhead**: tiny arrays (1x2x2) to isolate Python-layer cost
    - **Element-wise**: add, multiply, subtract at multiple sizes
    - **Structural**: zeros_like, ones_like
    - **Reductions**: sum_over_samples, mean_over_samples
    - **Join**: join along samples at different batch sizes
    - **JAX JIT**: compilation time and cached execution
    - **JAX grad**: gradient computation throughput

``bench_labels.py``
    Measures Labels creation, lookup, and property access:

    - **Creation**: from numpy, torch, and jax arrays (with and without assume_unique)
    - **Lookup**: position() at different entry positions
    - **Values access**: cost of the .values property
    - **Equality**: comparison between Labels

Interpreting results
--------------------

The key metric is **dispatch overhead** (tiny arrays). This isolates the cost of
``array_namespace()`` lookup and the dispatch layer from actual array computation.
For the Array API Standard dispatch, this overhead should be comparable to (within
2x of) the old isinstance-based dispatch for numpy and torch, and should be
present but small for JAX.

For larger arrays, the dispatch overhead is amortized by the actual computation
and should be negligible (<1% of total time).
