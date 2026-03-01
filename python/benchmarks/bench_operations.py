"""Benchmarks for metatensor operations across backends.

Measures dispatch overhead and operation throughput for the Array API
Standard dispatch layer vs. the old isinstance-based dispatch. Run with:

    python python/benchmarks/bench_operations.py

Optionally pass --jax to include JAX benchmarks (requires jax + metatensor_jax).
Optionally pass --torch to include torch benchmarks (requires torch).
"""

import argparse
import sys
import timeit

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tensor(backend, n_blocks, n_samples, n_properties, n_components=0):
    """Create a TensorMap with the given shape and backend."""
    if backend == "numpy":
        make_array = lambda shape: np.random.randn(*shape).astype(np.float64)
    elif backend == "torch":
        import torch
        make_array = lambda shape: torch.randn(*shape, dtype=torch.float64)
    elif backend == "jax":
        import jax.numpy as jnp
        import jax
        key = jax.random.PRNGKey(42)
        def make_array(shape, _key=[key]):
            _key[0], subkey = jax.random.split(_key[0])
            return jax.random.normal(subkey, shape=shape, dtype=jnp.float64)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    blocks = []
    for i in range(n_blocks):
        if n_components > 0:
            shape = (n_samples, n_components, n_properties)
            comp_labels = [
                Labels(["c"], np.array([[j] for j in range(n_components)]))
            ]
        else:
            shape = (n_samples, n_properties)
            comp_labels = []

        blocks.append(TensorBlock(
            values=make_array(shape),
            samples=Labels(
                ["system", "atom"],
                np.array([[s // 10, s % 10] for s in range(n_samples)]),
            ),
            components=comp_labels,
            properties=Labels(["p"], np.array([[p] for p in range(n_properties)])),
        ))

    keys = Labels(["key"], np.array([[i] for i in range(n_blocks)]))
    return TensorMap(keys, blocks)


def _bench(label, fn, number=100, repeat=5):
    """Run a benchmark and print results."""
    times = timeit.repeat(fn, number=number, repeat=repeat)
    best = min(times) / number
    median = sorted(times)[len(times) // 2] / number
    print(f"  {label:50s}  best={best*1e6:8.1f} us  median={median*1e6:8.1f} us")
    return best


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------


def bench_elementwise(backends, sizes):
    """Element-wise operations: add, multiply, subtract."""
    print("\n=== Element-wise operations ===")
    for n_blocks, n_samples, n_props in sizes:
        print(f"\n  Shape: {n_blocks} blocks x {n_samples} samples x {n_props} properties")
        for backend in backends:
            tm = _make_tensor(backend, n_blocks, n_samples, n_props)
            _bench(f"[{backend}] add", lambda: metatensor.add(tm, tm))
            _bench(f"[{backend}] multiply", lambda: metatensor.multiply(tm, tm))
            _bench(f"[{backend}] subtract", lambda: metatensor.subtract(tm, tm))


def bench_structural(backends, sizes):
    """Structural operations: join, zeros_like, ones_like."""
    print("\n=== Structural operations ===")
    for n_blocks, n_samples, n_props in sizes:
        print(f"\n  Shape: {n_blocks} blocks x {n_samples} samples x {n_props} properties")
        for backend in backends:
            tm = _make_tensor(backend, n_blocks, n_samples, n_props)
            _bench(f"[{backend}] zeros_like", lambda: metatensor.zeros_like(tm))
            _bench(f"[{backend}] ones_like", lambda: metatensor.ones_like(tm))


def bench_reductions(backends, sizes):
    """Reduction operations: sum_over_samples, mean_over_samples."""
    print("\n=== Reduction operations ===")
    for n_blocks, n_samples, n_props in sizes:
        print(f"\n  Shape: {n_blocks} blocks x {n_samples} samples x {n_props} properties")
        for backend in backends:
            tm = _make_tensor(backend, n_blocks, n_samples, n_props)
            _bench(
                f"[{backend}] sum_over_samples",
                lambda: metatensor.sum_over_samples(tm, sample_names="atom"),
            )
            _bench(
                f"[{backend}] mean_over_samples",
                lambda: metatensor.mean_over_samples(tm, sample_names="atom"),
            )


def bench_join(backends):
    """Join operations at different batch sizes."""
    print("\n=== Join operations ===")
    for n_tensors in [2, 5, 10]:
        print(f"\n  Joining {n_tensors} TensorMaps (4 blocks x 50 samples x 20 properties)")
        for backend in backends:
            tms = [_make_tensor(backend, 4, 50, 20) for _ in range(n_tensors)]
            _bench(
                f"[{backend}] join(axis='samples')",
                lambda: metatensor.join(tms, axis="samples"),
                number=50,
            )


def bench_dispatch_overhead(backends):
    """Measure dispatch overhead with tiny arrays to isolate framework overhead."""
    print("\n=== Dispatch overhead (tiny arrays, isolates Python-layer cost) ===")
    for backend in backends:
        tm = _make_tensor(backend, 1, 2, 2)
        _bench(f"[{backend}] add (1x2x2)", lambda: metatensor.add(tm, tm), number=500)
        _bench(
            f"[{backend}] zeros_like (1x2x2)",
            lambda: metatensor.zeros_like(tm),
            number=500,
        )


def bench_jax_jit():
    """JAX JIT compilation and execution benchmarks."""
    import jax
    import jax.numpy as jnp
    import metatensor_jax  # noqa: F401

    print("\n=== JAX JIT benchmarks ===")

    tm = _make_tensor("jax", 4, 100, 20)

    @jax.jit
    def jit_add(t):
        return metatensor.add(t, t)

    @jax.jit
    def jit_chain(t):
        doubled = metatensor.add(t, t)
        return metatensor.multiply(doubled, doubled)

    # Warmup (triggers compilation)
    _ = jit_add(tm)
    _ = jit_chain(tm)

    print("\n  After compilation (cached execution)")
    _bench("[jax] jit add (4x100x20)", lambda: jit_add(tm))
    _bench("[jax] jit chain add+mul (4x100x20)", lambda: jit_chain(tm))

    # Measure compilation time
    print("\n  Including compilation (cold start)")
    def cold_add():
        @jax.jit
        def f(t):
            return metatensor.add(t, t)
        return f(tm)

    _bench("[jax] jit add cold (4x100x20)", cold_add, number=5, repeat=3)


def bench_jax_grad():
    """JAX gradient computation benchmarks."""
    import jax
    import jax.numpy as jnp
    import metatensor_jax  # noqa: F401

    print("\n=== JAX gradient benchmarks ===")

    tm = _make_tensor("jax", 4, 100, 20)

    def loss_fn(t):
        doubled = metatensor.add(t, t)
        total = 0.0
        for i in range(len(doubled)):
            total = total + jnp.sum(doubled.block_by_id(i).values)
        return total

    grad_fn = jax.jit(jax.grad(loss_fn))

    # Warmup
    _ = grad_fn(tm)

    _bench("[jax] grad (4x100x20)", lambda: grad_fn(tm))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Metatensor operations benchmarks")
    parser.add_argument("--torch", action="store_true", help="Include torch backend")
    parser.add_argument("--jax", action="store_true", help="Include JAX backend")
    parser.add_argument("--all", action="store_true", help="Include all backends")
    args = parser.parse_args()

    backends = ["numpy"]
    if args.torch or args.all:
        try:
            import torch  # noqa: F401
            backends.append("torch")
        except ImportError:
            print("Warning: torch not available, skipping torch benchmarks")
    if args.jax or args.all:
        try:
            import jax  # noqa: F401
            import metatensor_jax  # noqa: F401
            backends.append("jax")
        except ImportError:
            print("Warning: jax/metatensor_jax not available, skipping JAX benchmarks")

    print(f"Benchmarking backends: {', '.join(backends)}")
    print(f"metatensor version: {metatensor.__version__}")

    sizes = [
        (4, 50, 20),     # small: typical per-atom descriptor
        (10, 200, 50),   # medium: spherical expansion
        (20, 500, 100),  # large: full basis set
    ]

    bench_dispatch_overhead(backends)
    bench_elementwise(backends, sizes)
    bench_structural(backends, sizes)
    bench_reductions(backends, sizes)
    bench_join(backends)

    if "jax" in backends:
        bench_jax_jit()
        bench_jax_grad()

    print("\nDone.")


if __name__ == "__main__":
    main()
