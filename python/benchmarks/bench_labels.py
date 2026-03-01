"""Benchmarks for Labels creation and lookup across array types.

Measures the cost of constructing Labels from different array backends
and looking up entries, to verify that the array-primary Labels (backed
by mts_array_t) do not regress performance compared to the numpy-only path.

Run with:

    python python/benchmarks/bench_labels.py
"""

import argparse
import timeit

import numpy as np

from metatensor import Labels


def _bench(label, fn, number=100, repeat=5):
    """Run a benchmark and print results."""
    times = timeit.repeat(fn, number=number, repeat=repeat)
    best = min(times) / number
    median = sorted(times)[len(times) // 2] / number
    print(f"  {label:55s}  best={best*1e6:8.1f} us  median={median*1e6:8.1f} us")
    return best


def bench_creation(backends, sizes):
    """Labels construction from different array types."""
    print("\n=== Labels creation ===")
    for size in sizes:
        print(f"\n  Size: {size} entries, 3 columns")
        values_np = np.array([[i, i + 1, i + 2] for i in range(size)], dtype=np.int32)

        for backend in backends:
            if backend == "numpy":
                values = values_np
            elif backend == "torch":
                import torch
                values = torch.from_numpy(values_np)
            elif backend == "jax":
                import jax.numpy as jnp
                values = jnp.array(values_np)

            _bench(
                f"[{backend}] Labels(['a','b','c'], {size})",
                lambda v=values: Labels(["a", "b", "c"], v),
            )

            _bench(
                f"[{backend}] Labels(['a','b','c'], {size}, unique)",
                lambda v=values: Labels(["a", "b", "c"], v, assume_unique=True),
            )


def bench_lookup(sizes):
    """Labels.position() lookup speed."""
    print("\n=== Labels.position() lookup ===")
    for size in sizes:
        print(f"\n  Labels with {size} entries, 3 columns")
        values = np.array([[i, i + 1, i + 2] for i in range(size)], dtype=np.int32)
        labels = Labels(["a", "b", "c"], values)

        # Lookup existing entry at different positions
        first = [0, 1, 2]
        mid = [size // 2, size // 2 + 1, size // 2 + 2]
        last = [size - 1, size, size + 1]

        _bench(
            f"position(first)",
            lambda: labels.position(first),
            number=1000,
        )
        _bench(
            f"position(middle)",
            lambda: labels.position(mid),
            number=1000,
        )
        _bench(
            f"position(last)",
            lambda: labels.position(last),
            number=1000,
        )


def bench_values_access(backends, sizes):
    """Measure cost of accessing .values property."""
    print("\n=== Labels.values access ===")
    for size in sizes:
        print(f"\n  Labels with {size} entries")
        values_np = np.array([[i, i + 1, i + 2] for i in range(size)], dtype=np.int32)

        for backend in backends:
            if backend == "numpy":
                values = values_np
            elif backend == "torch":
                import torch
                values = torch.from_numpy(values_np)
            elif backend == "jax":
                import jax.numpy as jnp
                values = jnp.array(values_np)

            labels = Labels(["a", "b", "c"], values)

            _bench(
                f"[{backend}] .values ({size} entries)",
                lambda l=labels: l.values,
                number=1000,
            )


def bench_equality(sizes):
    """Labels equality comparison."""
    print("\n=== Labels equality ===")
    for size in sizes:
        values = np.array([[i, i + 1, i + 2] for i in range(size)], dtype=np.int32)
        labels_a = Labels(["a", "b", "c"], values)
        labels_b = Labels(["a", "b", "c"], values)

        _bench(
            f"labels_a == labels_b ({size} entries)",
            lambda: labels_a == labels_b,
            number=500,
        )


def main():
    parser = argparse.ArgumentParser(description="Metatensor Labels benchmarks")
    parser.add_argument("--torch", action="store_true", help="Include torch arrays")
    parser.add_argument("--jax", action="store_true", help="Include JAX arrays")
    parser.add_argument("--all", action="store_true", help="Include all backends")
    args = parser.parse_args()

    backends = ["numpy"]
    if args.torch or args.all:
        try:
            import torch  # noqa: F401
            backends.append("torch")
        except ImportError:
            print("Warning: torch not available, skipping")
    if args.jax or args.all:
        try:
            import jax  # noqa: F401
            backends.append("jax")
        except ImportError:
            print("Warning: jax not available, skipping")

    print(f"Benchmarking backends: {', '.join(backends)}")

    sizes = [100, 1_000, 10_000]

    bench_creation(backends, sizes)
    bench_lookup(sizes)
    bench_values_access(backends, sizes)
    bench_equality(sizes)

    print("\nDone.")


if __name__ == "__main__":
    main()
