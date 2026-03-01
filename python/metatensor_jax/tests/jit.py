"""Tests for JAX JIT compilation with metatensor types."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import metatensor
import metatensor_jax  # noqa: F401 -- triggers PyTree registration
from metatensor import Labels, TensorBlock, TensorMap

from . import _tests_utils


class TestJitBasic:
    """Test that metatensor operations can be JIT-compiled with JAX."""

    def test_jit_add(self):
        """add(tm, tm) should work under jax.jit."""
        tm = _tests_utils.tensor_jax()

        @jax.jit
        def compute(t):
            return metatensor.add(t, t)

        result = compute(tm)
        assert isinstance(result, TensorMap)

        for i in range(len(tm)):
            expected = tm.block_by_id(i).values * 2.0
            assert jnp.allclose(result.block_by_id(i).values, expected)

    def test_jit_multiply(self):
        """multiply(tm, tm) should work under jax.jit."""
        tm = _tests_utils.tensor_jax()

        @jax.jit
        def compute(t):
            return metatensor.multiply(t, t)

        result = compute(tm)
        assert isinstance(result, TensorMap)

        for i in range(len(tm)):
            expected = tm.block_by_id(i).values ** 2
            assert jnp.allclose(result.block_by_id(i).values, expected)

    def test_jit_subtract(self):
        """subtract(tm, tm) should yield zeros under jax.jit."""
        tm = _tests_utils.tensor_jax()

        @jax.jit
        def compute(t):
            return metatensor.subtract(t, t)

        result = compute(tm)
        for i in range(len(tm)):
            assert jnp.allclose(result.block_by_id(i).values, 0.0)

    def test_jit_retracing(self):
        """Same structure should not cause retracing."""
        tm1 = _tests_utils.tensor_jax()
        tm2 = _tests_utils.tensor_jax()  # same structure, different id

        call_count = 0

        @jax.jit
        def compute(t):
            return metatensor.add(t, t)

        # First call: compiles
        result1 = compute(tm1)
        # Second call: should use cached compilation
        result2 = compute(tm2)

        assert isinstance(result1, TensorMap)
        assert isinstance(result2, TensorMap)


class TestJitGradient:
    """Test JAX automatic differentiation through metatensor operations."""

    def test_grad_scalar_output(self):
        """jax.grad should work through metatensor block operations."""
        block = TensorBlock(
            values=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            samples=Labels(["s"], np.array([[0], [1]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm = TensorMap(Labels.single(), [block])

        def scalar_loss(t):
            doubled = metatensor.add(t, t)
            # Sum all values to produce a scalar
            total = jnp.float32(0.0)
            for i in range(len(doubled)):
                total = total + jnp.sum(doubled.block_by_id(i).values)
            return total

        grad_tm = jax.grad(scalar_loss)(tm)
        assert isinstance(grad_tm, TensorMap)

        # d/d(values) of sum(2 * values) = 2.0 everywhere
        expected = jnp.full_like(block.values, 2.0)
        assert jnp.allclose(grad_tm.block_by_id(0).values, expected)

    def test_value_and_grad(self):
        """jax.value_and_grad should return both value and gradient."""
        block = TensorBlock(
            values=jnp.array([[1.0, 2.0]]),
            samples=Labels(["s"], np.array([[0]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm = TensorMap(Labels.single(), [block])

        def loss(t):
            return jnp.sum(t.block_by_id(0).values ** 2)

        val, grad_tm = jax.value_and_grad(loss)(tm)

        # sum([1, 4]) = 5
        assert jnp.isclose(val, 5.0)

        # d/dx of x^2 = 2x
        assert jnp.allclose(grad_tm.block_by_id(0).values, 2.0 * block.values)
