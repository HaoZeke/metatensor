"""Tests for metatensor operations with JAX arrays.

Verifies that the array-API-based dispatch layer works correctly when
block values are jax.Array instances.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import metatensor
import metatensor_jax  # noqa: F401
from metatensor import Labels, TensorBlock, TensorMap

from . import _tests_utils


class TestElementWiseOps:
    """Element-wise operations should produce correct results with JAX arrays."""

    def test_add(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.add(tm, tm)

        for i in range(len(tm)):
            expected = tm.block_by_id(i).values * 2
            assert jnp.allclose(result.block_by_id(i).values, expected)

    def test_multiply(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.multiply(tm, tm)

        for i in range(len(tm)):
            expected = tm.block_by_id(i).values ** 2
            assert jnp.allclose(result.block_by_id(i).values, expected)

    def test_subtract(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.subtract(tm, tm)

        for i in range(len(tm)):
            assert jnp.allclose(result.block_by_id(i).values, 0.0)

    def test_divide(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.divide(tm, tm)

        for i in range(len(tm)):
            assert jnp.allclose(result.block_by_id(i).values, 1.0)

    def test_abs(self):
        block = TensorBlock(
            values=jnp.array([[-1.0, 2.0], [3.0, -4.0]]),
            samples=Labels(["s"], np.array([[0], [1]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm = TensorMap(Labels.single(), [block])
        result = metatensor.abs(tm)

        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(result.block_by_id(0).values, expected)


class TestStructuralOps:
    """Structural operations (join, slice) with JAX arrays."""

    def test_join_samples(self):
        """Joining along samples should concatenate block values."""
        block_a = TensorBlock(
            values=jnp.array([[1.0, 2.0]]),
            samples=Labels(["s"], np.array([[0]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        block_b = TensorBlock(
            values=jnp.array([[3.0, 4.0]]),
            samples=Labels(["s"], np.array([[1]])),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm_a = TensorMap(Labels.single(), [block_a])
        tm_b = TensorMap(Labels.single(), [block_b])

        result = metatensor.join([tm_a, tm_b], axis="samples")

        expected = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(result.block_by_id(0).values, expected)
        assert result.block_by_id(0).values.shape == (2, 2)

    def test_join_properties(self):
        """Joining along properties should concatenate along last axis."""
        block_a = TensorBlock(
            values=jnp.array([[1.0], [2.0]]),
            samples=Labels(["s"], np.array([[0], [1]])),
            components=[],
            properties=Labels(["p"], np.array([[0]])),
        )
        block_b = TensorBlock(
            values=jnp.array([[3.0], [4.0]]),
            samples=Labels(["s"], np.array([[0], [1]])),
            components=[],
            properties=Labels(["p"], np.array([[1]])),
        )
        tm_a = TensorMap(Labels.single(), [block_a])
        tm_b = TensorMap(Labels.single(), [block_b])

        result = metatensor.join([tm_a, tm_b], axis="properties")

        expected = jnp.array([[1.0, 3.0], [2.0, 4.0]])
        assert jnp.allclose(result.block_by_id(0).values, expected)


class TestReductionOps:
    """Reduction operations with JAX arrays."""

    def test_sum_over_samples(self):
        block = TensorBlock(
            values=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
            samples=Labels(
                ["system", "atom"],
                np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm = TensorMap(
            Labels(["key"], np.array([[0]])),
            [block],
        )

        result = metatensor.sum_over_samples(tm, sample_names="atom")

        # system 0: [1+3, 2+4] = [4, 6]
        # system 1: [5+7, 6+8] = [12, 14]
        expected = jnp.array([[4.0, 6.0], [12.0, 14.0]])
        assert jnp.allclose(result.block_by_id(0).values, expected)

    def test_mean_over_samples(self):
        block = TensorBlock(
            values=jnp.array([[2.0], [4.0], [6.0], [8.0]]),
            samples=Labels(
                ["system", "atom"],
                np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
            ),
            components=[],
            properties=Labels(["p"], np.array([[0]])),
        )
        tm = TensorMap(
            Labels(["key"], np.array([[0]])),
            [block],
        )

        result = metatensor.mean_over_samples(tm, sample_names="atom")

        # system 0: mean(2, 4) = 3
        # system 1: mean(6, 8) = 7
        expected = jnp.array([[3.0], [7.0]])
        assert jnp.allclose(result.block_by_id(0).values, expected)


class TestLikeOps:
    """zeros_like, ones_like, etc. with JAX arrays."""

    def test_zeros_like(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.zeros_like(tm)

        for i in range(len(tm)):
            assert jnp.allclose(result.block_by_id(i).values, 0.0)
            assert result.block_by_id(i).values.shape == tm.block_by_id(i).values.shape
            assert isinstance(result.block_by_id(i).values, jax.Array)

    def test_ones_like(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.ones_like(tm)

        for i in range(len(tm)):
            assert jnp.allclose(result.block_by_id(i).values, 1.0)
            assert isinstance(result.block_by_id(i).values, jax.Array)

    def test_equal_metadata_after_zeros_like(self):
        tm = _tests_utils.tensor_jax()
        result = metatensor.zeros_like(tm)

        assert result.keys == tm.keys
        for i in range(len(tm)):
            assert result.block_by_id(i).samples == tm.block_by_id(i).samples
            assert result.block_by_id(i).properties == tm.block_by_id(i).properties


class TestComparisonOps:
    """Comparison operations with JAX arrays."""

    def test_equal(self):
        tm = _tests_utils.tensor_jax()
        assert metatensor.equal(tm, tm)

    def test_allclose(self):
        tm = _tests_utils.tensor_jax()
        assert metatensor.allclose(tm, tm)
