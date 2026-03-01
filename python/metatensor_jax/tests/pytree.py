"""Tests for JAX PyTree registration of TensorMap and TensorBlock."""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

import metatensor
import metatensor_jax  # noqa: F401 -- triggers PyTree registration
from metatensor import Labels, TensorBlock, TensorMap

from . import _tests_utils


class TestTensorBlockPyTree:
    """Tests for TensorBlock as a JAX PyTree node."""

    def test_flatten_unflatten_simple(self):
        block = _tests_utils.simple_block_jax()
        leaves, treedef = jax.tree_util.tree_flatten(block)

        # Only values should be a leaf (no gradients)
        assert len(leaves) == 1
        assert jnp.array_equal(leaves[0], block.values)

        # Round-trip
        block2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(block2, TensorBlock)
        assert jnp.array_equal(block2.values, block.values)
        assert block2.samples == block.samples
        assert block2.properties == block.properties

    def test_flatten_unflatten_with_gradient(self):
        tm = _tests_utils.tensor_jax()
        block = tm.block_by_id(0)

        leaves, treedef = jax.tree_util.tree_flatten(block)

        # block values + 1 gradient's values = 2 leaves
        assert len(leaves) == 2

        block2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(block2, TensorBlock)
        assert jnp.array_equal(block2.values, block.values)

        # Check gradient round-trip
        grad_params = list(block2.gradients())
        assert len(grad_params) == 1
        param, grad = grad_params[0]
        assert param == "g"

        orig_grad = block.gradient("g")
        assert jnp.array_equal(grad.values, orig_grad.values)
        assert grad.samples == orig_grad.samples

    def test_tree_map_scales_values(self):
        block = _tests_utils.simple_block_jax()

        # tree_map should scale only the dynamic leaves (values)
        scaled = jax.tree_util.tree_map(lambda x: x * 2.0, block)
        assert isinstance(scaled, TensorBlock)
        assert jnp.allclose(scaled.values, block.values * 2.0)
        # Labels should be unchanged
        assert scaled.samples == block.samples
        assert scaled.properties == block.properties


class TestTensorMapPyTree:
    """Tests for TensorMap as a JAX PyTree node."""

    def test_flatten_unflatten(self):
        tm = _tests_utils.tensor_jax()
        leaves, treedef = jax.tree_util.tree_flatten(tm)

        # block_0: 1 values + 1 gradient = 2
        # block_1: 1 values + 1 gradient = 2
        # total = 4 leaves
        assert len(leaves) == 4

        tm2 = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(tm2, TensorMap)
        assert len(tm2) == len(tm)
        assert tm2.keys == tm.keys

        for i in range(len(tm)):
            orig = tm.block_by_id(i)
            new = tm2.block_by_id(i)
            assert jnp.array_equal(new.values, orig.values)
            assert new.samples == orig.samples
            assert new.properties == orig.properties

    def test_tree_map_scales_all_blocks(self):
        tm = _tests_utils.tensor_jax()

        scaled = jax.tree_util.tree_map(lambda x: x * 3.0, tm)
        assert isinstance(scaled, TensorMap)
        assert scaled.keys == tm.keys

        for i in range(len(tm)):
            orig = tm.block_by_id(i)
            new = scaled.block_by_id(i)
            assert jnp.allclose(new.values, orig.values * 3.0)

    def test_treedef_equality_for_same_structure(self):
        """Same structure should produce equal treedefs (enables JIT caching)."""
        tm1 = _tests_utils.tensor_jax()
        tm2 = _tests_utils.tensor_jax()

        _, treedef1 = jax.tree_util.tree_flatten(tm1)
        _, treedef2 = jax.tree_util.tree_flatten(tm2)

        assert treedef1 == treedef2

    def test_empty_block(self):
        """TensorMap with zero-sample blocks should round-trip."""
        block = TensorBlock(
            values=jnp.zeros((0, 2)),
            samples=Labels(["s"], np.empty((0, 1), dtype=np.int32)),
            components=[],
            properties=Labels(["p"], np.array([[0], [1]])),
        )
        tm = TensorMap(Labels.single(), [block])

        leaves, treedef = jax.tree_util.tree_flatten(tm)
        tm2 = jax.tree_util.tree_unflatten(treedef, leaves)

        assert tm2.block_by_id(0).values.shape == (0, 2)
