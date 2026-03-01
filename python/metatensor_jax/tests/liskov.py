"""Cross-backend Liskov substitution tests.

Verifies that metatensor.TensorMap is the SAME type regardless of which
array backend stores the block values. Operations produce numerically
equivalent results across numpy, torch, and jax.
"""

import numpy as np
import pytest

import jax.numpy as jnp

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from . import _tests_utils

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _tensor_torch():
    """Same structure as tensor_jax but with torch tensors."""
    block_1 = TensorBlock(
        values=torch.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            values=torch.full((2, 1, 1), 11.0),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=torch.full((3, 1, 3), 2.0),
        samples=Labels(["s"], np.array([[0], [1], [3]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((3, 1, 3), 12.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [0, 3], [2, -2]])),
            components=block_2.components,
            properties=block_2.properties,
        ),
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0]]),
    )

    return TensorMap(keys, [block_1, block_2])


class TestLiskovEquivalence:
    """One TensorMap type, any array backend."""

    def test_same_type(self):
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        assert type(tm_np) is type(tm_jax)
        assert type(tm_np) is TensorMap

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_same_type_torch(self):
        tm_np = _tests_utils.tensor_numpy()
        tm_torch = _tensor_torch()

        assert type(tm_np) is type(tm_torch)
        assert type(tm_np) is TensorMap

    def test_add_numpy_vs_jax(self):
        """add() produces numerically equivalent results for numpy and jax."""
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.add(tm_np, tm_np)
        result_jax = metatensor.add(tm_jax, tm_jax)

        for i in range(len(result_np)):
            np_vals = np.asarray(result_np.block_by_id(i).values)
            jax_vals = np.asarray(result_jax.block_by_id(i).values)
            np.testing.assert_allclose(np_vals, jax_vals)

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_add_all_backends(self):
        """add() produces equivalent results across numpy, torch, and jax."""
        tm_np = _tests_utils.tensor_numpy()
        tm_torch = _tensor_torch()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.add(tm_np, tm_np)
        result_torch = metatensor.add(tm_torch, tm_torch)
        result_jax = metatensor.add(tm_jax, tm_jax)

        for i in range(len(result_np)):
            np_vals = np.asarray(result_np.block_by_id(i).values)
            torch_vals = result_torch.block_by_id(i).values.detach().numpy()
            jax_vals = np.asarray(result_jax.block_by_id(i).values)

            np.testing.assert_allclose(np_vals, jax_vals)
            np.testing.assert_allclose(np_vals, torch_vals)

    def test_multiply_numpy_vs_jax(self):
        """multiply() produces equivalent results for numpy and jax."""
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.multiply(tm_np, tm_np)
        result_jax = metatensor.multiply(tm_jax, tm_jax)

        for i in range(len(result_np)):
            np_vals = np.asarray(result_np.block_by_id(i).values)
            jax_vals = np.asarray(result_jax.block_by_id(i).values)
            np.testing.assert_allclose(np_vals, jax_vals)

    def test_subtract_numpy_vs_jax(self):
        """subtract(tm, tm) yields zeros for both backends."""
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.subtract(tm_np, tm_np)
        result_jax = metatensor.subtract(tm_jax, tm_jax)

        for i in range(len(result_np)):
            np_vals = np.asarray(result_np.block_by_id(i).values)
            jax_vals = np.asarray(result_jax.block_by_id(i).values)
            np.testing.assert_allclose(np_vals, 0.0)
            np.testing.assert_allclose(jax_vals, 0.0)

    def test_keys_preserved(self):
        """Operations preserve TensorMap keys across backends."""
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.add(tm_np, tm_np)
        result_jax = metatensor.add(tm_jax, tm_jax)

        assert result_np.keys == result_jax.keys

    def test_labels_preserved(self):
        """Operations preserve sample/property Labels across backends."""
        tm_np = _tests_utils.tensor_numpy()
        tm_jax = _tests_utils.tensor_jax()

        result_np = metatensor.add(tm_np, tm_np)
        result_jax = metatensor.add(tm_jax, tm_jax)

        for i in range(len(result_np)):
            assert result_np.block_by_id(i).samples == result_jax.block_by_id(i).samples
            assert (
                result_np.block_by_id(i).properties
                == result_jax.block_by_id(i).properties
            )
