"""Utility functions for metatensor-jax tests."""

import numpy as np

import jax
import jax.numpy as jnp

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


def tensor_jax():
    """A test TensorMap with JAX arrays in block values."""
    block_1 = TensorBlock(
        values=jnp.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            values=jnp.full((2, 1, 1), 11.0),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=jnp.full((3, 1, 3), 2.0),
        samples=Labels(["s"], np.array([[0], [1], [3]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=jnp.full((3, 1, 3), 12.0),
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


def tensor_numpy():
    """Same structure as tensor_jax but with numpy arrays."""
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            values=np.full((2, 1, 1), 11.0),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["s"], np.array([[0], [1], [3]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((3, 1, 3), 12.0),
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


def simple_block_jax():
    """A simple TensorBlock with JAX arrays (no gradients, no components)."""
    return TensorBlock(
        values=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        samples=Labels(["s"], np.array([[0], [1], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
