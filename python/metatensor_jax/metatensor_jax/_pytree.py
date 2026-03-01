"""
PyTree registration for metatensor types.

TensorMap and TensorBlock are registered as JAX PyTree nodes. This enables
JAX transformations (jit, grad, vmap) to trace through metatensor operations.

Design:
- Block values and gradient values are dynamic leaves (traced by JAX)
- Labels and structural metadata are static auxiliary data (not traced)
- Labels are semantically immutable; their CPU values are lazily materialized
"""

import numpy as np

import jax
import metatensor


def _labels_to_hashable(labels):
    """Convert Labels to a hashable representation for PyTree aux_data."""
    names = tuple(labels.names)
    values = tuple(tuple(int(v) for v in row) for row in labels.values)
    return (names, values)


def _labels_from_hashable(data):
    """Reconstruct Labels from hashable representation."""
    names, values = data
    if len(values) == 0:
        return metatensor.Labels(list(names), np.empty((0, len(names)), dtype=np.int32))
    return metatensor.Labels(list(names), np.array(values, dtype=np.int32))


def _tensorblock_flatten(block):
    """Flatten TensorBlock for PyTree.

    Dynamic leaves: block values
    Static aux: samples, components, properties labels
    """
    children = [block.values]

    # Collect gradients
    grad_params = list(block.gradients())
    grad_children = []
    grad_metadata = []
    for param, grad_block in grad_params:
        grad_children.append(grad_block.values)
        grad_metadata.append((
            param,
            _labels_to_hashable(grad_block.samples),
            [_labels_to_hashable(c) for c in grad_block.components],
            _labels_to_hashable(grad_block.properties),
        ))

    children.extend(grad_children)

    aux = (
        _labels_to_hashable(block.samples),
        [_labels_to_hashable(c) for c in block.components],
        _labels_to_hashable(block.properties),
        grad_metadata,
    )
    return children, aux


def _tensorblock_unflatten(aux, children):
    """Reconstruct TensorBlock from PyTree leaves and aux."""
    samples_h, components_h, properties_h, grad_metadata = aux

    values = children[0]
    samples = _labels_from_hashable(samples_h)
    components = [_labels_from_hashable(c) for c in components_h]
    properties = _labels_from_hashable(properties_h)

    block = metatensor.TensorBlock(
        values=values,
        samples=samples,
        components=components,
        properties=properties,
    )

    # Reconstruct gradients
    for i, (param, g_samples_h, g_components_h, g_properties_h) in enumerate(
        grad_metadata
    ):
        grad_values = children[1 + i]
        grad_block = metatensor.TensorBlock(
            values=grad_values,
            samples=_labels_from_hashable(g_samples_h),
            components=[_labels_from_hashable(c) for c in g_components_h],
            properties=_labels_from_hashable(g_properties_h),
        )
        block.add_gradient(param, grad_block)

    return block


def _tensormap_flatten(tm):
    """Flatten TensorMap for PyTree.

    Dynamic leaves: all block values + gradient values (JAX traces these)
    Static aux: keys + all Labels metadata (immutable, Rust-managed)
    """
    children = []
    block_structures = []

    for i in range(len(tm)):
        block = tm.block_by_id(i)
        block_children, block_aux = _tensorblock_flatten(block)
        children.extend(block_children)
        block_structures.append((len(block_children), block_aux))

    aux = (
        _labels_to_hashable(tm.keys),
        block_structures,
        tm.info(),
    )
    return children, aux


def _tensormap_unflatten(aux, children):
    """Reconstruct TensorMap from PyTree leaves and aux."""
    keys_h, block_structures, info = aux

    keys = _labels_from_hashable(keys_h)
    blocks = []
    offset = 0

    for n_children, block_aux in block_structures:
        block_children = children[offset : offset + n_children]
        offset += n_children
        block = _tensorblock_unflatten(block_aux, block_children)
        blocks.append(block)

    tensor = metatensor.TensorMap(keys, blocks)
    for key, value in info.items():
        tensor.set_info(key, value)

    return tensor


def register_pytrees():
    """Register metatensor types as JAX PyTree nodes."""
    jax.tree_util.register_pytree_node(
        metatensor.TensorBlock,
        _tensorblock_flatten,
        _tensorblock_unflatten,
    )
    jax.tree_util.register_pytree_node(
        metatensor.TensorMap,
        _tensormap_flatten,
        _tensormap_unflatten,
    )
