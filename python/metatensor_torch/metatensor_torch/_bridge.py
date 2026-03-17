"""
Bridge between unified metatensor types and TorchScript ScriptObjects.

This module provides conversion functions for model export/import boundaries:
``torch.jit.save``, ``torch.jit.load``, and the metatensor.torch.atomistic
deployment pipeline. The unified TensorMap is THE type for all computation;
TorchScript ScriptObjects are only used for serialization.
"""

import torch

import metatensor


def to_torch_script(tensor_map: metatensor.TensorMap) -> torch.ScriptObject:
    """Convert a unified TensorMap to a TorchScript ScriptObject.

    Used at model export boundaries (e.g. before ``torch.jit.save``).

    :param tensor_map: the unified TensorMap to convert
    :return: a TorchScript-compatible ScriptObject (torch.classes.metatensor.TensorMap)
    """
    # Save to buffer, then load as TorchScript object
    buffer = tensor_map.save_buffer()
    return torch.ops.metatensor.load_buffer(bytes(buffer))


def from_torch_script(ts_map: torch.ScriptObject) -> metatensor.TensorMap:
    """Convert a TorchScript ScriptObject back to a unified TensorMap.

    Used at model import boundaries (e.g. after ``torch.jit.load``).

    :param ts_map: a TorchScript ScriptObject
    :return: a unified TensorMap
    """
    # Save TorchScript object to buffer, then load as unified type
    buffer = torch.ops.metatensor.save_buffer(ts_map)
    return metatensor.io.load_buffer(buffer)


def to_torch_script_block(
    tensor_block: metatensor.TensorBlock,
) -> torch.ScriptObject:
    """Convert a unified TensorBlock to a TorchScript ScriptObject."""
    from metatensor.io import _save_block_buffer_raw, load_block_buffer

    buffer = _save_block_buffer_raw(tensor_block)
    return torch.ops.metatensor.load_block_buffer(bytes(buffer))


def to_torch_script_labels(labels: metatensor.Labels) -> torch.ScriptObject:
    """Convert unified Labels to a TorchScript ScriptObject."""
    buffer = labels.save_buffer()
    return torch.ops.metatensor.load_labels_buffer(bytes(buffer))


def from_torch_script_labels(ts_labels: torch.ScriptObject) -> metatensor.Labels:
    """Convert TorchScript Labels back to unified Labels."""
    buffer = torch.ops.metatensor.save_buffer(ts_labels)
    return metatensor.Labels.load_buffer(buffer)
