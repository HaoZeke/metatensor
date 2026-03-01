"""
TorchScript-compiled operations that call ``torch.ops.metatensor.*`` directly.

These are only used by the atomistic module for model deployment where
TorchScript compilation is required. Normal Python operations go through
metatensor-operations (re-exported in ``operations.py``).

This module provides thin wrappers around the C++ ops registered by the
metatensor-torch shared library.
"""

import torch


def add_ts(
    a: torch.classes.metatensor.TensorMap,
    b: torch.classes.metatensor.TensorMap,
) -> torch.classes.metatensor.TensorMap:
    """TorchScript-compatible add operation."""
    return torch.ops.metatensor.add(a, b)


def subtract_ts(
    a: torch.classes.metatensor.TensorMap,
    b: torch.classes.metatensor.TensorMap,
) -> torch.classes.metatensor.TensorMap:
    """TorchScript-compatible subtract operation."""
    return torch.ops.metatensor.subtract(a, b)


def multiply_ts(
    a: torch.classes.metatensor.TensorMap,
    b: torch.classes.metatensor.TensorMap,
) -> torch.classes.metatensor.TensorMap:
    """TorchScript-compatible multiply operation."""
    return torch.ops.metatensor.multiply(a, b)
