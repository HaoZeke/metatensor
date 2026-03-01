"""
Operations for metatensor-torch.

With the unified TensorMap type (LSP unification), operations are defined ONCE
in metatensor-operations. This module simply re-exports them.

For TorchScript-compiled models that need operations at the C++ level, use
``_scripted_ops.py`` which wraps ``torch.ops.metatensor.*`` directly.
"""

from metatensor_operations import *  # noqa: F401, F403
