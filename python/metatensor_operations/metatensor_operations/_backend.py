# This file defines the default set of classes to use in metatensor-operations
#
# The code from metatensor-operations can be used in two different modes: either pure
# Python mode or TorchScript mode. In the second case, we need to use a different
# version of the classes above; to do without having to rewrite everything, we re-import
# metatensor-operations in a special context.
#
# See metatensor-torch/metatensor/torch/operations.py for more information.
#
# Any change to this file MUST be also be made to `metatensor/torch/operations.py`.
import re
from typing import Union

import numpy as np

import metatensor


try:
    import torch

    Array = Union[np.ndarray, torch.Tensor]
    _HAS_TORCH = True
except ImportError:
    Array = np.ndarray
    _HAS_TORCH = False


Labels = metatensor.Labels
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap

# type used by the values in Labels
LabelsValues = np.ndarray


def torch_jit_is_scripting():
    return False


def torch_jit_script(function):
    """
    This function takes the place of ``@torch.jit.script`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).

    When used as a decorator (``@torch_jit_script def foo()``), it does nothing to the
    function.
    """
    return function


def torch_jit_annotate(type, value):
    """
    This function takes the place of ``torch.jit.annotate`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).
    """
    return value


_VERSION_REGEX = re.compile(r"(\d+)\.(\d+)\.*.")


def _version_at_least(version, expected):
    version = tuple(map(int, _VERSION_REGEX.match(version).groups()))
    expected = tuple(map(int, _VERSION_REGEX.match(expected).groups()))

    return version >= expected


# Warning: this function (as all functions in this module) is part of the public API of
# metatensor-operations, updating it means that new versions of metatensor-torch will
# not be able to work with old versions of metatensor-operations, so any update should
# be treated as a breaking change.
def isinstance_metatensor(value, typename):
    assert typename in ("Labels", "TensorBlock", "TensorMap")

    # With unified TensorMap (LSP unification), metatensor.torch re-exports the
    # core types. TorchScript ScriptObjects are handled by the bridge layer
    # (_bridge.py) and should be converted before reaching operations.

    if typename == "Labels":
        return isinstance(value, Labels)
    elif typename == "TensorBlock":
        return isinstance(value, TensorBlock)
    elif typename == "TensorMap":
        return isinstance(value, TensorMap)
    else:
        return False
