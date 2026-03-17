import os
import sys
from typing import TYPE_CHECKING

import torch

from . import utils  # noqa: F401
from ._c_lib import _load_library
from .version import __version__  # noqa: F401

# Re-export unified core types -- these ARE the canonical types
import metatensor as _metatensor_core

Labels = _metatensor_core.Labels
TensorBlock = _metatensor_core.TensorBlock
TensorMap = _metatensor_core.TensorMap

sys.modules["metatensor.torch"] = sys.modules[__name__]

if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") != "0" or TYPE_CHECKING:
    from .documentation import (
        Labels,  # noqa: F811
        LabelsEntry,
        TensorBlock,  # noqa: F811
        TensorMap,  # noqa: F811
        dtype_name,
        load_block_buffer,
        load_buffer,
        load_labels_buffer,
        save_buffer,
        version,
    )
else:
    _load_library()

    # TorchScript C++ classes available for export path
    _TorchScriptLabels = torch.classes.metatensor.Labels
    _TorchScriptLabelsEntry = torch.classes.metatensor.LabelsEntry
    _TorchScriptTensorBlock = torch.classes.metatensor.TensorBlock
    _TorchScriptTensorMap = torch.classes.metatensor.TensorMap
    LabelsEntry = _metatensor_core.LabelsEntry

    version = torch.ops.metatensor.version
    dtype_name = torch.ops.metatensor.dtype_name
    load_buffer = torch.ops.metatensor.load_buffer
    load_block_buffer = torch.ops.metatensor.load_block_buffer
    load_labels_buffer = torch.ops.metatensor.load_labels_buffer
    save_buffer = torch.ops.metatensor.save_buffer

# Bridge functions for TorchScript <-> unified type conversion
from ._bridge import (  # noqa: E402, F401
    from_torch_script,
    to_torch_script,
)

from .serialization import (  # noqa: F401, E402
    load,
    load_block,
    load_labels,
    save,
)


try:
    import metatensor_operations  # noqa: F401, E402

    HAS_METATENSOR_OPERATIONS = True
except ImportError:
    HAS_METATENSOR_OPERATIONS = False

if HAS_METATENSOR_OPERATIONS:
    from . import operations  # noqa: F401

    _ops = sys.modules["metatensor.torch.operations"]
    for _name in getattr(
        _ops, "__all__", [n for n in dir(_ops) if not n.startswith("_")]
    ):
        globals()[_name] = getattr(_ops, _name)
else:
    # __getattr__ is called when a module attribute can not be found, we use it to
    # give the user a better error message if they don't have metatensor-operations
    def __getattr__(name):
        raise AttributeError(
            f"metatensor.torch.{name} is not defined, are you sure you have the "
            "metatensor-operations package installed?"
        )


try:
    import metatensor_learn  # noqa: F401

    from . import learn  # noqa: F401
except ImportError:
    pass


from . import atomistic  # noqa: F401, E402
from ._module import _patch_torch_jit_module  # noqa:  E402


_patch_torch_jit_module()

__all__ = [
    "Labels",
    "TensorBlock",
    "TensorMap",
]
