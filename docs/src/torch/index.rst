.. _metatensor-torch:

TorchScript backend
===================

.. toctree::
    :maxdepth: 2
    :hidden:

    reference/index

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md

We provide a special PyTorch C++ extension exporting all of the core metatensor
types in a way compatible with the TorchScript compiler, allowing users to save
and load models based on metatensor everywhere TorchScript is supported. This
allow to define, train and save a model from Python, and then load it with pure
C++ code, without requiring a Python interpreter. Please refer to the
:ref:`installation instructions <install-torch>` to know how to install the
Python and C++ sides of this library.

.. grid::

    .. grid-item-card:: |Python-16x16| TorchScript Python API reference
        :link: python-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions in the
        ``metatensor-torch`` Python package.

        +++
        Documentation for version |metatensor-torch-version|

    .. grid-item-card:: |Cxx-16x16| TorchScript C++ API reference
        :link: cxx-api-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions in the
        ``metatensor/torch.hpp`` C++ header.

        +++
        Documentation for version |metatensor-torch-version|

Unified type model
^^^^^^^^^^^^^^^^^^

Since the LSP unification, ``metatensor.torch`` re-exports the core types:
``metatensor.torch.TensorMap`` is now the same class as
``metatensor.TensorMap``. Code written for any backend (numpy, torch, JAX)
uses the same types and operations.

The ``metatensor-torch`` package provides two additional capabilities beyond
the core:

1. **TorchScript export**: The ``to_torch_script()`` / ``from_torch_script()``
   bridge converts ``TensorMap`` to/from TorchScript ``ScriptObject`` types for
   model deployment via ``torch.jit.save`` / ``torch.jit.load``.

2. **C++ integration**: The TorchScript C++ API (``metatensor/torch.hpp``)
   enables non-Python software to load and run metatensor-based models.

When to use metatensor-torch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Model deployment**: If your model needs to be exported to TorchScript
  (for use in C++ simulation engines, for example), install ``metatensor-torch``
  and use ``to_torch_script()`` at the export boundary.

- **Training and development**: Use the core ``metatensor.TensorMap`` with
  regular PyTorch tensors as block values. ``torch.compile`` and autograd work
  through the unified operations layer.

- **Backend-agnostic code**: Operations like ``metatensor.add``,
  ``metatensor.join``, etc. work identically with numpy, torch, and :ref:`JAX
  <metatensor-jax>` arrays. Write once, run anywhere.

TorchScript bridge
^^^^^^^^^^^^^^^^^^

For deploying models in non-Python environments:

.. code-block:: python

    import metatensor.torch as mts_torch

    # Convert to TorchScript for export
    ts_map = mts_torch.to_torch_script(tensor_map)
    torch.jit.save(ts_map, "model_output.pt")

    # Load back
    ts_map_loaded = torch.jit.load("model_output.pt")
    tensor_map = mts_torch.from_torch_script(ts_map_loaded)

.. _Jax: https://jax.readthedocs.io/en/latest/
.. _cupy: https://cupy.dev
