"""
JAX array callbacks for metatensor's ``mts_array_t``.

JAX arrays support ``__dlpack__`` in eager mode, so the existing DLPack
bridge in metatensor-core already handles data interchange. This module
provides the registration hook that tells metatensor-core how to recognize
and wrap jax arrays.

For JIT mode (Tracer objects), the callbacks return shape/dtype from the
abstract value. ``as_dlpack()`` raises if called during tracing because the
Rust core only needs shape/dtype/device during operations, not physical
memory.
"""

from metatensor.data.array import _is_jax_array


def register_jax_array_callbacks():
    """Register jax array support with metatensor-core's data extraction layer.

    This is called automatically on ``import metatensor_jax``. It registers
    the jax data origin so that ``mts_array_to_python_array`` can extract
    jax arrays from ``mts_array_t``.
    """
    try:
        import jax

        from metatensor.data.extract import register_external_data_wrapper
        from metatensor.data.array import _origin_jax

        class ExternalJaxArray:
            """Wrapper for external jax array data from mts_array_t."""

            def __init__(self, dl_tensor, parent):
                import jax.dlpack

                self._parent = parent
                self.array = jax.dlpack.from_dlpack(dl_tensor)

        # Register so that mts_array_to_python_array recognizes jax origins
        register_external_data_wrapper(
            "metatensor.data.array.jax",
            ExternalJaxArray,
        )

    except ImportError:
        pass  # jax not available, nothing to register
