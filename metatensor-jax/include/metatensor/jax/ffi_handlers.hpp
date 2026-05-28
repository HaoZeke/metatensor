#ifndef METATENSOR_JAX_FFI_HANDLERS_HPP
#define METATENSOR_JAX_FFI_HANDLERS_HPP

#include <cstdint>

#include "metatensor/jax/exports.h"

extern "C" {
struct XLA_FFI_CallFrame;
struct XLA_FFI_Error;
}

namespace metatensor_jax {

/// Each operation exposes one extern "C" symbol of the shape
/// `XLA_FFI_Error* metatensor_<op>(XLA_FFI_CallFrame*)`. The signatures live
/// in src/register.cpp via XLA_FFI_DEFINE_HANDLER_SYMBOL. The impl_* free
/// functions in src/register.cpp take typed `ffi::Buffer<DataType::S64, 0>`
/// arguments carrying metatensor opaque pointer values, call the matching
/// `mts_*` C ABI function from <metatensor.h>, and return `ffi::Error`.
///
/// The Python side (`python/metatensor_jax/metatensor_jax/_xla_ffi.py`)
/// hands each symbol to `jax.ffi.register_ffi_target`, after which a jit-
/// traced function can call into metatensor-core through `jax.ffi.ffi_call`
/// and the symbol name appears in `jax.make_jaxpr` output.

// Labels handlers
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_clone(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_free(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_count(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_size(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_position(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_union(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_intersection(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_difference(XLA_FFI_CallFrame*);

// Block handlers
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_block_free(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_block_copy(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_block_labels(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_block_gradient(XLA_FFI_CallFrame*);

// TensorMap handlers
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_free(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_copy(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_keys(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_block_by_id(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_blocks_matching(XLA_FFI_CallFrame*);

// Merge plan handlers (split keys_to_properties / keys_to_samples from R5)
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_keys_to_properties_plan(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_keys_to_samples_plan(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_free(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_new_keys(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_block_count(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_block_samples(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_block_properties(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_merge_plan_block_input_count(XLA_FFI_CallFrame*);

// Operations layer (R2 explicitly names sort, slice, mean_over_samples / sum_over_samples)
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_sort(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_slice(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_mean_over_samples(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_sum_over_samples(XLA_FFI_CallFrame*);

// IO handlers
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_load(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_labels_save(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_load(XLA_FFI_CallFrame*);
extern "C" METATENSOR_JAX_EXPORT XLA_FFI_Error* metatensor_tensormap_save(XLA_FFI_CallFrame*);

}  // namespace metatensor_jax

#endif  // METATENSOR_JAX_FFI_HANDLERS_HPP
