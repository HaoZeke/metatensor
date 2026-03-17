//! XLA FFI wrappers for metatensor C API functions.
//!
//! Provides thin wrappers that encode metatensor opaque pointers as i64 scalar
//! values, suitable for passing through XLA's buffer protocol. These functions
//! are registered as XLA custom call targets via `jax.ffi.register_ffi_target`.
//!
//! See PJRT_XLA_FFI_Integration.org in obsidian-notes for design details.

use metatensor_sys::{
    mts_labels_t, mts_tensormap_t, mts_status_t,
    mts_labels_clone, mts_labels_free, mts_labels_count, mts_labels_size,
};

// Merge plan type and functions are new additions not yet in metatensor-sys
#[repr(C)]
pub struct mts_merge_plan_t {
    _unused: [u8; 0],
}

extern "C" {
    fn mts_merge_plan_block_count(
        plan: *const mts_merge_plan_t,
        count: *mut usize,
    ) -> mts_status_t;
    fn mts_merge_plan_free(
        plan: *mut mts_merge_plan_t,
    ) -> mts_status_t;
    fn mts_tensormap_keys_to_properties_plan(
        tensor: *const mts_tensormap_t,
        keys_to_move: *const mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_merge_plan_t;
    fn mts_tensormap_keys_to_samples_plan(
        tensor: *const mts_tensormap_t,
        keys_to_move: *const mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_merge_plan_t;
}

/// Convert i64 to raw pointer.
#[inline]
unsafe fn i64_to_ptr<T>(value: i64) -> *const T {
    value as usize as *const T
}

#[inline]
unsafe fn i64_to_mut_ptr<T>(value: i64) -> *mut T {
    value as usize as *mut T
}

#[inline]
fn ptr_to_i64<T>(ptr: *const T) -> i64 {
    ptr as usize as i64
}

// ============================================================================
// XLA FFI handlers -- Labels
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_clone(labels_ptr: i64) -> i64 {
    let labels = i64_to_ptr::<mts_labels_t>(labels_ptr);
    if labels.is_null() { return 0; }
    ptr_to_i64(mts_labels_clone(labels))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_free(labels_ptr: i64) -> i32 {
    let labels = i64_to_mut_ptr::<mts_labels_t>(labels_ptr);
    mts_labels_free(labels)
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_count(labels_ptr: i64) -> i64 {
    let labels = i64_to_ptr::<mts_labels_t>(labels_ptr);
    if labels.is_null() { return -1; }
    let mut count: usize = 0;
    if mts_labels_count(labels, &mut count) != 0 { return -1; }
    count as i64
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_size(labels_ptr: i64) -> i64 {
    let labels = i64_to_ptr::<mts_labels_t>(labels_ptr);
    if labels.is_null() { return -1; }
    let mut size: usize = 0;
    if mts_labels_size(labels, &mut size) != 0 { return -1; }
    size as i64
}

// ============================================================================
// XLA FFI handlers -- Merge Plan
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_keys_to_properties_plan(
    tensor_ptr: i64, keys_ptr: i64, sort_samples: bool,
) -> i64 {
    let tensor = i64_to_ptr::<mts_tensormap_t>(tensor_ptr);
    let keys = i64_to_ptr::<mts_labels_t>(keys_ptr);
    ptr_to_i64(mts_tensormap_keys_to_properties_plan(tensor, keys, sort_samples))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_keys_to_samples_plan(
    tensor_ptr: i64, keys_ptr: i64, sort_samples: bool,
) -> i64 {
    let tensor = i64_to_ptr::<mts_tensormap_t>(tensor_ptr);
    let keys = i64_to_ptr::<mts_labels_t>(keys_ptr);
    ptr_to_i64(mts_tensormap_keys_to_samples_plan(tensor, keys, sort_samples))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_free(plan_ptr: i64) -> i32 {
    let plan = i64_to_mut_ptr::<mts_merge_plan_t>(plan_ptr);
    mts_merge_plan_free(plan)
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_block_count(plan_ptr: i64) -> i64 {
    let plan = i64_to_ptr::<mts_merge_plan_t>(plan_ptr);
    if plan.is_null() { return -1; }
    let mut count: usize = 0;
    if mts_merge_plan_block_count(plan, &mut count) != 0 { return -1; }
    count as i64
}
