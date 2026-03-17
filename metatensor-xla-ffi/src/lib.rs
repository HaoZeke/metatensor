//! XLA FFI wrappers for metatensor C API functions.
//!
//! Provides thin wrappers that encode metatensor opaque pointers as i64 scalar
//! values, suitable for passing through XLA's buffer protocol. These functions
//! are registered as XLA custom call targets via `jax.ffi.register_ffi_target`.
//!
//! See PJRT_XLA_FFI_Integration.org in obsidian-notes for design details.

use metatensor_sys::{
    mts_labels_t, mts_tensormap_t, mts_block_t, mts_status_t, mts_array_t,
    // Labels
    mts_labels_clone, mts_labels_free, mts_labels_count, mts_labels_size,
    mts_labels_position, mts_labels_values,
    mts_labels_union, mts_labels_intersection, mts_labels_difference,
    // Blocks
    mts_block_free, mts_block_copy, mts_block_labels,
    mts_block_data,
    // TensorMap
    mts_tensormap_free, mts_tensormap_copy, mts_tensormap_keys,
    mts_tensormap_block_by_id, mts_tensormap_blocks_matching,
    // IO
    mts_labels_load, mts_labels_save,
    mts_tensormap_load, mts_tensormap_save,
};

// Merge plan type -- new addition not yet in metatensor-sys
#[repr(C)]
pub struct mts_merge_plan_t {
    _unused: [u8; 0],
}

extern "C" {
    fn mts_merge_plan_block_count(plan: *const mts_merge_plan_t, count: *mut usize) -> mts_status_t;
    fn mts_merge_plan_free(plan: *mut mts_merge_plan_t) -> mts_status_t;
    fn mts_merge_plan_new_keys(plan: *const mts_merge_plan_t) -> *mut mts_labels_t;
    fn mts_merge_plan_block_samples(plan: *const mts_merge_plan_t, block_idx: usize) -> *mut mts_labels_t;
    fn mts_merge_plan_block_properties(plan: *const mts_merge_plan_t, block_idx: usize) -> *mut mts_labels_t;
    fn mts_merge_plan_block_shape(plan: *const mts_merge_plan_t, block_idx: usize, shape: *mut *const usize, shape_count: *mut usize) -> mts_status_t;
    fn mts_merge_plan_block_input_count(plan: *const mts_merge_plan_t, block_idx: usize, count: *mut usize) -> mts_status_t;
    fn mts_merge_plan_block_movements(plan: *const mts_merge_plan_t, block_idx: usize, input_idx: usize, source_block_index: *mut usize, movements: *mut *const metatensor_sys::mts_data_movement_t, movements_count: *mut usize) -> mts_status_t;
    fn mts_tensormap_keys_to_properties_plan(tensor: *const mts_tensormap_t, keys_to_move: *const mts_labels_t, sort_samples: bool) -> *mut mts_merge_plan_t;
    fn mts_tensormap_keys_to_samples_plan(tensor: *const mts_tensormap_t, keys_to_move: *const mts_labels_t, sort_samples: bool) -> *mut mts_merge_plan_t;
}

#[inline]
unsafe fn i64_to_ptr<T>(value: i64) -> *const T { value as usize as *const T }
#[inline]
unsafe fn i64_to_mut_ptr<T>(value: i64) -> *mut T { value as usize as *mut T }
#[inline]
fn ptr_to_i64<T>(ptr: *const T) -> i64 { ptr as usize as i64 }
#[inline]
fn mut_ptr_to_i64<T>(ptr: *mut T) -> i64 { ptr as usize as i64 }

// ============================================================================
// Labels (7 handlers)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_clone(p: i64) -> i64 {
    mut_ptr_to_i64(mts_labels_clone(i64_to_ptr(p)))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_free(p: i64) -> i32 {
    mts_labels_free(i64_to_mut_ptr(p))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_count(p: i64) -> i64 {
    let mut c: usize = 0;
    if mts_labels_count(i64_to_ptr(p), &mut c) != 0 { return -1; }
    c as i64
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_size(p: i64) -> i64 {
    let mut s: usize = 0;
    if mts_labels_size(i64_to_ptr(p), &mut s) != 0 { return -1; }
    s as i64
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_position(p: i64, values: *const i32, values_count: usize) -> i64 {
    let mut result: i64 = -1;
    if mts_labels_position(i64_to_ptr(p), values, values_count, &mut result) != 0 { return -2; }
    result
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_union(first: i64, second: i64) -> i64 {
    let mut result: *mut mts_labels_t = std::ptr::null_mut();
    let status = mts_labels_union(
        i64_to_ptr(first), i64_to_ptr(second),
        &mut result, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0,
    );
    if status != 0 { return 0; }
    mut_ptr_to_i64(result)
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_intersection(first: i64, second: i64) -> i64 {
    let mut result: *mut mts_labels_t = std::ptr::null_mut();
    let status = mts_labels_intersection(
        i64_to_ptr(first), i64_to_ptr(second),
        &mut result, std::ptr::null_mut(), 0, std::ptr::null_mut(), 0,
    );
    if status != 0 { return 0; }
    mut_ptr_to_i64(result)
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_difference(first: i64, second: i64) -> i64 {
    let mut result: *mut mts_labels_t = std::ptr::null_mut();
    let status = mts_labels_difference(
        i64_to_ptr(first), i64_to_ptr(second),
        &mut result, std::ptr::null_mut(), 0,
    );
    if status != 0 { return 0; }
    mut_ptr_to_i64(result)
}

// ============================================================================
// Blocks (4 handlers)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_block_free(p: i64) -> i32 {
    mts_block_free(i64_to_mut_ptr(p))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_block_copy(p: i64) -> i64 {
    mut_ptr_to_i64(mts_block_copy(i64_to_ptr(p)))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_block_labels(p: i64, axis: usize) -> i64 {
    mut_ptr_to_i64(mts_block_labels(i64_to_ptr(p), axis))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_block_data(p: i64, data: *mut mts_array_t) -> i32 {
    mts_block_data(i64_to_mut_ptr(p), data)
}

// ============================================================================
// TensorMap (7 handlers)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_free(p: i64) -> i32 {
    mts_tensormap_free(i64_to_mut_ptr(p))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_copy(p: i64) -> i64 {
    mut_ptr_to_i64(mts_tensormap_copy(i64_to_ptr(p)))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_keys(p: i64) -> i64 {
    mut_ptr_to_i64(mts_tensormap_keys(i64_to_ptr(p)))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_block_by_id(p: i64, index: usize) -> i64 {
    let mut block: *mut mts_block_t = std::ptr::null_mut();
    if mts_tensormap_block_by_id(i64_to_mut_ptr(p), &mut block, index) != 0 { return 0; }
    mut_ptr_to_i64(block)
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_blocks_matching(
    p: i64, selection_ptr: i64, out_indices: *mut usize, out_count: *mut usize,
) -> i32 {
    mts_tensormap_blocks_matching(i64_to_ptr(p), out_indices, out_count, i64_to_ptr(selection_ptr))
}

// ============================================================================
// Merge Plan (8 handlers)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_keys_to_properties_plan(
    tensor: i64, keys: i64, sort_samples: bool,
) -> i64 {
    mut_ptr_to_i64(mts_tensormap_keys_to_properties_plan(i64_to_ptr(tensor), i64_to_ptr(keys), sort_samples))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_keys_to_samples_plan(
    tensor: i64, keys: i64, sort_samples: bool,
) -> i64 {
    mut_ptr_to_i64(mts_tensormap_keys_to_samples_plan(i64_to_ptr(tensor), i64_to_ptr(keys), sort_samples))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_free(p: i64) -> i32 {
    mts_merge_plan_free(i64_to_mut_ptr(p))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_new_keys(p: i64) -> i64 {
    mut_ptr_to_i64(mts_merge_plan_new_keys(i64_to_ptr(p)))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_block_count(p: i64) -> i64 {
    let mut c: usize = 0;
    if mts_merge_plan_block_count(i64_to_ptr(p), &mut c) != 0 { return -1; }
    c as i64
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_block_samples(p: i64, idx: usize) -> i64 {
    mut_ptr_to_i64(mts_merge_plan_block_samples(i64_to_ptr(p), idx))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_block_properties(p: i64, idx: usize) -> i64 {
    mut_ptr_to_i64(mts_merge_plan_block_properties(i64_to_ptr(p), idx))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_merge_plan_block_input_count(p: i64, idx: usize) -> i64 {
    let mut c: usize = 0;
    if mts_merge_plan_block_input_count(i64_to_ptr(p), idx, &mut c) != 0 { return -1; }
    c as i64
}

// ============================================================================
// IO (4 handlers -- load/save for labels and tensormaps)
// ============================================================================

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_load(path: *const std::os::raw::c_char) -> i64 {
    mut_ptr_to_i64(mts_labels_load(path))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_labels_save(path: *const std::os::raw::c_char, labels: i64) -> i32 {
    mts_labels_save(path, i64_to_ptr(labels))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_load(
    path: *const std::os::raw::c_char,
    create_array: metatensor_sys::mts_create_array_callback_t,
) -> i64 {
    mut_ptr_to_i64(mts_tensormap_load(path, create_array))
}

#[no_mangle]
pub unsafe extern "C" fn mts_xla_tensormap_save(
    path: *const std::os::raw::c_char, tensor: i64,
) -> i32 {
    mts_tensormap_save(path, i64_to_ptr(tensor))
}
