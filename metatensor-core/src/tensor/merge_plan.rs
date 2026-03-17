//! Merge plan types for keys_to_properties and keys_to_samples.
//!
//! A merge plan captures the metadata (labels, sample mappings, property ranges,
//! movement instructions) needed to merge blocks, WITHOUT touching arrays.
//! This separation enables JAX traceability: Rust computes the plan (concrete),
//! Python executes the array operations (traced by JAX).

use std::ops::Range;
use std::sync::Arc;

use crate::Labels;
use crate::data::mts_data_movement_t;

/// Plan for merging a single block's data into the output array.
#[derive(Debug, Clone)]
pub struct BlockMergePlan {
    /// Index of the source block in the original TensorMap
    pub block_index: usize,
    /// Mapping from old sample indices to new sample indices
    pub samples_mapping: Vec<usize>,
    /// Range of properties in the output array for this block's data.
    /// None if the block's key is not in the requested keys_to_move.
    pub property_range: Option<Range<usize>>,
    /// Pre-computed movement instructions for array scatter
    pub movements: Vec<mts_data_movement_t>,
}

/// Plan for merging gradient data.
#[derive(Debug, Clone)]
pub struct GradientMergePlan {
    /// Name of the gradient parameter
    pub parameter: String,
    /// New gradient sample labels
    pub samples: Arc<Labels>,
    /// Pre-computed movement instructions for gradient scatter
    pub block_movements: Vec<Vec<mts_data_movement_t>>,
}

/// Complete merge plan for a single output block (result of merging multiple
/// input blocks along properties or samples).
#[derive(Debug, Clone)]
pub struct OutputBlockPlan {
    /// New sample labels for the output block
    pub samples: Arc<Labels>,
    /// New component labels for the output block
    pub components: Vec<Arc<Labels>>,
    /// New property labels for the output block
    pub properties: Arc<Labels>,
    /// Shape of the output array (samples, components..., properties)
    pub output_shape: Vec<usize>,
    /// Per-input-block merge instructions
    pub block_plans: Vec<BlockMergePlan>,
    /// Gradient merge plans
    pub gradient_plans: Vec<GradientMergePlan>,
}

/// Complete merge plan for a keys_to_properties or keys_to_samples operation.
#[derive(Debug, Clone)]
pub struct MergePlan {
    /// New keys for the output TensorMap
    pub new_keys: Arc<Labels>,
    /// One OutputBlockPlan per output block
    pub output_blocks: Vec<OutputBlockPlan>,
}
