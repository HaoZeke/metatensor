use std::sync::Arc;

use indexmap::IndexSet;

use crate::labels::Labels;
use crate::{Error, TensorBlock, LabelValue};

use crate::data::{mts_array_t, mts_data_movement_t};

use super::TensorMap;
use super::utils::{KeyAndBlock, remove_dimensions_from_keys, merge_samples, merge_gradient_samples};
use super::merge_plan::{MergePlan, OutputBlockPlan, BlockMergePlan, GradientMergePlan};


impl TensorMap {
    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (`keys_to_move.count()
    /// == 0`), then the new property labels will contain entries corresponding
    /// to the merged blocks only. For example, merging a block with key `a=0`
    /// and properties `p=1, 2` with a block with key `a=2` and properties `p=1,
    /// 3` will produce a block with properties `a, p = (0, 1), (0, 2), (2, 1),
    /// (2, 3)`.
    ///
    /// If `keys_to_move` contains entries, then the property labels must be the
    /// same for all the merged blocks. In that case, the merged property labels
    /// will contains each of the entries of `keys_to_move` and then the current
    /// property labels. For example, using `a=2, 3` in `keys_to_move`, and
    /// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
    /// (3, 1), (3, 2)`.
    ///
    /// The new sample labels will contains all of the merged blocks sample
    /// labels. The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    pub fn keys_to_properties(&self, keys_to_move: &Labels, fill_value: mts_array_t, sort_samples: bool) -> Result<TensorMap, Error> {
        let plan = self.keys_to_properties_plan(keys_to_move, sort_samples)?;
        execute_properties_merge_plan(self, &plan, &fill_value)
    }

    /// Compute the merge plan for keys_to_properties without touching arrays.
    ///
    /// This returns a `MergePlan` that describes how to merge blocks along the
    /// property axis. The plan can be executed later with array-specific code,
    /// enabling JAX traceability (Rust computes metadata, Python does array ops).
    pub fn keys_to_properties_plan(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<MergePlan, Error> {
        if self.keys.is_empty() {
            return Err(Error::InvalidParameter(
                "there are no keys to move in an empty TensorMap".into()
            ));
        }

        let names_to_move = keys_to_move.names();
        let splitted_keys = remove_dimensions_from_keys(&self.keys, &names_to_move)?;

        let keys_to_move = if keys_to_move.count() == 0 {
            None
        } else {
            Some(keys_to_move)
        };

        let mut output_blocks = Vec::new();
        if splitted_keys.new_keys.count() == 1 {
            let blocks_to_merge = self.keys.iter()
                .enumerate()
                .zip(&self.blocks)
                .map(|((block_index, key), block)| {
                    let mut moved_key = Vec::new();
                    for &i in &splitted_keys.dimensions_positions {
                        moved_key.push(key[i]);
                    }
                    (block_index, KeyAndBlock { key: moved_key, block })
                })
                .collect::<Vec<_>>();

            let plan = compute_properties_block_plan(
                &blocks_to_merge,
                keys_to_move,
                &names_to_move,
                sort_samples,
            )?;
            output_blocks.push(plan);
        } else {
            assert!(splitted_keys.new_keys.count() > 1);
            for entry in &splitted_keys.new_keys {
                let selection = Labels::new(
                    &splitted_keys.new_keys.names(),
                    entry.to_vec(),
                ).expect("invalid labels");

                let matching = self.blocks_matching(&selection)?;
                let blocks_to_merge = matching.iter()
                    .map(|&i| {
                        let block = &self.blocks[i];
                        let key = &self.keys[i];
                        let mut moved_key = Vec::new();
                        for &j in &splitted_keys.dimensions_positions {
                            moved_key.push(key[j]);
                        }
                        (i, KeyAndBlock { key: moved_key, block })
                    })
                    .collect::<Vec<_>>();

                let plan = compute_properties_block_plan(
                    &blocks_to_merge,
                    keys_to_move,
                    &names_to_move,
                    sort_samples,
                )?;
                output_blocks.push(plan);
            }
        }

        Ok(MergePlan {
            new_keys: Arc::new(splitted_keys.new_keys),
            output_blocks,
        })
    }
}


/// Compute the merge plan for a group of blocks to be merged along properties.
/// This is pure metadata computation -- no arrays are touched.
fn compute_properties_block_plan(
    blocks_to_merge: &[(usize, KeyAndBlock)],
    keys_to_move: Option<&Labels>,
    extracted_names: &[&str],
    sort_samples: bool,
) -> Result<OutputBlockPlan, Error> {
    assert!(!blocks_to_merge.is_empty());

    let first_block = blocks_to_merge[0].1.block;
    for gradient in first_block.gradients().values() {
        if !gradient.gradients().is_empty() {
            return Err(Error::InvalidParameter(
                "gradient of gradients are not supported yet in keys_to_properties".into()
            ));
        }
    }

    let first_components_label = &first_block.components;
    let first_property_labels = &first_block.properties;
    for (_, KeyAndBlock{block, ..}) in blocks_to_merge {
        if &block.components != first_components_label {
            return Err(Error::InvalidParameter(
                "can not move keys to properties if the blocks have \
                different components labels, call components_to_properties first".into()
            ));
        }

        if keys_to_move.is_some() && &block.properties != first_property_labels {
            return Err(Error::InvalidParameter(
                "can not provide values for the keys to move to properties if \
                the blocks have different property labels".into()
            ));
        }
    }

    // strip block indices for merge_samples (which expects &[KeyAndBlock])
    let key_and_blocks: Vec<KeyAndBlock> = blocks_to_merge.iter()
        .map(|(_, kb)| KeyAndBlock { key: kb.key.clone(), block: kb.block })
        .collect();

    // collect and merge samples across the blocks
    let (merged_samples, samples_mappings) = merge_samples(
        &key_and_blocks,
        &first_block.samples.names(),
        sort_samples,
    );

    // compute new properties
    let mut new_properties_set = IndexSet::new();
    if let Some(keys_to_move) = keys_to_move {
        for new_property in keys_to_move {
            for (_, KeyAndBlock{block, ..}) in blocks_to_merge {
                for old_property in &*block.properties {
                    let mut property = new_property.to_vec();
                    property.extend_from_slice(old_property);
                    new_properties_set.insert(property);
                }
            }
        }
        assert_eq!(new_properties_set.len(), first_property_labels.count() * keys_to_move.count());
    } else {
        for (_, KeyAndBlock{key, block}) in blocks_to_merge {
            for old_property in &*block.properties {
                let mut property = key.clone();
                property.extend_from_slice(old_property);
                new_properties_set.insert(property);
            }
        }
    }

    let new_property_names = extracted_names.iter()
        .chain(first_block.properties.names().iter())
        .copied()
        .collect::<Vec<_>>();

    let new_properties = unsafe {
        Arc::new(Labels::new_unchecked_uniqueness(
            &new_property_names,
            new_properties_set.iter().flatten().copied().collect()
        ).expect("invalid labels"))
    };

    let new_components = first_block.components.to_vec();
    let new_properties_count = new_properties.count();

    // compute output shape from first block's shape
    let first_shape = first_block.values.shape()?;
    let mut output_shape = first_shape.to_vec();
    output_shape[0] = merged_samples.count();
    let property_axis = output_shape.len() - 1;
    output_shape[property_axis] = new_properties_count;

    // compute property ranges and movement instructions for each block
    let mut block_plans = Vec::new();
    for (i, ((block_index, KeyAndBlock{key, block}), samples_mapping)) in
        blocks_to_merge.iter().zip(&samples_mappings).enumerate()
    {
        let property_range = if block.properties.is_empty() {
            None
        } else {
            let mut first_prop = key.clone();
            first_prop.extend_from_slice(&block.properties[0]);
            let start = new_properties.position(&first_prop);
            let size = block.properties.count();
            start.map(|s| s..(s + size))
        };

        let movements = if let Some(ref pr) = property_range {
            samples_mapping.iter().enumerate().map(|(sample_i, &new_sample_i)| {
                mts_data_movement_t {
                    sample_in: sample_i,
                    sample_out: new_sample_i,
                    properties_start_in: 0,
                    properties_start_out: pr.start,
                    properties_length: pr.len(),
                }
            }).collect()
        } else {
            Vec::new()
        };

        block_plans.push(BlockMergePlan {
            block_index: *block_index,
            samples_mapping: samples_mapping.clone(),
            property_range,
            movements,
        });
    }

    // compute gradient plans
    let mut gradient_plans = Vec::new();
    for (parameter, _first_gradient) in first_block.gradients() {
        let new_gradient_samples = merge_gradient_samples(
            &key_and_blocks, parameter, &samples_mappings
        );

        let mut block_movements = Vec::new();
        for (bp, (_, KeyAndBlock{block, ..})) in block_plans.iter().zip(blocks_to_merge) {
            if bp.property_range.is_none() {
                block_movements.push(Vec::new());
                continue;
            }

            let property_range = bp.property_range.as_ref().unwrap();
            let gradient = block.gradient(parameter).expect("missing gradient");

            let movements: Vec<mts_data_movement_t> = gradient.samples.iter().enumerate().map(|(sample_i, grad_sample)| {
                let mut grad_sample = grad_sample.to_vec();
                let old_sample_i = usize::try_from(grad_sample[0]).expect("could not convert to usize");
                let new_sample_i = bp.samples_mapping[old_sample_i];
                grad_sample[0] = LabelValue::from(i32::try_from(new_sample_i).expect("could not convert to i32"));

                let new_grad_sample_i = new_gradient_samples.position(&grad_sample).expect("missing entry in merged samples");

                mts_data_movement_t {
                    sample_in: sample_i,
                    sample_out: new_grad_sample_i,
                    properties_start_in: 0,
                    properties_start_out: property_range.start,
                    properties_length: property_range.len(),
                }
            }).collect();

            block_movements.push(movements);
        }

        gradient_plans.push(GradientMergePlan {
            parameter: parameter.clone(),
            samples: new_gradient_samples,
            block_movements,
        });
    }

    Ok(OutputBlockPlan {
        samples: merged_samples,
        components: new_components,
        properties: new_properties,
        output_shape,
        block_plans,
        gradient_plans,
    })
}


/// Execute a merge plan along properties using array operations.
/// This is the "Phase 2" that touches arrays.
fn execute_properties_merge_plan(
    tensor: &TensorMap,
    plan: &MergePlan,
    fill_value: &mts_array_t,
) -> Result<TensorMap, Error> {
    let mut new_blocks = Vec::new();

    for output_plan in &plan.output_blocks {
        // find first block to use as template for array creation
        let first_block_idx = output_plan.block_plans[0].block_index;
        let first_block = &tensor.blocks()[first_block_idx];

        // create output array filled with fill_value
        let mut new_data = first_block.values.create(&output_plan.output_shape, fill_value)?;

        // execute movements for each input block
        for bp in &output_plan.block_plans {
            if !bp.movements.is_empty() {
                let source_block = &tensor.blocks()[bp.block_index];
                new_data.move_data(&source_block.values, &bp.movements)?;
            }
        }

        let mut new_block = TensorBlock::new(
            new_data,
            Arc::clone(&output_plan.samples),
            output_plan.components.clone(),
            Arc::clone(&output_plan.properties),
        ).expect("constructed an invalid block");

        // execute gradient movements
        for gp in &output_plan.gradient_plans {
            let first_gradient = first_block.gradient(&gp.parameter).expect("missing gradient");
            let mut grad_shape = first_gradient.values.shape()?.to_vec();
            grad_shape[0] = gp.samples.count();
            let property_axis = grad_shape.len() - 1;
            grad_shape[property_axis] = output_plan.properties.count();

            let mut new_gradient_data = first_block.values.create(&grad_shape, fill_value)?;
            let new_grad_components = first_gradient.components.to_vec();

            for (bp, movements) in output_plan.block_plans.iter().zip(&gp.block_movements) {
                if movements.is_empty() {
                    continue;
                }
                let source_block = &tensor.blocks()[bp.block_index];
                let gradient = source_block.gradient(&gp.parameter).expect("missing gradient");
                new_gradient_data.move_data(&gradient.values, movements)?;
            }

            let new_gradient = TensorBlock::new(
                new_gradient_data,
                Arc::clone(&gp.samples),
                new_grad_components,
                Arc::clone(&output_plan.properties),
            ).expect("created invalid gradient");

            new_block.add_gradient(&gp.parameter, new_gradient).expect("could not add gradient");
        }

        new_blocks.push(new_block);
    }

    let mut result = TensorMap::new(Arc::clone(&plan.new_keys), new_blocks)?;
    for (k, v) in tensor.info() {
        result.add_info(k, v.clone());
    }
    Ok(result)
}
