use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::io::Read;
use std::sync::Arc;

use memmap2::Mmap;
use zip::ZipArchive;

use dlpk::sys::DLDataType;

use super::labels::load_labels;
use super::Endianness;
use super::block::npy_descr_to_dtype;
use super::mmap_array::{MmapArray, element_size, new_created_array};
use super::npy_header::{Header, DataType};

use crate::utils::ConstCString;
use crate::{TensorMap, TensorBlock, Labels, LabelValue, Error, mts_array_t};

/// Load a `TensorMap` from the file at the given path using memory mapping.
///
/// This provides zero-copy loading for data arrays: instead of reading and
/// copying the array data, the file is memory-mapped and the array data
/// points directly into the mapped region. This can significantly reduce
/// memory usage and loading time for large datasets.
///
/// Labels (samples, components, properties) are still loaded normally since
/// they need to be owned by the `Labels` struct.
///
/// The file must use the STORED (uncompressed) ZIP format, which is the
/// default when saving with metatensor.
pub fn load_mmap(path: &str) -> Result<TensorMap, Error> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    let path_str = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&path_str).map_err(|e| (path_str, e))?)?;

    let mut blocks = Vec::new();
    for block_i in 0..keys.count() {
        let prefix = format!("blocks/{}/", block_i);
        let block = read_mmap_block(&mut archive, &mmap, &prefix, None)?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;

    // Load info.json, if it exists
    let info_path = String::from("info.json");
    if archive.file_names().any(|name| name == info_path) {
        let mut info_file = archive.by_name(&info_path).map_err(|e| (info_path, e))?;
        let mut info = String::new();
        info_file.read_to_string(&mut info)?;
        let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
        let info = info.as_object().ok_or_else(|| {
            Error::Serialization("'info.json' should contain an object".into())
        })?;

        for (key, value) in info.iter() {
            let value = value.as_str().ok_or_else(|| {
                Error::Serialization("values in 'info.json' should be strings".into())
            })?;
            tensor.add_info(
                key,
                ConstCString::new(
                    CString::new(value).expect("value in 'info.json' should not contain a NUL byte"),
                ),
            );
        }
    }

    Ok(tensor)
}

/// Load a `TensorBlock` from the file at the given path using memory mapping.
pub fn load_block_mmap(path: &str) -> Result<TensorBlock, Error> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, &mmap, "", None)
}

fn read_mmap_block(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
) -> Result<TensorBlock, Error> {
    // Load values array via mmap
    let values_path = format!("{}values.npy", prefix);
    let (data, shape) = read_mmap_data(archive, mmap, &values_path)?;

    // Load labels normally
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let props_path = format!("{}properties.npy", prefix);
        let props_file = archive.by_name(&props_path).map_err(|e| (props_path, e))?;
        Arc::new(load_labels(props_file)?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    // Find and load gradients
    let mut parameters = HashSet::new();
    let gradient_prefix = format!("{}gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter.split('/').next().expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    for parameter in &parameters {
        let gradient = read_mmap_block(
            archive,
            mmap,
            &format!("{}gradients/{}/", prefix, parameter),
            Some(properties.clone()),
        )?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}

fn read_mmap_data(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    path: &str,
) -> Result<(mts_array_t, Vec<usize>), Error> {
    // Get the entry to find its data offset and verify it's STORED
    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression method {:?}, but mmap loading requires STORED (uncompressed) entries",
            path, entry.compression()
        )));
    }

    let entry_size = entry.size() as usize;
    let data_start = entry.data_start() as usize;

    // Drop the entry to release the borrow on archive
    drop(entry);

    if data_start + entry_size > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file", path
        )));
    }

    // Parse the NPY header from the mmap slice
    let npy_bytes = &mmap[data_start..data_start + entry_size];
    let (header, npy_header_len) = Header::from_slice(npy_bytes)?;

    if header.fortran_order {
        return Err(Error::Serialization(
            "data can not be loaded from fortran-order arrays".into(),
        ));
    }

    let descr = if let DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization(
            "structured arrays are not supported for mmap loading".into(),
        ));
    };

    let (code, bits, endian) = npy_descr_to_dtype(descr)?;

    // Verify endianness matches native
    match endian {
        Endianness::Native => {}
        Endianness::Little => {
            if cfg!(target_endian = "big") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has little-endian data on a big-endian system".into(),
                ));
            }
        }
        Endianness::Big => {
            if cfg!(target_endian = "little") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has big-endian data on a little-endian system".into(),
                ));
            }
        }
    }

    let shape = header.shape;
    let num_elements: usize = shape.iter().product();
    let element_bytes = (bits as usize / 8) * 1; // lanes = 1
    let raw_data_offset = data_start + npy_header_len;
    let data_len = num_elements * element_bytes;

    if raw_data_offset + data_len > mmap.len() {
        return Err(Error::Serialization(format!(
            "NPY data in '{}' extends beyond the end of the file", path
        )));
    }

    let dl_dtype = DLDataType {
        code,
        bits,
        lanes: 1,
    };

    let array = MmapArray::new(
        Arc::clone(mmap),
        raw_data_offset,
        data_len,
        shape.clone(),
        dl_dtype,
    );

    Ok((array.into_mts_array(), shape))
}

// ============================================================================
// Partial mmap loading
// ============================================================================

/// Load a `TensorMap` from the file at the given path using memory mapping,
/// selecting only a subset of the data based on keys, samples, and properties.
///
/// This function memory-maps the file, reads the metadata, applies the
/// selections, then copies only the matching data into new owned arrays.
///
/// - `keys`: if `Some`, only blocks whose key matches the selection are loaded.
/// - `samples`: if `Some`, only rows matching the selection are kept in each
///   block.
/// - `properties`: if `Some`, only columns matching the selection are kept in
///   each block.
///
/// Each selection `Labels` uses the standard `Labels::select` semantics: the
/// selection's names must be a subset of the target's names, and all matching
/// entries are included.
pub fn load_mmap_partial(
    path: &str,
    keys: Option<&Labels>,
    samples: Option<&Labels>,
    properties: Option<&Labels>,
) -> Result<TensorMap, Error> {
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }
        .map_err(|e| Error::Io(e))?;
    let mmap = Arc::new(mmap);

    let cursor = std::io::Cursor::new(mmap.as_ref().as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    // Load all keys from the file
    let path_str = String::from("keys.npy");
    let all_keys = load_labels(archive.by_name(&path_str).map_err(|e| (path_str, e))?)?;

    // Determine which block indices to load
    let block_indices = if let Some(key_sel) = keys {
        let mut selected = vec![0i64; all_keys.count()];
        let n_selected = all_keys.select(key_sel, &mut selected)?;
        selected[..n_selected].iter().map(|&i| i as usize).collect::<Vec<_>>()
    } else {
        (0..all_keys.count()).collect()
    };

    // Build filtered keys
    let filtered_keys = labels_subset(&all_keys, &block_indices)?;

    let mut blocks = Vec::with_capacity(block_indices.len());
    for &block_i in &block_indices {
        let prefix = format!("blocks/{}/", block_i);
        let block = read_mmap_partial_block(
            &mut archive,
            &mmap,
            &prefix,
            samples,
            properties,
            None,
        )?;
        blocks.push(block);
    }

    let mut tensor = TensorMap::new(Arc::new(filtered_keys), blocks)?;

    // Load info.json, if it exists
    let info_path = String::from("info.json");
    if archive.file_names().any(|name| name == info_path) {
        let mut info_file = archive.by_name(&info_path).map_err(|e| (info_path, e))?;
        let mut info = String::new();
        info_file.read_to_string(&mut info)?;
        let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
        let info = info.as_object().ok_or_else(|| {
            Error::Serialization("'info.json' should contain an object".into())
        })?;

        for (key, value) in info.iter() {
            let value = value.as_str().ok_or_else(|| {
                Error::Serialization("values in 'info.json' should be strings".into())
            })?;
            tensor.add_info(
                key,
                ConstCString::new(
                    CString::new(value).expect("value in 'info.json' should not contain a NUL byte"),
                ),
            );
        }
    }

    Ok(tensor)
}

/// Build a new `Labels` containing only the rows at the given indices.
fn labels_subset(labels: &Labels, indices: &[usize]) -> Result<Labels, Error> {
    let names_owned = labels.names();
    let names: Vec<&str> = names_owned.iter().map(|s| &**s).collect();
    if names.is_empty() {
        return Labels::new(&names, Vec::new());
    }
    let mut values = Vec::with_capacity(indices.len() * names.len());
    for &idx in indices {
        values.extend_from_slice(&labels[idx]);
    }
    // The indices come from select() or sequential iteration, which
    // guarantees uniqueness since the source Labels are unique.
    unsafe { Labels::new_unchecked_uniqueness(&names, values) }
}

/// Parse the NPY header from a ZIP entry without reading the data, returning
/// the DLDataType, shape, element size in bytes, and the byte offset of the
/// raw data within the mmap.
fn parse_npy_metadata(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Mmap,
    path: &str,
) -> Result<(DLDataType, Vec<usize>, usize, usize), Error> {
    let entry = archive.by_name(path).map_err(|e| (path.to_string(), e))?;

    if entry.compression() != zip::CompressionMethod::Stored {
        return Err(Error::Serialization(format!(
            "entry '{}' uses compression method {:?}, but mmap loading requires STORED (uncompressed) entries",
            path, entry.compression()
        )));
    }

    let entry_size = entry.size() as usize;
    let data_start = entry.data_start() as usize;
    drop(entry);

    if data_start + entry_size > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file", path
        )));
    }

    let npy_bytes = &mmap[data_start..data_start + entry_size];
    let (header, npy_header_len) = Header::from_slice(npy_bytes)?;

    if header.fortran_order {
        return Err(Error::Serialization(
            "data can not be loaded from fortran-order arrays".into(),
        ));
    }

    let descr = if let DataType::Scalar(s) = &header.type_descriptor {
        s.as_str()
    } else {
        return Err(Error::Serialization(
            "structured arrays are not supported for mmap loading".into(),
        ));
    };

    let (code, bits, endian) = npy_descr_to_dtype(descr)?;

    match endian {
        Endianness::Native => {}
        Endianness::Little => {
            if cfg!(target_endian = "big") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has little-endian data on a big-endian system".into(),
                ));
            }
        }
        Endianness::Big => {
            if cfg!(target_endian = "little") {
                return Err(Error::Serialization(
                    "mmap loading requires native endianness; file has big-endian data on a little-endian system".into(),
                ));
            }
        }
    }

    let dl_dtype = DLDataType { code, bits, lanes: 1 };
    let elem_bytes = (bits as usize / 8) * 1;
    let raw_data_offset = data_start + npy_header_len;

    Ok((dl_dtype, header.shape, elem_bytes, raw_data_offset))
}

/// Copy selected rows and columns from source mmap data into a new owned array.
///
/// `src_shape` is the full shape `[n_samples, comp..., n_properties]`.
/// `sample_indices` maps new_row → old_row.
/// `prop_indices` maps new_col → old_col (or `None` to select all properties).
fn gather_selected_data(
    mmap: &Mmap,
    raw_data_offset: usize,
    dl_dtype: DLDataType,
    src_shape: &[usize],
    sample_indices: &[usize],
    prop_indices: Option<&[usize]>,
) -> Result<mts_array_t, Error> {
    let elem = element_size(&dl_dtype);

    let src_n_props = *src_shape.last().unwrap();
    let n_components: usize = if src_shape.len() > 2 {
        src_shape[1..src_shape.len() - 1].iter().product()
    } else {
        1
    };
    let src_row_bytes = n_components * src_n_props * elem;

    let new_n_samples = sample_indices.len();
    let new_n_props = prop_indices.map_or(src_n_props, |p| p.len());

    // Build new shape: [new_n_samples, comp..., new_n_props]
    let mut new_shape = Vec::with_capacity(src_shape.len());
    new_shape.push(new_n_samples);
    if src_shape.len() > 2 {
        new_shape.extend_from_slice(&src_shape[1..src_shape.len() - 1]);
    }
    new_shape.push(new_n_props);

    let mut dst_array = new_created_array(&new_shape, dl_dtype);

    if new_n_samples == 0 || new_n_props == 0 {
        return Ok(dst_array.into_mts_array());
    }

    // Get mutable access to the destination bytes
    let dst_data = Arc::make_mut(&mut dst_array.data);
    let dst_bytes = dst_data.as_mut_slice();

    let src_base = &mmap[raw_data_offset..];

    if let Some(prop_idx) = prop_indices {
        // Selective property copying
        let dst_row_bytes = n_components * new_n_props * elem;
        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_row_off = old_row * src_row_bytes;
            let dst_row_off = new_row * dst_row_bytes;
            for comp in 0..n_components {
                let src_comp_off = src_row_off + comp * src_n_props * elem;
                let dst_comp_off = dst_row_off + comp * new_n_props * elem;
                for (new_col, &old_col) in prop_idx.iter().enumerate() {
                    let src_off = src_comp_off + old_col * elem;
                    let dst_off = dst_comp_off + new_col * elem;
                    dst_bytes[dst_off..dst_off + elem]
                        .copy_from_slice(&src_base[src_off..src_off + elem]);
                }
            }
        }
    } else {
        // All properties selected — copy full rows (contiguous)
        let row_bytes = n_components * src_n_props * elem;
        for (new_row, &old_row) in sample_indices.iter().enumerate() {
            let src_off = old_row * row_bytes;
            let dst_off = new_row * row_bytes;
            dst_bytes[dst_off..dst_off + row_bytes]
                .copy_from_slice(&src_base[src_off..src_off + row_bytes]);
        }
    }

    Ok(dst_array.into_mts_array())
}

/// Read a single block from the archive with partial sample/property selection.
fn read_mmap_partial_block(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    samples_sel: Option<&Labels>,
    properties_sel: Option<&Labels>,
    parent_properties: Option<(Arc<Labels>, Option<&[usize]>)>,
) -> Result<TensorBlock, Error> {
    // Load labels
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let all_samples = load_labels(samples_file)?;

    // Parse data header (don't read data yet)
    let values_path = format!("{}values.npy", prefix);
    let (dl_dtype, src_shape, _elem_bytes, raw_data_offset) =
        parse_npy_metadata(archive, mmap, &values_path)?;

    // Determine sample selection
    let sample_indices = if let Some(sel) = samples_sel {
        let mut selected = vec![0i64; all_samples.count()];
        let n = all_samples.select(sel, &mut selected)?;
        selected[..n].iter().map(|&i| i as usize).collect::<Vec<_>>()
    } else {
        (0..all_samples.count()).collect()
    };

    // Determine property selection
    let (new_properties, prop_indices): (Arc<Labels>, Option<Vec<usize>>) =
        if let Some((ref parent_props, ref parent_prop_idx)) = parent_properties {
            // Gradient: inherit property selection from parent
            (parent_props.clone(), parent_prop_idx.map(|p| p.to_vec()))
        } else {
            let props_path = format!("{}properties.npy", prefix);
            let props_file = archive.by_name(&props_path).map_err(|e| (props_path, e))?;
            let all_props = load_labels(props_file)?;

            if let Some(sel) = properties_sel {
                let mut selected = vec![0i64; all_props.count()];
                let n = all_props.select(sel, &mut selected)?;
                let idx: Vec<usize> = selected[..n].iter().map(|&i| i as usize).collect();
                let filtered = labels_subset(&all_props, &idx)?;
                (Arc::new(filtered), Some(idx))
            } else {
                (Arc::new(all_props), None)
            }
        };

    // Load components
    let mut components = Vec::new();
    for i in 0..(src_shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    // Gather selected data
    let data = gather_selected_data(
        mmap,
        raw_data_offset,
        dl_dtype,
        &src_shape,
        &sample_indices,
        prop_indices.as_deref(),
    )?;

    // Build filtered labels
    let new_samples = Arc::new(labels_subset(&all_samples, &sample_indices)?);

    let mut block = TensorBlock::new(data, new_samples, components, new_properties.clone())?;

    // Handle gradients (only for values blocks, not gradient blocks themselves)
    if parent_properties.is_none() {
        let mut parameters = HashSet::new();
        let gradient_prefix = format!("{}gradients/", prefix);
        for name in archive.file_names() {
            if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
                let (_, parameter) = name.split_at(gradient_prefix.len());
                let parameter = parameter.split('/').next()
                    .expect("could not find gradient parameter");
                parameters.insert(parameter.to_string());
            }
        }

        for parameter in &parameters {
            let grad_prefix = format!("{}gradients/{}/", prefix, parameter);
            let gradient = read_mmap_partial_gradient(
                archive,
                mmap,
                &grad_prefix,
                &sample_indices,
                &new_properties,
                prop_indices.as_deref(),
            )?;
            block.add_gradient(parameter, gradient)?;
        }
    }

    Ok(block)
}

/// Read a gradient block with correct sample reindexing and property filtering.
///
/// Gradient samples have `["sample", ...]` where column 0 indexes parent
/// samples. We build a map from old parent sample index to new sequential
/// index, filter gradient samples where `entry[0]` is in the map, and
/// replace `entry[0]` with the new index.
fn read_mmap_partial_gradient(
    archive: &mut ZipArchive<std::io::Cursor<&[u8]>>,
    mmap: &Arc<Mmap>,
    prefix: &str,
    parent_sample_indices: &[usize],
    new_properties: &Arc<Labels>,
    prop_indices: Option<&[usize]>,
) -> Result<TensorBlock, Error> {
    // Build old→new parent sample index map
    let mut parent_map: HashMap<i32, i32> = HashMap::new();
    for (new_idx, &old_idx) in parent_sample_indices.iter().enumerate() {
        parent_map.insert(old_idx as i32, new_idx as i32);
    }

    // Load gradient samples
    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let grad_samples = load_labels(samples_file)?;

    // Parse gradient data header
    let values_path = format!("{}values.npy", prefix);
    let (dl_dtype, src_shape, _elem_bytes, raw_data_offset) =
        parse_npy_metadata(archive, mmap, &values_path)?;

    // Load components
    let mut components = Vec::new();
    for i in 0..(src_shape.len() - 2) {
        let comp_path = format!("{}components/{}.npy", prefix, i);
        let comp_file = archive.by_name(&comp_path).map_err(|e| (comp_path, e))?;
        components.push(Arc::new(load_labels(comp_file)?));
    }

    // Filter gradient samples: keep only those whose sample[0] (parent sample)
    // is in our selection, and build new sample values with reindexed sample[0]
    let grad_names = grad_samples.names();
    let grad_names_ref: Vec<&str> = grad_names.iter().map(|s| &**s).collect();
    let mut kept_grad_indices = Vec::new();
    let mut new_grad_values = Vec::new();

    for (i, entry) in grad_samples.iter().enumerate() {
        let parent_sample_idx = entry[0].i32();
        if let Some(&new_parent_idx) = parent_map.get(&parent_sample_idx) {
            kept_grad_indices.push(i);
            // First value is the reindexed parent sample
            new_grad_values.push(LabelValue::from(new_parent_idx));
            // Rest of the gradient sample dimensions stay as-is
            for &val in &entry[1..] {
                new_grad_values.push(val);
            }
        }
    }

    let new_grad_samples = Arc::new(
        unsafe { Labels::new_unchecked_uniqueness(&grad_names_ref, new_grad_values)? }
    );

    // Gather selected gradient data
    let data = gather_selected_data(
        mmap,
        raw_data_offset,
        dl_dtype,
        &src_shape,
        &kept_grad_indices,
        prop_indices,
    )?;

    TensorBlock::new(data, new_grad_samples, components, new_properties.clone())
}
