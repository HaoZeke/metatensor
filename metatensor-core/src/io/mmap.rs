//! Memory-mapped loading of `TensorBlock` / `TensorMap` from `.mts` files.
//!
//! Parses NPY headers via mmap, then dispatches each array to a
//! caller-supplied `create_array` callback with `(shape, dtype, file_offset)`.
//! The caller decides how to materialise the array: mmap-backed view, plain
//! read, GPU Direct Storage upload, etc.
//!
//! Requirements on the input file:
//! - STORED (uncompressed) ZIP entries (this is what `mts_*_save` writes).
//! - Native byte order for all numeric arrays (mmap views must be directly
//!   reinterpretable).

use std::collections::HashSet;
use std::ffi::CString;
use std::io::{Cursor, Read};
use std::sync::Arc;

use memmap2::Mmap;
use zip::ZipArchive;

use dlpk::sys::DLDataType;

use super::block::npy_descr_to_dtype;
use super::labels::load_labels;
use super::npy_header::{DataType, Header};
use super::Endianness;

use crate::utils::ConstCString;
use crate::{mts_array_t, Error, Labels, TensorBlock, TensorMap};


/// Metadata extracted by parsing the NPY header of an entry without copying
/// its data.
struct ArrayFileInfo {
    shape: Vec<usize>,
    dl_dtype: DLDataType,
    /// Byte offset of the raw array data within the mmap-ed file.
    file_offset: usize,
}


/// Parse an NPY entry's header (shape + dtype + file offset of raw data) by
/// reading bytes directly from the mmap-backed slice.
fn parse_npy_entry(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    mmap: &[u8],
    path: &str,
) -> Result<ArrayFileInfo, Error> {
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

    let end = data_start.checked_add(entry_size).ok_or_else(|| {
        Error::Serialization(format!("entry '{}' has overflowing size+offset", path))
    })?;
    if end > mmap.len() {
        return Err(Error::Serialization(format!(
            "entry '{}' extends beyond the end of the file",
            path
        )));
    }

    let npy_bytes = &mmap[data_start..end];

    let mut cursor = Cursor::new(npy_bytes);
    let header = Header::from_reader(&mut cursor).map_err(|e| {
        Error::Serialization(format!("invalid NPY header in '{}': {}", path, e))
    })?;

    if header.fortran_order {
        return Err(Error::Serialization(format!(
            "fortran-order arrays are not supported (in '{}')",
            path
        )));
    }

    let descr = match &header.type_descriptor {
        DataType::Scalar(s) => s.as_str(),
        _ => {
            return Err(Error::Serialization(format!(
                "structured arrays are not supported (in '{}')",
                path
            )))
        }
    };

    let (code, bits, endian) = npy_descr_to_dtype(descr)?;

    let native_ok = match endian {
        Endianness::Native => true,
        Endianness::Little => cfg!(target_endian = "little"),
        Endianness::Big => cfg!(target_endian = "big"),
    };
    if !native_ok {
        return Err(Error::Serialization(format!(
            "mmap loading requires native byte order, but entry '{}' uses '{}'",
            path, descr
        )));
    }

    let header_len = cursor.position() as usize;
    let file_offset = data_start + header_len;

    Ok(ArrayFileInfo {
        shape: header.shape,
        dl_dtype: DLDataType { code, bits, lanes: 1 },
        file_offset,
    })
}


/// Load a `TensorMap` from the file at the given path using memory mapping.
///
/// The implementation memory-maps the file, parses NPY headers to discover
/// array shapes, dtypes, and byte offsets, and then dispatches each array
/// (values + gradient values) to `create_array(shape, dtype, file_offset)`.
///
/// `create_array` decides how to materialise the `mts_array_t`: it can wrap
/// the mmap-ed bytes as a zero-copy view, copy them into an owned buffer,
/// or stream them to a GPU via GPU Direct Storage. The byte length is
/// `shape.iter().product() * (dtype.bits / 8) * dtype.lanes`.
///
/// Labels are loaded normally (decompressed into owned `Labels` values).
///
/// # File format constraints
/// - The file must use `STORED` (uncompressed) ZIP entries.
/// - Numeric arrays must use native byte order.
pub fn load_mmap<F>(path: &str, create_array: F) -> Result<TensorMap, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    // SAFETY: we treat the mmap as a read-only view; the underlying file is
    // owned by `file` for the duration of this call.
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    let keys_path = String::from("keys.npy");
    let keys = load_labels(archive.by_name(&keys_path).map_err(|e| (keys_path, e))?)?;

    let mut blocks = Vec::with_capacity(keys.count());
    for block_i in 0..keys.count() {
        let prefix = format!("blocks/{}/", block_i);
        blocks.push(read_mmap_block(
            &mut archive,
            mmap.as_ref(),
            &prefix,
            None,
            &create_array,
        )?);
    }

    let mut tensor = TensorMap::new(Arc::new(keys), blocks)?;

    let info_path = String::from("info.json");
    if archive.file_names().any(|name| name == info_path) {
        let mut info_file = archive.by_name(&info_path).map_err(|e| (info_path, e))?;
        let mut info = String::new();
        info_file.read_to_string(&mut info)?;
        let info = jzon::parse(&info).map_err(|e| Error::Serialization(e.to_string()))?;
        let info = info
            .as_object()
            .ok_or_else(|| Error::Serialization("'info.json' should contain an object".into()))?;

        for (key, value) in info.iter() {
            let value = value
                .as_str()
                .ok_or_else(|| Error::Serialization("values in 'info.json' should be strings".into()))?;
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


/// Load a single `TensorBlock` from the file at the given path using memory
/// mapping. See [`load_mmap`] for callback semantics and file format
/// constraints.
pub fn load_block_mmap<F>(path: &str, create_array: F) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let file = std::fs::File::open(path)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(Error::Io)?;

    let cursor = Cursor::new(mmap.as_ref());
    let mut archive = ZipArchive::new(cursor).map_err(|e| ("<root>".into(), e))?;

    read_mmap_block(&mut archive, mmap.as_ref(), "", None, &create_array)
}


fn read_mmap_block<F>(
    archive: &mut ZipArchive<Cursor<&[u8]>>,
    mmap: &[u8],
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
where
    F: Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error>,
{
    let values_path = format!("{}values.npy", prefix);
    let info = parse_npy_entry(archive, mmap, &values_path)?;
    let shape_len = info.shape.len();
    let data = create_array(info.shape, info.dl_dtype, info.file_offset)?;

    let samples_path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&samples_path).map_err(|e| (samples_path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..shape_len.saturating_sub(2) {
        let path = format!("{}components/{}.npy", prefix, i);
        let component_file = archive.by_name(&path).map_err(|e| (path, e))?;
        components.push(Arc::new(load_labels(component_file)?));
    }

    let properties = if let Some(ref properties) = properties {
        properties.clone()
    } else {
        let path = format!("{}properties.npy", prefix);
        let properties_file = archive.by_name(&path).map_err(|e| (path, e))?;
        Arc::new(load_labels(properties_file)?)
    };

    let mut block = TensorBlock::new(data, samples, components, properties.clone())?;

    let mut parameters = HashSet::new();
    let gradient_prefix = format!("{}gradients/", prefix);
    for name in archive.file_names() {
        if name.starts_with(&gradient_prefix) && name.ends_with("/samples.npy") {
            let (_, parameter) = name.split_at(gradient_prefix.len());
            let parameter = parameter
                .split('/')
                .next()
                .expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    for parameter in &parameters {
        let gradient = read_mmap_block(
            archive,
            mmap,
            &format!("{}gradients/{}/", prefix, parameter),
            Some(properties.clone()),
            create_array,
        )?;
        block.add_gradient(parameter, gradient)?;
    }

    Ok(block)
}


#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::data::TestArray;

    const DATA_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data.mts");
    const BLOCK_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/block.mts");

    /// What metatensor handed back to the callback for one array.
    #[derive(Clone, Debug)]
    struct CallbackRecord {
        shape: Vec<usize>,
        dtype: DLDataType,
        file_offset: usize,
    }

    /// Build a callback that records the (shape, dtype, file_offset) it was
    /// invoked with and returns a metadata-only `TestArray`. Real bindings
    /// (numpy, torch, cuFile) would instead materialise an array at
    /// `file_offset`; the metadata-only flavour is enough to assert that
    /// metatensor is dispatching consistently.
    fn record_and_test_array(
        records: Arc<Mutex<Vec<CallbackRecord>>>,
    ) -> impl Fn(Vec<usize>, DLDataType, usize) -> Result<mts_array_t, Error> {
        move |shape, dtype, file_offset| {
            records.lock().unwrap().push(CallbackRecord {
                shape: shape.clone(),
                dtype,
                file_offset,
            });
            Ok(TestArray::new(shape))
        }
    }

    #[test]
    fn callback_metadata_is_consistent() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let cb = record_and_test_array(Arc::clone(&records));
        let _ = load_mmap(DATA_PATH, cb).expect("mmap load failed");

        let recs = records.lock().unwrap();
        assert!(!recs.is_empty(), "expected at least one callback invocation");
        for rec in recs.iter() {
            assert!(!rec.shape.is_empty(), "callback shape should be non-empty");
            assert!(rec.file_offset > 0, "file_offset should be inside the file");
            assert_eq!(rec.dtype.lanes, 1, "test fixture uses scalar dtype only");
        }
    }

    #[test]
    fn two_mmap_loads_are_structurally_identical() {
        // Two independent mmap loads of the same file must produce
        // identical key/sample/property/gradient structure. Data equality
        // is verified at the binding layer (Python/Torch) where real
        // arrays are materialised; here we only have metadata-only
        // TestArrays, which is enough for structural cross-checking.
        let records_a: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let a = load_mmap(DATA_PATH, record_and_test_array(Arc::clone(&records_a)))
            .expect("mmap load A failed");

        let records_b: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let b = load_mmap(DATA_PATH, record_and_test_array(Arc::clone(&records_b)))
            .expect("mmap load B failed");

        assert_eq!(a.keys().count(), b.keys().count());
        assert_eq!(a.keys().dimensions(), b.keys().dimensions());
        assert_eq!(records_a.lock().unwrap().len(), records_b.lock().unwrap().len());

        for (ba, bb) in a.blocks().iter().zip(b.blocks().iter()) {
            assert_eq!(ba.samples.count(), bb.samples.count());
            assert_eq!(ba.properties.count(), bb.properties.count());
            assert_eq!(ba.gradients().len(), bb.gradients().len());
        }
    }

    #[test]
    fn load_block_mmap_invokes_callback() {
        let records: Arc<Mutex<Vec<CallbackRecord>>> = Arc::new(Mutex::new(Vec::new()));
        let _block = load_block_mmap(BLOCK_PATH, record_and_test_array(Arc::clone(&records)))
            .expect("mmap block load failed");

        let count = records.lock().unwrap().len();
        assert!(count >= 1, "expected at least one callback invocation, got {}", count);
    }
}
