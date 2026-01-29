use std::io::BufReader;
use std::collections::HashSet;
use std::sync::Arc;

use byteorder::{LittleEndian, ReadBytesExt, BigEndian, WriteBytesExt};
use zip::{ZipArchive, ZipWriter};
use dlpk::sys::{DLDataTypeCode, DLDevice, DLDeviceType, DLPackVersion, DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
use dlpk::sys::DLPACK_FLAG_BITMASK_READ_ONLY;
use ndarray::ArrayViewD;
use std::convert::TryInto;

use super::npy_header::{Header, DataType};
use super::{check_for_extra_bytes, PathOrBuffer};
use super::labels::{load_labels, save_labels};

use crate::{TensorBlock, Labels, Error, mts_array_t};


/// Check if the file/buffer in `data` looks like it could contain serialized
/// `TensorBlock`.
pub fn looks_like_block_data(mut data: PathOrBuffer) -> bool {
    match data {
        PathOrBuffer::Path(path) => {
            match std::fs::File::open(path) {
                Ok(file) => {
                    let mut buffer = BufReader::new(file);
                    return looks_like_block_data(PathOrBuffer::Buffer(&mut buffer));
                },
                Err(_) => { return false; }
            }
        },
        PathOrBuffer::Buffer(ref mut buffer) => {
            match ZipArchive::new(buffer) {
                Ok(mut archive) => {
                    return archive.by_name("values.npy").is_ok()
                }
                Err(_) => { return false; }
            }
        },
    }
}

/// Load the serialized tensor block from the given path.
///
/// Arrays for the values and gradient data will be created with the given
/// `create_array` callback, and filled by this function with the corresponding
/// data.
///
/// See the [`load`] for more information about the format used to serialize
/// `TensorBlock`.
pub fn load_block<R, F>(reader: R, create_array: F) -> Result<TensorBlock, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let mut archive = ZipArchive::new(reader).map_err(|e| ("<root>".into(), e))?;

    return read_single_block(&mut archive, "", None, &create_array);
}

/// Save the given block to a file (or any other writer).
///
/// The format used is documented in the [`load`] function, and consists of a
/// zip archive containing NPY files. The recomended file extension when saving
/// data is `.mts`, to prevent confusion with generic `.npz` files.
pub fn save_block<W: std::io::Write + std::io::Seek>(writer: W, block: &TensorBlock) -> Result<(), Error> {
    let mut archive = ZipWriter::new(writer);
    write_single_block(&mut archive, "", true, block)?;
    archive.finish().map_err(|e| ("<root>".into(), e))?;

    return Ok(());
}


/******************************************************************************/

#[allow(clippy::needless_pass_by_value)]
pub(super) fn read_single_block<R, F>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    properties: Option<Arc<Labels>>,
    create_array: &F,
) -> Result<TensorBlock, Error>
    where R: std::io::Read + std::io::Seek,
          F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let path = format!("{}values.npy", prefix);
    let data_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let (data, shape) = read_data(data_file, &create_array)?;

    let path = format!("{}samples.npy", prefix);
    let samples_file = archive.by_name(&path).map_err(|e| (path, e))?;
    let samples = Arc::new(load_labels(samples_file)?);

    let mut components = Vec::new();
    for i in 0..(shape.len() - 2) {
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
            let parameter = parameter.split('/').next().expect("could not find gradient parameter");
            parameters.insert(parameter.to_string());
        }
    }

    for parameter in &parameters {
        let gradient = read_single_block(
            archive,
            &format!("{}gradients/{}/", prefix, parameter),
            Some(properties.clone()),
            create_array
        )?;

        block.add_gradient(parameter, gradient)?;
    }

    return Ok(block);
}

// Read a data array from the given reader, using numpy's NPY format
fn read_data<R, F>(mut reader: R, create_array: &F) -> Result<(mts_array_t, Vec<usize>), Error>
    where R: std::io::Read, F: Fn(Vec<usize>) -> Result<mts_array_t, Error>
{
    let header = Header::from_reader(&mut reader)?;
    if header.fortran_order {
        return Err(Error::Serialization("data can not be loaded from fortran-order arrays".into()));
    }

    let shape = header.shape;
    let mut array = create_array(shape.clone())?;

    match header.type_descriptor {
        DataType::Scalar(s) if s == "<f8" => {
            reader.read_f64_into::<LittleEndian>(array.data_mut()?)?;
        }
        DataType::Scalar(s) if s == ">f8" => {
            reader.read_f64_into::<BigEndian>(array.data_mut()?)?;
        }
        _ => {
            return Err(Error::Serialization(format!(
                "unknown type for data array, expected 64-bit floating points, got {}",
                header.type_descriptor
            )));
        }
    }

    check_for_extra_bytes(&mut reader)?;

    return Ok((array, shape));
}

pub(super) fn write_single_block<W: std::io::Write + std::io::Seek>(
    archive: &mut ZipWriter<W>,
    prefix: &str,
    values: bool,
    block: &TensorBlock,
) -> Result<(), Error> {
    let options = zip::write::FileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .large_file(true)
        .last_modified_time(zip::DateTime::from_date_and_time(2000, 1, 1, 0, 0, 0).expect("invalid datetime"));

    let path = format!("{}values.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    write_data(archive, &block.values)?;

    let path = format!("{}samples.npy", prefix);
    archive.start_file(&path, options).map_err(|e| (path, e))?;
    save_labels(archive, &block.samples)?;

    for (i, component) in block.components.iter().enumerate() {
        let path = format!("{}components/{}.npy", prefix, i);
        archive.start_file(&path, options).map_err(|e| (path, e))?;
        save_labels(archive, component)?;
    }

    if values {
        let path = format!("{}properties.npy", prefix);
        archive.start_file(&path, options).map_err(|e| (path.clone(), e))?;
        save_labels(archive, &block.properties)?;
    }

    for (parameter, gradient) in block.gradients() {
        let prefix = format!("{}gradients/{}/", prefix, parameter);
        write_single_block(archive, &prefix, false, gradient)?;
    }

    Ok(())
}

fn dlpack_to_npy_descr(code: DLDataTypeCode, bits: u8) -> Result<String, Error> {
    let endian = if cfg!(target_endian = "little") { "<" } else { ">" };

    let (type_char, type_size) = match (code, bits) {
        (DLDataTypeCode::kDLInt, 8) => ("i", 1),
        (DLDataTypeCode::kDLInt, 16) => ("i", 2),
        (DLDataTypeCode::kDLInt, 32) => ("i", 4),
        (DLDataTypeCode::kDLInt, 64) => ("i", 8),
        (DLDataTypeCode::kDLUInt, 8) => ("u", 1),
        (DLDataTypeCode::kDLUInt, 16) => ("u", 2),
        (DLDataTypeCode::kDLUInt, 32) => ("u", 4),
        (DLDataTypeCode::kDLUInt, 64) => ("u", 8),
        (DLDataTypeCode::kDLFloat, 32) => ("f", 4),
        (DLDataTypeCode::kDLFloat, 64) => ("f", 8),
        _ => return Err(Error::Serialization(
            format!("unsupported DLPack dtype: code {:?}, bits {:?}", code, bits)
                                            )
        ),
    };

    Ok(format!("{}{}{}", endian, type_char, type_size))
}

// Write an array to the given writer, using numpy's NPY format
fn write_data<W: std::io::Write>(writer: &mut W, array: &mts_array_t) -> Result<(), Error> {
    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    
    // Get DLPack Tensor
    let dl_tensor = array.as_dlpack(device, None, version)?;
    let tensor_ref = dl_tensor.as_ref();
    let dtype = tensor_ref.raw.dtype;
    let (code, bits) = (dtype.code, dtype.bits);

    // Validate Lanes
    if dtype.lanes != 1 {
        return Err(Error::Serialization(format!(
            "unsupported DLPack dtype: lanes != 1 ({})", dtype.lanes
        )));
    }

    // Write Header
    let tdesc = super::block::dlpack_to_npy_descr(code, bits)?;
    let header = Header {
        type_descriptor: DataType::Scalar(tdesc.into()),
        fortran_order: false,
        shape: array.shape()?.to_vec(),
    };

    header.write(&mut *writer)?;

    // Get metadata for size and pointer for data
    let num_elements: usize = header.shape.iter().product();
    if num_elements == 0 {
        return Ok(());
    }

    // Dispatch based on Dtype
    // Generic type inference; the closure dictates the type T.
    match (code, bits) {
        (DLDataTypeCode::kDLFloat, 32) => write_values(writer, tensor_ref, |w, v| w.write_f32::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLFloat, 64) => write_values(writer, tensor_ref, |w, v| w.write_f64::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLInt, 8) => write_values(writer, tensor_ref, |w, v| w.write_i8(v))?,
        (DLDataTypeCode::kDLInt, 16) => write_values(writer, tensor_ref, |w, v| w.write_i16::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLInt, 32) => write_values(writer, tensor_ref, |w, v| w.write_i32::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLInt, 64) => write_values(writer, tensor_ref, |w, v| w.write_i64::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLUInt, 8) => write_values(writer, tensor_ref, |w, v| w.write_u8(v))?,
        (DLDataTypeCode::kDLUInt, 16) => write_values(writer, tensor_ref, |w, v| w.write_u16::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLUInt, 32) => write_values(writer, tensor_ref, |w, v| w.write_u32::<LittleEndian>(v))?,
        (DLDataTypeCode::kDLUInt, 64) => write_values(writer, tensor_ref, |w, v| w.write_u64::<LittleEndian>(v))?,
        _ => return Err(Error::Serialization(format!("unsupported dtype: {:?} {}", code, bits))),
    }

    Ok(())
}

fn write_values<'a, W, T, F, Tensor>(writer: &mut W, tensor: Tensor, write_fn: F) -> Result<(), Error>
where
    W: std::io::Write,
    T: Copy + 'static,
    F: Fn(&mut W, T) -> std::io::Result<()>,
    Tensor: TryInto<ArrayViewD<'a, T>>,
    <Tensor as TryInto<ArrayViewD<'a, T>>>::Error: std::fmt::Debug,
{
    let view: ArrayViewD<T> = tensor
        .try_into()
        .map_err(|e| Error::Serialization(format!("DLPack conversion failed: {:?}", e)))?;

    for &val in view.iter() {
        write_fn(writer, val)?;
    }

    Ok(())
}
