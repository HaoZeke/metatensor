use std::io::BufReader;
use std::collections::HashSet;
use std::sync::Arc;

use byteorder::{LittleEndian, BigEndian, NativeEndian, WriteBytesExt, ReadBytesExt};
use zip::{ZipArchive, ZipWriter};
use dlpk::sys::{DLDataTypeCode, DLDevice, DLPackVersion};

use super::npy_header::{Header, DataType};
use super::{check_for_extra_bytes, PathOrBuffer};
use super::labels::{load_labels, save_labels};

use crate::{TensorBlock, Labels, Error, mts_array_t};

enum Endianness {
    Little,
    Big,
}

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

fn npy_descr_to_dtype(descr: &str) -> Result<(DLDataTypeCode, u8, Endianness), Error> {
    if descr.len() < 3 {
        return Err(Error::Serialization(format!("invalid type descriptor: {}", descr)));
    }

    let endian = match &descr[0..1] {
        "<" | "|" => Endianness::Little,
        ">" => Endianness::Big,
        _ => return Err(Error::Serialization(format!("unknown endianness in type descriptor: {}", descr))),
    };

    let type_char = &descr[1..2];
    let size_str = &descr[2..];
    let size: u8 = size_str.parse().map_err(|_| {
        Error::Serialization(format!("invalid size in type descriptor: {}", descr))
    })?;

    let (code, bits) = match (type_char, size) {
        ("f", 4) => (DLDataTypeCode::kDLFloat, 32),
        ("f", 8) => (DLDataTypeCode::kDLFloat, 64),
        ("i", 1) => (DLDataTypeCode::kDLInt, 8),
        ("i", 2) => (DLDataTypeCode::kDLInt, 16),
        ("i", 4) => (DLDataTypeCode::kDLInt, 32),
        ("i", 8) => (DLDataTypeCode::kDLInt, 64),
        ("u", 1) => (DLDataTypeCode::kDLUInt, 8),
        ("u", 2) => (DLDataTypeCode::kDLUInt, 16),
        ("u", 4) => (DLDataTypeCode::kDLUInt, 32),
        ("u", 8) => (DLDataTypeCode::kDLUInt, 64),
        ("b", 1) => (DLDataTypeCode::kDLBool, 8),
        ("c", 8) => (DLDataTypeCode::kDLComplex, 64),
        ("c", 16) => (DLDataTypeCode::kDLComplex, 128),
        ("f", 2) => (DLDataTypeCode::kDLFloat, 16),
        _ => return Err(Error::Serialization(format!("unsupported type descriptor: {}", descr))),
    };

    Ok((code, bits, endian))
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
    let array = create_array(shape.clone())?;

    let num_elements: usize = shape.iter().product();
    if num_elements == 0 {
        check_for_extra_bytes(&mut reader)?;
        return Ok((array, shape));
    }

    let device = DLDevice::cpu();
    let version = DLPackVersion::current();
    let dl_tensor = array.as_dlpack(device, None, version)?;

    let descr = match &header.type_descriptor {
        DataType::Scalar(s) => s.as_str(),
        _ => return Err(Error::Serialization("structured arrays are not supported".into())),
    };

    let (file_code, file_bits, endian) = npy_descr_to_dtype(descr)?;

    let tensor_ref = dl_tensor.as_ref();
    // NOTE: don't check dtype, since we have non-standard casting (e.g. Bool -> u8)
    // instead, handle errors in the macro

    macro_rules! read_as {
        ($ty:ty, $reader:expr, $tensor:expr, $read_call:expr) => {{
            // Convert shape from DLPack (i64) to ndarray (usize)
            let shape: Vec<usize> = $tensor.shape()
                .iter()
                .map(|&x| x as usize)
                .collect();

            // Unsafe Cast
            // SAFETY: We trust the caller (match block) to match the byte size correctly.
            let mut view = unsafe {
                let data_ptr = $tensor.raw.data as *mut $ty;
                ndarray::ArrayViewMutD::from_shape_ptr(shape, data_ptr)
            };

            for value in view.iter_mut() {
                // EXPLICIT ERROR HANDLING
                match $read_call($reader) {
                    Ok(v) => *value = v,
                    Err(e) => {
                        let err = <crate::Error as From<std::io::Error>>::from(e);
                        return Err(err);
                    }
                }
            }
            Ok::<(), crate::Error>(())
        }}
    }

    // Explicitly annotating the closure arg |r: &mut R| to fix E0282
    match (file_code, file_bits, endian) {
        // Standard Floats
        (DLDataTypeCode::kDLFloat, 32, Endianness::Little) => read_as!(f32, &mut reader, tensor_ref, |r: &mut R| r.read_f32::<LittleEndian>()),
        (DLDataTypeCode::kDLFloat, 32, Endianness::Big) => read_as!(f32, &mut reader, tensor_ref, |r: &mut R| r.read_f32::<BigEndian>()),
        (DLDataTypeCode::kDLFloat, 64, Endianness::Little) => read_as!(f64, &mut reader, tensor_ref, |r: &mut R| r.read_f64::<LittleEndian>()),
        (DLDataTypeCode::kDLFloat, 64, Endianness::Big) => read_as!(f64, &mut reader, tensor_ref, |r: &mut R| r.read_f64::<BigEndian>()),

        // Standard Ints
        (DLDataTypeCode::kDLInt, 8, _) => read_as!(i8, &mut reader, tensor_ref, |r: &mut R| r.read_i8()),
        (DLDataTypeCode::kDLInt, 16, Endianness::Little) => read_as!(i16, &mut reader, tensor_ref, |r: &mut R| r.read_i16::<LittleEndian>()),
        (DLDataTypeCode::kDLInt, 16, Endianness::Big) => read_as!(i16, &mut reader, tensor_ref, |r: &mut R| r.read_i16::<BigEndian>()),
        (DLDataTypeCode::kDLInt, 32, Endianness::Little) => read_as!(i32, &mut reader, tensor_ref, |r: &mut R| r.read_i32::<LittleEndian>()),
        (DLDataTypeCode::kDLInt, 32, Endianness::Big) => read_as!(i32, &mut reader, tensor_ref, |r: &mut R| r.read_i32::<BigEndian>()),
        (DLDataTypeCode::kDLInt, 64, Endianness::Little) => read_as!(i64, &mut reader, tensor_ref, |r: &mut R| r.read_i64::<LittleEndian>()),
        (DLDataTypeCode::kDLInt, 64, Endianness::Big) => read_as!(i64, &mut reader, tensor_ref, |r: &mut R| r.read_i64::<BigEndian>()),

        // Unsigned Ints
        (DLDataTypeCode::kDLUInt, 8, _) => read_as!(u8, &mut reader, tensor_ref, |r: &mut R| r.read_u8()),
        (DLDataTypeCode::kDLUInt, 16, Endianness::Little) => read_as!(u16, &mut reader, tensor_ref, |r: &mut R| r.read_u16::<LittleEndian>()),
        (DLDataTypeCode::kDLUInt, 16, Endianness::Big) => read_as!(u16, &mut reader, tensor_ref, |r: &mut R| r.read_u16::<BigEndian>()),
        (DLDataTypeCode::kDLUInt, 32, Endianness::Little) => read_as!(u32, &mut reader, tensor_ref, |r: &mut R| r.read_u32::<LittleEndian>()),
        (DLDataTypeCode::kDLUInt, 32, Endianness::Big) => read_as!(u32, &mut reader, tensor_ref, |r: &mut R| r.read_u32::<BigEndian>()),
        (DLDataTypeCode::kDLUInt, 64, Endianness::Little) => read_as!(u64, &mut reader, tensor_ref, |r: &mut R| r.read_u64::<LittleEndian>()),
        (DLDataTypeCode::kDLUInt, 64, Endianness::Big) => read_as!(u64, &mut reader, tensor_ref, |r: &mut R| r.read_u64::<BigEndian>()),

        // Boolean (Read as u8)
        (DLDataTypeCode::kDLBool, 8, _) => read_as!(u8, &mut reader, tensor_ref, |r: &mut R| r.read_u8()),

        // Complex Numbers (Read as array of 2 floats)
        (DLDataTypeCode::kDLComplex, 64, Endianness::Little) => read_as!([f32; 2], &mut reader, tensor_ref, |r: &mut R| Ok([r.read_f32::<LittleEndian>()?, r.read_f32::<LittleEndian>()?])),
        (DLDataTypeCode::kDLComplex, 64, Endianness::Big) => read_as!([f32; 2], &mut reader, tensor_ref, |r: &mut R| Ok([r.read_f32::<BigEndian>()?, r.read_f32::<BigEndian>()?])),
        (DLDataTypeCode::kDLComplex, 128, Endianness::Little) => read_as!([f64; 2], &mut reader, tensor_ref, |r: &mut R| Ok([r.read_f64::<LittleEndian>()?, r.read_f64::<LittleEndian>()?])),
        (DLDataTypeCode::kDLComplex, 128, Endianness::Big) => read_as!([f64; 2], &mut reader, tensor_ref, |r: &mut R| Ok([r.read_f64::<BigEndian>()?, r.read_f64::<BigEndian>()?])),

        // Float16 (Read as u16)
        (DLDataTypeCode::kDLFloat, 16, Endianness::Little) => read_as!(u16, &mut reader, tensor_ref, |r: &mut R| r.read_u16::<LittleEndian>()),
        (DLDataTypeCode::kDLFloat, 16, Endianness::Big) => read_as!(u16, &mut reader, tensor_ref, |r: &mut R| r.read_u16::<BigEndian>()),

        _ => Err(Error::Serialization(format!(
            "unsupported dtype for reading: {:?} {} bits", file_code, file_bits
        ))),
    }?; // match arms return Result<(), Error>

    check_for_extra_bytes(&mut reader)?;
    Ok((array, shape))
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
        (DLDataTypeCode::kDLBool, 8) => ("b", 1),
        (DLDataTypeCode::kDLComplex, 64) => ("c", 8),
        (DLDataTypeCode::kDLComplex, 128) => ("c", 16),
        (DLDataTypeCode::kDLFloat, 16) => ("f", 2),
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

    macro_rules! write_as {
        ($ty:ty, $writer:expr, $tensor:expr, $write_call:expr) => {{
            let shape: Vec<usize> = $tensor.shape()
                .iter()
                .map(|&x| x as usize)
                .collect();

            // Forcefully cast for Bool -> u8
            let view = unsafe {
                let data_ptr = $tensor.raw.data as *const $ty;
                ndarray::ArrayViewD::from_shape_ptr(shape, data_ptr)
            };

            for &val in view.iter() {
                // Explicit match to handle the error without relying on '?' inference
                match $write_call($writer, val) {
                    Ok(_) => {},
                    Err(e) => {
                        let err = <crate::Error as From<std::io::Error>>::from(e);
                        return Err(err);
                    }
                }
            }
            // Explicitly hint the return type to the compiler
            Ok::<(), crate::Error>(())
        }}
    }

    match (code, bits) {
        (DLDataTypeCode::kDLFloat, 32) => write_as!(f32, writer, tensor_ref, |w: &mut W, v| w.write_f32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLFloat, 64) => write_as!(f64, writer, tensor_ref, |w: &mut W, v| w.write_f64::<NativeEndian>(v)),
        
        (DLDataTypeCode::kDLInt, 8) => write_as!(i8, writer, tensor_ref, |w: &mut W, v| w.write_i8(v)),
        (DLDataTypeCode::kDLInt, 16) => write_as!(i16, writer, tensor_ref, |w: &mut W, v| w.write_i16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 32) => write_as!(i32, writer, tensor_ref, |w: &mut W, v| w.write_i32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLInt, 64) => write_as!(i64, writer, tensor_ref, |w: &mut W, v| w.write_i64::<NativeEndian>(v)),
        
        (DLDataTypeCode::kDLUInt, 8) => write_as!(u8, writer, tensor_ref, |w: &mut W, v| w.write_u8(v)),
        (DLDataTypeCode::kDLUInt, 16) => write_as!(u16, writer, tensor_ref, |w: &mut W, v| w.write_u16::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 32) => write_as!(u32, writer, tensor_ref, |w: &mut W, v| w.write_u32::<NativeEndian>(v)),
        (DLDataTypeCode::kDLUInt, 64) => write_as!(u64, writer, tensor_ref, |w: &mut W, v| w.write_u64::<NativeEndian>(v)),

        (DLDataTypeCode::kDLBool, 8) => write_as!(u8, writer, tensor_ref, |w: &mut W, v| w.write_u8(v)),
        
        (DLDataTypeCode::kDLComplex, 64) => write_as!([f32; 2], writer, tensor_ref, |w: &mut W, v: [f32; 2]| { 
            w.write_f32::<NativeEndian>(v[0])?; 
            w.write_f32::<NativeEndian>(v[1]) 
        }),
        (DLDataTypeCode::kDLComplex, 128) => write_as!([f64; 2], writer, tensor_ref, |w: &mut W, v: [f64; 2]| { 
            w.write_f64::<NativeEndian>(v[0])?; 
            w.write_f64::<NativeEndian>(v[1]) 
        }),
        
        (DLDataTypeCode::kDLFloat, 16) => write_as!(u16, writer, tensor_ref, |w: &mut W, v| w.write_u16::<NativeEndian>(v)),

        _ => Err(Error::Serialization(format!("unsupported dtype: {:?} {}", code, bits))),
    }
}
