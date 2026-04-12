use crate::loader::{CoreMLModelInfo, CoreMLModelLoader};
use crate::CoreMLError;
use flate2::Compression;
use std::io::{Read, Write};
use std::path::PathBuf;
use tempfile::NamedTempFile;

/// Decompresses a model buffer from disk.
pub fn decompress_buffer_from_disk(path: &PathBuf) -> Result<Vec<u8>, CoreMLError> {
    let file = std::fs::File::open(path)?;
    let mut vec = vec![];
    let mut decoder = flate2::read::ZlibDecoder::new(file);
    decoder.read_to_end(&mut vec)?;
    Ok(vec)
}

/// Compresses a model buffer and saves it to the specified cache directory.
pub fn compress_buffer_to_disk(
    vec: &[u8],
    cache_dir: &mut PathBuf,
) -> Result<PathBuf, CoreMLError> {
    if cache_dir.as_os_str().is_empty() {
        *cache_dir = PathBuf::from(".");
    }
    if !cache_dir.exists() {
        let _ = std::fs::remove_dir_all(&cache_dir);
        std::fs::create_dir_all(&cache_dir)?;
    }
    let m = if !cache_dir.is_dir() {
        cache_dir.clone()
    } else {
        cache_dir.join("model_cache")
    };
    let file = std::fs::File::create(&m)?;
    let mut encoder = flate2::write::ZlibEncoder::new(file, Compression::best());
    encoder.write_all(vec)?;
    Ok(m)
}

/// Unloads a buffer by writing it to a temporary file and reading it back.
/// This is used to ensure the memory is backed by disk if needed by the OS.
pub fn unload_buffer_to_temp_file(v: &[u8]) -> Result<CoreMLModelLoader, CoreMLError> {
    let mut temp_file = NamedTempFile::new()?;
    temp_file.write_all(v)?;
    let res = std::fs::read(temp_file.path())?;
    Ok(CoreMLModelLoader::Buffer(res))
}
