use crate::error::CoreMLError;

/// Validate input shape dynamically against expected shape
pub fn validate_coreml_shape(
    expected_shape: &[usize],
    actual_shape: &[usize],
    feature_name: &str,
) -> Result<(), CoreMLError> {
    if expected_shape.is_empty() && !actual_shape.is_empty() {
        return Err(CoreMLError::BadInputShape(format!(
            "Input feature name '{}' not expected!",
            feature_name
        )));
    }
    // Flexible shape matching: 0 means any dimension
    if expected_shape.len() != actual_shape.len()
        || !expected_shape
            .iter()
            .zip(actual_shape.iter())
            .all(|(&c, &a)| c == 0 || c == a)
    {
        return Err(CoreMLError::BadInputShape(format!(
            "expected shape {:?} found {:?}",
            expected_shape, actual_shape
        )));
    }
    Ok(())
}
use flate2::Compression;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn save_buffer_to_disk(vec: &[u8], cache_dir: &mut PathBuf) -> Result<PathBuf, CoreMLError> {
    if cache_dir.as_os_str().is_empty() {
        *cache_dir = PathBuf::from(".");
    }
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)?;
    }
    // pick the file specified, if it's a folder/dir append model_cache
    let m = if !cache_dir.is_dir() {
        cache_dir.clone()
    } else {
        cache_dir.join("model_cache")
    };

    std::fs::File::create(&m)
        .and_then(|file| {
            flate2::write::ZlibEncoder::new(file, Compression::best())
                .write_all(vec)
        })?;

    Ok(m)
}

pub fn load_buffer_from_disk(path: &Path) -> Result<Vec<u8>, CoreMLError> {
    let file = std::fs::File::open(path)?;
    let mut vec = vec![];
    std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(file), &mut vec)?;
    Ok(vec)
}
