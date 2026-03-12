use crate::error::CoreMLError;
use flate2::Compression;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn save_buffer_to_disk(vec: &[u8], cache_dir: &mut PathBuf) -> Result<PathBuf, CoreMLError> {
    if cache_dir.as_os_str().is_empty() {
        *cache_dir = PathBuf::from(".");
    }
    // Determine target path before creating directories: if path has an extension,
    // treat it as a file path (create parent dir); otherwise treat as directory.
    let m = if cache_dir.extension().is_some() {
        if let Some(parent) = cache_dir.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        cache_dir.clone()
    } else {
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }
        cache_dir.join("model_cache")
    };

    std::fs::File::create(&m).and_then(|file| {
        flate2::write::ZlibEncoder::new(file, Compression::best()).write_all(vec)
    })?;

    Ok(m)
}

pub fn load_buffer_from_disk(path: &Path) -> Result<Vec<u8>, CoreMLError> {
    let file = std::fs::File::open(path)?;
    let mut vec = vec![];
    std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(file), &mut vec)?;
    Ok(vec)
}
