use crate::error::CoreMLError;
use flate2::Compression;
use std::io::Write;
use std::path::{Path, PathBuf};

pub fn save_buffer_to_disk(vec: &[u8], cache_dir: &mut PathBuf) -> Result<PathBuf, CoreMLError> {
    if cache_dir.as_os_str().is_empty() {
        *cache_dir = PathBuf::from(".");
    }
    // Determine target path: if the path already exists as a directory or has no extension,
    // treat as directory and append "model_cache_{hash}". Otherwise treat as a file path.
    let m = if cache_dir.is_dir() || (!cache_dir.is_file() && cache_dir.extension().is_none()) {
        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)?;
        }
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(vec);
        let hash = hasher.finalize();
        cache_dir.join(format!("model_cache_{:x}", hash))
    } else {
        if let Some(parent) = cache_dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
            }
        }
        cache_dir.clone()
    };

    std::fs::File::create(&m).and_then(|file| {
        let mut encoder = flate2::write::ZlibEncoder::new(file, Compression::best());
        encoder.write_all(vec)?;
        encoder.finish()?;
        Ok(())
    })?;

    Ok(m)
}

pub fn load_buffer_from_disk(path: &Path) -> Result<Vec<u8>, CoreMLError> {
    let file = std::fs::File::open(path)?;
    let mut vec = vec![];
    std::io::Read::read_to_end(&mut flate2::read::ZlibDecoder::new(file), &mut vec)?;
    Ok(vec)
}
