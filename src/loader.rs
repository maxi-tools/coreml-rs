use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum CoreMLModelLoader {
    /// Model to be loaded from the given path
    ModelPath(PathBuf),
    /// Model cache built and stored at path, to be used for faster reload
    CompiledPath(PathBuf),
    /// Model with buffer to manage the buffer locally
    Buffer(Vec<u8>),
    BufferToDisk(PathBuf),
}
