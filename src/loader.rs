use crate::ffi::ComputePlatform;
use std::path::PathBuf;

#[derive(Default, Clone)]
pub struct CoreMLModelOptions {
    pub compute_platform: ComputePlatform,
    pub cache_dir: PathBuf,
}

impl std::fmt::Debug for CoreMLModelOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoreMLModelOptions")
            .field(
                "compute_platform",
                match self.compute_platform {
                    ComputePlatform::Cpu => &"CPU",
                    ComputePlatform::CpuAndANE => &"CpuAndAne",
                    ComputePlatform::CpuAndGpu => &"CpuAndGpu",
                },
            )
            .finish()
    }
}

#[derive(Debug)]
pub enum CoreMLModelLoader {
    /// Model to be loaded from the given path
    ModelPath(PathBuf),
    /// Model cache built and stored at path, to be used for faster reload
    CompiledPath(PathBuf),
    /// Model with buffer to manage the buffer locally
    Buffer(Vec<u8>),
    BufferToDisk(PathBuf),
}

// Info required to create a coreml model
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    pub opts: CoreMLModelOptions,
}

impl CoreMLModelOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_compute_platform(mut self, compute_platform: ComputePlatform) -> Self {
        self.compute_platform = compute_platform;
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = cache_dir.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_model_options_builder() {
        let opts = CoreMLModelOptions::new()
            .with_compute_platform(ComputePlatform::CpuAndANE)
            .with_cache_dir("/tmp/cache");

        assert!(matches!(opts.compute_platform, ComputePlatform::CpuAndANE));
        assert_eq!(opts.cache_dir, PathBuf::from("/tmp/cache"));
    }
}
