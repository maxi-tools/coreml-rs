use crate::loader::CoreMLModelLoader;
use crate::options::CoreMLModelInfo;
use crate::CoreMLError;
use flate2::Compression;
use std::io::Write;
use std::path::PathBuf;

pub trait ModelState: Sized {
    type Model;

    fn info(&self) -> &CoreMLModelInfo;
    fn loader(&self) -> &CoreMLModelLoader;
    fn model(&self) -> Option<&Self::Model>;

    fn into_parts(self) -> (CoreMLModelInfo, CoreMLModelLoader, Option<Self::Model>);

    fn load(self) -> Result<Self, CoreMLError>;
    fn unload(self) -> Result<Self, CoreMLError>;

    fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        let (mut info, loader, _) = self.into_parts();
        let new_loader = match loader {
            CoreMLModelLoader::Buffer(vec) => {
                if info.opts.cache_dir.as_os_str().is_empty() {
                    info.opts.cache_dir = PathBuf::from(".");
                }
                if !info.opts.cache_dir.exists() {
                    _ = std::fs::remove_dir_all(&info.opts.cache_dir);
                    _ = std::fs::create_dir_all(&info.opts.cache_dir);
                }
                let m = if !info.opts.cache_dir.is_dir() {
                    info.opts.cache_dir.clone()
                } else {
                    info.opts.cache_dir.join("model_cache")
                };

                std::fs::File::create(&m)
                    .map_err(CoreMLError::IoError)
                    .and_then(|file| {
                        let mut encoder =
                            flate2::write::ZlibEncoder::new(file, Compression::best());
                        encoder.write_all(&vec).map_err(CoreMLError::IoError)
                    })?;
                CoreMLModelLoader::BufferToDisk(m)
            }
            l => l,
        };

        // If it was loaded, it remains loaded but we've cached it to disk?
        // Actually unload_to_disk usually implies transitioning to Unloaded.
        // The original implementation returned Unloaded.
        Ok(Self::from_parts(info, new_loader, None))
    }

    fn from_parts(
        info: CoreMLModelInfo,
        loader: CoreMLModelLoader,
        model: Option<Self::Model>,
    ) -> Self;
}
