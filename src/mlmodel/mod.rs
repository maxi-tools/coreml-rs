use crate::{
    ffi::{modelWithAssets, modelWithPath, Model, ModelOutput},
    loader::CoreMLModelLoader,
    mlarray::MLArray,
    options::{CoreMLModelInfo, CoreMLModelOptions},
    CoreMLError,
};
use flate2::Compression;
use ndarray::Array;
#[cfg(target_os = "macos")]
use std::collections::HashSet;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};

pub use crate::swift::MLModelOutput;

/// Maximum tensor size (512MB) - prevents memory exhaustion attacks from oversized inputs.
const MAX_TENSOR_SIZE_BYTES: usize = 512 * 1024 * 1024;

/// Represents a Core ML model and its associated state (either loaded or unloaded).
///
/// This enum manages the lifecycle of a model, allowing it to be loaded into memory for
/// inference and unloaded to save resources.
#[derive(Debug)]
pub enum CoreMLModelWithState {
    /// The model is not currently in memory. It contains information on how to load it.
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    /// The model is loaded into memory and ready for inference.
    Loaded(CoreMLModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl crate::state::ModelState for CoreMLModelWithState {
    type Model = CoreMLModel;

    fn info(&self) -> &CoreMLModelInfo {
        match self {
            Self::Unloaded(info, _) => info,
            Self::Loaded(_, info, _) => info,
        }
    }

    fn loader(&self) -> &CoreMLModelLoader {
        match self {
            Self::Unloaded(_, loader) => loader,
            Self::Loaded(_, _, loader) => loader,
        }
    }

    fn model(&self) -> Option<&Self::Model> {
        match self {
            Self::Unloaded(_, _) => None,
            Self::Loaded(model, _, _) => Some(model),
        }
    }

    fn into_parts(self) -> (CoreMLModelInfo, CoreMLModelLoader, Option<Self::Model>) {
        match self {
            Self::Unloaded(info, loader) => (info, loader, None),
            Self::Loaded(model, info, loader) => (info, loader, Some(model)),
        }
    }

    fn from_parts(
        info: CoreMLModelInfo,
        loader: CoreMLModelLoader,
        model: Option<Self::Model>,
    ) -> Self {
        if let Some(model) = model {
            Self::Loaded(model, info, loader)
        } else {
            Self::Unloaded(info, loader)
        }
    }

    /// Transitions the model from the `Unloaded` state to the `Loaded` state.
    fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                Self::validate_path(&path_buf)?;
                let mut coreml_model = CoreMLModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoad(
                        "Failed to load model; model path not valid".to_string(),
                        Self::Unloaded(info, CoreMLModelLoader::ModelPath(path_buf)),
                    ));
                }
                Ok(Self::Loaded(
                    coreml_model,
                    info,
                    CoreMLModelLoader::ModelPath(path_buf),
                ))
            }
            CoreMLModelLoader::CompiledPath(path_buf) => {
                Self::validate_path(&path_buf)?;
                let mut coreml_model =
                    CoreMLModel::load_from_path(path_buf.display().to_string(), info.clone(), true);
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoad(
                        "Failed to load model; compiled model cache got purged".to_string(),
                        Self::Unloaded(info, CoreMLModelLoader::CompiledPath(path_buf)),
                    ));
                }
                Ok(Self::Loaded(
                    coreml_model,
                    info,
                    CoreMLModelLoader::CompiledPath(path_buf),
                ))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoad(
                        "Failed to load model; likely not a CoreML mlmodel file".to_string(),
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::BufferToDisk(u) => {
                match std::fs::File::open(&u)
                    .map_err(CoreMLError::IoError)
                    .and_then(|file| {
                        let mut vec = vec![];
                        flate2::read::ZlibDecoder::new(file)
                            .read_to_end(&mut vec)
                            .map_err(CoreMLError::IoError)?;
                        Ok(vec)
                    }) {
                    Ok(vec) => {
                        let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(_err) => Err(CoreMLError::FailedToLoad(
                        "failed to load the model from cached buffer path".to_string(),
                        CoreMLModelWithState::Unloaded(info, CoreMLModelLoader::BufferToDisk(u)),
                    )),
                }
            }
        }
    }

    /// Unload the model from memory, returning it to the Unloaded state.
    fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(model, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => CoreMLModelLoader::Buffer(v),
                    CoreMLModelLoader::ModelPath(_) => {
                        if let Some(path) = model.model.compiled_path() {
                            CoreMLModelLoader::CompiledPath(path.into())
                        } else {
                            loader
                        }
                    }
                    x => x,
                },
            ))
        } else {
            Ok(self)
        }
    }

    /// Unloads the model buffer to the disk, at cache_dir.
    fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, mut info, loader) | Self::Unloaded(mut info, loader) => {
                let loader = {
                    match loader {
                        CoreMLModelLoader::Buffer(vec) => {
                            if info.opts.cache_dir.as_os_str().is_empty() {
                                info.opts.cache_dir = PathBuf::from(".");
                            }

                            if info.opts.cache_dir.exists() {
                                if !info.opts.cache_dir.is_dir() {
                                    return Err(CoreMLError::IoError(std::io::Error::new(
                                        std::io::ErrorKind::AlreadyExists,
                                        "cache_dir exists but is not a directory",
                                    )));
                                }
                            } else {
                                std::fs::create_dir_all(&info.opts.cache_dir)
                                    .map_err(CoreMLError::IoError)?;
                            }

                            let m = info.opts.cache_dir.join("model_cache");

                            match std::fs::File::create(&m)
                                .map_err(CoreMLError::IoError)
                                .and_then(|file| {
                                    let mut encoder =
                                        flate2::write::ZlibEncoder::new(file, Compression::best());
                                    encoder.write_all(&vec).map_err(CoreMLError::IoError)?;
                                    encoder.finish().map_err(CoreMLError::IoError)?;
                                    Ok(())
                                }) {
                                Ok(_) => {}
                                Err(err) => {
                                    return Err(CoreMLError::FailedToLoad(
                                        format!("failed to load the model from the buffer: {err}"),
                                        CoreMLModelWithState::Unloaded(
                                            info,
                                            CoreMLModelLoader::Buffer(vec),
                                        ),
                                    ));
                                }
                            }
                            CoreMLModelLoader::BufferToDisk(m)
                        }
                        loader => loader,
                    }
                };
                Ok(Self::Unloaded(info, loader))
            }
        }
    }
}

impl CoreMLModelWithState {
    fn validate_path(path: &Path) -> Result<(), CoreMLError> {
        if path
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            // Security: Use file_name to avoid leaking full system paths in error messages.
            let _name = path
                .file_name()
                .unwrap_or(std::ffi::OsStr::new("<redacted>"));
            let name = path.file_name().unwrap_or(path.as_os_str());
            return Err(CoreMLError::UnknownError(format!(
                "Invalid model path: path traversal detected in {:?}",
                name
            )));
        }
        Ok(())
    }

    pub fn new(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::ModelPath(path.as_ref().to_path_buf()),
        )
    }

    pub fn new_compiled(path: impl AsRef<Path>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(
            CoreMLModelInfo { opts },
            CoreMLModelLoader::CompiledPath(path.as_ref().to_path_buf()),
        )
    }

    pub fn from_buf(buf: Vec<u8>, opts: CoreMLModelOptions) -> Self {
        Self::Unloaded(CoreMLModelInfo { opts }, CoreMLModelLoader::Buffer(buf))
    }

    pub fn load(self) -> Result<Self, CoreMLError> {
        use crate::state::ModelState;
        ModelState::load(self)
    }

    pub fn unload(self) -> Result<Self, CoreMLError> {
        use crate::state::ModelState;
        ModelState::unload(self)
    }

    pub fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        use crate::state::ModelState;
        ModelState::unload_to_disk(self)
    }

    pub fn description(&self) -> Result<crate::description::ModelDescription, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.add_input(tag, input),
        }
    }

    pub fn add_input_cvpixelbuffer(
        &mut self,
        tag: impl AsRef<str>,
        width: usize,
        height: usize,
        bgra_data: Vec<u8>,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.add_input_cvpixelbuffer(tag, width, height, bgra_data)
            }
        }
    }

    /// Bind an IOSurface-backed tensor as a model input (#828 P0a).
    ///
    /// This is the zero-copy equivalent of `add_input` for callers that
    /// already hold a pooled `IOSurface` — no data is copied into an
    /// owned `Vec` or `ndarray::Array`.
    ///
    /// # Lock contract
    ///
    /// The caller must lock the surface (e.g., with
    /// `kIOSurfaceLockReadOnly`) before calling this function and must
    /// keep it locked through the next [`predict`](Self::predict) call.
    /// CoreML stores a raw pointer to the locked base address; unlocking
    /// early will corrupt inference.
    ///
    /// # Errors
    ///
    /// - `ModelNotLoaded` — model has not been loaded.
    /// - `BadInputShape` — shape contains zero dimensions or exceeds the
    ///   surface's allocation size.
    /// - `UnknownError` — the Swift-side bridge call failed (invalid
    ///   data type tag, `MLMultiArray` construction error, etc.).
    ///
    /// # Safety
    ///
    /// `surface` must be a valid `IOSurfaceRef` produced by
    /// `IOSurfaceCreate` (or equivalent) and currently locked by the
    /// caller.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_input_iosurface(
        &mut self,
        tag: impl AsRef<str>,
        surface: crate::iosurface::IOSurfaceRef,
        dtype: crate::mlarray::MLDataType,
        shape: &[usize],
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                // SAFETY: forwarded to caller's contract.
                unsafe { core_mlmodel.add_input_iosurface(tag, surface, dtype, shape) }
            }
        }
    }

    /// Bind a borrowed `CVPixelBuffer` as a model input (#828 P0a).
    ///
    /// Unlike [`add_input_cvpixelbuffer`](Self::add_input_cvpixelbuffer),
    /// this does not take ownership of a `Vec<u8>`. The pixel buffer is
    /// retained by CoreML for the lifetime of the feature value.
    ///
    /// # Errors
    ///
    /// - `ModelNotLoaded` — model has not been loaded.
    /// - `UnknownError` — the Swift-side bridge call failed.
    ///
    /// # Safety
    ///
    /// `pixel_buffer` must be a valid `CVPixelBufferRef`.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_input_cvpixelbuffer_ref(
        &mut self,
        tag: impl AsRef<str>,
        pixel_buffer: crate::iosurface::CVPixelBufferRef,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                // SAFETY: forwarded to caller's contract.
                unsafe { core_mlmodel.add_input_cvpixelbuffer_ref(tag, pixel_buffer) }
            }
        }
    }

    /// Bind an `IOSurface` as the destination for a named model output
    /// (#828 P0d).
    ///
    /// CoreML writes the prediction result for `tag` directly into the
    /// caller-provided surface rather than allocating its own
    /// `MLMultiArray`. This is the zero-copy mirror of
    /// [`add_input_iosurface`](Self::add_input_iosurface) on the write
    /// side — used by the P1 hybrid decode rewrite to hand the previous
    /// layer's FFN output straight to the next layer's Metal attention
    /// input without a Rust-side copy.
    ///
    /// The binding is single-shot: the Swift-side output-backings
    /// dictionary is cleared after each `predict()` call, so the caller
    /// must re-bind before every prediction.
    ///
    /// IOSurface-bound outputs are intentionally **not** projected into
    /// the post-predict `MLModelOutput.outputs` map — the caller already
    /// owns the surface and reads the bytes out-of-band via
    /// `IOSurfaceGetBaseAddress`. Projecting them would force a copy
    /// that defeats the whole point of the API.
    ///
    /// # Lock contract
    ///
    /// The caller must lock the surface read-write (NOT
    /// `kIOSurfaceLockReadOnly` — CoreML writes into it) before calling
    /// this function and must keep it locked through the subsequent
    /// [`predict`](Self::predict) /
    /// [`predict_with_state`](Self::predict_with_state) call. Unlocking
    /// early corrupts inference. See the P0d spec doc
    /// (`docs/superpowers/specs/2026-04-08-828-p0d-coreml-output-iosurface-design.md`)
    /// for the full lifecycle.
    ///
    /// # Errors
    ///
    /// - `ModelNotLoaded` — model has not been loaded.
    /// - `BadInputShape` — `surface` is null, `shape` contains zero
    ///   dimensions, shape arithmetic overflows, or the computed
    ///   byte-count exceeds the surface's allocation size.
    /// - `UnknownError` — the Swift-side bridge call failed (invalid
    ///   dtype tag, `MLMultiArray` construction error, etc.).
    ///
    /// # Safety
    ///
    /// `surface` must be a valid `IOSurfaceRef` produced by
    /// `IOSurfaceCreate` (or equivalent) and currently locked
    /// read-write by the caller.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_output_iosurface(
        &mut self,
        tag: impl AsRef<str>,
        surface: crate::iosurface::IOSurfaceRef,
        dtype: crate::mlarray::MLDataType,
        shape: &[usize],
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                // SAFETY: forwarded to caller's contract.
                unsafe { core_mlmodel.add_output_iosurface(tag, surface, dtype, shape) }
            }
        }
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }

    pub fn make_state(&mut self) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                if core_mlmodel.make_state() {
                    Ok(())
                } else {
                    Err(CoreMLError::UnknownError(
                        "make_state failed (model may not support state or macOS < 15)".to_string(),
                    ))
                }
            }
        }
    }

    pub fn predict_with_state(&mut self) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict_with_state(),
        }
    }

    pub fn has_state(&self) -> bool {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => false,
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.has_state(),
        }
    }

    pub fn reset_state(&mut self) {
        if let CoreMLModelWithState::Loaded(core_mlmodel, _, _) = self {
            core_mlmodel.reset_state();
        }
    }

    pub fn compiled_path(&self) -> Option<String> {
        match self {
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.model.compiled_path(),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Model,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
    cached_predict_info: Option<(bool, Vec<(String, Vec<usize>, String)>)>,
    cached_predict_with_state_info: Option<(bool, Vec<(String, Vec<usize>, String)>)>,
    output_buffers: HashMap<String, Vec<u8>>,
    /// Set of output tags that have been pre-bound to a caller-provided
    /// `IOSurfaceRef` via [`CoreMLModelWithState::add_output_iosurface`]
    /// (#828 P0d). Tags in this set are skipped by the default
    /// auto-backing allocation loop in `predict_inner` and are NOT
    /// projected into the post-predict `MLModelOutput.outputs` map —
    /// the caller reads the surface out-of-band.
    ///
    /// The set is cleared after each `predict()` call to match the
    /// Swift-side `self.outputs = [:]` reset; bindings are single-shot.
    #[cfg(target_os = "macos")]
    iosurface_bound_outputs: HashSet<String>,
    /// Keep model asset buffer alive for the duration of the model's life.
    _model_asset_buffer: Option<Vec<u8>>,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl CoreMLModel {
    fn apply_options(mut model: Model, opts: &CoreMLModelOptions) -> Model {
        if let Some(enabled) = opts.allow_low_precision_accumulation_on_gpu {
            model.setAllowLowPrecisionAccumulationOnGPU(enabled);
        }
        if let Some(enabled) = opts.prediction_uses_cpu_only {
            model.setPredictionUsesCPUOnly(enabled);
        }
        model
    }

    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        let path_buf = PathBuf::from(&path);
        if path_buf
            .components()
            .any(|c| matches!(c, std::path::Component::RootDir))
            && !path.starts_with("/Users/")
            && !path_buf.starts_with(std::env::temp_dir())
            && !path.starts_with("./")
        {
            eprintln!("WARNING: Loading model from potentially sensitive system path: {}. Ensure this is intended.", path);
        }
        let model = Self::apply_options(
            modelWithPath(path, info.opts.compute_platform, compiled),
            &info.opts,
        );

        Self {
            model,
            outputs: Default::default(),
            cached_predict_info: None,
            cached_predict_with_state_info: None,
            output_buffers: Default::default(),
            #[cfg(target_os = "macos")]
            iosurface_bound_outputs: Default::default(),
            _model_asset_buffer: None,
        }
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let model = Self::apply_options(
            modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            &info.opts,
        );

        Self {
            model,
            outputs: Default::default(),
            cached_predict_info: None,
            cached_predict_with_state_info: None,
            output_buffers: Default::default(),
            #[cfg(target_os = "macos")]
            iosurface_bound_outputs: Default::default(),
            _model_asset_buffer: Some(buf),
        }
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        let input: MLArray = input.into();

        // Security: Limit input tensor size to prevent DoS via memory exhaustion.
        let total_elements = input
            .shape()
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim));
        let element_size = match &input {
            MLArray::Float32Array(_) | MLArray::Int32Array(_) | MLArray::UInt32Array(_) => 4,
            MLArray::Float16Array(_) | MLArray::Int16Array(_) | MLArray::UInt16Array(_) => 2,
            MLArray::Int8Array(_) | MLArray::UInt8Array(_) => 1,
            // IOSurface-backed arrays go through the dedicated
            // `add_input_iosurface` path — if one ends up here the
            // unsupported-type arm below will reject it.
            #[cfg(target_os = "macos")]
            MLArray::IOSurface(wrap) => wrap.dtype().size_bytes(),
        };
        let is_too_large = match total_elements {
            Some(total) => total.saturating_mul(element_size) > MAX_TENSOR_SIZE_BYTES,
            None => true,
        };
        if is_too_large {
            return Err(CoreMLError::BadInputShape(format!(
                "Input tensor for '{}' is too large (max 512MB)",
                tag.as_ref()
            )));
        }

        let name = tag.as_ref().to_string();
        let shape: Vec<usize> = input.shape().to_vec();

        match &input {
            MLArray::Float32Array(array_base) => {
                let owned = array_base.as_standard_layout().into_owned();
                let (data, offset) = owned.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                let capacity = data.capacity();
                let mut data_bytes = unsafe {
                    let ptr = data.as_ptr() as *mut u8;
                    let len = data.len() * 4;
                    let cap = data.capacity() * 4;
                    std::mem::forget(data);
                    Vec::from_raw_parts(ptr, len, cap)
                };

                if !self.model.bindInputF32(
                    shape,
                    &name,
                    data_bytes.as_mut_ptr() as *mut f32,
                    capacity,
                ) {
                    return Err(CoreMLError::UnknownError(
                        "failed to bind input to model".to_string(),
                    ));
                }
                // Swift's MLMultiArray deallocator now owns this buffer.
                std::mem::forget(data_bytes);
            }
            MLArray::Float16Array(array_base) => {
                let owned = array_base.as_standard_layout().into_owned();
                let (data, offset) = owned.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                let capacity = data.capacity();
                let mut data_bytes = unsafe {
                    let ptr = data.as_ptr() as *mut u8;
                    let len = data.len() * 2;
                    let cap = data.capacity() * 2;
                    std::mem::forget(data);
                    Vec::from_raw_parts(ptr, len, cap)
                };

                if !self.model.bindInputU16(
                    shape,
                    &name,
                    data_bytes.as_mut_ptr() as *mut u16,
                    capacity,
                ) {
                    return Err(CoreMLError::UnknownError(
                        "failed to bind input to model".to_string(),
                    ));
                }
                // Swift's MLMultiArray deallocator now owns this buffer.
                std::mem::forget(data_bytes);
            }
            MLArray::Int32Array(array_base) => {
                let owned = array_base.as_standard_layout().into_owned();
                let (data, offset) = owned.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                let capacity = data.capacity();
                let mut data_bytes = unsafe {
                    let ptr = data.as_ptr() as *mut u8;
                    let len = data.len() * 4;
                    let cap = data.capacity() * 4;
                    std::mem::forget(data);
                    Vec::from_raw_parts(ptr, len, cap)
                };

                if !self.model.bindInputI32(
                    shape,
                    &name,
                    data_bytes.as_mut_ptr() as *mut i32,
                    capacity,
                ) {
                    return Err(CoreMLError::UnknownError(
                        "failed to bind input to model".to_string(),
                    ));
                }
                // Swift's MLMultiArray deallocator now owns this buffer.
                std::mem::forget(data_bytes);
            }
            _ => {
                return Err(CoreMLError::UnknownError(
                    "unsupported input type for bindInput".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn add_input_cvpixelbuffer(
        &mut self,
        tag: impl AsRef<str>,
        width: usize,
        height: usize,
        bgra_data: Vec<u8>,
    ) -> Result<(), CoreMLError> {
        let name = tag.as_ref().to_string();
        let expected_len = width * height * 4;

        if bgra_data.len() != expected_len {
            return Err(CoreMLError::BadInputShape(format!(
                "Expected {} bytes for {}x{} BGRA image, got {}",
                expected_len,
                width,
                height,
                bgra_data.len()
            )));
        }

        if bgra_data.len() > MAX_TENSOR_SIZE_BYTES {
            return Err(CoreMLError::BadInputShape(format!(
                "Input buffer too large: {} bytes (max {} bytes)",
                bgra_data.len(),
                MAX_TENSOR_SIZE_BYTES
            )));
        }

        let mut data = bgra_data;
        if !self.model.bindInputCVPixelBuffer(
            width,
            height,
            &name,
            data.as_mut_ptr(),
            data.capacity(),
        ) {
            return Err(CoreMLError::UnknownError(
                "failed to bind CVPixelBuffer input to model".to_string(),
            ));
        }
        // Swift's CVPixelBuffer release callback now owns this buffer.
        std::mem::forget(data);
        Ok(())
    }

    /// Internal zero-copy IOSurface bind (#828 P0a).
    ///
    /// See [`CoreMLModelWithState::add_input_iosurface`] for the public
    /// API and safety contract.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_input_iosurface(
        &mut self,
        tag: impl AsRef<str>,
        surface: crate::iosurface::IOSurfaceRef,
        dtype: crate::mlarray::MLDataType,
        shape: &[usize],
    ) -> Result<(), CoreMLError> {
        if surface.is_null() {
            return Err(CoreMLError::BadInputShape(
                "IOSurfaceRef is null".to_string(),
            ));
        }
        if shape.is_empty() || shape.iter().any(|&d| d == 0) {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface shape must be non-empty with no zero dims, got {shape:?}"
            )));
        }
        // Use checked arithmetic throughout: silent `saturating_mul`
        // could mask an overflow as a too-large-tensor error which is
        // misleading. Propagate the overflow as its own explicit error.
        let elem_count = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| {
                CoreMLError::BadInputShape(format!(
                    "IOSurface shape element-count overflow for '{}': {shape:?}",
                    tag.as_ref()
                ))
            })?;
        let expected_bytes = elem_count.checked_mul(dtype.size_bytes()).ok_or_else(|| {
            CoreMLError::BadInputShape(format!(
                "IOSurface byte-count overflow for '{}': {shape:?} * {} bytes",
                tag.as_ref(),
                dtype.size_bytes()
            ))
        })?;
        if expected_bytes > MAX_TENSOR_SIZE_BYTES {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface tensor for '{}' exceeds max tensor size ({} > {})",
                tag.as_ref(),
                expected_bytes,
                MAX_TENSOR_SIZE_BYTES
            )));
        }
        // SAFETY: caller's contract — surface is a valid IOSurfaceRef.
        let alloc_size = unsafe { crate::iosurface::IOSurfaceGetAllocSize(surface) };
        if alloc_size < expected_bytes {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface alloc_size={alloc_size} < expected={expected_bytes} for shape={shape:?} dtype={dtype:?}"
            )));
        }

        let name = tag.as_ref().to_string();
        let shape_vec: Vec<usize> = shape.to_vec();
        let ptr = surface as *mut u8;
        if !self
            .model
            .bindInputIOSurface(ptr, dtype.raw_tag(), shape_vec, &name)
        {
            return Err(CoreMLError::UnknownError(format!(
                "failed to bind IOSurface input '{}' to model",
                name
            )));
        }
        Ok(())
    }

    /// Internal zero-copy CVPixelBuffer bind (#828 P0a).
    ///
    /// See [`CoreMLModelWithState::add_input_cvpixelbuffer_ref`] for the
    /// public API and safety contract.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_input_cvpixelbuffer_ref(
        &mut self,
        tag: impl AsRef<str>,
        pixel_buffer: crate::iosurface::CVPixelBufferRef,
    ) -> Result<(), CoreMLError> {
        if pixel_buffer.is_null() {
            return Err(CoreMLError::BadInputShape(
                "CVPixelBufferRef is null".to_string(),
            ));
        }
        let name = tag.as_ref().to_string();
        let ptr = pixel_buffer as *mut u8;
        if !self.model.bindInputCVPixelBufferRef(ptr, &name) {
            return Err(CoreMLError::UnknownError(format!(
                "failed to bind CVPixelBuffer ref '{}' to model",
                name
            )));
        }
        Ok(())
    }

    /// Internal zero-copy IOSurface output bind (#828 P0d).
    ///
    /// See [`CoreMLModelWithState::add_output_iosurface`] for the public
    /// API and safety contract.
    #[cfg(target_os = "macos")]
    pub unsafe fn add_output_iosurface(
        &mut self,
        tag: impl AsRef<str>,
        surface: crate::iosurface::IOSurfaceRef,
        dtype: crate::mlarray::MLDataType,
        shape: &[usize],
    ) -> Result<(), CoreMLError> {
        if surface.is_null() {
            return Err(CoreMLError::BadInputShape(
                "IOSurfaceRef is null".to_string(),
            ));
        }
        if shape.is_empty() || shape.iter().any(|&d| d == 0) {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface output shape must be non-empty with no zero dims, got {shape:?}"
            )));
        }
        // Use checked arithmetic throughout — pathological shapes
        // should produce an explicit overflow error rather than a
        // misleading "too large" diagnostic.
        let elem_count = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| {
                CoreMLError::BadInputShape(format!(
                    "IOSurface output shape element-count overflow for '{}': {shape:?}",
                    tag.as_ref()
                ))
            })?;
        let expected_bytes = elem_count.checked_mul(dtype.size_bytes()).ok_or_else(|| {
            CoreMLError::BadInputShape(format!(
                "IOSurface output byte-count overflow for '{}': {shape:?} * {} bytes",
                tag.as_ref(),
                dtype.size_bytes()
            ))
        })?;
        if expected_bytes > MAX_TENSOR_SIZE_BYTES {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface output tensor for '{}' exceeds max tensor size ({} > {})",
                tag.as_ref(),
                expected_bytes,
                MAX_TENSOR_SIZE_BYTES
            )));
        }
        // SAFETY: caller's contract — surface is a valid IOSurfaceRef.
        let alloc_size = unsafe { crate::iosurface::IOSurfaceGetAllocSize(surface) };
        if alloc_size < expected_bytes {
            return Err(CoreMLError::BadInputShape(format!(
                "IOSurface output alloc_size={alloc_size} < expected={expected_bytes} for shape={shape:?} dtype={dtype:?}"
            )));
        }

        let name = tag.as_ref().to_string();
        // `bindOutput*` signatures all take `Vec<i32>` for shape, so
        // convert here to keep the FFI table consistent.
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let ptr = surface as *mut u8;
        if !self
            .model
            .bindOutputIOSurface(ptr, dtype.raw_tag(), shape_i32, &name)
        {
            return Err(CoreMLError::UnknownError(format!(
                "failed to bind IOSurface output '{}' to model",
                name
            )));
        }
        // Record the tag so the default auto-backing allocation loop
        // in `predict_inner` skips it, preserving the IOSurface
        // backing installed on the Swift side.
        //
        // Also clear any existing regular output backing for the same
        // tag — otherwise the post-prediction output map could contain
        // stale data from a prior `add_output` call.
        self.outputs.remove(&name);
        self.output_buffers.remove(&name);
        self.iosurface_bound_outputs.insert(name);
        Ok(())
    }

    fn add_output_generic<T: crate::mlarray::MLType>(
        &mut self,
        tag: impl AsRef<str>,
        out: impl Into<MLArray>,
        ty_label: &'static str,
        bind_fn: impl FnOnce(&Model, Vec<i32>, &str, *mut T, usize) -> bool,
    ) -> bool {
        let arr: MLArray = out.into();

        if arr.len_bytes() > MAX_TENSOR_SIZE_BYTES {
            eprintln!("output buffer too large: {} bytes", arr.len_bytes());
            return false;
        }

        let shape = arr.shape().to_vec();
        self.outputs
            .insert(tag.as_ref().to_string(), (ty_label, shape.clone()));
        let i32_shape: Vec<i32> = shape.iter().map(|&i| i as i32).collect();
        let tensor = match T::extract_from_mlarray(arr) {
            Some(t) => t,
            None => return false,
        };
        let (data, offset) = tensor.into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let mut data = data.into_boxed_slice().into_vec();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.len();
        if !bind_fn(&self.model, i32_shape, &name, ptr, len) {
            return false;
        }
        // Safety: Reinterpret Vec<T> to Vec<u8> to store it in output_buffers
        let data_bytes = unsafe {
            let ptr = data.as_ptr() as *mut u8;
            let len = data.len() * std::mem::size_of::<T>();
            let cap = data.capacity() * std::mem::size_of::<T>();
            std::mem::forget(data);
            Vec::from_raw_parts(ptr, len, cap)
        };
        self.output_buffers.insert(name, data_bytes);
        true
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        self.add_output_generic::<f32>(tag, out, "f32", |m, s, n, p, l| m.bindOutputF32(s, n, p, l))
    }

    pub fn add_output_u16(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        self.add_output_generic::<u16>(tag, out, "f16", |m, s, n, p, l| m.bindOutputU16(s, n, p, l))
    }

    pub fn add_output_i32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        self.add_output_generic::<i32>(tag, out, "i32", |m, s, n, p, l| m.bindOutputI32(s, n, p, l))
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        self.predict_inner(false, |model: &Model| model.predict())
    }

    pub fn predict_with_state(&mut self) -> Result<MLModelOutput, CoreMLError> {
        self.predict_inner(true, |model: &Model| model.predictWithState())
    }

    pub fn description(&self) -> crate::description::ModelDescription {
        self.model.description().into()
    }

    pub fn make_state(&mut self) -> bool {
        self.model.makeState()
    }

    pub fn has_state(&self) -> bool {
        self.model.hasState()
    }

    pub fn reset_state(&mut self) {
        self.model.resetState()
    }

    fn predict_inner(
        &mut self,
        stateful: bool,
        predict_fn: impl FnOnce(&Model) -> ModelOutput,
    ) -> Result<MLModelOutput, CoreMLError> {
        let cache = if stateful {
            &mut self.cached_predict_with_state_info
        } else {
            &mut self.cached_predict_info
        };

        if cache.is_none() {
            let desc = self.model.description();
            let mut use_output_backing = true;
            if std::env::var("COREML_DISABLE_OUTPUT_BACKING").is_ok() {
                use_output_backing = false;
            }
            let mut output_info = Vec::new();

            let output_names = desc.output_names();
            for name in output_names {
                let output_shape = desc.output_shape(&name);
                let ty = desc.output_type(&name);

                let is_dynamic = if stateful {
                    output_shape.is_empty() || output_shape.contains(&0)
                } else {
                    output_shape.contains(&0)
                };

                if is_dynamic {
                    use_output_backing = false;
                }

                output_info.push((name, output_shape, ty));
            }
            *cache = Some((use_output_backing, output_info));
        }

        let (use_output_backing, output_info) = if stateful {
            self.cached_predict_with_state_info
                .as_ref()
                .unwrap()
                .clone()
        } else {
            self.cached_predict_info.as_ref().unwrap().clone()
        };

        if use_output_backing {
            for (name, output_shape, ty) in &output_info {
                // #828 P0d: if the caller pre-bound this output to an
                // IOSurface via `add_output_iosurface`, skip the
                // default auto-backing allocation so we don't
                // overwrite the IOSurface MLMultiArray that the Swift
                // shim already installed in `self.outputs[name]`.
                #[cfg(target_os = "macos")]
                if self.iosurface_bound_outputs.contains(name) {
                    continue;
                }
                match ty.as_str() {
                    "f32" | "bool" | "boolean" => {
                        self.add_output_f32(
                            name.clone(),
                            Array::<f32, _>::zeros(output_shape.clone()),
                        );
                    }
                    "f16" | "float16" => {
                        self.add_output_u16(
                            name.clone(),
                            Array::<u16, _>::zeros(output_shape.clone()),
                        );
                    }
                    "int32" | "int64" | "int16" | "uint32" | "uint64" | "uint16" => {
                        self.add_output_i32(
                            name.clone(),
                            Array::<i32, _>::zeros(output_shape.clone()),
                        );
                    }
                    _ => {
                        return Err(CoreMLError::UnknownError(format!(
                            "non-f32/f16/i32 output types are not supported (yet)! type: {}",
                            ty
                        )));
                    }
                }
            }
        }

        let output = predict_fn(&self.model);

        if let Some(err) = output.getError() {
            // Clear IOSurface bindings even on failure so a stale
            // backing doesn't persist into the next predict() call.
            #[cfg(target_os = "macos")]
            self.iosurface_bound_outputs.clear();
            return Err(CoreMLError::UnknownError(err));
        }

        if !use_output_backing {
            let mut outputs = fxhash::FxHashMap::default();
            for (name, _output_shape, ty) in &output_info {
                // Skip outputs that were bound to an IOSurface — the
                // real data lives in the caller's surface, not in
                // ModelOutput's standard backing. Extracting here would
                // produce zero/garbage bytes and overwrite the caller's
                // out-of-band data flow.
                #[cfg(target_os = "macos")]
                if self.iosurface_bound_outputs.contains(name.as_str()) {
                    continue;
                }
                let actual_shape: Vec<usize> = output.outputShape(name).into_iter().collect();
                if actual_shape.is_empty() {
                    continue;
                }

                match ty.as_str() {
                    "f32" | "bool" | "boolean" => {
                        let out = output.outputF32(name);
                        if !out.is_empty() {
                            match Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                                Ok(array) => {
                                    outputs.insert(name.clone(), array.into());
                                }
                                Err(e) => eprintln!(
                                    "WARNING: output '{}' shape reconstruction failed: {}",
                                    name, e
                                ),
                            }
                        }
                    }
                    "f16" | "float16" => {
                        let out = output.outputU16(name);
                        if !out.is_empty() {
                            match Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                                Ok(array) => {
                                    let f16_array = reinterpret_u16_to_f16(array);
                                    outputs.insert(name.clone(), f16_array.into());
                                }
                                Err(e) => eprintln!(
                                    "WARNING: output '{}' shape reconstruction failed: {}",
                                    name, e
                                ),
                            }
                        }
                    }
                    "int32" | "int64" | "int16" | "uint32" | "uint64" | "uint16" => {
                        let out = output.outputI32(name);
                        if !out.is_empty() {
                            match Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                                Ok(array) => {
                                    outputs.insert(name.clone(), array.into());
                                }
                                Err(e) => eprintln!(
                                    "WARNING: output '{}' shape reconstruction failed: {}",
                                    name, e
                                ),
                            }
                        }
                    }
                    _ => {}
                }
            }
            // Clear the IOSurface binding set AFTER the extraction
            // loop has had a chance to skip bound tags.
            #[cfg(target_os = "macos")]
            self.iosurface_bound_outputs.clear();
            return Ok(MLModelOutput { outputs });
        }

        // use_output_backing path — self.outputs has the backings.
        // Clear IOSurface bindings now (they were consumed).
        #[cfg(target_os = "macos")]
        self.iosurface_bound_outputs.clear();

        Ok(MLModelOutput {
            outputs: self
                .outputs
                .clone()
                .into_iter()
                .filter_map(|(key, (ty, shape))| {
                    let name = key.as_str();
                    match ty {
                        "f32" => {
                            let out = output.outputF32(name);
                            match Array::from_shape_vec(shape, out) {
                                Ok(array) => Some((key, array.into())),
                                Err(e) => {
                                    eprintln!(
                                        "WARNING: output '{}' shape reconstruction failed: {}",
                                        name, e
                                    );
                                    None
                                }
                            }
                        }
                        "f16" => {
                            let out = output.outputU16(name);
                            match Array::from_shape_vec(shape, out) {
                                Ok(array) => Some((key, reinterpret_u16_to_f16(array).into())),
                                Err(e) => {
                                    eprintln!(
                                        "WARNING: output '{}' shape reconstruction failed: {}",
                                        name, e
                                    );
                                    None
                                }
                            }
                        }
                        "i32" => {
                            let out = output.outputI32(name);
                            match Array::from_shape_vec(shape, out) {
                                Ok(array) => Some((key, array.into())),
                                Err(e) => {
                                    eprintln!(
                                        "WARNING: output '{}' shape reconstruction failed: {}",
                                        name, e
                                    );
                                    None
                                }
                            }
                        }
                        _ => None,
                    }
                })
                .collect(),
        })
    }
}

fn reinterpret_u16_to_f16(input: ndarray::ArrayD<u16>) -> ndarray::ArrayD<half::f16> {
    let shape = input.shape().to_vec();
    let len = input.len();
    let (raw_vec, offset) = input.into_raw_vec_and_offset();
    assert!(
        matches!(offset, Some(0) | None),
        "array base offset is not zero; bad aligned data reinterpret"
    );
    let raw_vec_f16 = unsafe {
        let ptr = raw_vec.as_ptr() as *mut half::f16;
        let capacity = raw_vec.capacity();
        std::mem::forget(raw_vec);
        Vec::from_raw_parts(ptr, len, capacity)
    };
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), raw_vec_f16).unwrap()
}
