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
    time::Duration,
};

/// Backoff schedule applied between retry attempts.
///
/// Mirrors `RetryBackoff` in upstream `swarnimarun/coreml-rs` (commit
/// `82f6fa3` — feat: add support for local retry) so existing upstream
/// callers can port without renaming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RetryBackoff {
    /// No delay between attempts.
    None,
    /// Constant delay between attempts.
    Fixed(Duration),
    /// Exponential growth, capped at `max`.
    Exponential {
        initial: Duration,
        multiplier: f64,
        max: Duration,
    },
}

impl Default for RetryBackoff {
    fn default() -> Self {
        RetryBackoff::None
    }
}

impl RetryBackoff {
    fn delay(self, retry_index: usize) -> Duration {
        match self {
            RetryBackoff::None => Duration::ZERO,
            RetryBackoff::Fixed(delay) => delay,
            RetryBackoff::Exponential {
                initial,
                multiplier,
                max,
            } => {
                let multiplier = multiplier.max(1.0);
                // Scale in f64 seconds rather than with `Duration::mul_f64`.
                //
                // `mul_f64` panics on a non-finite factor or on overflow, and
                // `multiplier.powi(retry_index)` produces exactly that once the
                // index or the multiplier is large enough -- which turns a slow
                // retry loop into a crash in a library that maxi-ml links.
                // Clamping against `max` before constructing the Duration means
                // the f64 intermediate can be saturating without ever reaching
                // the panicking constructor.
                //
                // `retry_index` is clamped to i32::MAX because the `as i32` cast
                // is a wrapping cast for indices above 2^31-1; saturating there
                // is harmless because the result is clamped to `max` anyway.
                let exponent = retry_index.min(i32::MAX as usize) as i32;
                let scaled = initial.as_secs_f64() * multiplier.powi(exponent);
                if scaled.is_finite() && scaled >= 0.0 && scaled < max.as_secs_f64() {
                    Duration::from_secs_f64(scaled)
                } else {
                    max
                }
            }
        }
    }
}

/// Configures the retry methods on [`CoreMLModel`] and
/// [`CoreMLModelWithState`]: `predict_with_retry{,_if}` and
/// `predict_with_rebind_retry{,_if}`.
///
/// For a model that takes inputs, use the `rebind` variants -- a failed
/// predict clears the input bindings, so a retry that does not re-bind
/// cannot succeed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictRetryOptions {
    pub max_retries: usize,
    pub backoff: RetryBackoff,
}

impl Default for PredictRetryOptions {
    fn default() -> Self {
        Self::none()
    }
}

/// The attempt / predicate / backoff / propagate loop behind every
/// `predict_with_retry*` method, as a free function over an arbitrary
/// operation.
///
/// It is deliberately free of CoreML types so the retry semantics can be
/// unit-tested without a loaded model -- which matters because no CI lane
/// in this repo runs `cargo test` against real models, and because the
/// semantics (does the predicate gate a retry? does the final error
/// propagate? is the delay applied between attempts and not after the
/// last?) are exactly the things worth pinning down.
///
/// `operation` is invoked once per attempt, so an implementation that
/// needs to restore state between attempts can do so inside the closure.
pub fn retry_with_backoff<T, E, F>(
    options: PredictRetryOptions,
    mut should_retry: impl FnMut(&E) -> bool,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut(usize) -> Result<T, E>,
{
    for attempt in 0..=options.max_retries {
        match operation(attempt) {
            Ok(value) => return Ok(value),
            Err(err) if attempt < options.max_retries && should_retry(&err) => {
                let delay = options.backoff.delay(attempt);
                if !delay.is_zero() {
                    std::thread::sleep(delay);
                }
            }
            Err(err) => return Err(err),
        }
    }
    unreachable!("retry loop always returns from success or final failure")
}

impl PredictRetryOptions {
    pub const fn none() -> Self {
        Self {
            max_retries: 0,
            backoff: RetryBackoff::None,
        }
    }

    pub const fn fixed(max_retries: usize, delay: Duration) -> Self {
        Self {
            max_retries,
            backoff: RetryBackoff::Fixed(delay),
        }
    }

    pub const fn exponential(
        max_retries: usize,
        initial: Duration,
        multiplier: f64,
        max: Duration,
    ) -> Self {
        Self {
            max_retries,
            backoff: RetryBackoff::Exponential {
                initial,
                multiplier,
                max,
            },
        }
    }
}

pub use crate::swift::MLModelOutput;

/// Per-device operation counts from an `MLComputePlan` (macOS 14.4+).
///
/// `total` is the number of program operations in the compiled model;
/// `ane`/`gpu`/`cpu` count operations whose *preferred* dispatch device is
/// that class. Counts may not sum to `total` when the OS reports no device
/// usage for an operation (e.g. const ops).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePlanDeviceCounts {
    pub total: usize,
    pub ane: usize,
    pub gpu: usize,
    pub cpu: usize,
}

impl ComputePlanDeviceCounts {
    /// Fraction of counted operations preferring the Neural Engine (0.0 when empty).
    pub fn ane_fraction(&self) -> f64 {
        let counted = self.ane + self.gpu + self.cpu;
        if counted == 0 {
            0.0
        } else {
            self.ane as f64 / counted as f64
        }
    }
}

/// Load the `MLComputePlan` for a compiled model (`.mlmodelc`) and report
/// per-device preferred-operation counts under `platform`.
///
/// Returns `None` when the plan cannot be loaded — older OS (< macOS 14.4),
/// invalid path, or a non-program model. This is the ground-truth check for
/// silent CPU fallback: throughput numbers alone cannot distinguish "fast
/// enough on CPU" from "actually resident on the ANE".
pub fn compute_plan_device_counts(
    compiled_path: impl AsRef<Path>,
    platform: crate::ffi::ComputePlatform,
) -> Option<ComputePlanDeviceCounts> {
    let path = compiled_path.as_ref().to_str()?.to_string();
    let counts = crate::ffi::computePlanDeviceCounts(path, platform);
    if counts.len() != 4 {
        return None;
    }
    Some(ComputePlanDeviceCounts {
        total: counts[0],
        ane: counts[1],
        gpu: counts[2],
        cpu: counts[3],
    })
}

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
                        // Same defect as the batch loader's cached path: a
                        // cache file can decompress cleanly and still contain a
                        // model CoreML rejects. The Buffer branch above checks
                        // `failed()`; this one must too, or a broken model is
                        // reported as `Loaded`.
                        if coreml_model.model.failed() {
                            return Err(CoreMLError::FailedToLoad(
                                "Failed to load model from cached buffer path; likely not a CoreML mlmodel file".to_string(),
                                Self::Unloaded(info, loader),
                            ));
                        }
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

    /// Returns the shape of every input feature, or `Err(ModelNotLoaded)`
    /// when the model has not been loaded yet.
    pub fn input_shapes(&self) -> Result<HashMap<String, Vec<usize>>, CoreMLError> {
        Ok(self.description()?.input_shapes())
    }

    /// Returns the shape of every output feature, or `Err(ModelNotLoaded)`
    /// when the model has not been loaded yet.
    pub fn output_shapes(&self) -> Result<HashMap<String, Vec<usize>>, CoreMLError> {
        Ok(self.description()?.output_shapes())
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

    /// Predict with a retry policy. Re-runs `predict()` up to
    /// `options.max_retries` additional times on failure, sleeping between
    /// attempts according to `options.backoff`.
    pub fn predict_with_retry(
        &mut self,
        options: PredictRetryOptions,
    ) -> Result<MLModelOutput, CoreMLError> {
        self.predict_with_retry_if(options, |_| true)
    }

    /// Like [`predict_with_retry`](Self::predict_with_retry), but only
    /// retries when `should_retry` returns `true` for the failure.
    ///
    /// See [`CoreMLModel::predict_with_retry_if`] for why a model with
    /// inputs needs [`predict_with_rebind_retry`](Self::predict_with_rebind_retry)
    /// instead: a failed predict clears the bindings, so retrying cannot
    /// restore them from inside this type.
    pub fn predict_with_retry_if(
        &mut self,
        options: PredictRetryOptions,
        should_retry: impl FnMut(&CoreMLError) -> bool,
    ) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.predict_with_retry_if(options, should_retry)
            }
        }
    }

    /// Re-bind inputs and predict, retrying the whole cycle.
    ///
    /// The loaded-model case delegates to
    /// [`CoreMLModel::predict_with_rebind_retry`]; the unloaded case
    /// reports `ModelNotLoaded` rather than invoking the closure against
    /// a model that does not exist.
    pub fn predict_with_rebind_retry(
        &mut self,
        options: PredictRetryOptions,
        rebind_and_predict: impl FnMut(&mut CoreMLModel) -> Result<MLModelOutput, CoreMLError>,
    ) -> Result<MLModelOutput, CoreMLError> {
        self.predict_with_rebind_retry_if(options, |_| true, rebind_and_predict)
    }

    /// [`predict_with_rebind_retry`](Self::predict_with_rebind_retry) with
    /// a caller-supplied retry predicate.
    pub fn predict_with_rebind_retry_if(
        &mut self,
        options: PredictRetryOptions,
        should_retry: impl FnMut(&CoreMLError) -> bool,
        mut rebind_and_predict: impl FnMut(&mut CoreMLModel) -> Result<MLModelOutput, CoreMLError>,
    ) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.predict_with_rebind_retry_if(options, should_retry, rebind_and_predict)
            }
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

    /// Per-device preferred-operation counts for this model's compute plan
    /// under the compute platform it was loaded with (macOS 14.4+).
    ///
    /// `None` when the model is unloaded, has no compiled path, or the OS
    /// cannot produce a plan. See [`compute_plan_device_counts`].
    pub fn compute_plan_device_counts(&self) -> Option<ComputePlanDeviceCounts> {
        match self {
            CoreMLModelWithState::Loaded(core_mlmodel, info, _) => {
                let path = core_mlmodel.model.compiled_path()?;
                // A prediction-time CPU-only override beats the load-time
                // compute platform — report what predictions actually use.
                let platform = if info.opts.prediction_uses_cpu_only == Some(true) {
                    crate::ffi::ComputePlatform::Cpu
                } else {
                    info.opts.compute_platform
                };
                compute_plan_device_counts(path, platform)
            }
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
    // NOTE: there is deliberately no `_model_asset_buffer` field here.
    // Buffer-backed models transfer their allocation to Swift, whose
    // `Data(bytesNoCopy:deallocator:)` frees it via `rust_vec_free_u8` — see
    // `CoreMLModel::load_buffer`. Retaining the `Vec` here as well gave the
    // allocation two owners and double-freed it.
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
        if let Some(disabled) = opts.disable_experimental_mle {
            model.setDisableExperimentalMLE(disabled);
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
        }
    }

    /// Load a model from an in-memory `.mlmodel` buffer.
    ///
    /// # Buffer ownership
    ///
    /// The allocation is *transferred* to Swift; it has exactly one owner at
    /// every point and is never owned on both sides at once.
    ///
    /// 1. Before the call, `buf` owns the allocation.
    /// 2. `modelWithAssets` is the Swift `initWithCompiledAsset`, which
    ///    immediately wraps the pointer in `Data(bytesNoCopy:deallocator:)`
    ///    with a custom deallocator that calls `rust_vec_free_u8`. That
    ///    `Data` is constructed *before* the `do`/`catch` around
    ///    `MLModelAsset(specification:)`, so Swift takes ownership on both
    ///    the success and the load-failure path — there is no path on which
    ///    Swift declines the buffer and hands it back.
    /// 3. `Box::into_raw` therefore leaks the allocation on the Rust side on
    ///    purpose: from here on Swift is the sole owner, and `rust_vec_free_u8`
    ///    is the sole free.
    ///
    /// Consequently the model must **not** also retain the buffer in a field.
    /// Doing so gave the allocation two owners — Swift's deallocator and the
    /// retained `Vec` — and every buffer-backed model double-freed once both
    /// sides dropped.
    ///
    /// The buffer is normalized to a boxed slice first because
    /// `rust_vec_free_u8(ptr, len)` reconstructs `Vec::from_raw_parts(ptr,
    /// len, len)`, i.e. it assumes capacity equals length. A `Vec` with spare
    /// capacity would otherwise be deallocated under a smaller layout than it
    /// was allocated with (and a `Vec::with_capacity(n)` that is still empty
    /// would leak all `n` bytes). `into_boxed_slice` shrinks to fit,
    /// reallocating only when there is spare capacity, so `len == capacity`
    /// holds for the pointer Swift receives.
    pub fn load_buffer(buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let buf = buf.into_boxed_slice();
        let len = buf.len();
        // Ownership moves to Swift here — see the note above.
        let ptr = Box::into_raw(buf) as *mut u8;

        let model = Self::apply_options(
            modelWithAssets(ptr, len as isize, info.opts.compute_platform),
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
            MLArray::UInt16Array(array_base) => {
                // `From<ArrayD<u16>>` produces this variant, and callers use it
                // to hand over raw f16 bit patterns. It shares the Swift
                // binding with Float16Array (`bindInputU16` builds an
                // `MLMultiArrayDataType.float16` array), so route it the same
                // way — dropping this arm silently broke every existing u16
                // caller with an "unsupported input" error.
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
                        "failed to bind u16 input to model".to_string(),
                    ));
                }
                // Swift's MLMultiArray deallocator now owns this buffer.
                std::mem::forget(data_bytes);
            }
            _ => {
                return Err(CoreMLError::BadInputShape(format!(
                    "unsupported input type for '{}': only f32, f16, i32, u16, and CVPixelBuffer inputs are supported",
                    tag.as_ref()
                )));
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

    /// Run `predict` under a retry policy. Each retry re-runs the full
    /// inner predict path (including output buffer rebinding), so
    /// transient CoreML/ANE errors are cleared before the next attempt.
    pub fn predict_with_retry(
        &mut self,
        options: PredictRetryOptions,
    ) -> Result<MLModelOutput, CoreMLError> {
        self.predict_with_retry_if(options, |_| true)
    }

    /// Retry a predict that needs no input re-binding, retrying only when
    /// `should_retry` accepts the failure.
    ///
    /// A failed `predict()` clears the Swift-side input dictionary
    /// (`clearBindings()` in the `predict()` catch) and the Rust-side
    /// `iosurface_bound_outputs` set, so this method cannot re-bind
    /// anything -- `CoreMLModel` does not retain input data, because
    /// `add_input` moves ownership into Swift's `self.dict`.
    ///
    /// That makes this method correct only where the retry does not need
    /// the inputs back: models with no inputs, and models whose failure
    /// path leaves the bindings installed. For a model with inputs, use
    /// [`predict_with_rebind_retry`](Self::predict_with_rebind_retry),
    /// which re-binds through a caller-supplied closure between attempts.
    /// Retrying a bound-input model with this method will fail
    /// identically on every attempt.
    pub fn predict_with_retry_if(
        &mut self,
        options: PredictRetryOptions,
        should_retry: impl FnMut(&CoreMLError) -> bool,
    ) -> Result<MLModelOutput, CoreMLError> {
        retry_with_backoff(options, should_retry, |_| self.predict())
    }

    /// Re-bind inputs and predict, retrying the whole cycle.
    ///
    /// `rebind_and_predict` is called once per attempt and is expected to
    /// re-install the model's inputs before calling
    /// [`predict`](Self::predict). Use this for any model that takes
    /// inputs: the first attempt's failure clears the bindings, so a
    /// retry that only re-runs `predict()` would run with an empty input
    /// dictionary and fail the same way every time.
    ///
    /// ```no_run
    /// # use coreml_rs_fork::{CoreMLModelWithState, PredictRetryOptions, CoreMLError};
    /// # fn demo(mut model: CoreMLModelWithState, tensor: ndarray::ArrayD<f32>) -> Result<(), CoreMLError> {
    /// let options = PredictRetryOptions::fixed(3, std::time::Duration::from_millis(50));
    /// let out = model.predict_with_rebind_retry(options, |model| {
    ///     model.add_input("image", tensor.clone())?;
    ///     model.predict()
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn predict_with_rebind_retry(
        &mut self,
        options: PredictRetryOptions,
        rebind_and_predict: impl FnMut(&mut Self) -> Result<MLModelOutput, CoreMLError>,
    ) -> Result<MLModelOutput, CoreMLError> {
        self.predict_with_rebind_retry_if(options, |_| true, rebind_and_predict)
    }

    /// [`predict_with_rebind_retry`](Self::predict_with_rebind_retry) with
    /// a caller-supplied retry predicate.
    pub fn predict_with_rebind_retry_if(
        &mut self,
        options: PredictRetryOptions,
        should_retry: impl FnMut(&CoreMLError) -> bool,
        mut rebind_and_predict: impl FnMut(&mut Self) -> Result<MLModelOutput, CoreMLError>,
    ) -> Result<MLModelOutput, CoreMLError> {
        retry_with_backoff(options, should_retry, |_| rebind_and_predict(self))
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

#[cfg(test)]
mod retry_backoff_tests {
    use super::*;

    #[test]
    fn none_backoff_is_always_zero() {
        let backoff = RetryBackoff::None;
        assert_eq!(backoff.delay(0), Duration::ZERO);
        assert_eq!(backoff.delay(1000), Duration::ZERO);
    }

    #[test]
    fn fixed_backoff_ignores_index() {
        let backoff = RetryBackoff::Fixed(Duration::from_millis(25));
        assert_eq!(backoff.delay(0), Duration::from_millis(25));
        assert_eq!(backoff.delay(7), Duration::from_millis(25));
    }

    #[test]
    fn exponential_backoff_grows_then_clamps() {
        let backoff = RetryBackoff::Exponential {
            initial: Duration::from_millis(10),
            multiplier: 2.0,
            max: Duration::from_millis(100),
        };
        assert_eq!(backoff.delay(0), Duration::from_millis(10));
        assert_eq!(backoff.delay(1), Duration::from_millis(20));
        assert_eq!(backoff.delay(2), Duration::from_millis(40));
        // Clamped at max from here on.
        assert_eq!(backoff.delay(3), Duration::from_millis(80));
        assert_eq!(backoff.delay(4), Duration::from_millis(100));
        assert_eq!(backoff.delay(50), Duration::from_millis(100));
    }

    #[test]
    fn exponential_backoff_multiplier_below_one_is_treated_as_one() {
        let backoff = RetryBackoff::Exponential {
            initial: Duration::from_millis(10),
            multiplier: 0.25,
            max: Duration::from_millis(100),
        };
        // multiplier.max(1.0) => 1.0, so no decay.
        assert_eq!(backoff.delay(0), Duration::from_millis(10));
        assert_eq!(backoff.delay(3), Duration::from_millis(10));
    }

    /// Regression: `Duration::mul_f64` panics on a non-finite factor.
    /// `powi` overflows to infinity well before the retry index is
    /// exhausted, so the delay must saturate at `max` instead.
    #[test]
    fn exponential_backoff_saturates_instead_of_panicking() {
        let backoff = RetryBackoff::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 10.0,
            max: Duration::from_secs(30),
        };
        // 10^309 overflows f64 to infinity.
        assert_eq!(backoff.delay(309), Duration::from_secs(30));
        assert_eq!(backoff.delay(4096), Duration::from_secs(30));
        assert_eq!(backoff.delay(usize::MAX), Duration::from_secs(30));
    }

    /// Regression: the `retry_index as i32` cast wraps for indices above
    /// 2^31-1. A wrapped (possibly negative) exponent must still saturate
    /// at `max` rather than producing a nonsense small delay.
    #[test]
    fn exponential_backoff_handles_indices_above_i32_max() {
        let backoff = RetryBackoff::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::from_secs(60),
        };
        let above = i32::MAX as usize + 1;
        assert_eq!(backoff.delay(above), Duration::from_secs(60));
        assert_eq!(backoff.delay(usize::MAX), Duration::from_secs(60));
    }

    #[test]
    fn exponential_backoff_with_zero_max_never_sleeps() {
        let backoff = RetryBackoff::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::ZERO,
        };
        assert_eq!(backoff.delay(0), Duration::ZERO);
        assert_eq!(backoff.delay(10), Duration::ZERO);
    }

    // ---- retry_with_backoff: the attempt loop itself --------------------
    //
    // No CoreML model is involved here, so these pin the retry semantics
    // directly: how many times the operation runs, whether the predicate
    // gates a retry, and which error comes out.

    #[test]
    fn retry_with_backoff_succeeds_on_first_attempt_without_retrying() {
        let mut calls = 0;
        let out = retry_with_backoff(
            PredictRetryOptions::fixed(5, Duration::ZERO),
            |_| true,
            |_| {
                calls += 1;
                Ok::<_, CoreMLError>(42u32)
            },
        )
        .unwrap();
        assert_eq!(out, 42);
        assert_eq!(calls, 1, "must not retry after a success");
    }

    #[test]
    fn retry_with_backoff_retries_until_success_and_reports_attempt_count() {
        let mut calls = 0;
        let mut seen_attempts = Vec::new();
        let out = retry_with_backoff(
            PredictRetryOptions::fixed(5, Duration::ZERO),
            |_| true,
            |attempt| {
                seen_attempts.push(attempt);
                calls += 1;
                if calls < 3 {
                    Err(CoreMLError::UnknownError("transient".to_string()))
                } else {
                    Ok(calls as u32)
                }
            },
        )
        .unwrap();
        assert_eq!(out, 3);
        assert_eq!(calls, 3);
        // The closure is told which attempt it is, so a rebind
        // implementation can vary per attempt.
        assert_eq!(seen_attempts, vec![0, 1, 2]);
    }

    #[test]
    fn retry_with_backoff_stops_after_max_retries_and_propagates_final_error() {
        let mut calls = 0;
        let err = retry_with_backoff(
            PredictRetryOptions::fixed(3, Duration::ZERO),
            |_| true,
            |_| {
                calls += 1;
                Err::<u32, _>(CoreMLError::UnknownError(format!("fail {calls}")))
            },
        )
        .unwrap_err();
        // 1 initial attempt + max_retries.
        assert_eq!(calls, 4);
        assert_eq!(err.to_string(), "UnknownError: fail 4");
    }

    #[test]
    fn retry_with_backoff_predicate_can_short_circuit_retries() {
        let mut calls = 0;
        let err = retry_with_backoff(
            PredictRetryOptions::fixed(10, Duration::ZERO),
            // Only retry the first failure; a non-retryable error stops immediately.
            |err| matches!(err, CoreMLError::UnknownError(m) if m == "retryable"),
            |_| {
                calls += 1;
                Err::<u32, _>(CoreMLError::UnknownError("fatal".to_string()))
            },
        )
        .unwrap_err();
        assert_eq!(calls, 1, "a fatal error must not be retried");
        assert_eq!(err.to_string(), "UnknownError: fatal");
    }

    #[test]
    fn retry_with_backoff_zero_retries_runs_the_operation_once() {
        let mut calls = 0;
        let err = retry_with_backoff(
            PredictRetryOptions::none(),
            |_| true,
            |_| {
                calls += 1;
                Err::<u32, _>(CoreMLError::ModelNotLoaded)
            },
        )
        .unwrap_err();
        assert_eq!(calls, 1);
        assert!(matches!(err, CoreMLError::ModelNotLoaded));
    }

    /// The re-bind contract: a fresh attempt must re-establish whatever
    /// the previous failure tore down. The closure here models what
    /// `predict_with_rebind_retry` asks callers to do -- re-install
    /// inputs, then predict -- and asserts the retry actually re-ran it
    /// rather than predicting with stale (or empty) bindings.
    #[test]
    fn retry_with_backoff_reruns_the_whole_operation_per_attempt() {
        let mut bindings = Vec::<&'static str>::new();
        let mut predictions = 0;

        let out = retry_with_backoff(
            PredictRetryOptions::fixed(4, Duration::ZERO),
            |_| true,
            |_| {
                // Stand-in for `model.add_input(..)?`.
                bindings.push("image");
                predictions += 1;
                if predictions < 3 {
                    Err(CoreMLError::UnknownError("transient".to_string()))
                } else {
                    Ok(bindings.len())
                }
            },
        )
        .unwrap();

        assert_eq!(out, 3);
        assert_eq!(bindings.len(), 3, "each attempt must re-bind");
        assert_eq!(predictions, 3);
    }

    #[test]
    fn retry_with_backoff_applies_backoff_between_attempts_only() {
        // Zero-duration backoff keeps the test fast; this asserts the
        // loop structure (max_retries + 1 attempts) rather than wall time.
        let mut attempts = 0;
        let _ = retry_with_backoff(
            PredictRetryOptions::exponential(2, Duration::ZERO, 2.0, Duration::from_secs(1)),
            |_| true,
            |_| {
                attempts += 1;
                Err::<(), _>(CoreMLError::UnknownError("x".to_string()))
            },
        );
        assert_eq!(attempts, 3);
    }

    #[test]
    fn retry_with_backoff_handles_a_recovering_operation_with_exponential_backoff() {
        let start = std::time::Instant::now();
        let mut calls = 0;
        let out = retry_with_backoff(
            PredictRetryOptions::exponential(
                3,
                Duration::from_millis(5),
                2.0,
                Duration::from_millis(100),
            ),
            |_| true,
            |_| {
                calls += 1;
                if calls < 3 {
                    Err(CoreMLError::UnknownError("transient".to_string()))
                } else {
                    Ok("recovered")
                }
            },
        )
        .unwrap();
        assert_eq!(out, "recovered");
        // 5ms + 10ms of sleep, so at least 15ms must have elapsed.
        assert!(
            start.elapsed() >= Duration::from_millis(15),
            "backoff sleeps must actually be applied between attempts"
        );
    }

    #[test]
    fn retry_options_constructors_match_their_backoff() {
        assert_eq!(PredictRetryOptions::none().max_retries, 0);
        assert_eq!(PredictRetryOptions::none().backoff, RetryBackoff::None);

        let fixed = PredictRetryOptions::fixed(3, Duration::from_millis(10));
        assert_eq!(fixed.max_retries, 3);
        assert_eq!(
            fixed.backoff,
            RetryBackoff::Fixed(Duration::from_millis(10))
        );

        let exp = PredictRetryOptions::exponential(
            5,
            Duration::from_millis(1),
            3.0,
            Duration::from_secs(2),
        );
        assert_eq!(exp.max_retries, 5);
        assert_eq!(
            exp.backoff,
            RetryBackoff::Exponential {
                initial: Duration::from_millis(1),
                multiplier: 3.0,
                max: Duration::from_secs(2),
            }
        );

        // Default is the no-retry policy, and `predict()` uses it.
        assert_eq!(PredictRetryOptions::default(), PredictRetryOptions::none());
        assert_eq!(RetryBackoff::default(), RetryBackoff::None);
    }
}
