use crate::{
    ffi::{modelWithAssets, modelWithPath, ComputePlatform, Model},
    mlarray::MLArray,
    mlbatchmodel::CoreMLBatchModelWithState,
};
use flate2::Compression;
use ndarray::Array;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

pub use crate::swift::MLModelOutput;

use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreMLError {
    #[error("CoreML Cache IoError: {0}")]
    IoError(std::io::Error),
    #[error("BadInputShape: {0}")]
    BadInputShape(String),
    // #[error("Lz4 Decompression Error: {0}")]
    // Lz4DecompressError(DecompressError),
    #[error("UnknownError: {0}")]
    UnknownError(String),
    #[error("UnknownError: {0}")]
    UnknownErrorStatic(&'static str),
    #[error("ModelNotLoaded: coreml model not loaded into session")]
    ModelNotLoaded,
    #[error("FailedToLoad: coreml model couldn't be loaded: {0}")]
    FailedToLoadStatic(&'static str, CoreMLModelWithState),
    #[error("FailedToLoad: coreml model couldn't be loaded: {0}")]
    FailedToLoad(String, CoreMLModelWithState),
    #[error("FailedToLoadBatch: coreml model couldn't be loaded: {0}")]
    FailedToLoadBatchStatic(&'static str, CoreMLBatchModelWithState),
    #[error("FailedToLoadBatch: coreml model couldn't be loaded: {0}")]
    FailedToBatchLoad(String, CoreMLBatchModelWithState),
}

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

#[derive(Debug)]
pub enum CoreMLModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl CoreMLModelWithState {
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
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                let mut coreml_model = CoreMLModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; model path not valid",
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
                let mut coreml_model =
                    CoreMLModel::load_from_path(path_buf.display().to_string(), info.clone(), true);
                if !coreml_model.model.load() {
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; compiled model cache got purged",
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
                    return Err(CoreMLError::FailedToLoadStatic(
                        "Failed to load model; likely not a CoreML mlmodel file",
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::BufferToDisk(u) => {
                match std::fs::File::open(&u)
                    .map_err(|io| CoreMLError::IoError(io))
                    .and_then(|file| {
                        let mut vec = vec![];
                        _ = flate2::read::ZlibDecoder::new(file)
                            .read_to_end(&mut vec)
                            .map_err(|io| CoreMLError::IoError(io))?;
                        Ok(vec)
                    }) {
                    Ok(vec) => {
                        let mut coreml_model = CoreMLModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => Err(CoreMLError::FailedToLoad(
                        format!("failed to load the model from cached buffer path: {err}"),
                        CoreMLModelWithState::Unloaded(info, CoreMLModelLoader::BufferToDisk(u)),
                    )),
                }
            }
        }
    }

    /// Might fail irrecoverably if the system is too low on disk space(very unlikely)
    pub fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(model, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => {
                        let mut temp_file = NamedTempFile::new().map_err(CoreMLError::IoError)?;
                        temp_file.write_all(&v).map_err(CoreMLError::IoError)?;
                        let res = std::fs::read(temp_file.path()).map_err(CoreMLError::IoError)?;
                        CoreMLModelLoader::Buffer(res)
                    }
                    CoreMLModelLoader::ModelPath(_) => {
                        // if the model is loaded from modelPath it has to have compiled path
                        let path = model.model.compiled_path().unwrap();
                        CoreMLModelLoader::CompiledPath(path.into())
                    }
                    x => x,
                },
            ))
        } else {
            Ok(self)
        }
    }

    /// Unloads the model buffer to the disk, at cache_dir
    pub fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, mut info, loader) | Self::Unloaded(mut info, loader) => {
                let loader = {
                    match loader {
                        CoreMLModelLoader::Buffer(vec) => {
                            if info.opts.cache_dir.as_os_str().is_empty() {
                                info.opts.cache_dir = PathBuf::from(".");
                            }
                            if !info.opts.cache_dir.exists() {
                                _ = std::fs::remove_dir_all(&info.opts.cache_dir);
                                _ = std::fs::create_dir_all(&info.opts.cache_dir);
                            }
                            // pick the file specified, if it's a folder/dir append model_cache
                            let m = if !info.opts.cache_dir.is_dir() {
                                info.opts.cache_dir.clone()
                            } else {
                                info.opts.cache_dir.join("model_cache")
                            };
                            match std::fs::File::create(&m)
                                .map_err(|io| CoreMLError::IoError(io))
                                .map(|file| {
                                    flate2::write::ZlibEncoder::new(file, Compression::best())
                                        .write_all(&vec)
                                        .map_err(CoreMLError::IoError)
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

    pub fn description(&self) -> Result<HashMap<&str, Vec<String>>, CoreMLError> {
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

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }

    /// Initialize CoreML State for stateful models (macOS 15+).
    pub fn make_state(&mut self) -> Result<(), CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => {
                if core_mlmodel.make_state() {
                    Ok(())
                } else {
                    Err(CoreMLError::UnknownErrorStatic(
                        "make_state failed (model may not support state or macOS < 15)",
                    ))
                }
            }
        }
    }

    /// Run prediction using CoreML State.
    pub fn predict_with_state(&mut self) -> Result<MLModelOutput, CoreMLError> {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict_with_state(),
        }
    }

    /// Check if this model has an active CoreML State.
    pub fn has_state(&self) -> bool {
        match self {
            CoreMLModelWithState::Unloaded(_, _) => false,
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.has_state(),
        }
    }

    /// Reset CoreML State.
    pub fn reset_state(&mut self) {
        if let CoreMLModelWithState::Loaded(core_mlmodel, _, _) = self {
            core_mlmodel.reset_state();
        }
    }

    /// Get the compiled model path (if available after loading from .mlpackage).
    pub fn compiled_path(&self) -> Option<String> {
        match self {
            CoreMLModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.model.compiled_path(),
            _ => None,
        }
    }
}

// Info required to create a coreml model
#[derive(Debug, Clone)]
pub struct CoreMLModelInfo {
    pub opts: CoreMLModelOptions,
}

#[derive(Debug)]
pub struct CoreMLModel {
    model: Model,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLModel {}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model").finish()
    }
}

impl CoreMLModel {
    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        let coreml_model = Self {
            model: modelWithPath(path, info.opts.compute_platform, compiled),
            outputs: Default::default(),
        };
        coreml_model
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let coreml_model = Self {
            model: modelWithAssets(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            outputs: Default::default(),
        };
        std::mem::forget(buf);
        coreml_model
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
    ) -> Result<(), CoreMLError> {
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let desc = self.model.description();
        let shape: Vec<usize> = input.shape().to_vec();
        let arr = desc.input_shape(name.clone());
        if arr.is_empty() && !shape.is_empty() {
            return Err(CoreMLError::BadInputShape(format!(
                "Input feature name '{name}' not expected!"
            )));
        }
        // Flexible shape matching: 0 means any dimension (ct.RangeDim)
        if arr.len() != shape.len()
            || !arr
                .iter()
                .zip(shape.iter())
                .all(|(&c, &a)| c == 0 || c == a)
        {
            return Err(CoreMLError::BadInputShape(format!(
                "expected shape {arr:?} found {shape:?}"
            )));
        }
        match input {
            MLArray::Float32Array(array_base) => {
                // Ensure C-contiguous layout before extracting raw data,
                // since bindInput assumes C-contiguous strides.
                let contiguous = if array_base.is_standard_layout() {
                    array_base
                } else {
                    array_base.as_standard_layout().into_owned()
                };
                let (mut data, offset) = contiguous.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self
                    .model
                    .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity())
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            MLArray::Float16Array(array_base) => {
                // Ensure C-contiguous layout before extracting raw data,
                // since bindInput assumes C-contiguous strides.
                let contiguous = if array_base.is_standard_layout() {
                    array_base
                } else {
                    array_base.as_standard_layout().into_owned()
                };
                let (mut data, offset) = contiguous.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self.model.bindInputU16(
                    shape,
                    name,
                    data.as_mut_ptr() as *mut u16,
                    data.capacity(),
                ) {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            MLArray::Int32Array(array_base) => {
                // Ensure C-contiguous layout before extracting raw data,
                // since bindInput assumes C-contiguous strides.
                let contiguous = if array_base.is_standard_layout() {
                    array_base
                } else {
                    array_base.as_standard_layout().into_owned()
                };
                let (mut data, offset) = contiguous.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self
                    .model
                    .bindInputI32(shape, name, data.as_mut_ptr(), data.capacity())
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            MLArray::UInt16Array(array_base) => {
                // UInt16 uses the same Swift binding as Float16 (bindInputU16)
                let contiguous = if array_base.is_standard_layout() {
                    array_base
                } else {
                    array_base.as_standard_layout().into_owned()
                };
                let (mut data, offset) = contiguous.into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                if !self.model.bindInputU16(
                    shape,
                    name,
                    data.as_mut_ptr(),
                    data.capacity(),
                ) {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind u16 input to model",
                    ));
                }
                std::mem::forget(data);
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
        let expected_len = width * height * 4; // 4 bytes per pixel (BGRA)

        if bgra_data.len() != expected_len {
            return Err(CoreMLError::BadInputShape(format!(
                "Expected {} bytes for {}x{} BGRA image, got {}",
                expected_len,
                width,
                height,
                bgra_data.len()
            )));
        }

        let mut data = bgra_data;
        if !self.model.bindInputCVPixelBuffer(
            width,
            height,
            name,
            data.as_mut_ptr(),
            data.capacity(),
        ) {
            return Err(CoreMLError::UnknownErrorStatic(
                "failed to bind CVPixelBuffer input to model",
            ));
        }
        std::mem::forget(data);
        Ok(())
    }

    pub fn add_output_f32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", shape.to_vec()));
        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let (mut data, offset) = arr.extract_to_tensor::<f32>().into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        if !self.model.bindOutputF32(shape, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
    }

    pub fn add_output_u16(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f16", shape.to_vec()));
        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let (mut data, offset) = arr.extract_to_tensor::<u16>().into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        if !self.model.bindOutputU16(shape, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
    }

    pub fn add_output_i32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        let shape = arr.shape();
        self.outputs
            .insert(tag.as_ref().to_string(), ("i32", shape.to_vec()));
        let shape: Vec<i32> = shape.into_iter().map(|i| *i as i32).collect();
        let (mut data, offset) = arr.extract_to_tensor::<i32>().into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned output buffer"
        );
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        if !self.model.bindOutputI32(shape, name, ptr, len) {
            return false;
        }
        std::mem::forget(data);
        true
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        let desc = self.model.description();

        // Check if we should use output backing (only for fixed-size outputs)
        let mut use_output_backing = true;
        let mut output_info = Vec::new();

        for name in desc.output_names() {
            let output_shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());

            // Skip output backing if any dimension is 0 (flexible size)
            if output_shape.iter().any(|&dim| dim == 0) {
                use_output_backing = false;
            }

            output_info.push((name, output_shape, ty));
        }

        // Only set up output backing for fixed-size outputs
        if use_output_backing {
            for (name, output_shape, ty) in &output_info {
                match ty.as_str() {
                    "f32" => {
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
                    "bool" | "boolean" => {
                        // Cast boolean to f32 for output
                        self.add_output_f32(
                            name.clone(),
                            Array::<f32, _>::zeros(output_shape.clone()),
                        );
                    }
                    _ => {
                        return Err(CoreMLError::UnknownErrorStatic(
                            "non-f32/f16/i32 output types are not supported (yet)!",
                        ));
                    }
                }
            }
        }

        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }

        // For flexible outputs, extract directly from Core ML output
        if !use_output_backing {
            let mut outputs = HashMap::new();
            for (name, _output_shape, ty) in output_info {
                match ty.as_str() {
                    "f32" => {
                        let actual_shape = output.outputShape(name.clone());
                        let out = output.outputF32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            eprintln!(
                                "warning: output '{}' has no shape or data (shape={:?}, len={})",
                                &name,
                                actual_shape,
                                out.len()
                            );
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    "f16" | "float16" => {
                        let actual_shape = output.outputShape(name.clone());
                        let out = output.outputU16(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            eprintln!(
                                "warning: output '{}' has no shape or data (shape={:?}, len={})",
                                &name,
                                actual_shape,
                                out.len()
                            );
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            let f16_array = reinterpret_u16_to_f16(array);
                            outputs.insert(name, f16_array.into());
                        }
                    }
                    "int32" | "int64" | "int16" | "uint32" | "uint64" | "uint16" => {
                        let actual_shape = output.outputShape(name.clone());
                        let out = output.outputI32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            eprintln!(
                                "warning: output '{}' has no shape or data (shape={:?}, len={})",
                                &name,
                                actual_shape,
                                out.len()
                            );
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    "bool" | "boolean" => {
                        let actual_shape = output.outputShape(name.clone());
                        let out = output.outputF32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            eprintln!(
                                "warning: output '{}' has no shape or data (shape={:?}, len={})",
                                &name,
                                actual_shape,
                                out.len()
                            );
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    _ => {
                        eprintln!("warning: type not supported, and will be skipped in the output");
                    }
                }
            }
            return Ok(MLModelOutput { outputs });
        }

        // For fixed-size outputs, use the pre-allocated buffers
        Ok(MLModelOutput {
            outputs: self
                .outputs
                .clone()
                .into_iter()
                .filter_map(|(key, (ty, shape))| {
                    let name = key.clone();
                    match ty {
                        "f32" => {
                            let out = output.outputF32(name);
                            let array = Array::from_shape_vec(shape, out).ok()?;
                            Some((key, array.into()))
                        }
                        "f16" => {
                            let out = output.outputU16(name);
                            let array =
                                reinterpret_u16_to_f16(Array::from_shape_vec(shape, out).ok()?);
                            Some((key, array.into()))
                        }
                        "i32" => {
                            let out = output.outputI32(name);
                            let array = Array::from_shape_vec(shape, out).ok()?;
                            Some((key, array.into()))
                        }
                        _ => {
                            eprintln!(
                                "warning: type not supported, and will be skipped in the output"
                            );
                            return None;
                        }
                    }
                })
                .collect(),
        })
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.description();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }

    /// Initialize CoreML State (MLState) for stateful models.
    /// Returns true if the model supports state (macOS 15+, stateful .mlpackage).
    pub fn make_state(&mut self) -> bool {
        self.model.makeState()
    }

    /// Run prediction using CoreML State (KV cache managed in-place by CoreML).
    pub fn predict_with_state(&mut self) -> Result<MLModelOutput, CoreMLError> {
        let desc = self.model.description();

        let mut use_output_backing = true;
        let mut output_info = Vec::new();

        for name in desc.output_names() {
            let output_shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());
            // Empty shape (e.g. stateful models) or flexible dims → skip output backing
            if output_shape.is_empty() || output_shape.iter().any(|&dim| dim == 0) {
                use_output_backing = false;
            }
            output_info.push((name, output_shape, ty));
        }

        if use_output_backing {
            for (name, output_shape, ty) in &output_info {
                match ty.as_str() {
                    "f32" => {
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
                    "bool" | "boolean" => {
                        // Cast boolean to f32 for output
                        self.add_output_f32(
                            name.clone(),
                            Array::<f32, _>::zeros(output_shape.clone()),
                        );
                    }
                    _ => {
                        return Err(CoreMLError::UnknownErrorStatic(
                            "non-f32/f16/i32 output types are not supported (yet)!",
                        ));
                    }
                }
            }
        }

        let output = self.model.predictWithState();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }

        if !use_output_backing {
            let mut outputs = HashMap::new();
            for (name, _output_shape, ty) in output_info {
                match ty.as_str() {
                    "f32" => {
                        let actual_shape = output.outputShape(name.clone());
                        let actual_shape: Vec<usize> = actual_shape.into_iter().collect();
                        let out = output.outputF32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    "f16" | "float16" => {
                        let actual_shape = output.outputShape(name.clone());
                        let actual_shape: Vec<usize> = actual_shape.into_iter().collect();
                        let out = output.outputU16(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            let f16_array = reinterpret_u16_to_f16(array);
                            outputs.insert(name, f16_array.into());
                        }
                    }
                    "int32" | "int64" | "int16" | "uint32" | "uint64" | "uint16" => {
                        let actual_shape = output.outputShape(name.clone());
                        let actual_shape: Vec<usize> = actual_shape.into_iter().collect();
                        let out = output.outputI32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    "bool" | "boolean" => {
                        let actual_shape = output.outputShape(name.clone());
                        let actual_shape: Vec<usize> = actual_shape.into_iter().collect();
                        let out = output.outputF32(name.clone());
                        if actual_shape.is_empty() || out.is_empty() {
                            continue;
                        }
                        if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out)
                        {
                            outputs.insert(name, array.into());
                        }
                    }
                    _ => {}
                }
            }
            return Ok(MLModelOutput { outputs });
        }

        Ok(MLModelOutput {
            outputs: self
                .outputs
                .clone()
                .into_iter()
                .filter_map(|(key, (ty, shape))| {
                    let name = key.clone();
                    match ty {
                        "f32" => {
                            let out = output.outputF32(name);
                            let array = Array::from_shape_vec(shape, out).ok()?;
                            Some((key, array.into()))
                        }
                        "f16" => {
                            let out = output.outputU16(name);
                            let array =
                                reinterpret_u16_to_f16(Array::from_shape_vec(shape, out).ok()?);
                            Some((key, array.into()))
                        }
                        "i32" => {
                            let out = output.outputI32(name);
                            let array = Array::from_shape_vec(shape, out).ok()?;
                            Some((key, array.into()))
                        }
                        _ => None,
                    }
                })
                .collect(),
        })
    }

    /// Check if this model has an active CoreML State.
    pub fn has_state(&self) -> bool {
        self.model.hasState()
    }

    /// Reset CoreML State (e.g., for a new conversation).
    pub fn reset_state(&mut self) {
        self.model.resetState()
    }
}

fn reinterpret_u16_to_f16(input: ndarray::ArrayD<u16>) -> ndarray::ArrayD<half::f16> {
    let shape = input.shape().to_vec();
    let len = input.len();

    // Consume input and get the raw Vec<u32>
    let (raw_vec, offset) = input.into_raw_vec_and_offset();
    assert!(
        matches!(offset, Some(0) | None),
        "array base offset is not zero; bad aligned data reinterpret"
    );

    // SAFETY:
    // - u16 and f16 have the same size and alignment
    // - The underlying data is valid to reinterpret as f16
    // - This creates a new Vec<f16> with the same bytes
    let raw_vec_f16 = {
        let ptr = raw_vec.as_ptr() as *mut half::f16;
        let capacity = raw_vec.capacity();
        std::mem::forget(raw_vec); // prevent drop of original vec
        unsafe { Vec::from_raw_parts(ptr, len, capacity) }
    };

    match ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), raw_vec_f16) {
        Ok(array) => array,
        Err(err) => panic!("failed to rebuild f16 array from raw parts: {err}"),
    }
}
