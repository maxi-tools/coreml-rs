use crate::mlarray::MLArrayBaseExt;
use crate::{
    ffi::{modelWithAssets, modelWithPath, Model, ModelOutput},
    mlarray::MLArray,
};
use ndarray::Array;
use std::{
    collections::HashMap,
    path::Path,
};
use tempfile::NamedTempFile;

pub use crate::swift::MLModelOutput;

pub use crate::error::CoreMLError;

pub use crate::loader::CoreMLModelOptions;

pub use crate::loader::CoreMLModelLoader;

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
            CoreMLModelLoader::BufferToDisk(u) => match crate::utils::load_buffer_from_disk(&u) {
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
            },
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
                            match crate::utils::save_buffer_to_disk(&vec, &mut info.opts.cache_dir)
                            {
                                Ok(m) => CoreMLModelLoader::BufferToDisk(m),
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

pub use crate::loader::CoreMLModelInfo;

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
        Self {
            model: modelWithPath(path, info.opts.compute_platform, compiled),
            outputs: Default::default(),
        }
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
        crate::utils::validate_coreml_shape(&arr, &shape, &name)?;
        match input {
            MLArray::Float32Array(array_base) => {
                // Ensure C-contiguous layout before extracting raw data,
                // since bindInput assumes C-contiguous strides.
                let mut data = array_base.into_contiguous_raw_vec();
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
                let mut data = array_base.into_contiguous_raw_vec();
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
                let mut data = array_base.into_contiguous_raw_vec();
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
                let mut data = array_base.into_contiguous_raw_vec();
                if !self
                    .model
                    .bindInputU16(shape, name, data.as_mut_ptr(), data.capacity())
                {
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
        self.outputs
            .insert(tag.as_ref().to_string(), ("f32", arr.shape().to_vec()));
        let (mut data, shape) = arr.into_contiguous_raw_vec_and_shape::<f32>();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        let res = self.model.bindOutputF32(shape, name, ptr, len);
        std::mem::forget(data);
        res
    }

    pub fn add_output_u16(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        self.outputs
            .insert(tag.as_ref().to_string(), ("f16", arr.shape().to_vec()));
        let (mut data, shape) = arr.into_contiguous_raw_vec_and_shape::<u16>();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        let res = self.model.bindOutputU16(shape, name, ptr, len);
        std::mem::forget(data);
        res
    }

    pub fn add_output_i32(&mut self, tag: impl AsRef<str>, out: impl Into<MLArray>) -> bool {
        let arr: MLArray = out.into();
        self.outputs
            .insert(tag.as_ref().to_string(), ("i32", arr.shape().to_vec()));
        let (mut data, shape) = arr.into_contiguous_raw_vec_and_shape::<i32>();
        let name = tag.as_ref().to_string();
        let ptr = data.as_mut_ptr();
        let len = data.capacity();
        let res = self.model.bindOutputI32(shape, name, ptr, len);
        std::mem::forget(data);
        res
    }

    /// Shared output setup: inspect model description, bind output buffers.
    /// `allow_empty_shape` controls whether empty shapes also disable output backing
    /// (needed for stateful models where shapes may be unknown until runtime).
    fn setup_outputs(
        &mut self,
        allow_empty_shape: bool,
    ) -> Result<(bool, Vec<(String, Vec<usize>, String)>), CoreMLError> {
        let desc = self.model.description();
        let mut use_output_backing = true;
        let mut output_info = Vec::new();

        for name in desc.output_names() {
            let output_shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());

            let is_flexible = output_shape.iter().any(|&dim| dim == 0);
            let is_empty = allow_empty_shape && output_shape.is_empty();
            if is_flexible || is_empty {
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
        Ok((use_output_backing, output_info))
    }

    /// Extract flexible outputs from a CoreML prediction result.
    fn extract_flexible_outputs(
        output_info: Vec<(String, Vec<usize>, String)>,
        output: &ModelOutput,
    ) -> HashMap<String, MLArray> {
        let mut outputs = HashMap::new();
        for (name, _output_shape, ty) in output_info {
            let actual_shape: Vec<usize> = output.outputShape(name.clone()).into_iter().collect();
            if actual_shape.is_empty() {
                eprintln!("warning: output '{}' has no shape data", name);
                continue;
            }

            match ty.as_str() {
                "f32" | "bool" | "boolean" => {
                    let out = output.outputF32(name.clone());
                    if out.is_empty() {
                        eprintln!("warning: output '{}' returned empty data", name);
                        continue;
                    }
                    if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                        outputs.insert(name, array.into());
                    }
                }
                "f16" | "float16" => {
                    let out = output.outputU16(name.clone());
                    if out.is_empty() {
                        eprintln!("warning: output '{}' returned empty data", name);
                        continue;
                    }
                    if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                        outputs.insert(name, reinterpret_u16_to_f16(array).into());
                    }
                }
                "int32" | "int64" | "int16" | "uint32" | "uint64" | "uint16" => {
                    let out = output.outputI32(name.clone());
                    if out.is_empty() {
                        eprintln!("warning: output '{}' returned empty data", name);
                        continue;
                    }
                    if let Ok(array) = Array::from_shape_vec(ndarray::IxDyn(&actual_shape), out) {
                        outputs.insert(name, array.into());
                    }
                }
                _ => {
                    eprintln!("warning: type not supported, and will be skipped in the output");
                }
            }
        }
        outputs
    }

    /// Collect fixed-size outputs from pre-allocated buffers.
    fn collect_fixed_outputs(&self, output: &ModelOutput) -> HashMap<String, MLArray> {
        self.outputs
            .clone()
            .into_iter()
            .filter_map(|(key, (ty, shape))| {
                let name = key.clone();
                match ty {
                    "f32" => {
                        let out = output.outputF32(name);
                        Array::from_shape_vec(shape, out)
                            .ok()
                            .map(|a| (key, a.into()))
                    }
                    "f16" => {
                        let out = output.outputU16(name);
                        Array::from_shape_vec(shape, out)
                            .ok()
                            .map(|a| (key, reinterpret_u16_to_f16(a).into()))
                    }
                    "i32" => {
                        let out = output.outputI32(name);
                        Array::from_shape_vec(shape, out)
                            .ok()
                            .map(|a| (key, a.into()))
                    }
                    _ => None,
                }
            })
            .collect()
    }

    pub fn predict(&mut self) -> Result<MLModelOutput, CoreMLError> {
        let (use_output_backing, output_info) = self.setup_outputs(false)?;
        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let outputs = if !use_output_backing {
            Self::extract_flexible_outputs(output_info, &output)
        } else {
            self.collect_fixed_outputs(&output)
        };
        Ok(MLModelOutput { outputs })
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
        let (use_output_backing, output_info) = self.setup_outputs(true)?;
        let output = self.model.predictWithState();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let outputs = if !use_output_backing {
            Self::extract_flexible_outputs(output_info, &output)
        } else {
            self.collect_fixed_outputs(&output)
        };
        Ok(MLModelOutput { outputs })
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
