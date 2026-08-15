//! Batch inference support for Core ML.
//!
//! This module provides `CoreMLBatchModelWithState`, which allows running inference
//! on multiple inputs in a single call, potentially improving throughput on hardware
//! like the Apple Neural Engine.

use crate::{
    ffi::{modelWithAssetsBatch, modelWithPathBatch, BatchModel},
    loader::CoreMLModelLoader,
    mlarray::MLArray,
    options::{CoreMLModelInfo, CoreMLModelOptions},
    swift::MLBatchModelOutput,
    CoreMLError,
};
use ndarray::Array;
use std::{
    collections::HashMap,
    io::{Read, Write},
    path::Path,
};
use tempfile::NamedTempFile;

pub use crate::swift::MLModelOutput;

/// A wrapper around a Core ML batch model that tracks its loading state.
///
/// Similar to `CoreMLModelWithState`, this enum enables flexible lifecycle
/// management for batch-capable models.
#[derive(Debug)]
pub enum CoreMLBatchModelWithState {
    /// The batch model is configured but not currently loaded.
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    /// The batch model is loaded and ready for inference.
    Loaded(CoreMLBatchModel, CoreMLModelInfo, CoreMLModelLoader),
}

impl crate::state::ModelState for CoreMLBatchModelWithState {
    type Model = CoreMLBatchModel;

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

    fn load(self) -> Result<Self, CoreMLError> {
        let Self::Unloaded(info, loader) = self else {
            return Ok(self);
        };
        match loader {
            CoreMLModelLoader::ModelPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    false,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::ModelPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatch(
                        "Failed to load model; likely not a CoreML model file".to_string(),
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::CompiledPath(path_buf) => {
                let mut coreml_model = CoreMLBatchModel::load_from_path(
                    path_buf.display().to_string(),
                    info.clone(),
                    true,
                );
                coreml_model.model.load();
                let loader = CoreMLModelLoader::CompiledPath(path_buf);
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatch(
                        "Failed to load model; likely not a CoreML model file".to_string(),
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLBatchModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatch(
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
                        let mut coreml_model = CoreMLBatchModel::load_buffer(vec, info.clone());
                        coreml_model.model.load();
                        let loader = CoreMLModelLoader::BufferToDisk(u);
                        Ok(Self::Loaded(coreml_model, info, loader))
                    }
                    Err(err) => Err(CoreMLError::FailedToLoadBatch(
                        format!("failed to load the model from cached buffer path: {err}"),
                        CoreMLBatchModelWithState::Unloaded(
                            info,
                            CoreMLModelLoader::BufferToDisk(u),
                        ),
                    )),
                }
            }
        }
    }

    /// Might fail if system disk space too low(very unlikely)
    fn unload(self) -> Result<Self, CoreMLError> {
        if let Self::Loaded(_, info, loader) = self {
            Ok(Self::Unloaded(
                info,
                match loader {
                    CoreMLModelLoader::Buffer(v) => {
                        let mut temp_file = NamedTempFile::new().map_err(CoreMLError::IoError)?;
                        temp_file.write_all(&v).map_err(CoreMLError::IoError)?;
                        CoreMLModelLoader::Buffer(
                            std::fs::read(temp_file.path()).map_err(CoreMLError::IoError)?,
                        )
                    }
                    x => x,
                },
            ))
        } else {
            Ok(self)
        }
    }
}

impl CoreMLBatchModelWithState {
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
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

    /// Adds an input feature to the batch at a specific index.
    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => {
                core_mlmodel.add_input(tag, input, idx)
            }
        }
    }

    /// Performs batch inference on all added inputs.
    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        match self {
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => core_mlmodel.predict(),
        }
    }
}

#[derive(Debug)]
pub struct CoreMLBatchModel {
    model: BatchModel,
    outputs: HashMap<String, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLBatchModel {}

impl std::fmt::Debug for BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchModel").finish()
    }
}

impl CoreMLBatchModel {
    fn apply_options(mut model: BatchModel, opts: &CoreMLModelOptions) -> BatchModel {
        if let Some(enabled) = opts.allow_low_precision_accumulation_on_gpu {
            model.setAllowLowPrecisionAccumulationOnGPU(enabled);
        }
        if let Some(enabled) = opts.prediction_uses_cpu_only {
            model.setPredictionUsesCPUOnly(enabled);
        }
        model
    }

    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        let model = Self::apply_options(
            modelWithPathBatch(path, info.opts.compute_platform, compiled),
            &info.opts,
        );

        Self {
            model,
            outputs: Default::default(),
        }
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let model = Self::apply_options(
            modelWithAssetsBatch(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            &info.opts,
        );

        let coreml_model = Self {
            model,
            outputs: Default::default(),
        };
        std::mem::forget(buf);
        coreml_model
    }

    pub fn add_input(
        &mut self,
        tag: impl AsRef<str>,
        input: impl Into<MLArray>,
        idx: isize,
    ) -> Result<(), CoreMLError> {
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let shape: Vec<usize> = input.shape().to_vec();

        use std::mem::ManuallyDrop;
        let mut s = ManuallyDrop::new(input);
        match &mut *s {
            MLArray::Float32Array(array_base) => {
                let (data, offset) = array_base.clone().into_raw_vec_and_offset();
                assert!(
                    matches!(offset, Some(0) | None),
                    "array base offset is not zero; bad aligned input"
                );
                let mut data = data;
                let capacity = data.capacity();

                if !self
                    .model
                    .bindInputF32(shape, &name, data.as_mut_ptr(), capacity, idx)
                {
                    return Err(CoreMLError::UnknownError(
                        "failed to bind input to model".to_string(),
                    ));
                }
                std::mem::forget(data);
            }
            _ => {
                return Err(CoreMLError::UnknownError(
                    "unsupported input type for batch model".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        let desc = self.model.description();
        for name in desc.output_names() {
            let shape = desc.output_shape(&name);
            let ty = desc.output_type(&name);
            match ty.as_str() {
                "f32" => {
                    self.outputs.insert(name, ("f32", shape.to_vec()));
                }
                _ => {
                    return Err(CoreMLError::UnknownError(format!(
                        "non-f32 output types are not supported (yet)! type: {}",
                        ty
                    )))
                }
            }
        }

        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let n = output.count();
        Ok(MLBatchModelOutput {
            outputs: (0..n)
                .map(|i| {
                    let output = output.for_idx(i);
                    let mut element_map = fxhash::FxHashMap::default();
                    for (key, (ty, shape)) in &self.outputs {
                        if *ty != "f32" {
                            continue;
                        }
                        let name = key.as_str();
                        let out = output.outputF32(name);
                        if let Ok(array) = Array::from_shape_vec(shape.clone(), out) {
                            element_map.insert(key.clone(), array.into());
                        }
                    }
                    element_map
                })
                .collect(),
        })
    }

    pub fn description(&self) -> crate::description::ModelDescription {
        self.model.description().into()
    }
}
