use crate::mlarray::MLArrayBaseExt;
use crate::FeatureName;
use crate::{
    error::CoreMLError,
    ffi::{modelWithAssetsBatch, modelWithPathBatch, BatchModel},
    loader::{CoreMLModelInfo, CoreMLModelLoader},
    mlarray::MLArray,
    swift::MLBatchModelOutput,
    CoreMLModelOptions,
};
use ndarray::Array;
use std::{collections::HashMap, io::Write, path::Path};
use tempfile::NamedTempFile;

pub use crate::swift::MLModelOutput;

#[derive(Debug)]
pub enum CoreMLBatchModelWithState {
    Unloaded(CoreMLModelInfo, CoreMLModelLoader),
    Loaded(CoreMLBatchModel, CoreMLModelInfo, CoreMLModelLoader),
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
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
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
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML model file",
                        Self::Unloaded(info, loader),
                    ));
                }
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::Buffer(vec) => {
                let mut coreml_model = CoreMLBatchModel::load_buffer(vec.clone(), info.clone());
                coreml_model.model.load();
                if coreml_model.model.failed() {
                    return Err(CoreMLError::FailedToLoadBatchStatic(
                        "Failed to load model; likely not a CoreML mlmodel file",
                        Self::Unloaded(info, CoreMLModelLoader::Buffer(vec)),
                    ));
                }
                let loader = CoreMLModelLoader::Buffer(vec);
                Ok(Self::Loaded(coreml_model, info, loader))
            }
            CoreMLModelLoader::BufferToDisk(u) => match crate::storage::load_buffer_from_disk(&u) {
                Ok(vec) => {
                    let mut coreml_model = CoreMLBatchModel::load_buffer(vec, info.clone());
                    coreml_model.model.load();
                    if coreml_model.model.failed() {
                        return Err(CoreMLError::FailedToLoadBatchStatic(
                            "Failed to load model from cached buffer",
                            Self::Unloaded(info, CoreMLModelLoader::BufferToDisk(u)),
                        ));
                    }
                    let loader = CoreMLModelLoader::BufferToDisk(u);
                    Ok(Self::Loaded(coreml_model, info, loader))
                }
                Err(err) => Err(CoreMLError::FailedToLoadBatch(
                    format!("failed to load the model from cached buffer path: {err}"),
                    CoreMLBatchModelWithState::Unloaded(info, CoreMLModelLoader::BufferToDisk(u)),
                )),
            },
        }
    }

    /// Might fail if system disk space too low(very unlikely)
    pub fn unload(self) -> Result<Self, CoreMLError> {
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

    /// Unloads the model buffer to the disk, at cache_dir
    pub fn unload_to_disk(self) -> Result<Self, CoreMLError> {
        match self {
            Self::Loaded(_, mut info, loader) | Self::Unloaded(mut info, loader) => {
                let loader = {
                    match loader {
                        CoreMLModelLoader::Buffer(vec) => {
                            match crate::storage::save_buffer_to_disk(
                                &vec,
                                &mut info.opts.cache_dir,
                            ) {
                                Ok(m) => CoreMLModelLoader::BufferToDisk(m),
                                Err(err) => {
                                    return Err(CoreMLError::FailedToLoadBatch(
                                        format!("failed to load the model from the buffer: {err}"),
                                        CoreMLBatchModelWithState::Unloaded(
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
            CoreMLBatchModelWithState::Unloaded(_, _) => Err(CoreMLError::ModelNotLoaded),
            CoreMLBatchModelWithState::Loaded(core_mlmodel, _, _) => Ok(core_mlmodel.description()),
        }
    }

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
    // save_path: Option<PathBuf>,
    outputs: HashMap<FeatureName, (&'static str, Vec<usize>)>,
}

unsafe impl Send for CoreMLBatchModel {}

impl std::fmt::Debug for BatchModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchModel").finish()
    }
}

impl CoreMLBatchModel {
    pub fn load_from_path(path: String, info: CoreMLModelInfo, compiled: bool) -> Self {
        Self {
            model: modelWithPathBatch(path, info.opts.compute_platform, compiled),
            // save_path: None,
            outputs: Default::default(),
        }
    }

    pub fn load_buffer(mut buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let coreml_model = Self {
            model: modelWithAssetsBatch(
                buf.as_mut_ptr(),
                buf.len() as isize,
                info.opts.compute_platform,
            ),
            // save_path: None,
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
        // route input correctly
        let input: MLArray = input.into();
        let name = tag.as_ref().to_string();
        let desc = self.model.description();
        let shape: Vec<usize> = input.shape().to_vec();
        let arr = desc.input_shape(name.clone());
        crate::utils::validate_coreml_shape(&arr, &shape, &name)?;
        match input {
            MLArray::Float32Array(array_base) => {
                let mut data = array_base.into_contiguous_raw_vec();
                if !self
                    .model
                    .bindInputF32(shape, name, data.as_mut_ptr(), data.capacity(), idx)
                {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "failed to bind input to model",
                    ));
                }
                std::mem::forget(data);
            }
            _ => {
                return Err(CoreMLError::UnknownErrorStatic(
                    "failed to bind input to model",
                ));
            }
        }
        Ok(())
    }

    pub fn predict(&mut self) -> Result<MLBatchModelOutput, CoreMLError> {
        let desc = self.model.description();
        for name in desc.output_names() {
            let shape = desc.output_shape(name.clone());
            let ty = desc.output_type(name.clone());
            match ty.as_str() {
                "f32" => {
                    self.outputs.insert(name, ("f32", shape.to_vec()));
                }
                _ => {
                    return Err(CoreMLError::UnknownErrorStatic(
                        "non-f32 output types are not supported (yet)!",
                    ))
                }
            }
        }

        let output = self.model.predict();
        if let Some(err) = output.getError() {
            return Err(CoreMLError::UnknownError(err));
        }
        let n = output.count();
        let mut all_outputs = Vec::with_capacity(n as usize);

        for i in 0..n {
            let output_at_idx = output.for_idx(i);
            let mut batch_item_outputs = HashMap::new();

            for (key, (ty, shape)) in &self.outputs {
                if *ty != "f32" {
                    eprintln!("warning: non-f32 types aren't supported, and will be skipped in the output");
                    continue;
                }

                let name = key.clone();
                let out = output_at_idx.outputF32(name);

                if let Ok(array) = Array::from_shape_vec(shape.clone(), out) {
                    batch_item_outputs.insert(key.clone(), array.into());
                }
            }
            all_outputs.push(batch_item_outputs);
        }

        Ok(MLBatchModelOutput {
            outputs: all_outputs,
        })
    }

    pub fn description(&self) -> HashMap<&str, Vec<String>> {
        let desc = self.model.description();
        let mut map = HashMap::new();
        map.insert("input", desc.inputs());
        map.insert("output", desc.outputs());
        map
    }
}
