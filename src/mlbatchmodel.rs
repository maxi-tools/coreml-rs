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

/// Move `array`'s `elem_count` elements into a `Vec` in C-standard (row-major)
/// order, without copying when the array is already laid out that way.
///
/// Swift builds row-major strides from the *logical* shape it is handed (see
/// `bindInputF32` in `swift_library.swift`), so the allocation it receives must
/// be in C-standard order and must start at the first element the shape
/// describes. `ArrayBase::into_raw_vec_and_offset` returns the allocation in
/// *memory* order, which for a transposed (or otherwise non-standard) array is
/// a different permutation of the values than the strides claim — CoreML would
/// silently consume scrambled input under a shape that still looks valid.
///
/// Three cases, in decreasing order of frequency:
///
/// * **Standard layout at offset 0** — the overwhelmingly common case. The
///   allocation already matches what Swift expects, so it is moved out
///   untouched and no allocation or copy occurs.
/// * **Standard layout at a nonzero offset** — an *owned but sliced* array is
///   C-contiguous yet begins part way into its allocation. The elements are
///   compacted to the front in place. The previous code asserted `offset == 0`
///   here and panicked on this perfectly legal input.
/// * **Non-standard layout** — e.g. an owned transposed array.
///   `as_standard_layout` materializes a row-major copy; this is the only
///   branch that allocates.
fn into_standard_layout_vec<A: Clone>(array: ndarray::ArrayD<A>, elem_count: usize) -> Vec<A> {
    if array.is_standard_layout() {
        let (mut data, offset) = array.into_raw_vec_and_offset();
        let offset = offset.unwrap_or(0);
        if offset != 0 || data.len() != elem_count {
            data.drain(..offset);
            data.truncate(elem_count);
        }
        data
    } else {
        array
            .as_standard_layout()
            .into_owned()
            .into_raw_vec_and_offset()
            .0
    }
}

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
                        // A cache file can decompress cleanly and still hold a
                        // model CoreML refuses — match the in-memory and
                        // path branches, which both reject `failed()` instead
                        // of reporting a broken model as `Loaded`.
                        if coreml_model.model.failed() {
                            return Err(CoreMLError::FailedToLoadBatch(
                                "Failed to load model from cached buffer path; likely not a CoreML mlmodel file".to_string(),
                                Self::Unloaded(info, loader),
                            ));
                        }
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

    /// Load a batch model from an in-memory `.mlmodel` buffer.
    ///
    /// # Buffer ownership
    ///
    /// Identical to [`crate::mlmodel::CoreMLModel::load_buffer`]: the
    /// allocation is transferred to Swift, which wraps it in
    /// `Data(bytesNoCopy:deallocator:)` and frees it through
    /// `rust_vec_free_u8`. Swift takes ownership before the `do`/`catch`, so
    /// it owns the buffer on the load-failure path too. `Box::into_raw` leaks
    /// it on the Rust side deliberately — Swift is the sole owner and the sole
    /// free, and this type stores no reference to it.
    ///
    /// The boxed-slice normalization matters for the same reason: the Swift
    /// deallocator reconstructs `Vec::from_raw_parts(ptr, len, len)`, so
    /// capacity must equal length or the allocation is freed under the wrong
    /// layout.
    pub fn load_buffer(buf: Vec<u8>, info: CoreMLModelInfo) -> Self {
        let buf = buf.into_boxed_slice();
        let len = buf.len();
        // Ownership moves to Swift here — see the note above.
        let ptr = Box::into_raw(buf) as *mut u8;

        let model = Self::apply_options(
            modelWithAssetsBatch(ptr, len as isize, info.opts.compute_platform),
            &info.opts,
        );

        Self {
            model,
            outputs: Default::default(),
        }
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
        // `MLArray` implements `Drop` (it zeroes its backing store), so the
        // owned tensor cannot be moved out of the enum by pattern match.
        // Suppress the drop glue and take the payload out by hand instead —
        // the previous code cloned the tensor (an O(N) copy of the whole input
        // buffer, per call, in the hot path) purely to satisfy the borrow
        // checker, and then never destroyed the original, leaking one full
        // input tensor on every successful bind.
        let mut s = ManuallyDrop::new(input);
        let mut unsupported = false;
        match &mut *s {
            MLArray::Float32Array(array_base) => {
                // SAFETY: `array_base` points at the live, initialized payload
                // of the `Float32Array` variant. `ptr::read` moves it out
                // bitwise into `array_owned`, which is now the sole owner of
                // the tensor. `s` is a `ManuallyDrop`, so `MLArray::drop` never
                // runs, and no path below reads or drops `s` again.
                //
                // Ownership from here, on both layout branches:
                //   * standard layout — `into_raw_vec_and_offset` consumes
                //     `array_owned` and moves the same allocation into `data`.
                //   * non-standard layout — `array_owned` is only borrowed to
                //     produce a fresh standard-layout `data`, then dropped
                //     normally at the end of this arm, freeing the original
                //     exactly once.
                // Either way `data` is the unique owner at the bind call, and
                // it passes to Swift's `MLMultiArray` deallocator on the
                // `mem::forget` success path, or is dropped by Rust on the
                // bind-failure path. No double free, no leak.
                let array_owned = unsafe { std::ptr::read(array_base) };

                let elem_count: usize = shape.iter().product();
                let mut data = into_standard_layout_vec(array_owned, elem_count);
                debug_assert_eq!(data.len(), elem_count);
                let capacity = data.capacity();

                if !self
                    .model
                    .bindInputF32(shape, &name, data.as_mut_ptr(), capacity, idx)
                {
                    // Swift took no ownership, so `data` drops here and frees
                    // the tensor we just moved out of `s`.
                    return Err(CoreMLError::UnknownError(
                        "failed to bind input to model".to_string(),
                    ));
                }
                // Swift's MLMultiArray deallocator now owns this buffer.
                std::mem::forget(data);
            }
            _ => unsupported = true,
        }
        if unsupported {
            // SAFETY: nothing was moved out of `s` on this path, its payload is
            // still fully initialized, and `s` is not used again afterwards —
            // so running the enum's own drop glue exactly once here is
            // correct. Without it the rejected tensor would leak.
            unsafe { ManuallyDrop::drop(&mut s) };
            return Err(CoreMLError::UnknownError(
                "unsupported input type for batch model".to_string(),
            ));
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

#[cfg(test)]
mod tests {
    use super::into_standard_layout_vec;
    use ndarray::{array, s, Array2};

    /// The hot path: an already C-contiguous array must be moved, not copied.
    /// The returned `Vec` should reuse the original allocation, so the base
    /// pointer is unchanged.
    #[test]
    fn standard_layout_input_is_moved_not_copied() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let arr = arr.into_dyn();
        let expected_ptr = arr.as_ptr();

        let out = into_standard_layout_vec(arr, 6);

        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            out.as_ptr(),
            expected_ptr,
            "already-standard input must not be reallocated"
        );
    }

    /// Regression: an owned *transposed* array is non-standard layout. Handing
    /// its raw allocation to Swift gave CoreML memory-order values under
    /// row-major strides, i.e. a silent transposition of the input.
    #[test]
    fn transposed_input_is_materialized_in_row_major_order() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let transposed = arr.t().to_owned().into_dyn();
        assert_eq!(transposed.shape(), &[3, 2]);

        let out = into_standard_layout_vec(transposed, 6);

        // Row-major traversal of the 3x2 transpose.
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    /// Regression: an owned array that was sliced is C-contiguous but starts
    /// at a nonzero offset into its allocation. This used to trip the
    /// `offset == 0` assertion and panic.
    #[test]
    fn sliced_owned_input_with_nonzero_offset_is_compacted() {
        let arr: Array2<f32> = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        // Rows 1..3 — contiguous, but beginning 3 elements into the buffer.
        let sliced = arr.slice_move(s![1..3, ..]).into_dyn();
        assert_eq!(sliced.shape(), &[2, 3]);

        let out = into_standard_layout_vec(sliced, 6);

        assert_eq!(out, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    /// A sliced *column* range is neither contiguous nor standard layout; it
    /// must still come back in row-major order.
    #[test]
    fn non_contiguous_column_slice_is_materialized() {
        let arr: Array2<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let sliced = arr.slice_move(s![.., 0..2]).into_dyn();
        assert_eq!(sliced.shape(), &[2, 2]);

        let out = into_standard_layout_vec(sliced, 4);

        assert_eq!(out, vec![1.0, 2.0, 4.0, 5.0]);
    }
}
