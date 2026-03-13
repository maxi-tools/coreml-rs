use crate::mlbatchmodel::CoreMLBatchModelWithState;
use crate::mlmodel::CoreMLModelWithState;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum CoreMLError {
    #[error("CoreML Cache IoError: {0}")]
    IoError(#[from] std::io::Error),
    #[error("BadInputShape: {0}")]
    BadInputShape(String),
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
    FailedToLoadBatch(String, CoreMLBatchModelWithState),
    #[error("BindInputFailed: input '{name}' ({dtype}) with shape {shape:?} could not be bound")]
    BindInputFailed {
        name: String,
        shape: Vec<usize>,
        dtype: &'static str,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coreml_error_display_formatting() {
        let err_bad_shape = CoreMLError::BadInputShape("expected shape [1, 2]".to_string());
        assert_eq!(
            err_bad_shape.to_string(),
            "BadInputShape: expected shape [1, 2]"
        );

        let err_unknown = CoreMLError::UnknownError("out of memory".to_string());
        assert_eq!(err_unknown.to_string(), "UnknownError: out of memory");

        let err_unknown_static = CoreMLError::UnknownErrorStatic("macOS unsupported");
        assert_eq!(
            err_unknown_static.to_string(),
            "UnknownError: macOS unsupported"
        );

        let err_not_loaded = CoreMLError::ModelNotLoaded;
        assert_eq!(
            err_not_loaded.to_string(),
            "ModelNotLoaded: coreml model not loaded into session"
        );

        let err_bind = CoreMLError::BindInputFailed {
            name: "input_ids".to_string(),
            shape: vec![1, 128],
            dtype: "f32",
        };
        assert_eq!(
            err_bind.to_string(),
            "BindInputFailed: input 'input_ids' (f32) with shape [1, 128] could not be bound"
        );

        // Construct an IoError to test `From` and `Display`
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err_io: CoreMLError = io_err.into();
        assert_eq!(err_io.to_string(), "CoreML Cache IoError: file not found");
    }
}
