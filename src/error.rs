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
    #[error("BindInputFailed: failed to bind {0} input to model")]
    BindInputFailed(&'static str),
    #[error("UnsupportedOutputType: {0}")]
    UnsupportedOutputType(&'static str),
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
}
