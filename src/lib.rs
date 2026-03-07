#![allow(
    clippy::missing_transmute_annotations,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::result_large_err,
    clippy::len_zero,
    clippy::useless_conversion,
    clippy::let_and_return,
    clippy::type_complexity,
    clippy::manual_contains,
    clippy::derivable_impls
)]
pub mod error;
pub mod loader;
pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;
pub mod utils;

mod swift;

// re-exports
pub use ffi::ComputePlatform;
pub use loader::CoreMLModelOptions;
pub use mlmodel::CoreMLModelWithState;

pub use swift::swift as ffi;

pub use error::CoreMLError;
pub use mlarray::MLArray;
pub use mlbatchmodel::CoreMLBatchModelWithState;
