// Specific clippy allows for FFI and CoreML interop
#![allow(
    clippy::all,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::result_large_err,
    clippy::type_complexity,
    clippy::missing_transmute_annotations
)]
pub mod description;
pub mod error;
#[cfg(target_os = "macos")]
pub mod iosurface;
pub mod loader;
pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;
pub mod options;
pub mod state;

mod swift;

// re-exports
pub use error::CoreMLError;
pub use ffi::ComputePlatform;
pub use mlarray::MLDataType;
pub use mlmodel::{CoreMLModel, CoreMLModelWithState};
pub use options::{CoreMLModelInfo, CoreMLModelOptions};
pub use swift::ffi;
