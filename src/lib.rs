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
pub mod mlarray;
pub mod mlbatchmodel;
pub mod mlmodel;

mod swift;

// re-exports
pub use ffi::ComputePlatform;
pub use mlmodel::{CoreMLModelOptions, CoreMLModelWithState};

pub use swift::swift as ffi;
