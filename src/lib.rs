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
pub use mlmodel::{
    compute_plan_device_counts, ComputePlanDeviceCounts, CoreMLModel, CoreMLModelWithState,
};
pub use options::{CoreMLModelInfo, CoreMLModelOptions};
pub use swift::ffi;

/// Print the required cargo directives for linking Swift Concurrency on macOS.
///
/// Since `rustc-link-arg` is not transitive, downstream consumers must emit
/// this rpath themselves in their own `build.rs` to avoid launch-time
/// "library not loaded" errors for `libswift_Concurrency`.
///
/// # Example
///
/// In your `build.rs`:
/// ```rust
/// coreml_rs_fork::print_swift_linking_directives();
/// ```
pub fn print_swift_linking_directives() {
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    }
}
