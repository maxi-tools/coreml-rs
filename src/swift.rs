#![allow(non_camel_case_types, clippy::not_unsafe_ptr_arg_deref)]

use crate::mlarray::MLArray;

#[swift_bridge::bridge]
pub mod ffi {
    enum ComputePlatform {
        Cpu,
        CpuAndANE,
        CpuAndGpu,
    }
    extern "Rust" {
        fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32>;
        fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32>;
        fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16>;
        fn rust_vec_from_ptr_i32_cpy(ptr: *mut i32, len: usize) -> Vec<i32>;
        fn rust_vec_from_ptr_f32_cpy(ptr: *mut f32, len: usize) -> Vec<f32>;
        fn rust_vec_from_ptr_u16_cpy(ptr: *mut u16, len: usize) -> Vec<u16>;
        fn rust_vec_free_f32(ptr: *mut f32, len: usize);
        fn rust_vec_free_i32(ptr: *mut i32, len: usize);
        fn rust_vec_free_u16(ptr: *mut u16, len: usize);
        fn rust_vec_free_u8(ptr: *mut u8, len: usize);
    }

    extern "Swift" {
        #[swift_bridge(swift_name = "initWithPath")]
        pub fn modelWithPath(path: String, compute: ComputePlatform, compiled: bool) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAsset")]
        pub fn modelWithAssets(ptr: *mut u8, len: isize, compute: ComputePlatform) -> Model;
        #[swift_bridge(swift_name = "initWithCompiledAssetBatch")]
        pub fn modelWithAssetsBatch(
            ptr: *mut u8,
            len: isize,
            compute: ComputePlatform,
        ) -> BatchModel;
        #[swift_bridge(swift_name = "initWithPathBatch")]
        pub fn modelWithPathBatch(
            path: String,
            compute: ComputePlatform,
            compiled: bool,
        ) -> BatchModel;
    }

    extern "Swift" {
        type BatchOutput;

        #[swift_bridge(swift_name = "getOutputAtIndex")]
        pub fn for_idx(&self, at: isize) -> ModelOutput;
        pub fn getError(&self) -> Option<String>;
        pub fn count(&self) -> isize;
    }

    extern "Swift" {
        type BatchModel;

        fn load(&mut self) -> bool;
        fn unload(&mut self) -> bool;
        fn setAllowLowPrecisionAccumulationOnGPU(&mut self, enabled: bool);
        fn setPredictionUsesCPUOnly(&mut self, enabled: bool);
        fn description(&self) -> ModelDescription;
        fn predict(&self) -> BatchOutput;
        fn bindInputF32(
            &self,
            shape: Vec<usize>,
            featureName: &str,
            data: *mut f32,
            len: usize,
            idx: isize,
        ) -> bool;
        #[swift_bridge(swift_name = "hasFailedToLoad")]
        fn failed(&self) -> bool;
    }

    extern "Swift" {
        type Model;

        fn bindOutputF32(
            &self,
            shape: Vec<i32>,
            featureName: &str,
            data: *mut f32,
            len: usize,
        ) -> bool;
        fn bindOutputU16(
            &self,
            shape: Vec<i32>,
            featureName: &str,
            data: *mut u16,
            len: usize,
        ) -> bool;
        fn bindOutputI32(
            &self,
            shape: Vec<i32>,
            featureName: &str,
            data: *mut i32,
            len: usize,
        ) -> bool;
        fn bindInputF32(
            &self,
            shape: Vec<usize>,
            featureName: &str,
            data: *mut f32,
            len: usize,
        ) -> bool;
        fn bindInputI32(
            &self,
            shape: Vec<usize>,
            featureName: &str,
            data: *mut i32,
            len: usize,
        ) -> bool;
        fn bindInputU16(
            &self,
            shape: Vec<usize>,
            featureName: &str,
            data: *mut u16,
            len: usize,
        ) -> bool;
        fn bindInputCVPixelBuffer(
            &self,
            width: usize,
            height: usize,
            featureName: &str,
            data: *mut u8,
            len: usize,
        ) -> bool;

        // #828 P0a: zero-copy IOSurface input binding.
        //
        // `surface` is an IOSurfaceRef passed as a raw pointer (opaque
        // CFTypeRef). The caller is responsible for locking the surface
        // around any subsequent `predict()` call. `dtypeRaw` matches the
        // `MLDataType::raw_tag()` values from `mlarray.rs`.
        fn bindInputIOSurface(
            &self,
            surface: *mut u8,
            dtypeRaw: i32,
            shape: Vec<usize>,
            featureName: &str,
        ) -> bool;

        // #828 P0a: zero-copy CVPixelBuffer input binding (borrow path).
        //
        // Unlike `bindInputCVPixelBuffer`, this does not take ownership
        // of a `Vec<u8>`. It wraps the passed CVPixelBufferRef in an
        // `MLFeatureValue(pixelBuffer:)` directly — swift-bridge retains
        // it for the lifetime of the feature value.
        fn bindInputCVPixelBufferRef(&self, pixelBuffer: *mut u8, featureName: &str) -> bool;

        // #828 P0d: zero-copy IOSurface OUTPUT binding.
        //
        // Binds a caller-provided `IOSurfaceRef` (passed as an opaque
        // raw pointer) as the destination backing for a named model
        // output. Mirrors `bindInputIOSurface` on the write side: the
        // caller must hold a read-write lock on the surface for the
        // duration of the subsequent `predict()` call. `dtypeRaw`
        // matches `MLDataType::raw_tag()`.
        //
        // Shape uses `Vec<i32>` to stay consistent with the rest of
        // the `bindOutput*` family (bindOutputF32/U16/I32).
        fn bindOutputIOSurface(
            &self,
            surface: *mut u8,
            dtypeRaw: i32,
            shape: Vec<i32>,
            featureName: &str,
        ) -> bool;

        #[swift_bridge(swift_name = "getCompiledPath")]
        fn compiled_path(&self) -> Option<String>;

        fn load(&mut self) -> bool;
        fn unload(&mut self) -> bool;
        fn setAllowLowPrecisionAccumulationOnGPU(&mut self, enabled: bool);
        fn setPredictionUsesCPUOnly(&mut self, enabled: bool);
        fn description(&self) -> ModelDescription;
        fn predict(&self) -> ModelOutput;
        #[swift_bridge(swift_name = "hasFailedToLoad")]
        fn failed(&self) -> bool;

        // CoreML State (MLState) for stateful KV cache
        fn makeState(&mut self) -> bool;
        fn predictWithState(&self) -> ModelOutput;
        fn hasState(&self) -> bool;
        fn resetState(&mut self);
    }

    extern "Swift" {
        type ModelDescription;

        fn inputs(&self) -> Vec<String>;
        fn outputs(&self) -> Vec<String>;
        fn output_names(&self) -> Vec<String>;
        fn input_names(&self) -> Vec<String>;
        fn output_type(&self, name: &str) -> String;
        fn output_shape(&self, name: &str) -> Vec<usize>;
        fn input_shape(&self, name: &str) -> Vec<usize>;
    }

    extern "Swift" {
        type ModelOutput;

        fn outputDescription(&self) -> Vec<String>;
        fn outputShape(&self, name: &str) -> Vec<usize>;
        fn outputF32(&self, name: &str) -> Vec<f32>;
        fn outputU16(&self, name: &str) -> Vec<u16>;
        fn outputI32(&self, name: &str) -> Vec<i32>;
        fn getError(&self) -> Option<String>;
    }
}

fn rust_vec_from_ptr_f32(ptr: *mut f32, len: usize) -> Vec<f32> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
fn rust_vec_from_ptr_u16(ptr: *mut u16, len: usize) -> Vec<u16> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}
fn rust_vec_from_ptr_i32(ptr: *mut i32, len: usize) -> Vec<i32> {
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// performs a memcpy
fn rust_vec_from_ptr_f32_cpy(ptr: *mut f32, len: usize) -> Vec<f32> {
    (unsafe { std::slice::from_raw_parts(ptr, len) }).to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_u16_cpy(ptr: *mut u16, len: usize) -> Vec<u16> {
    (unsafe { std::slice::from_raw_parts(ptr, len) }).to_vec()
}
/// performs a memcpy
fn rust_vec_from_ptr_i32_cpy(ptr: *mut i32, len: usize) -> Vec<i32> {
    (unsafe { std::slice::from_raw_parts(ptr, len) }).to_vec()
}

fn rust_vec_free_f32(ptr: *mut f32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}

fn rust_vec_free_u16(ptr: *mut u16, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}

fn rust_vec_free_u8(ptr: *mut u8, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}

fn rust_vec_free_i32(ptr: *mut i32, len: usize) {
    unsafe {
        _ = Vec::from_raw_parts(ptr, len, len);
    }
}

pub type FxHashMap<K, V> = fxhash::FxHashMap<K, V>;

pub struct MLModelOutput {
    pub outputs: FxHashMap<String, MLArray>,
}

pub struct MLBatchModelOutput {
    pub outputs: Vec<FxHashMap<String, MLArray>>,
}
