#![allow(clippy::all)]
//! Regression tests for MLArray type conversions and contiguity handling.
//! These tests don't require a CoreML model file.

use coreml_rs_fork::mlarray::{mean_absolute_error, MLArray};
use ndarray::{Array, Array2, IxDyn};

#[test]
fn f32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2, 3]);
    let recovered: Array<f32, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn i32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2, 2]), vec![10i32, 20, 30, 40]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2, 2]);
    let recovered: Array<i32, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn u16_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[3]), vec![100u16, 200, 300]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[3]);
    let recovered: Array<u16, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn u8_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[4]), vec![0u8, 127, 200, 255]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[4]);
    let recovered: Array<u8, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn i16_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![-100i16, 100]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<i16, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn i8_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[3]), vec![-1i8, 0, 1]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[3]);
    let recovered: Array<i8, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn u32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![u32::MAX, 0u32]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<u32, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn f16_roundtrip() {
    let vals: Vec<half::f16> = vec![half::f16::from_f32(1.5), half::f16::from_f32(-0.25)];
    let arr = Array::from_shape_vec(IxDyn(&[2]), vals).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<half::f16, _> = ml.extract_to_tensor().expect("extract_to_tensor failed");
    assert_eq!(recovered, arr);
}

#[test]
fn mean_absolute_error_basic() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.5f32, 2.5, 3.5];
    let mae = mean_absolute_error(a, b);
    assert!((mae - 0.5).abs() < 1e-6);
}

#[test]
fn mean_absolute_error_identical() {
    let a = [1.0f32, 2.0, 3.0];
    let mae = mean_absolute_error(a, a);
    assert!((mae - 0.0).abs() < 1e-10);
}

#[test]
fn transposed_f32_is_standard_layout() {
    // Create a non-contiguous array (transposed)
    let arr: Array2<f32> =
        Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let transposed = arr.t().to_owned();
    // as_standard_layout should produce a C-contiguous copy
    let contiguous = transposed.as_standard_layout().into_owned();
    assert!(contiguous.is_standard_layout());
    // Values should match the transposed view
    assert_eq!(contiguous[[0, 0]], 1.0);
    assert_eq!(contiguous[[0, 1]], 4.0);
    assert_eq!(contiguous[[1, 0]], 2.0);
}

/// Regression: `MLType for u16` accepts a `Float16Array` and hands back the
/// raw IEEE-754 binary16 bit patterns. This used to go through
/// `std::mem::transmute` on the whole `ArrayBase`, which is UB — Rust
/// guarantees no layout relationship between instantiations of a generic
/// non-`#[repr(C)]` struct at different type parameters. The replacement is an
/// element-wise `f16::to_bits`, so this test pins the exact bit patterns.
#[test]
fn f16_array_extracts_as_u16_bit_patterns() {
    use half::f16;
    let values = [
        f16::from_f32(0.0),
        f16::from_f32(1.0),
        f16::from_f32(-2.0),
        f16::from_f32(65504.0), // f16::MAX
    ];
    let expected: Vec<u16> = values.iter().map(|v| v.to_bits()).collect();
    let arr = Array::from_shape_vec(IxDyn(&[4]), values.to_vec()).unwrap();
    let ml: MLArray = arr.into();
    let recovered: Array<u16, _> = ml.extract_to_tensor().expect("f16 -> u16 extract failed");
    assert_eq!(recovered.into_raw_vec_and_offset().0, expected);
}

/// Regression: extracting the wrong type must not consume the array into a
/// `ManuallyDrop` that is never destroyed. The observable contract is just the
/// error, but this pins the behaviour the leak fix preserves.
#[test]
fn mismatched_extract_reports_error() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0]).unwrap();
    let ml: MLArray = arr.into();
    let err = ml
        .extract_to_tensor::<i32>()
        .expect_err("expected a type mismatch");
    assert!(err.contains("type mismatch"), "unexpected error: {err}");

    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![1u8, 2]).unwrap();
    let ml: MLArray = arr.into();
    assert!(ml.extract_to_tensor::<u16>().is_err());
}

/// Regression: a null `IOSurfaceRef` must be rejected before any IOSurface
/// framework call. The alloc-size query used to run first, handing a null
/// reference straight into IOSurface.
#[cfg(target_os = "macos")]
#[test]
fn null_iosurface_is_rejected_without_calling_the_framework() {
    use coreml_rs_fork::mlarray::MLDataType;
    // SAFETY: the null pointer is exactly the input under test; the function
    // must detect it and return before dereferencing anything.
    let res = unsafe { MLArray::from_iosurface(std::ptr::null(), MLDataType::Float32, &[2, 2]) };
    let err = res.err().expect("null surface must be rejected");
    assert!(
        matches!(err, coreml_rs_fork::CoreMLError::BadInputShape(_)),
        "expected BadInputShape, got {err:?}"
    );
}
