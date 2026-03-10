#![allow(clippy::all)]
//! Regression tests for MLArray type conversions and contiguity handling.
//! These tests don't require a CoreML model file.

use coreml_rs::mlarray::{mean_absolute_error, mean_absolute_error_bytes, MLArray};
use ndarray::{Array, Array2, IxDyn};

#[test]
fn f32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2, 3]);
    let recovered: Array<f32, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn i32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2, 2]), vec![10i32, 20, 30, 40]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2, 2]);
    let recovered: Array<i32, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn u16_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[3]), vec![100u16, 200, 300]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[3]);
    let recovered: Array<u16, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn u8_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[4]), vec![0u8, 127, 200, 255]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[4]);
    let recovered: Array<u8, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn i16_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![-100i16, 100]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<i16, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn i8_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[3]), vec![-1i8, 0, 1]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[3]);
    let recovered: Array<i8, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn u32_roundtrip() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![u32::MAX, 0u32]).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<u32, _> = ml.extract_to_tensor();
    assert_eq!(recovered, arr);
}

#[test]
fn f16_roundtrip() {
    let vals: Vec<half::f16> = vec![half::f16::from_f32(1.5), half::f16::from_f32(-0.25)];
    let arr = Array::from_shape_vec(IxDyn(&[2]), vals).unwrap();
    let ml: MLArray = arr.clone().into();
    assert_eq!(ml.shape(), &[2]);
    let recovered: Array<half::f16, _> = ml.extract_to_tensor();
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
fn mean_absolute_error_bytes_test() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [1.5f32, 2.5, 3.5];
    let a_bytes: &[u8] = bytemuck::cast_slice(&a);
    let b_bytes: &[u8] = bytemuck::cast_slice(&b);
    let mae = mean_absolute_error_bytes::<f32>(a_bytes, b_bytes);
    assert!((mae - 0.5).abs() < 1e-6);
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

#[test]
fn test_try_as_view_success_and_failure() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![1.0f32, 2.0]).unwrap();
    let ml: MLArray = arr.clone().into();

    // Success on exact match
    let view = ml.try_as_view_f32().unwrap();
    assert_eq!(view, arr);

    // Failure on mismatch
    let err = ml.try_as_view_i32().unwrap_err();
    assert_eq!(
        err,
        "MLArray type mismatch: expected i32, found type_id=f32"
    );
}

#[test]
fn test_try_extract_to_tensor_failure() {
    let arr = Array::from_shape_vec(IxDyn(&[2]), vec![10i32, 20]).unwrap();
    let ml: MLArray = arr.into();

    let res: Result<Array<f32, _>, _> = ml.try_extract_to_tensor();
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("MLArray type mismatch"));
}

#[test]
fn test_type_id_str() {
    let arr = Array::from_shape_vec(IxDyn(&[1]), vec![10i32]).unwrap();
    let ml: MLArray = arr.into();
    assert_eq!(ml.type_id_str(), "i32");

    let arr2 = Array::from_shape_vec(IxDyn(&[1]), vec![1.0f32]).unwrap();
    let ml2: MLArray = arr2.into();
    assert_eq!(ml2.type_id_str(), "f32");
}
