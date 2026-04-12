//! Abstractions for passing multidimensional arrays between Rust and Core ML.
//!
//! This module provides the `MLArray` enum, which wraps various types of `ndarray::Array`,
//! and traits for mapping Rust types to Core ML compatible types.

use half::f16;
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};

#[cfg(target_os = "macos")]
use crate::iosurface::{IOSurfaceGetAllocSize, IOSurfaceRef, RetainedIOSurface};

/// Element data type for a Core ML tensor wrapping an external buffer.
///
/// Used by [`MLArray::from_iosurface`] and the new `add_input_iosurface`
/// code path (#828 P0a). Discriminants match `MLType::TY` for the
/// corresponding scalar types so both tables can be unified later if
/// wanted.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MLDataType {
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
}

impl MLDataType {
    /// Size of a single element of this dtype in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            MLDataType::Float32 | MLDataType::Int32 => 4,
            MLDataType::Float16 => 2,
        }
    }

    /// Raw integer tag passed through the Swift bridge so the shim can
    /// pick the matching `MLMultiArrayDataType`.
    pub const fn raw_tag(self) -> i32 {
        self as i32
    }
}

/// `MLArray` variant backing an IOSurface-owned tensor.
///
/// Holds a retained `IOSurfaceRef` (CFRelease on drop), the logical
/// shape, and the element dtype. The actual data pointer is *not*
/// captured here — it is resolved on demand via `IOSurfaceGetBaseAddress`
/// while the caller holds a lock on the surface.
///
/// **Lock contract:** the caller must have the surface locked with
/// `IOSurfaceLock` before passing it to [`MLArray::from_iosurface`] *and*
/// must keep it locked through any subsequent `predict()` call that
/// consumes it. See `docs/superpowers/specs/2026-04-08-828-p0a-...` for
/// the full lifecycle.
#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct IOSurfaceMLArray {
    pub(crate) surface: RetainedIOSurface,
    pub(crate) dtype: MLDataType,
    pub(crate) shape: Vec<usize>,
}

#[cfg(target_os = "macos")]
impl IOSurfaceMLArray {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> MLDataType {
        self.dtype
    }

    pub fn surface_ptr(&self) -> IOSurfaceRef {
        self.surface.as_ptr()
    }
}

/// Represents a multi-dimensional array passed to or from Core ML.
///
/// This enum wraps `ndarray::Array` of various numeric types, providing a unified
/// way to handle different tensor data types within the library.
///
/// `MLArray` is the primary data structure for inference inputs and outputs. It wraps
/// an `ndarray::Array` and ensures that data is in a format compatible with the
/// underlying Swift/Objective-C bindings.
#[derive(Debug)]
pub enum MLArray {
    Float32Array(ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>),
    Float16Array(ArrayBase<OwnedRepr<f16>, Dim<IxDynImpl>>),
    Int32Array(ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>>),
    Int16Array(ArrayBase<OwnedRepr<i16>, Dim<IxDynImpl>>),
    Int8Array(ArrayBase<OwnedRepr<i8>, Dim<IxDynImpl>>),
    UInt32Array(ArrayBase<OwnedRepr<u32>, Dim<IxDynImpl>>),
    UInt16Array(ArrayBase<OwnedRepr<u16>, Dim<IxDynImpl>>),
    UInt8Array(ArrayBase<OwnedRepr<u8>, Dim<IxDynImpl>>),
    /// Zero-copy input backed by an `IOSurface`. Only used by the new
    /// `add_input_iosurface` path (#828 P0a). Cannot be extracted to an
    /// owned `ndarray::Array`.
    #[cfg(target_os = "macos")]
    IOSurface(IOSurfaceMLArray),
}

impl MLArray {
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::Float32Array(a) => a.shape(),
            MLArray::Float16Array(a) => a.shape(),
            MLArray::Int32Array(a) => a.shape(),
            MLArray::Int16Array(a) => a.shape(),
            MLArray::Int8Array(a) => a.shape(),
            MLArray::UInt32Array(a) => a.shape(),
            MLArray::UInt16Array(a) => a.shape(),
            MLArray::UInt8Array(a) => a.shape(),
            #[cfg(target_os = "macos")]
            MLArray::IOSurface(wrap) => wrap.shape(),
        }
    }

    pub fn len_bytes(&self) -> usize {
        match self {
            MLArray::Float32Array(array_base) => array_base.len() * 4,
            MLArray::Float16Array(array_base) => array_base.len() * 2,
            MLArray::Int32Array(array_base) => array_base.len() * 4,
            MLArray::Int16Array(array_base) => array_base.len() * 2,
            MLArray::Int8Array(array_base) => array_base.len(),
            MLArray::UInt32Array(array_base) => array_base.len() * 4,
            MLArray::UInt16Array(array_base) => array_base.len() * 2,
            MLArray::UInt8Array(array_base) => array_base.len(),
            #[cfg(target_os = "macos")]
            MLArray::IOSurface(wrap) => {
                wrap.shape.iter().copied().product::<usize>() * wrap.dtype.size_bytes()
            }
        }
    }

    /// Construct an `MLArray` that points at an IOSurface-backed buffer.
    ///
    /// This retains the passed `IOSurfaceRef` (via `CFRetain`) and wraps
    /// it in the new [`MLArray::IOSurface`] variant. The returned value
    /// carries the logical shape and dtype; it does not resolve the base
    /// address until bound to a model via `add_input_iosurface`.
    ///
    /// # Lock contract
    ///
    /// The caller is responsible for locking the surface (typically with
    /// `kIOSurfaceLockReadOnly`) before calling this function and keeping
    /// it locked for the lifetime of any `predict()` call that consumes
    /// the resulting array. CoreML stores a raw pointer into the locked
    /// base address, so unlocking early will corrupt inference.
    ///
    /// # Errors
    ///
    /// Returns `BadInputShape` if:
    /// - `surface` is null;
    /// - `shape` contains a zero dimension;
    /// - `shape.product() * dtype.size_bytes()` exceeds
    ///   `IOSurfaceGetAllocSize(surface)`.
    ///
    /// # Safety
    ///
    /// `surface` must be a valid `IOSurfaceRef` produced by
    /// `IOSurfaceCreate` (or equivalent). Passing any other pointer is
    /// undefined behavior.
    #[cfg(target_os = "macos")]
    pub unsafe fn from_iosurface(
        surface: IOSurfaceRef,
        dtype: MLDataType,
        shape: &[usize],
    ) -> Result<Self, crate::CoreMLError> {
        if shape.is_empty() || shape.iter().any(|&d| d == 0) {
            return Err(crate::CoreMLError::BadInputShape(format!(
                "IOSurface shape must be non-empty with no zero dims, got {shape:?}"
            )));
        }
        // Use checked arithmetic throughout: the shape product itself can
        // overflow `usize` on pathological inputs, so fold with
        // `checked_mul` before the final element-to-byte multiply.
        let elem_count = shape
            .iter()
            .copied()
            .try_fold(1usize, |acc, d| acc.checked_mul(d))
            .ok_or_else(|| {
                crate::CoreMLError::BadInputShape(format!(
                    "IOSurface shape element-count overflow: {shape:?}"
                ))
            })?;
        let expected_bytes = elem_count.checked_mul(dtype.size_bytes()).ok_or_else(|| {
            crate::CoreMLError::BadInputShape(format!(
                "IOSurface shape byte-count overflow: {shape:?} * {} bytes",
                dtype.size_bytes()
            ))
        })?;

        // SAFETY: caller asserted the pointer is a valid IOSurfaceRef.
        let alloc_size = unsafe { IOSurfaceGetAllocSize(surface) };
        if alloc_size < expected_bytes {
            return Err(crate::CoreMLError::BadInputShape(format!(
                "IOSurface alloc_size={alloc_size} < expected={expected_bytes} for shape={shape:?} dtype={dtype:?}"
            )));
        }

        // SAFETY: caller asserted the pointer is a valid IOSurfaceRef.
        let retained = unsafe { RetainedIOSurface::retain(surface) }
            .map_err(|e| crate::CoreMLError::BadInputShape(e.to_string()))?;

        Ok(MLArray::IOSurface(IOSurfaceMLArray {
            surface: retained,
            dtype,
            shape: shape.to_vec(),
        }))
    }

    /// Safely extract the underlying ndarray from the MLArray.
    /// Panics if the requested type T does not match the actual type stored in MLArray.
    pub fn extract_to_tensor<T: MLType>(self) -> Result<Array<T, Dim<IxDynImpl>>, String> {
        T::extract_from_mlarray(self).ok_or_else(|| {
            format!(
                "MLArray type mismatch: expected {}",
                std::any::type_name::<T>()
            )
        })
    }
}

impl Drop for MLArray {
    fn drop(&mut self) {
        // Zero out memory on drop to prevent leaking sensitive data
        // NOTE: This only works if MLArray is the exclusive owner of the backing storage.
        // If it was created from a shared view, this might zero out data still in use elsewhere,
        // but for MLArray's typical use as an owned container, this provides a security baseline.
        match self {
            MLArray::Float32Array(a) => a.fill(0.0),
            MLArray::Float16Array(a) => a.fill(half::f16::ZERO),
            MLArray::Int32Array(a) => a.fill(0),
            MLArray::Int16Array(a) => a.fill(0),
            MLArray::Int8Array(a) => a.fill(0),
            MLArray::UInt32Array(a) => a.fill(0),
            MLArray::UInt16Array(a) => a.fill(0),
            MLArray::UInt8Array(a) => a.fill(0),
            // IOSurface-backed arrays don't own the underlying memory —
            // the IOSurface does. Zeroing here would corrupt data still
            // in use by other consumers of the pooled surface. The
            // retained surface reference is released via
            // `RetainedIOSurface::drop`.
            #[cfg(target_os = "macos")]
            MLArray::IOSurface(_) => {}
        }
    }
}

pub fn mean_absolute_error_bytes<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>
        + bytemuck::Pod
        + core::fmt::Debug,
>(
    lhs: &[u8],
    rhs: &[u8],
) -> f64 {
    let lhs = bytemuck::cast_slice(lhs);
    let rhs = bytemuck::cast_slice(rhs);

    assert_eq!(lhs.len(), rhs.len(), "lhs and rhs have different lengths");
    mean_absolute_error::<T>(lhs, rhs)
}

pub fn mean_absolute_error<
    T: core::ops::Sub<Output = T>
        + PartialOrd
        + Copy
        + core::ops::Add<Output = T>
        + num::cast::AsPrimitive<f64>,
>(
    lhs: impl AsRef<[T]>,
    rhs: impl AsRef<[T]>,
) -> f64 {
    let (sum, count) = lhs
        .as_ref()
        .iter()
        .zip(rhs.as_ref())
        .map(|(&l, &r)| if l > r { l - r } else { r - l })
        .fold((0f64, 0usize), |(acc, count), x| (acc + x.as_(), count + 1));
    sum / count as f64
}

/// Trait for types that can be safely converted to and from `MLArray`.
///
/// This trait provides a type-safe abstraction over the heterogeneous numeric types
/// supported by Core ML, replacing unsafe transmutes with compile-time checked
/// conversions.
pub trait MLType: Sized {
    /// Internal type identifier used by the Swift bridge.
    const TY: usize;

    fn from_array(array: ArrayBase<OwnedRepr<Self>, Dim<IxDynImpl>>) -> MLArray;
    fn extract_from_mlarray(ml_array: MLArray) -> Option<Array<Self, Dim<IxDynImpl>>>;
}

macro_rules! impl_mltype {
    ($type:ty, $variant:ident, $ty_const:expr) => {
        impl MLType for $type {
            const TY: usize = $ty_const;

            fn from_array(array: ArrayBase<OwnedRepr<Self>, Dim<IxDynImpl>>) -> MLArray {
                MLArray::$variant(array)
            }
            fn extract_from_mlarray(ml_array: MLArray) -> Option<Array<Self, Dim<IxDynImpl>>> {
                let mut ml_array = std::mem::ManuallyDrop::new(ml_array);
                if let MLArray::$variant(ref mut array) = *ml_array {
                    unsafe { Some(std::ptr::read(array)) }
                } else {
                    None
                }
            }
        }
    };
}

impl_mltype!(f32, Float32Array, 0);
impl_mltype!(f16, Float16Array, 1);
impl_mltype!(i32, Int32Array, 2);
impl_mltype!(u8, UInt8Array, 4);
impl_mltype!(i16, Int16Array, 5);
impl_mltype!(i8, Int8Array, 6);
impl_mltype!(u32, UInt32Array, 7);

impl MLType for u16 {
    const TY: usize = 3;

    fn from_array(array: ArrayBase<OwnedRepr<Self>, Dim<IxDynImpl>>) -> MLArray {
        MLArray::UInt16Array(array)
    }

    fn extract_from_mlarray(ml_array: MLArray) -> Option<Array<Self, Dim<IxDynImpl>>> {
        let mut ml_array = std::mem::ManuallyDrop::new(ml_array);
        match *ml_array {
            MLArray::UInt16Array(ref mut array) => unsafe { Some(std::ptr::read(array)) },
            MLArray::Float16Array(ref mut array) => unsafe {
                Some(std::mem::transmute(std::ptr::read(array)))
            },
            _ => None,
        }
    }
}

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        T::from_array(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_extract_to_tensor_success() {
        let array: ArrayD<f32> =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ml_array = MLArray::from(array.clone());

        let extracted = ml_array.extract_to_tensor::<f32>().unwrap();
        assert_eq!(extracted, array);
    }

    #[test]
    fn test_extract_to_tensor_type_mismatch() {
        let array: ArrayD<f32> =
            Array::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ml_array = MLArray::from(array);

        let result = ml_array.extract_to_tensor::<i32>();
        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(error_msg.contains("type mismatch") && error_msg.contains("expected i32"));
    }

    #[test]
    fn test_mean_absolute_error_f32() {
        let lhs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let rhs: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean_absolute_error(&lhs, &rhs), 0.0);

        let rhs2: Vec<f32> = vec![1.5, 2.5, 3.5, 4.5];
        assert_eq!(mean_absolute_error(&lhs, &rhs2), 0.5);

        let rhs3: Vec<f32> = vec![0.0, 4.0, 0.0, 8.0];
        assert_eq!(mean_absolute_error(&lhs, &rhs3), 2.5);
    }

    #[test]
    fn test_mean_absolute_error_i32() {
        let lhs: Vec<i32> = vec![10, 20, 30, 40];
        let rhs: Vec<i32> = vec![10, 20, 30, 40];
        assert_eq!(mean_absolute_error(&lhs, &rhs), 0.0);

        let rhs2: Vec<i32> = vec![15, 25, 35, 45];
        assert_eq!(mean_absolute_error(&lhs, &rhs2), 5.0);
    }

    #[test]
    fn test_mean_absolute_error_u8() {
        let lhs: Vec<u8> = vec![10, 20, 30, 40];
        let rhs: Vec<u8> = vec![10, 20, 30, 40];
        assert_eq!(mean_absolute_error(&lhs, &rhs), 0.0);

        let rhs2: Vec<u8> = vec![5, 25, 25, 45];
        assert_eq!(mean_absolute_error(&lhs, &rhs2), 5.0);
    }

    #[test]
    fn test_mean_absolute_error_one_element_different() {
        let lhs: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let rhs: Vec<f64> = vec![1.0, 2.0, 7.0, 4.0];
        assert_eq!(mean_absolute_error(&lhs, &rhs), 1.0);
    }
}
