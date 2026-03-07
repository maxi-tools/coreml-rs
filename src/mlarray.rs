use half::f16;
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr};

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
}

impl MLArray {
    pub fn shape(&self) -> &[usize] {
        match self {
            MLArray::Float32Array(array_base) => array_base.shape(),
            MLArray::Float16Array(array_base) => array_base.shape(),
            MLArray::Int32Array(array_base) => array_base.shape(),
            MLArray::Int16Array(array_base) => array_base.shape(),
            MLArray::Int8Array(array_base) => array_base.shape(),
            MLArray::UInt32Array(array_base) => array_base.shape(),
            MLArray::UInt16Array(array_base) => array_base.shape(),
            MLArray::UInt8Array(array_base) => array_base.shape(),
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

/// Type ID constants used by MLType trait and MLArray dispatch.
pub(crate) const TY_F32: usize = 0;
pub(crate) const TY_F16: usize = 1;
pub(crate) const TY_I32: usize = 2;
pub(crate) const TY_U16: usize = 3;
pub(crate) const TY_U8: usize = 4;
pub(crate) const TY_I16: usize = 5;
pub(crate) const TY_I8: usize = 6;
pub(crate) const TY_U32: usize = 7;

pub trait MLType {
    const TY: usize;
}

impl MLType for f32 {
    const TY: usize = TY_F32;
}
impl MLType for half::f16 {
    const TY: usize = TY_F16;
}
impl MLType for i32 {
    const TY: usize = TY_I32;
}
impl MLType for u16 {
    const TY: usize = TY_U16;
}
impl MLType for u8 {
    const TY: usize = TY_U8;
}
impl MLType for i16 {
    const TY: usize = TY_I16;
}
impl MLType for i8 {
    const TY: usize = TY_I8;
}
impl MLType for u32 {
    const TY: usize = TY_U32;
}

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray {
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        unsafe {
            match T::TY {
                TY_F32 => MLArray::Float32Array(std::mem::transmute(value)),
                TY_F16 => MLArray::Float16Array(std::mem::transmute(value)),
                TY_I32 => MLArray::Int32Array(std::mem::transmute(value)),
                TY_U16 => MLArray::UInt16Array(std::mem::transmute(value)),
                TY_U8 => MLArray::UInt8Array(std::mem::transmute(value)),
                TY_I16 => MLArray::Int16Array(std::mem::transmute(value)),
                TY_I8 => MLArray::Int8Array(std::mem::transmute(value)),
                TY_U32 => MLArray::UInt32Array(std::mem::transmute(value)),
                _ => panic!("not supported"),
            }
        }
    }
}

impl MLArray {
    fn type_id(&self) -> usize {
        match self {
            MLArray::Float32Array(_) => TY_F32,
            MLArray::Float16Array(_) => TY_F16,
            MLArray::Int32Array(_) => TY_I32,
            MLArray::Int16Array(_) => TY_I16,
            MLArray::Int8Array(_) => TY_I8,
            MLArray::UInt32Array(_) => TY_U32,
            MLArray::UInt16Array(_) => TY_U16,
            MLArray::UInt8Array(_) => TY_U8,
        }
    }

    /// Extract the array as a typed tensor. Panics if type doesn't match the variant.
    pub fn extract_to_tensor<T: MLType>(self) -> Array<T, Dim<IxDynImpl>> {
        self.try_extract_to_tensor()
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Try to extract as typed tensor. Returns Err if type doesn't match the variant.
    pub fn try_extract_to_tensor<T: MLType>(self) -> Result<Array<T, Dim<IxDynImpl>>, String> {
        let actual = self.type_id_str();
        let expected = T::TY;
        if actual != expected {
            return Err(format!(
                "MLArray type mismatch: array holds type_id={} but extract requested type_id={}",
                actual, expected
            ));
        }
        unsafe {
            Ok(match self {
                MLArray::Float32Array(a) => std::mem::transmute(a),
                MLArray::Float16Array(a) => std::mem::transmute(a),
                MLArray::Int32Array(a) => std::mem::transmute(a),
                MLArray::Int16Array(a) => std::mem::transmute(a),
                MLArray::Int8Array(a) => std::mem::transmute(a),
                MLArray::UInt32Array(a) => std::mem::transmute(a),
                MLArray::UInt16Array(a) => std::mem::transmute(a),
                MLArray::UInt8Array(a) => std::mem::transmute(a),
            })
        }
    }
}

impl<T> MLArrayBaseExt for ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>
where
    T: Clone,
{
    type Item = T;

    fn into_contiguous_raw_vec(self) -> Vec<Self::Item> {
        let contiguous = if self.is_standard_layout() {
            self
        } else {
            self.as_standard_layout().into_owned()
        };
        let (data, offset) = contiguous.into_raw_vec_and_offset();
        assert!(
            matches!(offset, Some(0) | None),
            "array base offset is not zero; bad aligned data"
        );
        data
    }
}

pub trait MLArrayBaseExt {
    type Item;
    fn into_contiguous_raw_vec(self) -> Vec<Self::Item>;
}

impl MLArray {
    pub fn into_contiguous_raw_vec_and_shape<T: MLType>(self) -> (Vec<T>, Vec<i32>) {
        let shape = self.shape().iter().map(|&i| i as i32).collect::<Vec<i32>>();
        let data = self.extract_to_tensor::<T>().into_contiguous_raw_vec();
        (data, shape)
    }
}

use ndarray::ArrayView;

impl MLArray {
    pub fn try_as_view_f32(&self) -> Result<ArrayView<f32, Dim<IxDynImpl>>, String> {
        if let MLArray::Float32Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected f32, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_f16(&self) -> Result<ArrayView<f16, Dim<IxDynImpl>>, String> {
        if let MLArray::Float16Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected f16, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_i32(&self) -> Result<ArrayView<i32, Dim<IxDynImpl>>, String> {
        if let MLArray::Int32Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected i32, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_i16(&self) -> Result<ArrayView<i16, Dim<IxDynImpl>>, String> {
        if let MLArray::Int16Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected i16, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_i8(&self) -> Result<ArrayView<i8, Dim<IxDynImpl>>, String> {
        if let MLArray::Int8Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected i8, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_u32(&self) -> Result<ArrayView<u32, Dim<IxDynImpl>>, String> {
        if let MLArray::UInt32Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected u32, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_u16(&self) -> Result<ArrayView<u16, Dim<IxDynImpl>>, String> {
        if let MLArray::UInt16Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected u16, found type_id={}",
                self.type_id_str()
            ))
        }
    }

    pub fn try_as_view_u8(&self) -> Result<ArrayView<u8, Dim<IxDynImpl>>, String> {
        if let MLArray::UInt8Array(a) = self {
            Ok(a.view())
        } else {
            Err(format!(
                "MLArray type mismatch: expected u8, found type_id={}",
                self.type_id_str()
            ))
        }
    }
}

impl MLArray {
    pub fn type_id_str(&self) -> &'static str {
        match self {
            MLArray::Float32Array(_) => "f32",
            MLArray::Float16Array(_) => "f16",
            MLArray::Int32Array(_) => "i32",
            MLArray::Int16Array(_) => "i16",
            MLArray::Int8Array(_) => "i8",
            MLArray::UInt32Array(_) => "u32",
            MLArray::UInt16Array(_) => "u16",
            MLArray::UInt8Array(_) => "u8",
        }
    }
}
