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

// Helper trait to allow safely converting typed ArrayBase to MLArray variant
pub(crate) trait IntoMLArray {
    fn into_mlarray(self) -> MLArray;
}

impl IntoMLArray for ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::Float32Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<half::f16>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::Float16Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<i32>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::Int32Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<u16>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::UInt16Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<u8>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::UInt8Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<i16>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::Int16Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<i8>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::Int8Array(self)
    }
}
impl IntoMLArray for ArrayBase<OwnedRepr<u32>, Dim<IxDynImpl>> {
    fn into_mlarray(self) -> MLArray {
        MLArray::UInt32Array(self)
    }
}

impl<T: MLType> From<ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>> for MLArray
where
    ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>: IntoMLArray,
{
    fn from(value: ArrayBase<OwnedRepr<T>, Dim<IxDynImpl>>) -> Self {
        value.into_mlarray()
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
    pub fn extract_to_tensor<T: MLType + 'static>(self) -> Array<T, Dim<IxDynImpl>> {
        self.try_extract_to_tensor()
            .unwrap_or_else(|e| panic!("{}", e))
    }

    /// Helper to convert standard ArrayBase via Any.
    fn downcast_any<A: 'static, T: 'static>(
        a: Array<A, Dim<IxDynImpl>>,
    ) -> Result<Array<T, Dim<IxDynImpl>>, String> {
        // Since we checked actual == expected, T is A. Safe to use boxed Any downcast.
        let boxed: Box<dyn std::any::Any> = Box::new(a);
        boxed
            .downcast::<Array<T, Dim<IxDynImpl>>>()
            .map(|b| *b)
            .map_err(|_| "Failed to downcast Any to Array".to_string())
    }

    /// Try to extract as typed tensor. Returns Err if type doesn't match the variant.
    pub fn try_extract_to_tensor<T: MLType + 'static>(
        self,
    ) -> Result<Array<T, Dim<IxDynImpl>>, String> {
        let actual = self.type_id();
        let expected = T::TY;
        if actual != expected {
            return Err(format!(
                "MLArray type mismatch: array holds type_id={} but extract requested type_id={}",
                actual, expected
            ));
        }

        match self {
            MLArray::Float32Array(a) => Self::downcast_any(a),
            MLArray::Float16Array(a) => Self::downcast_any(a),
            MLArray::Int32Array(a) => Self::downcast_any(a),
            MLArray::Int16Array(a) => Self::downcast_any(a),
            MLArray::Int8Array(a) => Self::downcast_any(a),
            MLArray::UInt32Array(a) => Self::downcast_any(a),
            MLArray::UInt16Array(a) => Self::downcast_any(a),
            MLArray::UInt8Array(a) => Self::downcast_any(a),
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
    pub fn into_contiguous_raw_vec_and_shape<T: MLType + Clone + 'static>(
        self,
    ) -> (Vec<T>, Vec<i32>) {
        let shape = self.shape().iter().map(|&i| i as i32).collect::<Vec<i32>>();
        let data = self.extract_to_tensor::<T>().into_contiguous_raw_vec();
        (data, shape)
    }
}

use ndarray::ArrayView;

macro_rules! impl_try_as_view {
    ($($fn_name:ident, $ty:ty, $variant:ident, $ty_str:expr);+ $(;)?) => {
        impl MLArray {
            $(
                pub fn $fn_name(&self) -> Result<ArrayView<'_, $ty, Dim<IxDynImpl>>, String> {
                    if let MLArray::$variant(a) = self {
                        Ok(a.view())
                    } else {
                        Err(format!(
                            concat!("MLArray type mismatch: expected ", $ty_str, ", found type_id={}"),
                            self.type_id_str()
                        ))
                    }
                }
            )+
        }
    };
}

impl_try_as_view! {
    try_as_view_f32, f32, Float32Array, "f32";
    try_as_view_f16, f16, Float16Array, "f16";
    try_as_view_i32, i32, Int32Array, "i32";
    try_as_view_i16, i16, Int16Array, "i16";
    try_as_view_i8, i8, Int8Array, "i8";
    try_as_view_u32, u32, UInt32Array, "u32";
    try_as_view_u16, u16, UInt16Array, "u16";
    try_as_view_u8, u8, UInt8Array, "u8";
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
