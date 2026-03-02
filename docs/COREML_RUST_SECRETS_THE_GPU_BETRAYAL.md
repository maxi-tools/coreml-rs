# CoreML in Rust: The Complete Guide (For Future Claude Opus Instances)

## Executive Summary

This document covers everything learned about using CoreML from Rust, including critical bugs in the `coreml-rs` library and their fixes.

## The Bugs We Found

### Bug #1: The INT32 Catastrophe (CRITICAL)

**Location**: `swift_library.swift` in `coreml-rs-fork`

**The Bug**: The `bindInputI32` function was using `MLMultiArrayDataType.float32` instead of `.int32`:

```swift
// BROKEN (original code)
let array = try MLMultiArray.init(
    dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.float32,  // WRONG!
    strides: stride, deallocator: deallocMultiArrayRust)

// FIXED
let array = try MLMultiArray.init(
    dataPointer: data, shape: arr, dataType: MLMultiArrayDataType.int32,  // CORRECT!
    strides: stride, deallocator: deallocMultiArrayRust)
```

**Impact**: All integer inputs (token IDs, attention masks) were being reinterpreted as float32 values. Token ID `[1, 2, 3]` became `[1.4e-45, 2.8e-45, 4.2e-45]` (denormalized floats). Models produced garbage/zero outputs.

**Symptoms**:
- Model outputs are all zeros
- Timing looks correct (model runs, just with garbage input)
- No error messages

### Bug #2: Missing Float16 Output Support

The original `coreml-rs` only supported f32 outputs. Many CoreML models (especially those converted with `compute_precision="float16"`) output f16.

**Fix**: Added `outputF16AsF32` function that:
1. Detects when output is stored as `MLFeatureValue` (non-backed outputs)
2. Reads raw f16 data from MLMultiArray
3. Converts to f32 using Swift's native `Float32(f16value)`

```swift
func outputF16AsF32(name: RustString) -> RustVec<Float32> {
    let out: MLMultiArray
    if let featureValue = output[nameStr] as? MLFeatureValue {
        out = featureValue.multiArrayValue!
    } else if let arr = output[nameStr] as? MLMultiArray {
        out = arr
    } else {
        return RustVec.init()
    }
    
    let l = out.count
    var v = RustVec<Float32>()
    out.withUnsafeMutableBytes { ptr, strides in
        if out.dataType == .float16 {
            let p = ptr.baseAddress!.assumingMemoryBound(to: Float16.self)
            for i in 0..<l {
                v.push(value: Float32(p[i]))
            }
        } else if out.dataType == .float32 {
            let p = ptr.baseAddress!.assumingMemoryBound(to: Float32.self)
            for i in 0..<l {
                v.push(value: p[i])
            }
        }
    }
    return v
}
```

## ComputePlatform Selection

### The Options

| Rust Enum | Swift MLComputeUnits | Description |
|-----------|---------------------|-------------|
| `Cpu` | `.cpuOnly` | CPU only |
| `CpuAndANE` | `.cpuAndNeuralEngine` | CPU + Neural Engine (NO GPU) |
| `CpuAndGpu` | `.cpuAndGPU` | CPU + GPU (NO ANE) |
| `All` | `.all` | CPU + GPU + ANE |

### Performance Results (M-series Mac)

**mia-low embedding model:**

| Platform    | Load Time | Inference | Throughput |
|-------------|-----------|-----------|------------|
| Cpu         | 2.4s      | 204ms     | 4.9 emb/s  |
| CpuAndANE   | 13.2s     | 82ms      | 12.2 emb/s |
| CpuAndGpu   | 2.5s      | **48ms**  | **20.6 emb/s** |
| All         | 13.5s     | 84ms      | 11.9 emb/s |

**All tiers with CpuAndGpu:**

| Tier     | Load Time | Inference | Throughput |
|----------|-----------|-----------|------------|
| mia-low  | 2.6s      | 48ms      | 20.8 emb/s |
| mia-mid  | 12.9s     | 299ms     | 3.3 emb/s  |
| mia-high | 44.2s     | 559ms     | 1.8 emb/s  |

**Key insight**: `CpuAndGpu` is the best choice - fastest inference AND fastest load times. ANE-based options (CpuAndANE, All) are ~70% slower with 5x longer load times.

### Why ANE is Slower for Transformers

The Neural Engine is optimized for convolutions and specific matrix operations. **Model architecture determines optimal compute unit:**

**MobileNetV2 (Conv-based image model):**
| Platform | Inference | Throughput |
|----------|-----------|------------|
| CPU      | 8.3ms     | 121/s      |
| CPU+ANE  | **1.2ms** | **848/s**  |
| CPU+GPU  | 2.7ms     | 375/s      |
| ALL      | **1.1ms** | **926/s**  |

**Embedding Model (Transformer-based):**
| Platform | Inference | Throughput |
|----------|-----------|------------|
| CPU      | 204ms     | 4.9/s      |
| CPU+ANE  | 82ms      | 12.2/s     |
| CPU+GPU  | **48ms**  | **20.6/s** |
| ALL      | 84ms      | 11.9/s     |

**Recommendation:**
- **Conv networks (ResNet, MobileNet, YOLO):** Use `All` or `CpuAndANE`
- **Transformers (BERT, GPT, embeddings):** Use `CpuAndGpu`

ANE excels at conv ops but struggles with attention mechanisms and large matrix multiplies common in transformers.

### Power Efficiency

Even when ANE is slower, it's more power efficient. Embedding model power measurements:

| Platform | Throughput | Power | Efficiency |
|----------|------------|-------|------------|
| CPU      | 4.98/s     | 7.1W  | 0.70 emb/W |
| CPU+ANE  | 11.96/s    | 8.6W  | **1.40 emb/W** |
| CPU+GPU  | 20.98/s    | 67.1W | 0.31 emb/W |
| ALL      | 11.59/s    | 8.3W  | **1.39 emb/W** |

**Key insight:** GPU uses **8x more power** for only 2x throughput. ANE is **4.5x more power efficient**.

**Recommendation:**
- **Battery/thermal constrained:** Use `CpuAndANE` or `All`
- **Plugged in, max speed:** Use `CpuAndGpu`

## Usage Example (Rust)

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::ArrayD;

// Load model
let mut options = CoreMLModelOptions::default();
options.compute_platform = ComputePlatform::CpuAndGpu;  // Best for transformers

let mut model = CoreMLModelWithState::new(mlpackage_path, options)
    .load()
    .expect("Failed to load model");

// Prepare inputs (INT32 for token IDs)
let input_ids: ArrayD<i32> = ArrayD::from_shape_vec(
    ndarray::IxDyn(&[1, 512]),
    token_ids_vec,
).unwrap();

let attention_mask: ArrayD<i32> = ArrayD::from_shape_vec(
    ndarray::IxDyn(&[1, 512]),
    mask_vec,
).unwrap();

// Add inputs
model.add_input("input_ids", input_ids)?;
model.add_input("attention_mask", attention_mask)?;

// Run inference
let output = model.predict()?;

// Get embeddings (f16 models automatically convert to f32)
if let Some(MLArray::Float32Array(emb)) = output.outputs.get("embeddings") {
    // Use embeddings
}
```

## Debugging Tips

### If outputs are all zeros:
1. Check input types - are they actually INT32 or being misinterpreted?
2. Add debug to Swift `predict()` to print input dict contents
3. Verify token values: `arr[0]` should show integers like `9707`, not tiny floats like `1.4e-45`

### Swift debug helper:
```swift
func dbg(_ msg: String) {
    FileHandle.standardError.write("SWIFT: \(msg)\n".data(using: .utf8)!)
}
```

Use `FileHandle.standardError` instead of `print()` - stdout may be buffered.

### MLMultiArrayDataType values:
- 65552 = float16
- 65568 = float32
- 131104 = int32

## Swift Library Linking

If you get `Library not loaded: @rpath/libswift_Concurrency.dylib`:

```bash
install_name_tool -change @rpath/libswift_Concurrency.dylib \
    /usr/lib/swift/libswift_Concurrency.dylib \
    ./your_binary
```

## Files Changed in coreml-rs Fork

1. `swift-library/Sources/swift-library/swift_library.swift`:
   - Fixed `bindInputI32` to use `.int32` datatype
   - Added `outputF16AsF32` function
   - Added `All` case to `ComputePlatform` switch

2. `src/swift.rs`:
   - Added `All` variant to `ComputePlatform` enum
   - Added `outputF16AsF32` binding

3. `src/mlmodel.rs`:
   - Modified `predict()` to handle f16 outputs without binding
   - Call `outputF16AsF32` for f16 outputs

## Summary

1. **Use `CpuAndGpu`** for transformer models - ANE is a trap
2. **Fix the INT32 bug** if using coreml-rs-fork < version with fix
3. **Add f16 support** for models with float16 outputs
4. **Debug with stderr** not stdout in Swift code