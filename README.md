## Core-ML Rust Bindings

`coreml-rs` is a high-performance Rust library providing safe bindings for Apple's Core ML framework. It enables Rust developers to load and run Core ML models (`.mlmodel`, `.mlpackage`) on iOS, macOS, watchOS, and tvOS with full support for neural engine (ANE) acceleration and batch inference.

Built using `swift-bridge` for zero-copy interop and backed by `bytemuck`/`half` for efficient data layout.

### Key Features

- **Safety Fixes** (Recent): ANE output backing use-after-free fix, buffer capacity leak fix for improved memory safety
- **Inference**: Single and batch inference with configurable compute platforms (CPU, GPU, ANE)
- **Input Introspection**: New `input_shape()` API to query model input dimensions at runtime
- **Model Loading**: From binary buffers or zip archives containing `.mlpackage` files
- **Data Handling**: `ndarray` integration for flexible tensor I/O

### Status

Production-ready for macOS/iOS inference. Core API stable. Recent focus on safety audits and comprehensive E2E testing.

## Roadmap

- Expand input format support (image preprocessing pipelines)
- Zerocopy tensor types for zero-allocation inference
- Advanced configuration (caching, priority settings)

## Features

- **Model Loading**: Load Core ML models into Rust applications.
- **Inference**: Perform inference using loaded models.
- **Data Handling**: Manage input and output data for model inference.

## Installation & Prerequisites

### Requirements

- **Swift Runtime**: `libswift_Concurrency.dylib` must be available. Place it alongside the built binary or in the system library path.
- **Platforms**: macOS 11.0+ for building; iOS 14.0+ for target deployment
- **Rust**: 1.70+ (MSRV tested via CI)

### Setup

```toml
[dependencies]
coreml-rs = { version = "0.5", git = "https://github.com/maxi-tools/coreml-rs" }
ndarray = "0.16"
```

Build with:
```bash
cargo build --release
# Ensure libswift_Concurrency.dylib is in target/release/ or LD_LIBRARY_PATH
```

## API Overview

### Core Types

| Type | Purpose |
|------|---------|
| `CoreMLModelWithState` | Single-instance model for straightforward inference |
| `CoreMLBatchModelWithState` | Batch-optimized model for processing multiple inputs |
| `ComputePlatform` | Enum: `CpuAndAne`, `CpuAndGpu`, `All` |
| `MLArray` | Wrapper around model output dictionaries |

### `input_shape()` — New in 0.5.5

Query input tensor dimensions without running inference:

```rust
let model = CoreMLModelWithState::from_buf(buf, options);
let (batch, height, width, channels) = model.input_shape("image")?;
println!("Input shape: {}x{}x{}x{}", batch, height, width, channels);
```

Useful for validating inputs before prediction or preallocating buffers.

## Usage Examples

### Simple Inference

Load a Core ML model from a `.mlmodel` file and perform a single inference:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

### Batch Inference

Perform batch inference by adding multiple inputs:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    // Add multiple inputs for batch processing
    for i in 0..10 {
        let _ = model.add_input("image", input.clone().into_dyn(), i);
    }

    let output = model.predict().unwrap();

    // Process batch outputs
    for i in 0..10 {
        let v = output.bytesFrom(&format!("output_1_{}", i));
        let batch_output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();
        // Use batch_output as needed
    }
}
```

### Loading from Zip Archive

Load a model from a zip archive containing an `.mlpackage`:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;
use std::path::PathBuf;

fn unzip_to_path_from_hash(buf: &[u8]) -> Option<PathBuf> {
    fn get_cache_filename(model_buffer: &[u8]) -> String {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(model_buffer);
        let hash = hasher.finalize();
        format!("{:x}.mlpackage", hash)
    }
    let name = get_cache_filename(buf);

    let path = PathBuf::from("/tmp/coreml-aftershoot/");
    let path = path.join(name);
    _ = std::fs::remove_dir_all(&path);
    _ = std::fs::remove_file(&path);

    let mut res = zip::ZipArchive::new(std::io::Cursor::new(buf)).ok()?;
    res.extract(&path).ok()?;

    let m = path.join("model.mlpackage");
    if m.exists() {
        Some(m)
    } else {
        None
    }
}

pub fn main() {
    let buf = std::fs::read("./demo/model_2.zip").unwrap();

    let model_path = unzip_to_path_from_hash(&buf).unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::new(model_path, model_options).load().unwrap();

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

### Memory Management

Unload and reload models to manage memory usage:

```rust
use coreml_rs::{ComputePlatform, CoreMLModelOptions, CoreMLModelWithState};
use ndarray::Array4;

pub fn main() {
    let file = std::fs::read("./demo/model_3.mlmodel").unwrap();

    let mut model_options = CoreMLModelOptions::default();
    model_options.compute_platform = ComputePlatform::CpuAndANE;

    let mut model = CoreMLModelWithState::from_buf(file, model_options);

    let mut input = Array4::<f32>::zeros((1, 3, 512, 512));
    input.fill(1.0f32);

    let Ok(_) = model.add_input("image", input.clone().into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Unload model to free memory
    let unloaded_model = model.unload().unwrap();

    // Later, reload the model
    let mut model = unloaded_model.load().unwrap();

    // Add input again and predict
    let Ok(_) = model.add_input("image", input.into_dyn()) else {
        panic!("failed to add input feature, `image` to the model");
    };

    let output = model.predict();

    let v = output.unwrap().bytesFrom("output_1".to_string());
    let output: Array4<f32> = Array4::from_shape_vec([1, 3, 2048, 2048], v).unwrap();

    // Use output as needed
}
```

**Note**: These examples assume specific model inputs/outputs. Adjust based on your model's specifications.

---

## Testing

Run the test suite (requires valid `.mlmodel` artifacts):

```bash
cargo test --lib
cargo test --test load
cargo test --test mlarray_types
```

### E2E Scenarios

Full-stack integration tests including ANE output backing and buffer lifecycle:

```bash
cargo nextest run --release
```

See `tests/` for fixture models and expected behavior around:
- ANE output handling (use-after-free fixes)
- Buffer capacity management
- Input shape queries
- Batch inference scenarios

---

## Safety & Memory Management

Recent releases (0.5.5+) address:
- **ANE Output Backing**: Fixed use-after-free when maintaining output buffers across predictions
- **Buffer Capacity**: Corrected capacity field in loaded buffers to prevent leaks
- **Safe Send/Sync**: All model types properly gated with `#[derive(Send, Sync)]` guards

All unsafe code is documented with `// SAFETY:` comments and validated by CI. Review `src/mlmodel.rs` for details.

---

## Contributing

Contributions welcome! Areas of focus:
- Input preprocessing (image normalization pipelines)
- Performance tuning (Metal shader integration)
- Extended platform support

See [CLAUDE.md](../CLAUDE.md) in the monorepo for contribution guidelines.