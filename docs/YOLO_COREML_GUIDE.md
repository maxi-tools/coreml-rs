# Guide: Optimizing YOLO Models for Core ML and Apple Neural Engine

This guide provides best practices for converting and optimizing YOLO (v26 or similar) object detection models for high-performance inference on iOS/macOS devices using Core ML and the Apple Neural Engine (ANE).

## 1. Best Practices for Converting YOLO Models to Core ML

The recommended conversion path is from a PyTorch model to a Core ML `mlprogram` model package. The modern `mlprogram` format is more flexible and offers better performance compared to the older `.mlmodel` neural network format.

### Key Conversion Steps:

1.  **Prepare the PyTorch Model**: Before conversion, always set your model to evaluation mode to disable training-specific layers like dropout.
    ```python
    import torch

    model = YourYOLOModel()
    model.load_state_dict(torch.load("yolo.pt"))
    model.eval()
    ```

2.  **Trace the Model**: Capture the model's graph using `torch.jit.trace`. This requires providing an example input tensor with a specific shape.
    ```python
    # Example input for a 640x640 image
    example_input = torch.rand(1, 3, 640, 640)
    traced_model = torch.jit.trace(model, example_input)
    ```
    *Note: The newer `torch.export` is also an option, but as of `coremltools` 8.0, `torch.jit.trace` is the more stable and recommended path.*

3.  **Convert with `coremltools`**: Use `coremltools.convert()` to perform the conversion. The most critical step here is to define the model's input type.

    ```python
    import coremltools as ct

    # Define the input as an image for better performance and ease of use
    image_input = ct.ImageType(
        name="image_input",
        shape=(1, 3, 640, 640), # Batch, Channels, Height, Width
        scale=1/255.0, # Normalize pixel values to [0,1]
        bias=[0.0, 0.0, 0.0],
        color_layout=ct.colorlayout.RGB
    )

    # Convert the model
    coreml_model = ct.convert(
        traced_model,
        inputs=[image_input],
        convert_to="mlprogram" # Modern, recommended format
    )

    # Save the model
    coreml_model.save("YOLO.mlpackage")
    ```

## 2. Image Input Preprocessing and MLShapedArray

Handling image inputs correctly is crucial for both accuracy and performance.

*   **Use `ct.ImageType`**: As shown above, specifying the input as `ct.ImageType` is highly recommended. This allows you to pass a `CVPixelBuffer` directly to the model on-device, avoiding inefficient memory copies. It also allows the Vision framework to handle resizing and formatting automatically.

*   **Embed Preprocessing**: `coremltools` lets you embed preprocessing operations into the model itself. For YOLO models commonly trained with pixel values normalized to the `[0, 1]` range, you can set `scale = 1/255.0`. If your model uses more complex normalization (like subtracting a mean and dividing by a standard deviation, common in `torchvision`), you can calculate the equivalent `scale` and `bias` values and include them.

    For standard torchvision normalization:
    `mean = [0.485, 0.456, 0.406]`
    `std = [0.229, 0.224, 0.225]`

    The equivalent Core ML parameters are:
    ```python
    scale = 1 / (255.0 * 0.226) # Use average std dev
    red_bias = -0.485 / 0.229
    green_bias = -0.456 / 0.224
    blue_bias = -0.406 / 0.225
    bias = [red_bias, green_bias, blue_bias]

    image_input = ct.ImageType(..., scale=scale, bias=bias)
    ```

*   **`MLShapedArray`**: While `CVPixelBuffer` (via `ImageType`) is best for image inputs, `MLShapedArray` is the Swift equivalent of `MLMultiArray` and is used for generic tensor data. You would typically interact with it if your model outputs raw tensor data (like bounding box coordinates) rather than a specialized output type. For YOLO, the primary input should be an image.

## 3. NMS (Non-Maximum Suppression) Implementation

How you handle NMS has a significant impact on performance. The `coremltools` documentation **does not** provide a built-in, one-click option to add NMS during conversion.

**Option 1 (Recommended): Embed NMS in the Model Graph**

The most performant approach is to have the NMS operations as part of the model itself, so they can be accelerated by the GPU or ANE.

*   Many modern YOLO frameworks (e.g., Ultralytics YOLOv5/v8) allow you to export the model (e.g., to ONNX format) with the NMS and post-processing layers included.
*   **Workflow**:
    1.  Export your YOLO model from PyTorch to a format like ONNX or TorchScript, ensuring the export flag to include NMS is enabled.
    2.  Convert this exported model to Core ML.
*   If all the operations used in the NMS implementation are supported by `coremltools`, they will be converted and optimized. The model will then directly output the final filtered bounding boxes.

**Option 2 (Fallback): Implement NMS in App Code**

If you cannot embed NMS into the model graph, the model will output raw prediction tensors (e.g., a tensor of `[batch, num_boxes, 5 + num_classes]`). You are then responsible for implementing the NMS logic in your application's Swift code.

*   This is less performant as the NMS computation runs on the CPU and involves transferring potentially large output tensors from the ANE/GPU to the CPU.
*   The Vision framework may offer some utilities, but a fully manual implementation might be necessary to parse the raw YOLO output format.

## 4. Quantization Strategies for Real-Time Inference

Quantization reduces model size and can dramatically speed up inference, especially on the ANE. The modern `coremltools.optimize` API works on `mlprogram` models.

**1. FP16 (Default)**
*   By default, converting to `mlprogram` uses float 16 precision, which provides a 2x size reduction over float 32 with minimal to no accuracy loss. This is the baseline and already highly optimized for the ANE.

**2. Post-Training Weight Quantization (8-bit)**
*   **What it is**: Quantizes only the model's weights to `int8`. This is a quick, data-free method to reduce model size further.
*   **How**: Use `coremltools.optimize.coreml.linear_quantize_weights`.
    ```python
    import coremltools.optimize as cto

    op_config = cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
    config = cto.coreml.OptimizationConfig(global_config=op_config)
    quantized_model = cto.coreml.linear_quantize_weights(coreml_model, config=config)
    ```
*   **Benefit**: Reduces model size by ~4x from FP32 and can improve performance by lowering memory bandwidth requirements.

**3. Post-Training Full Quantization (W8A8)**
*   **What it is**: Quantizes both weights and activations to `int8`. This is the key to unlocking the highest performance on the ANE on A17 Pro/M4 chips and newer.
*   **How**: This is a two-step process that requires a small set of representative calibration data.
    ```python
    # 1. Prepare calibration data (a few hundred sample images)
    #    You need a data loader that yields dictionaries of input names to numpy arrays.
    
    # 2. Quantize activations
    act_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.experimental.OpActivationLinearQuantizerConfig(mode="linear_symmetric")
    )
    model_a8 = cto.coreml.experimental.linear_quantize_activations(
        coreml_model, act_config, calibration_data_loader
    )

    # 3. Quantize weights on the activation-quantized model
    weight_config = cto.coreml.OptimizationConfig(
        global_config=cto.coreml.OpLinearQuantizerConfig(mode="linear_symmetric")
    )
    model_w8a8 = cto.coreml.linear_quantize_weights(model_a8, config=weight_config)
    ```
*   **Benefit**: Can significantly improve latency on compute-bound models on supported hardware.

**4. Quantization Aware Training (QAT)**
*   **What it is**: The most advanced technique. It simulates quantization during a short fine-tuning phase in PyTorch, allowing the model to adapt and recover accuracy that might be lost during post-training quantization.
*   **How**: Use `coremltools.optimize.torch.LinearQuantizer` on the PyTorch model *before* converting to Core ML. This is a more involved process integrated into a training loop.
*   **Benefit**: Best accuracy for `int8` models.

## 5. Flexible Input Shapes for Different Resolutions

YOLO models often need to run on images of various sizes. `coremltools` allows you to define flexible input shapes during conversion.

**Option 1 (Recommended): Enumerated Shapes**
*   Use this when you have a known, finite set of input resolutions. This is the most performant flexible option, as Core ML can pre-compile and optimize for each specific shape.
    ```python
    input_shape = ct.EnumeratedShapes(
        shapes=[[1, 3, 416, 416], [1, 3, 640, 640], [1, 3, 800, 800]],
        default=[1, 3, 640, 640]
    )

    image_input = ct.ImageType(name="image_input", shape=input_shape, ...)
    # then pass this to ct.convert()
    ```

**Option 2: Range of Shapes**
*   Use this for more dynamic scenarios where the input size can vary within a continuous range.
    ```python
    input_shape = ct.Shape(
        shape=(
            1, 3,
            ct.RangeDim(lower_bound=320, upper_bound=800, default=640),
            ct.RangeDim(lower_bound=320, upper_bound=800, default=640)
        )
    )

    image_input = ct.ImageType(name="image_input", shape=input_shape, ...)
    ```
*   **Note**: Bounded ranges give the compiler more optimization opportunities than unbounded ranges. Unbounded ranges are not supported for the modern `mlprogram` format.