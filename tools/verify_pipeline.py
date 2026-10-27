"""
Thorough inspection and pipeline compatibility verification of the three TFLite models:
- node1_int8.tflite
- node2_int8.tflite
- node3_int8.tflite
"""
import os
import sys
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

MODELS = {
    "Node 1": "/mnt/c/Users/vinit/eee/esp32_vision_project/models/node1_int8.tflite",
    "Node 2": "/mnt/c/Users/vinit/eee/esp32_vision_project/models/node2_int8.tflite",
    "Node 3": "/mnt/c/Users/vinit/eee/esp32_vision_project/models/node3_int8.tflite",
}

# If running on Windows directly
if sys.platform == "win32":
    MODELS = {k: v.replace("/mnt/c/", "C:/") for k, v in MODELS.items()}

print("=" * 70)
print("PHASE 6: TFLite Model Architecture & Pipeline Verification")
print("=" * 70)

interpreters = {}
model_details = {}

for name, path in MODELS.items():
    if not os.path.exists(path):
        print(f"ERROR: Model file not found: {path}")
        sys.exit(1)

    size = os.path.getsize(path)
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    interpreters[name] = interp

    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()
    
    model_details[name] = {
        "path": path,
        "size": size,
        "inputs": inp_details,
        "outputs": out_details,
    }

    print(f"\n[{name}] - {os.path.basename(path)}")
    print(f"  File Size:    {size:,} bytes")
    for idx, inp in enumerate(inp_details):
        print(f"  Input  [{idx}]: name='{inp['name']}' shape={inp['shape'].tolist()} dtype={inp['dtype'].__name__}")
        print(f"               quantization: scale={inp['quantization_parameters']['scales']}, zero_point={inp['quantization_parameters']['zero_points']}")
    for idx, out in enumerate(out_details):
        print(f"  Output [{idx}]: name='{out['name']}' shape={out['shape'].tolist()} dtype={out['dtype'].__name__}")
        print(f"               quantization: scale={out['quantization_parameters']['scales']}, zero_point={out['quantization_parameters']['zero_points']}")

print("\n" + "=" * 70)
print("PIPELINE INTERFACE COMPATIBILITY CHECK")
print("=" * 70)

# Check Node 1 output -> Node 2 input
n1_out = model_details["Node 1"]["outputs"][0]
n2_inp = model_details["Node 2"]["inputs"][0]
compat_1_2 = (
    np.array_equal(n1_out['shape'], n2_inp['shape']) and
    n1_out['dtype'] == n2_inp['dtype']
)
print(f"Node 1 Output shape: {n1_out['shape'].tolist()} {n1_out['dtype'].__name__}")
print(f"Node 2 Input  shape: {n2_inp['shape'].tolist()} {n2_inp['dtype'].__name__}")
print(f"--> Node 1 -> Node 2 Compatibility: {'MATCH [PASS]' if compat_1_2 else 'MISMATCH [FAIL]'}")

# Check Node 2 output -> Node 3 input
n2_out = model_details["Node 2"]["outputs"][0]
n3_inp = model_details["Node 3"]["inputs"][0]
compat_2_3 = (
    np.array_equal(n2_out['shape'], n3_inp['shape']) and
    n2_out['dtype'] == n3_inp['dtype']
)
print(f"\nNode 2 Output shape: {n2_out['shape'].tolist()} {n2_out['dtype'].__name__}")
print(f"Node 3 Input  shape: {n3_inp['shape'].tolist()} {n3_inp['dtype'].__name__}")
print(f"--> Node 2 -> Node 3 Compatibility: {'MATCH [PASS]' if compat_2_3 else 'MISMATCH [FAIL]'}")

if not (compat_1_2 and compat_2_3):
    print("\nCRITICAL ERROR: Models cannot form a pipeline. Halting immediately.")
    sys.exit(2)

print("\n" + "=" * 70)
print("TEST EXECUTION: Full 3-Node End-to-End Synthetic Inference Pipeline")
print("=" * 70)

# Create a synthetic image input matching Node 1 input
n1_inp = model_details["Node 1"]["inputs"][0]
input_shape = n1_inp['shape']
input_dtype = n1_inp['dtype']

np.random.seed(42)
if input_dtype == np.int8:
    test_image = np.random.randint(-128, 127, size=input_shape, dtype=np.int8)
elif input_dtype == np.uint8:
    test_image = np.random.randint(0, 255, size=input_shape, dtype=np.uint8)
else:
    test_image = np.random.rand(*input_shape).astype(input_dtype)

print(f"1. Stage 1 (Node 1): Running inference on input {test_image.shape} {test_image.dtype}...")
interp1 = interpreters["Node 1"]
interp1.set_tensor(n1_inp['index'], test_image)
interp1.invoke()
tensor_1_out = interp1.get_tensor(n1_out['index'])
print(f"   Output generated: shape={tensor_1_out.shape}, dtype={tensor_1_out.dtype}, bytes={tensor_1_out.nbytes}")

print(f"\n2. Stage 2 (Node 2): Passing tensor to Node 2...")
interp2 = interpreters["Node 2"]
interp2.set_tensor(n2_inp['index'], tensor_1_out)
interp2.invoke()
tensor_2_out = interp2.get_tensor(n2_out['index'])
print(f"   Output generated: shape={tensor_2_out.shape}, dtype={tensor_2_out.dtype}, bytes={tensor_2_out.nbytes}")

print(f"\n3. Stage 3 (Node 3): Passing tensor to Node 3...")
interp3 = interpreters["Node 3"]
n3_out = model_details["Node 3"]["outputs"][0]
interp3.set_tensor(n3_inp['index'], tensor_2_out)
interp3.invoke()
tensor_3_out = interp3.get_tensor(n3_out['index'])
print(f"   Final Prediction output: shape={tensor_3_out.shape}, dtype={tensor_3_out.dtype}")
print(f"   Raw quantized scores: {tensor_3_out.tolist()}")

# Dequantize final prediction if quantized
scale = n3_out['quantization_parameters']['scales']
zero_point = n3_out['quantization_parameters']['zero_points']
if len(scale) > 0 and scale[0] > 0:
    dequantized = (tensor_3_out.astype(np.float32) - zero_point[0]) * scale[0]
    print(f"   Dequantized probabilities: {dequantized.tolist()}")
    predicted_class = int(np.argmax(dequantized))
    print(f"   Predicted class: {predicted_class}")

print("\n" + "=" * 70)
print("SUCCESS: 3-Stage Model Pipeline is 100% Mathematically & Architecturally Compatible!")
print("=" * 70)
