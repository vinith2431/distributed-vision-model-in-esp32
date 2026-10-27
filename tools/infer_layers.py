"""
Inference runner for the 3-Node Distributed Vision System.
Loads node1_int8.tflite, node2_int8.tflite, node3_int8.tflite.
Runs inference stage by stage and saves intermediate activation tensors and predictions.
"""
import sys
import os
import json
import numpy as np

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 infer_layers.py <input_rgb_path> [output_dir]")
        sys.exit(1)

    input_rgb_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(input_rgb_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load input image
    raw_bytes = open(input_rgb_path, "rb").read()
    if len(raw_bytes) != 96 * 96 * 3:
        raise ValueError(f"Expected 27648 bytes (96x96x3), got {len(raw_bytes)}")

    # Convert uint8 [0..255] to int8 [-128..127]
    img = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((1, 96, 96, 3)).astype(np.int32) - 128
    img = img.astype(np.int8)

    # Base path for models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    m1_path = os.path.join(base_dir, "models", "node1_int8.tflite")
    m2_path = os.path.join(base_dir, "models", "node2_int8.tflite")
    m3_path = os.path.join(base_dir, "models", "node3_int8.tflite")

    # 2. Stage 1: Node 1 Inference
    i1 = Interpreter(model_path=m1_path)
    i1.allocate_tensors()
    i1.set_tensor(i1.get_input_details()[0]['index'], img)
    i1.invoke()
    t1_out = i1.get_tensor(i1.get_output_details()[0]['index'])  # Shape [1, 12, 12, 32]

    # Save Node 1 output tensor
    t1_file = os.path.join(output_dir, "node1_activation.npy")
    np.save(t1_file, t1_out)

    # 3. Stage 2: Node 2 Inference
    i2 = Interpreter(model_path=m2_path)
    i2.allocate_tensors()
    i2.set_tensor(i2.get_input_details()[0]['index'], t1_out)
    i2.invoke()
    t2_out = i2.get_tensor(i2.get_output_details()[0]['index'])  # Shape [1, 12, 12, 48]

    # Save Node 2 output tensor
    t2_file = os.path.join(output_dir, "node2_activation.npy")
    np.save(t2_file, t2_out)

    # 4. Stage 3: Node 3 Inference
    i3 = Interpreter(model_path=m3_path)
    i3.allocate_tensors()
    i3.set_tensor(i3.get_input_details()[0]['index'], t2_out)
    i3.invoke()
    t3_out = i3.get_tensor(i3.get_output_details()[0]['index'])  # Shape [1, 2]

    # Compute dequantized probabilities
    out_details = i3.get_output_details()[0]
    scale = out_details['quantization_parameters']['scales'][0]
    zp = out_details['quantization_parameters']['zero_points'][0]
    probs = (t3_out.astype(np.float32) - zp) * scale
    probs = np.clip(probs, 0.0, 1.0)
    # Normalize if sum > 0
    s = np.sum(probs)
    if s > 0:
        probs = probs / s

    pred_class = int(np.argmax(probs))
    class_names = ["With Mask", "Without Mask"]

    result = {
        "node1": {
            "output_shape": list(t1_out.shape),
            "output_bytes": t1_out.nbytes,
            "channels": 32,
            "min_val": int(np.min(t1_out)),
            "max_val": int(np.max(t1_out)),
            "mean_val": float(np.mean(t1_out)),
        },
        "node2": {
            "output_shape": list(t2_out.shape),
            "output_bytes": t2_out.nbytes,
            "channels": 48,
            "min_val": int(np.min(t2_out)),
            "max_val": int(np.max(t2_out)),
            "mean_val": float(np.mean(t2_out)),
        },
        "node3": {
            "output_shape": list(t3_out.shape),
            "raw_scores": t3_out.tolist()[0],
            "probabilities": probs.tolist()[0],
            "predicted_class_idx": pred_class,
            "predicted_class_name": class_names[pred_class],
            "confidence_percent": round(float(probs[0][pred_class]) * 100, 2),
        }
    }

    res_file = os.path.join(output_dir, "prediction_result.json")
    with open(res_file, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
