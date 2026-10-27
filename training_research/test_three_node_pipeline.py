import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import new

MODEL_DIR = Path(
    "/mnt/c/Users/vinit/OneDrive/Desktop/sem 3/project/eee/lowram_models"
)

MODEL_FILES = [
    MODEL_DIR / "node1_int8.tflite",
    MODEL_DIR / "node2_int8.tflite",
    MODEL_DIR / "node3_int8.tflite",
]

for path in MODEL_FILES:
    if not path.exists():
        raise FileNotFoundError(f"Model missing: {path}")

def load_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    return interpreter

interpreters = [load_interpreter(p) for p in MODEL_FILES]

def details(interpreter):
    return interpreter.get_input_details()[0], interpreter.get_output_details()[0]

def quantize(real_values, tensor_detail):
    scale, zero_point = tensor_detail["quantization"]
    if scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")
    values = np.round(real_values / scale + zero_point)
    dtype = tensor_detail["dtype"]
    limits = np.iinfo(dtype)
    return np.clip(values, limits.min, limits.max).astype(dtype)

def dequantize(values, tensor_detail):
    scale, zero_point = tensor_detail["quantization"]
    return (values.astype(np.float32) - zero_point) * scale

def invoke(interpreter, real_input):
    input_detail, output_detail = details(interpreter)
    quantized_input = quantize(real_input, input_detail)
    interpreter.set_tensor(input_detail["index"], quantized_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
    return dequantize(output, output_detail)

print("Model tensor details:")
for i, interpreter in enumerate(interpreters, start=1):
    inp, out = details(interpreter)
    print(f"Node {i}: input={inp['shape']} {inp['dtype']}, "
          f"output={out['shape']} {out['dtype']}")
    print(f"        input quantization={inp['quantization']}, "
          f"output quantization={out['quantization']}")

_, _, test_ds = new.load_datasets()

y_true = []
y_pred = []

print("\nRunning the three-node pipeline on test images...")
for images, labels in test_ds:
    images_np = images.numpy().astype(np.float32)
    labels_np = labels.numpy()

    for image, label in zip(images_np, labels_np):
        # Every stage receives real-valued tensors; invoke() quantizes them
        # using that stage's own input scale and zero point.
        x = image[np.newaxis, ...]
        x = invoke(interpreters[0], x)
        x = invoke(interpreters[1], x)
        scores = invoke(interpreters[2], x)

        y_pred.append(int(np.argmax(scores[0])))
        y_true.append(int(label))

y_true = np.asarray(y_true)
y_pred = np.asarray(y_pred)

print("\nThree-node pipeline results")
print("Images evaluated:", len(y_true))
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.4f}%")
print("Confusion matrix (rows=true, columns=predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["withmask", "withoutmask"],
    digits=4,
    zero_division=0,
))
