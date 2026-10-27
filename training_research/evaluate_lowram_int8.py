import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import new as base

MODEL_PATH = "esp32_split/lowram_exports/lowram_int8.tflite"

_, _, test_ds = base.load_datasets()

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
scale_in, zero_in = inp["quantization"]
scale_out, zero_out = out["quantization"]

y_true, y_pred = [], []

for images, labels in test_ds:
    images = images.numpy().astype(np.float32)
    labels = labels.numpy()

    for image, label in zip(images, labels):
        # Quantize normalized [0,1] RGB input to INT8.
        q_image = np.clip(
            np.round(image / scale_in + zero_in), -128, 127
        ).astype(np.int8)[None, ...]

        interpreter.set_tensor(inp["index"], q_image)
        interpreter.invoke()
        q_output = interpreter.get_tensor(out["index"])[0]

        # The output is quantized; argmax is unchanged by
        # the shared output scale and zero point.
        y_pred.append(int(np.argmax(q_output)))
        y_true.append(int(label))

print("INT8 test accuracy:", np.mean(np.array(y_true) == np.array(y_pred)) * 100, "%")
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification report:")
print(classification_report(
    y_true, y_pred,
    target_names=base.CLASS_NAMES,
    digits=4
))
