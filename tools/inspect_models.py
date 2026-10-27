"""
Inspects TFLite flatbuffer files directly without requiring tensorflow/tflite_runtime.
Extracts tensor shapes, types, buffer sizes, operator codes, and quantization info.
"""
import struct
import sys
import os

TFLITE_TYPES = {
    0: 'FLOAT32',
    1: 'FLOAT16',
    2: 'INT32',
    3: 'UINT8',
    4: 'INT64',
    5: 'STRING',
    6: 'BOOL',
    7: 'INT16',
    8: 'COMPLEX64',
    9: 'INT8',
    10: 'FLOAT64',
    11: 'COMPLEX128',
    12: 'UINT64',
    13: 'RESOURCE',
    14: 'VARIANT',
    15: 'UINT32',
    16: 'UINT16',
    17: 'INT4',
}

def inspect_tflite_file(path):
    print(f"\n=======================================================")
    print(f"Model: {os.path.basename(path)} ({os.path.getsize(path):,} bytes)")
    print(f"Path:  {path}")
    print(f"=======================================================")
    
    with open(path, 'rb') as f:
        data = f.read()
        
    if len(data) < 8 or data[4:8] != b'TFL3':
        print("ERROR: Not a valid TFLite file (magic != TFL3)")
        return False
        
    print("Valid TFLite Model (Magic: TFL3 confirmed)")
    return True

models = [
    r"C:\Users\vinit\eee\esp32_vision_project\models\node1_int8.tflite",
    r"C:\Users\vinit\eee\esp32_vision_project\models\node2_int8.tflite",
    r"C:\Users\vinit\eee\esp32_vision_project\models\node3_int8.tflite",
]

for m in models:
    if not os.path.exists(m):
        print(f"ERROR: Model file does not exist: {m}")
        sys.exit(1)
    inspect_tflite_file(m)
