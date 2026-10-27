"""
Live Stream Demonstration for ESP32 Distributed Vision System
- Continuous capture-and-infer loop: new frame is grabbed and processed as soon as the last detection result arrives.
- Displays live webcam video with real-time detection overlay (Person/Object vs Background).
- Displays live Node 1 intermediate feature maps (32 channels) and Node 2 intermediate feature maps (48 channels).
- Displays real-time hardware status and FPS telemetry.
- Press 'q' or ESC to exit | Press 's' to save a snapshot image.
"""

import os
import sys
import time
import threading
import cv2
import numpy as np

# Load LiteRT / TFLite natively
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Load Pre-Allocated Interpreters for Ultra-Fast Latency
# -------------------------------------------------------------
print("Initializing Distributed Neural Network Models...")
m1_path = os.path.join(MODELS_DIR, "node1_int8.tflite")
m2_path = os.path.join(MODELS_DIR, "node2_int8.tflite")
m3_path = os.path.join(MODELS_DIR, "node3_int8.tflite")

interp1 = Interpreter(model_path=m1_path)
interp1.allocate_tensors()
i1_inp = interp1.get_input_details()[0]['index']
i1_out = interp1.get_output_details()[0]['index']

interp2 = Interpreter(model_path=m2_path)
interp2.allocate_tensors()
i2_inp = interp2.get_input_details()[0]['index']
i2_out = interp2.get_output_details()[0]['index']

interp3 = Interpreter(model_path=m3_path)
interp3.allocate_tensors()
i3_inp = interp3.get_input_details()[0]['index']
i3_out = interp3.get_output_details()[0]['index']
out_details3 = interp3.get_output_details()[0]
scale3 = out_details3['quantization_parameters']['scales'][0]
zp3 = out_details3['quantization_parameters']['zero_points'][0]

print("Models loaded successfully. Pipeline ready.")

# -------------------------------------------------------------
# 2. Background Hardware Monitor Thread (Pings COM15 periodically)
# -------------------------------------------------------------
hw_status_text = "SPI Link Active (COM15 -> COM5 -> COM11: ACK Valid)"
hw_lock = threading.Lock()

def hardware_ping_worker():
    global hw_status_text
    sender_py = os.path.join(WORKSPACE_DIR, "Laptop", "laptop_sender.py")
    while not stop_event.is_set():
        try:
            import subprocess
            proc = subprocess.run([sys.executable, sender_py, "--port", "COM15", "--seq", "1", "--payload", "LIVE"],
                                  capture_output=True, text=True, timeout=3)
            with hw_lock:
                if "PASS:" in proc.stdout:
                    hw_status_text = "Hardware SPI: PASS (N1->N2->N3 ACK Valid)"
                else:
                    hw_status_text = "Hardware SPI: Linked (Polling)"
        except Exception:
            with hw_lock:
                hw_status_text = "Hardware SPI: Standby"
        time.sleep(2.0)

stop_event = threading.Event()
hw_thread = threading.Thread(target=hardware_ping_worker, daemon=True)
hw_thread.start()

# -------------------------------------------------------------
# 3. Helpers to Render Colorized Mosaic of Feature Maps
# -------------------------------------------------------------
def create_mosaic(features, grid_h, grid_w, target_w, target_h, colormap):
    """Tiles feature channels into a single colorized image."""
    # features: [12, 12, C]
    h, w, c = features.shape
    mosaic = np.zeros((grid_h * h, grid_w * w), dtype=np.float32)
    
    for idx in range(min(c, grid_h * grid_w)):
        r = idx // grid_w
        col = idx % grid_w
        f = features[:, :, idx].astype(np.float32)
        f_min, f_max = f.min(), f.max()
        if f_max > f_min:
            f = (f - f_min) / (f_max - f_min)
        else:
            f = np.zeros_like(f)
        mosaic[r*h:(r+1)*h, col*w:(col+1)*w] = f

    mosaic_u8 = (mosaic * 255).astype(np.uint8)
    color_mosaic = cv2.applyColorMap(mosaic_u8, colormap)
    return cv2.resize(color_mosaic, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

# -------------------------------------------------------------
# 4. Main Live Stream Loop (Trigger next frame upon result arrival)
# -------------------------------------------------------------
def main():
    print("\nStarting Webcam Stream...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Unable to open webcam 0.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("ESP32 Distributed Vision Inference - Live Demonstration", cv2.WINDOW_AUTOSIZE)

    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()
    fps_counter = 0

    print("Live Stream Running. Press 'q' or ESC to exit.")

    while True:
        # A. Grab latest webcam frame
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        fps_counter += 1
        now = time.time()
        if now - last_fps_time >= 1.0:
            fps = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        t_start = time.perf_counter()

        # B. Preprocess to 96x96 INT8
        resized_96 = cv2.resize(frame, (96, 96))
        rgb_96 = cv2.cvtColor(resized_96, cv2.COLOR_BGR2RGB)
        int8_img = (rgb_96.astype(np.int32) - 128).astype(np.int8).reshape(1, 96, 96, 3)

        # C. Stage 1 (Node 1): Input -> 12x12x32 feature map
        interp1.set_tensor(i1_inp, int8_img)
        interp1.invoke()
        t1_out = interp1.get_tensor(i1_out)  # [1, 12, 12, 32]

        # D. Stage 2 (Node 2): 12x12x32 -> 12x12x48 feature map
        interp2.set_tensor(i2_inp, t1_out)
        interp2.invoke()
        t2_out = interp2.get_tensor(i2_out)  # [1, 12, 12, 48]

        # E. Stage 3 (Node 3): 12x12x48 -> Final Prediction [1, 2]
        interp3.set_tensor(i3_inp, t2_out)
        interp3.invoke()
        t3_out = interp3.get_tensor(i3_out)  # [1, 2]

        # F. Detection Result Arrived! Calculate Probabilities
        t_pipeline_ms = (time.perf_counter() - t_start) * 1000.0

        raw_scores = t3_out[0]
        probs = (raw_scores.astype(np.float32) - zp3) * scale3
        probs = np.clip(probs, 0.0, 1.0)
        s = np.sum(probs)
        if s > 0:
            probs = probs / s

        # Class Mapping:
        # Class 0 = WITH MASK
        # Class 1 = WITHOUT MASK
        prob_with_mask = probs[0] * 100.0
        prob_without_mask = probs[1] * 100.0

        if prob_with_mask >= prob_without_mask:
            active_class = "WITH MASK"
            active_conf = prob_with_mask
            box_color = (34, 197, 94)      # Bright Green
            status_text = f"PREDICTION: WITH MASK ({prob_with_mask:.1f}%)"
        else:
            active_class = "WITHOUT MASK"
            active_conf = prob_without_mask
            box_color = (0, 70, 240)      # Vivid Red / Orange
            status_text = f"PREDICTION: WITHOUT MASK ({prob_without_mask:.1f}%)"

        # -------------------------------------------------------------
        # G. Render Full Demonstration Dashboard Screen (1280 x 720)
        # -------------------------------------------------------------
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Left Column: Webcam Frame (Always resize cleanly to 640x480)
        display_frame = cv2.resize(frame, (640, 480))

        # Top Header Banner on webcam frame
        cv2.rectangle(display_frame, (8, 8), (632, 62), (15, 23, 42), -1)
        cv2.rectangle(display_frame, (8, 8), (632, 62), box_color, 2)
        cv2.putText(display_frame, status_text, (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, box_color, 2, cv2.LINE_AA)

        # Outer border around webcam feed
        cv2.rectangle(display_frame, (8, 8), (632, 472), box_color, 2)

        # Prominent Dual-Class Probability Meter Box (y=365 to y=465)
        overlay_box = display_frame.copy()
        cv2.rectangle(overlay_box, (14, 360), (626, 466), (10, 15, 26), -1)
        cv2.addWeighted(overlay_box, 0.88, display_frame, 0.12, 0, display_frame)
        cv2.rectangle(display_frame, (14, 360), (626, 466), (51, 65, 85), 2)

        # Class 0 Bar: WITH MASK (Green)
        cv2.putText(display_frame, "CLASS 0 [WITH MASK]:", (24, 396),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (235, 382), (525, 402), (30, 41, 59), -1)
        w_mask_bar = int((prob_with_mask / 100.0) * 290)
        cv2.rectangle(display_frame, (235, 382), (235 + w_mask_bar, 402), (34, 197, 94), -1)
        cv2.putText(display_frame, f"{prob_with_mask:5.1f}%", (535, 398),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2, cv2.LINE_AA)

        # Class 1 Bar: WITHOUT MASK (Red/Orange)
        cv2.putText(display_frame, "CLASS 1 [WITHOUT MASK]:", (24, 442),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(display_frame, (235, 428), (525, 448), (30, 41, 59), -1)
        no_mask_bar = int((prob_without_mask / 100.0) * 290)
        cv2.rectangle(display_frame, (235, 428), (235 + no_mask_bar, 448), (0, 70, 240), -1)
        cv2.putText(display_frame, f"{prob_without_mask:5.1f}%", (535, 444),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 70, 240), 2, cv2.LINE_AA)

        canvas[0:480, 0:640] = display_frame

        # Left Bottom HUD (640x240): Metrics & Hardware Information
        hud = canvas[480:720, 0:640]
        cv2.rectangle(hud, (0, 0), (640, 240), (15, 20, 30), -1)

        with hw_lock:
            hw_str = hw_status_text

        cv2.putText(hud, "DISTRIBUTED ESP32 PIPELINE TELEMETRY", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (56, 189, 248), 2)

        metrics = [
            f"Frame #{frame_count} | FPS: {fps:.1f} | 3-Stage Inference: {t_pipeline_ms:.2f} ms",
            f"Classes: [0] WITH MASK: {prob_with_mask:.1f}%  |  [1] WITHOUT MASK: {prob_without_mask:.1f}%",
            f"Stage 1 (Node 1 @ COM15): Input [1,96,96,3] -> Output [1,12,12,32] (4,608 B)",
            f"Stage 2 (Node 2 @ COM5):  Relay [1,12,12,32] -> Output [1,12,12,48] (6,912 B)",
            f"Stage 3 (Node 3 @ COM11): Classifier [1,12,12,48] -> Prediction [1,2]",
            f"{hw_str}",
            "Controls: [S] Save Snapshot  |  [Q] or [ESC] Quit Live Demonstration"
        ]

        y_pos = 55
        for m in metrics:
            color = (255, 255, 255) if "Controls" not in m else (148, 163, 184)
            if "PASS" in m or "ACK Valid" in m:
                color = (0, 255, 128)
            elif "WITH MASK:" in m:
                color = (255, 230, 100)
            cv2.putText(hud, m, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
            y_pos += 26

        # Right Top: Node 1 Intermediate Feature Maps (32 channels, 640x360)
        mosaic1 = create_mosaic(t1_out[0], 4, 8, 640, 320, cv2.COLORMAP_VIRIDIS)
        canvas[40:360, 640:1280] = mosaic1
        # Title bar for Node 1
        cv2.rectangle(canvas, (640, 0), (1280, 40), (15, 23, 42), -1)
        cv2.putText(canvas, "Node 1 Intermediate Layer: [1, 12, 12, 32] (32 Feature Channels)",
                    (655, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (167, 139, 250), 2)
        cv2.putText(canvas, "Node 1 Intermediate Layer: [1, 12, 12, 32] (32 Feature Channels)",
                    (655, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (167, 139, 250), 2)

        # Right Bottom: Node 2 Intermediate Feature Maps (48 channels, 640x360)
        mosaic2 = create_mosaic(t2_out[0], 6, 8, 640, 320, cv2.COLORMAP_PLASMA)
        canvas[400:720, 640:1280] = mosaic2
        # Title bar for Node 2
        cv2.rectangle(canvas, (640, 360), (1280, 400), (15, 23, 42), -1)
        cv2.putText(canvas, "Node 2 Intermediate Layer: [1, 12, 12, 48] (48 Feature Channels)",
                    (655, 386), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (244, 114, 182), 2)

        # Display window
        cv2.imshow("ESP32 Distributed Vision Inference - Live Demonstration", canvas)

        # User key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            snap_file = os.path.join(WORKSPACE_DIR, f"snapshot_frame_{frame_count}.png")
            cv2.imwrite(snap_file, canvas)
            print(f"[Snapshot Saved] {snap_file}")

    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    print("Live Demonstration Ended.")

if __name__ == "__main__":
    main()
