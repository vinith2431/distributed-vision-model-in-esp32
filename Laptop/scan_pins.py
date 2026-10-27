"""
Captures pin scanner output from Node 2 (COM5) and Node 3 (COM11).
Node 2 drives GPIO14, 13, 25, 26, 27 HIGH one at a time.
Node 3 reports which GPIO it detects as HIGH.
Run for ~20 seconds to cover one full scan cycle.
"""
import serial
import time
import threading

logs = []

def monitor(port, name):
    try:
        with serial.Serial(port, 115200, timeout=0.1) as s:
            s.dtr = False; s.rts = False
            deadline = time.time() + 25
            while time.time() < deadline:
                line = s.readline()
                if line:
                    txt = line.decode('utf-8', errors='replace').strip()
                    if txt:
                        ts = time.time()
                        entry = f"[{ts:.2f}] [{name}] {txt}"
                        logs.append(entry)
                        print(entry)
    except Exception as e:
        print(f"[{name}] ERROR: {e}")

t2 = threading.Thread(target=monitor, args=('COM5',  'Node2'), daemon=True)
t3 = threading.Thread(target=monitor, args=('COM11', 'Node3'), daemon=True)
t2.start(); t3.start()
t2.join(); t3.join()

print("\n=== SCAN SUMMARY ===")
n2_driving = [l for l in logs if 'NODE2_DRIVING_HIGH' in l]
n3_detected = [l for l in logs if 'NODE3_DETECTED_HIGH' in l]
print("Node 2 drove:")
for l in n2_driving: print(" ", l)
print("Node 3 detected:")
for l in n3_detected: print(" ", l)
