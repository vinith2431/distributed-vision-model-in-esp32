"""Reverse scan: Node 3 drives SCLK/MOSI/CS pins, Node 2 listens for which GPIO receives them."""
import serial, time, threading

logs = []
lock = threading.Lock()

def monitor(port, name):
    try:
        with serial.Serial(port, 115200, timeout=0.1) as s:
            s.dtr = False; s.rts = False
            deadline = time.time() + 25
            while time.time() < deadline:
                raw = s.readline()
                if raw:
                    txt = raw.decode('latin-1', errors='replace').strip()
                    if txt:
                        entry = f"[{time.strftime('%H:%M:%S')}][{name}] {txt}"
                        with lock:
                            logs.append(entry)
                            print(entry)
    except Exception as e:
        print(f"[{name}] ERROR: {e}")

t2 = threading.Thread(target=monitor, args=('COM5',  'Node2'), daemon=True)
t3 = threading.Thread(target=monitor, args=('COM11', 'Node3'), daemon=True)
t2.start(); t3.start()
t2.join(); t3.join()

print("\n=== REVERSE SCAN SUMMARY ===")
driving   = [l for l in logs if 'NODE3_DRIVING_HIGH' in l]
detected  = [l for l in logs if 'NODE2_DETECTED_HIGH' in l]
print("Node 3 drove:")
for l in driving:  print(" ", l)
print("Node 2 detected:")
for l in detected: print(" ", l)
