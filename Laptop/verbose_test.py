"""
Verbose end-to-end test.
- Monitors Node 2 (COM5) and Node 3 (COM11) the entire time.
- Sends packet to Node 1 and waits 30s for a response.
- Prints every line from all 3 nodes in real time.
"""
import serial
import time
import threading

logs = []
lock = threading.Lock()

def monitor(port, name, stop_event):
    try:
        with serial.Serial(port, 115200, timeout=0.1) as s:
            s.dtr = False; s.rts = False
            while not stop_event.is_set():
                raw = s.readline()
                if raw:
                    txt = raw.decode('latin-1', errors='replace').strip()
                    if txt:
                        ts = time.strftime('%H:%M:%S')
                        msg = f"[{ts}][{name}] {txt}"
                        with lock:
                            logs.append(msg)
                            safe_msg = msg.encode('ascii', errors='backslashreplace').decode('ascii')
                            print(safe_msg)
    except Exception as e:
        print(f"[{name}] MONITOR ERROR: {e}")

stop = threading.Event()
t2 = threading.Thread(target=monitor, args=('COM5',  'N2', stop), daemon=True)
t3 = threading.Thread(target=monitor, args=('COM11', 'N3', stop), daemon=True)
t2.start(); t3.start()

time.sleep(3.0)  # let all boards boot fully

print("=== Opening Node 1 COM15 ===")
try:
    with serial.Serial('COM15', 115200, timeout=0.5) as ser1:
        time.sleep(0.5)
        # Drain stale data
        ser1.reset_input_buffer()

        packet = "PKT|1|HELLO_NODE_CHAIN|0CCB2243\n"
        print(f">>> Sending: {packet.strip()}")
        ser1.write(packet.encode('utf-8'))
        ser1.flush()

        deadline = time.time() + 25.0
        while time.time() < deadline:
            line = ser1.readline()
            if line:
                txt = line.decode('utf-8', errors='backslashreplace').strip()
                if txt:
                    ts = time.strftime('%H:%M:%S')
                    print(f"[{ts}][N1] {txt}")
                    if "Response:" in txt or "TIMEOUT" in txt:
                        break
except Exception as e:
    print(f"Node 1 error: {e}")

time.sleep(1.0)
stop.set()
t2.join(timeout=2); t3.join(timeout=2)

print("\n=== KEY LINES ===")
for l in logs:
    if any(kw in l for kw in ['READY','RX_FROM','PACKET','RESULT','TRANS','ERR','FATAL']):
        print(l)
