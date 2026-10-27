import serial
import time
import threading

def monitor_port(port, name, stop_event, logs):
    try:
        with serial.Serial(port, 115200, timeout=0.1) as ser:
            ser.dtr = False
            ser.rts = False
            while not stop_event.is_set():
                line = ser.readline()
                if line:
                    decoded = line.decode('utf-8', errors='replace').strip()
                    if decoded:
                        logs.append(f"[{name} {port}] {decoded}")
    except Exception as e:
        logs.append(f"[{name} {port}] ERROR: {e}")

def main():
    logs_node2 = []
    logs_node3 = []
    stop_event = threading.Event()

    t2 = threading.Thread(target=monitor_port, args=('COM5', 'Node2', stop_event, logs_node2))
    t3 = threading.Thread(target=monitor_port, args=('COM11', 'Node3', stop_event, logs_node3))
    t2.daemon = True
    t3.daemon = True
    t2.start()
    t3.start()

    time.sleep(1.0)

    print("Opening Node 1 (COM15)...")
    try:
        with serial.Serial('COM15', 115200, timeout=0.5) as ser1:
            time.sleep(1.5)
            # Drain boot messages
            while ser1.readline():
                pass

            packet = "PKT|1|HELLO_NODE_CHAIN|0CCB2243\n"
            print(f"Sending to Node 1: {packet.strip()}")
            ser1.write(packet.encode('utf-8'))
            ser1.flush()

            deadline = time.time() + 20.0
            while time.time() < deadline:
                line = ser1.readline()
                if line:
                    decoded = line.decode('utf-8', errors='backslashreplace').strip()
                    if decoded:
                        print(f"[Node1 COM15] {decoded}")
                        if "Response:" in decoded:
                            break  # Got ACK/NACK — done
    except Exception as e:
        print(f"Error on Node 1: {e}")

    time.sleep(1.0)
    stop_event.set()
    t2.join(timeout=1.0)
    t3.join(timeout=1.0)

    print("\n--- Summary of Logs ---")
    print("Node 2 logs:")
    for l in logs_node2:
        print("  ", l)

    print("Node 3 logs:")
    for l in logs_node3:
        print("  ", l)

if __name__ == '__main__':
    main()
