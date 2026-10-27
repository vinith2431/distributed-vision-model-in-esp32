import argparse
import time
import zlib
import serial


def crc_hex(text: str) -> str:
    return f"{zlib.crc32(text.encode('utf-8')) & 0xFFFFFFFF:08X}"


def valid_ack(line: str, expected_seq: str) -> bool:
    line = line.strip()
    parts = line.split("|")
    if len(parts) != 4 or parts[0] != "ACK" or parts[1] != expected_seq or parts[2] != "OK":
        return False
    return parts[3].upper() == crc_hex(f"{parts[1]}|{parts[2]}")


def main():
    parser = argparse.ArgumentParser(description="Send one CRC32 packet through ESP32 Node1 -> Node2 -> Node3")
    parser.add_argument("--port", default="COM5", help="Node 1 USB serial port (default COM5)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seq", default="1")
    parser.add_argument("--payload", default="HELLO_NODE_CHAIN")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    body = f"{args.seq}|{args.payload}"
    packet = f"PKT|{body}|{crc_hex(body)}"
    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25, write_timeout=2) as ser:
            time.sleep(2.0)  # Many ESP32 boards reset when the serial port opens.
            ser.reset_input_buffer()
            print(f"Sending: {packet}")
            ser.write((packet + "\n").encode("utf-8"))
            ser.flush()
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    print(f"Node1: {line}")
                if line.startswith("Response: "):
                    reply = line[len("Response: "):].strip()
                    if valid_ack(reply, args.seq):
                        print("PASS: ACK sequence and CRC32 are valid.")
                        return
                    if reply.startswith("NACK|"):
                        print("FAIL: Node 3 returned NACK.")
                        return
            print("TIMEOUT: No valid ACK received. Use the board logs to locate the failing stage.")
            raise SystemExit(1)
    except serial.SerialException as exc:
        print(f"SERIAL ERROR: {exc}")
        print("Close Arduino Serial Monitor for Node 1 and verify the COM port.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
