
import os
import time
import struct
import zlib

import cv2
import serial

PORT = "COM14"
BAUD = 115200

WIDTH = 96
HEIGHT = 96
CHANNELS = 3
FRAME_SIZE = WIDTH * HEIGHT * CHANNELS

# Must match UART_MAGIC in node1_serial.cc:
# 0x31474D49, representing "IMG1" in little-endian byte order.
UART_MAGIC = 0x31474D49


def read_line(ser, timeout=15):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        line = ser.readline()
        if line:
            return line.decode("ascii", errors="replace").strip()

    raise TimeoutError("Timed out waiting for Node1 response")


def wait_for_image_ready(ser, timeout=15):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        line = ser.readline()
        if not line:
            continue

        text = line.decode("ascii", errors="replace").strip()
        if text:
            print("ESP32:", text)

        if text == "IMAGE_READY":
            return

    raise TimeoutError("Node1 IMAGE_READY not received")


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open laptop webcam")

    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not capture webcam frame")

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        payload = rgb.tobytes()

        if len(payload) != FRAME_SIZE:
            raise RuntimeError(
                f"Unexpected frame size: {len(payload)}"
            )

        os.makedirs("data", exist_ok=True)

        with open("data/last_frame.rgb", "wb") as f:
            f.write(payload)

        if not cv2.imwrite("data/last_frame.png", frame):
            raise RuntimeError("Could not save PNG image")

        print("Image saved: data/last_frame.png")
        print("Raw RGB saved: data/last_frame.rgb")
        print(f"Image bytes: {len(payload)}")

        checksum = sum(payload) & 0xFFFFFFFF
        print(f"Host checksum: {checksum:08X}")
        print(f"Opening {PORT} at {BAUD} baud...")

        with serial.Serial(
            PORT, BAUD, timeout=1, write_timeout=10
        ) as ser:
            # Opening a serial port may reset Node1.
            time.sleep(3)
            ser.reset_input_buffer()

            wait_for_image_ready(ser)

            # Node1 expects:
            # uint32 magic, uint32 payload length, uint32 CRC32.
            header = struct.pack(
                "<III", UART_MAGIC, FRAME_SIZE, checksum
            )

            ser.write(header)
            ser.flush()

            ser.write(payload)
            ser.flush()

            print("Image payload sent; waiting for Node1...")

            deadline = time.monotonic() + 30
            image_received = False

            while time.monotonic() < deadline:
                line = ser.readline()
                if not line:
                    continue

                text = line.decode(
                    "ascii", errors="replace"
                ).strip()

                if not text:
                    continue

                print("ESP32:", text)

                if text == "IMAGE_RECEIVED" and not image_received:
                    image_received = True
                    print("SUCCESS: Node1 accepted the image.")
                    print(
                        "Monitoring Node1 inference and SPI output "
                        "for another 20 seconds..."
                    )
                    deadline = time.monotonic() + 20
                    continue

                if text in (
                    "BAD_HEADER",
                    "BAD_CHECKSUM",
                    "INCOMPLETE_IMAGE",
                    "INFERENCE_OR_SPI_FAILED",
                    "SPI_INIT_FAILED",
                ) or text.startswith("BAD_LENGTH"):
                    raise RuntimeError(
                        f"Node1 reported an error: {text}"
                    )

                if text == "IMAGE_READY" and image_received:
                    print(
                        "Node1 is ready for another image; "
                        "the monitoring window has ended."
                    )
                    break

            if not image_received:
                raise TimeoutError(
                    "Timed out waiting for Node1 image confirmation"
                )

            print("Serial monitoring finished.")

    finally:
        cap.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
