# Protocol Specification

## 1. Small Packet Protocol (ASCII Milestone)

Line-delimited protocol over USB-UART (115200 baud) and SPI (128-byte fixed-size frames):

### Request Packet Format
```
PKT|<sequence>|<payload>|<CRC32>
```
- `<sequence>`: Integer sequence number (e.g. `1`, `2`).
- `<payload>`: ASCII payload string (e.g. `HELLO_NODE_CHAIN`).
- `<CRC32>`: 8-digit uppercase hexadecimal CRC-32/IEEE computed over UTF-8 bytes of `<sequence>|<payload>`.

### Acknowledgment (ACK) Format
```
ACK|<sequence>|OK|<CRC32>
```
- `<CRC32>`: Computed over `<sequence>|OK`.
- Expected for sequence 1: `ACK|1|OK|A2D71A3E`.

### Negative Acknowledgment (NACK) Format
```
NACK|<sequence>|BADCRC|<CRC32>
NACK|<sequence>|FORMAT|<CRC32>
```

---

## 2. Binary Image & Feature Tensor Protocol (Vision Milestone)

For transmitting the 27,648-byte image ($96 \times 96 \times 3$) and intermediate feature tensors:

- **Chunk Size:** 56 bytes payload per 64-byte SPI frame (or 112 bytes payload per 128-byte frame).
- **Header:** 8 bytes:
  - Magic (2 bytes): `0xA55A`
  - Sequence ID (2 bytes)
  - Payload length (2 bytes)
  - Chunk Checksum (2 bytes)
- **Status / Bitmap:** Receiver returns bitmask of missing chunks for bounded selective retransmission.
