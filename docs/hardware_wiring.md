# Hardware Wiring Reference

## Board Specifications
- **Boards:** 3x ESP32 DevKit V1 (ESP32-WROOM-32, 4MB Flash, dual-core Xtensa 240 MHz).
- **Logic Level:** 3.3V logic throughout. Common ground connected across all three boards.

## COM Port Assignments
- **Node 1:** `COM15`
- **Node 2:** `COM5`
- **Node 3:** `COM11`

---

## Physical Wiring Map

### Node 1 (Master) <-> Node 2 (Slave) [Bus A: VSPI]
| Signal | Node 1 Pin | Node 2 Pin | Status |
|:---|:---|:---|:---|
| **SCLK** | GPIO18 | GPIO18 | Verified |
| **MOSI** | GPIO23 | GPIO23 | Verified |
| **MISO** | GPIO19 | GPIO19 | Verified |
| **CS** | GPIO5 | GPIO5 | Verified |
| **GND** | GND | GND | Common Ground |

### Node 2 (Slave) <-> Node 3 (Master) [Bus B: HSPI]
| Signal | Node 2 Pin | Node 3 Pin | Notes |
|:---|:---|:---|:---|
| **SCLK** | GPIO14 | GPIO14 | Verified (bidirectional scanner) |
| **MOSI** | GPIO13 | GPIO13 | Verified (Node 3 drives -> Node 2 receives) |
| **MISO** | **GPIO25** | **GPIO27** | **Physical cross-wire**: Node 2 drives GPIO25 -> arrives at Node 3 GPIO27 |
| **CS** | GPIO26 | GPIO26 | Verified (Node 3 drives -> Node 2 receives) |
| **GND** | GND | GND | Common Ground |

---

## Critical SPI Timing Setting
- **Mode:** **SPI Mode 1** (`CPOL=0, CPHA=1`) must be used across all nodes.
- **Clock Rate:** 1 MHz (`1,000,000 Hz`).
- **Reason:** The ESP-IDF `spi_slave` driver with DMA (`SPI_DMA_CH_AUTO`) requires Mode 1 to avoid a 1-bit shift on MISO output.
