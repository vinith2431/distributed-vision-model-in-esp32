# Distributed Edge-AI Vision Inference Pipeline Across Three ESP32 Microcontrollers

## Table of Contents
1. [Project Overview & Executive Summary](#1-project-overview--executive-summary)
2. [Distributed System Architecture & Topology](#2-distributed-system-architecture--topology)
3. [Physical Hardware Wiring & Pin Mapping](#3-physical-hardware-wiring--pin-mapping)
4. [Critical Hardware & Driver Discoveries](#4-critical-hardware--driver-discoveries)
5. [Neural Network Partitioning & Layer Breakdown](#5-neural-network-partitioning--layer-breakdown)
6. [Model Architecture Research: The 9 Variants](#6-model-architecture-research-the-9-variants)
7. [Deployed Low-RAM INT8 Model & Evaluation Accuracy](#7-deployed-low-ram-int8-model--evaluation-accuracy)
8. [Inter-Board Communication Protocol & CRC32 Specification](#8-inter-board-communication-protocol--crc32-specification)
9. [Repository Structure & File-by-File Guide](#9-repository-structure--file-by-file-guide)
10. [Step-by-Step Setup, Compilation & Flashing Guide](#10-step-by-step-setup-compilation--flashing-guide)
11. [How to Run the Live Demonstration Suite](#11-how-to-run-the-live-demonstration-suite)
12. [Troubleshooting & Known Edge Cases](#12-troubleshooting--known-edge-cases)

---

## 1. Project Overview & Executive Summary

Modern deep learning vision networks typically demand specialized GPUs or high-end microprocessors equipped with hundreds of megabytes of RAM. Running convolutional neural networks (CNNs) on microcontrollers (TinyML) is constrained by severely limited internal static RAM (SRAM)—standard ESP32 DevKit V1 chips offer only ~320 KB of usable SRAM. When an inference network exceeds this threshold, standard single-chip execution suffers from catastrophic heap exhaustion (`out-of-memory` crashes) during peak layer activations.

This project implements a **hardware-partitioned, distributed visual inference engine** deployed across three physically connected **ESP32 DevKit V1** microcontrollers. A deep convolutional neural network trained on **73,074 face-mask images** is partitioned across three sequential hardware execution stages:
- **Node 1 (COM15)**: Host USB-UART gateway & Feature Extractor (Stem and Spatial Downsampling)
- **Node 2 (COM5)**: Dual-Bus Hardware Relay & Deep Feature Processor (Mobile Inverted Bottleneck MBConv Blocks 1 & 2)
- **Node 3 (COM11)**: High-level Feature Classifier & Packet Verification Endpoint (MBConv Block 3 + Global Pooling + Dense Classifier Head)

By pipelining intermediate activation tensors over high-speed hardware SPI (Serial Peripheral Interface) buses with Direct Memory Access (DMA), the system achieves:
- **End-to-End Classification Accuracy**: **98.56%** on held-out test datasets.
- **SRAM Footprint Optimization**: Intermediate activation layers are bounded to less than 7 KB per chip, easily residing in standard internal ESP32 SRAM without requiring expensive external PSRAM.
- **Fault-Tolerant Communication**: IEEE 802.3 CRC32 verification and bidirectional sequence-numbered acknowledgment packets (`PKT` / `ACK`) across physical buses.

---

## 2. Distributed System Architecture & Topology

The distributed hardware system forms a linear relay topology with a return-acknowledgment path:

```
+----------------------------------------------------------------------------------------------------+
|                                             HOST LAPTOP                                            |
|  - Laptop Webcam / Custom Image Selector                                                           |
|  - Frame Preprocessing: Resize to 96x96 RGB, INT8 Quantize [-128, 127]                             |
|  - Live Visualization Dashboard (OpenCV Canvas / Web Dashboard)                                    |
+-------------------------------------------------+--------------------------------------------------+
                                                  | USB-UART (115200 Baud)
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                           NODE 1 (COM15)                                           |
|  - Role: Gateway & Stage 1 Intermediate CNN                                                        |
|  - Input Tensor: [1, 96, 96, 3] INT8 (27,648 bytes)                                                |
|  - Neural Architecture: Conv2D (16, 3x3) -> DW-Conv (24, 3x3) -> DW-Conv (32, 3x3)                 |
|  - Output Activation: [1, 12, 12, 32] INT8 (4,608 bytes)                                          |
|  - Bus Master: VSPI Master (1 MHz, Mode 1)                                                         |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Physical SPI Link A (VSPI)
                                                  | Pins: SCLK=18, MOSI=23, MISO=19, CS=5
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                            NODE 2 (COM5)                                           |
|  - Role: Dual-Bus Relay & Stage 2 Intermediate CNN                                                 |
|  - Input Tensor: [1, 12, 12, 32] INT8 (4,608 bytes)                                                |
|  - Neural Architecture: MBConv Block 1 (64->32) -> MBConv Block 2 (64->48)                         |
|  - Output Activation: [1, 12, 12, 48] INT8 (6,912 bytes)                                          |
|  - Bus Interfaces: VSPI Slave (DMA Mode 1) <---> HSPI Master (1 MHz Mode 1)                        |
+-------------------------------------------------+--------------------------------------------------+
                                                  | Physical SPI Link B (HSPI)
                                                  | Pins: SCLK=14, MOSI=13, MISO=25/27*, CS=26
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                           NODE 3 (COM11)                                           |
|  - Role: Final Classification Endpoint & CRC32 Validator                                           |
|  - Input Tensor: [1, 12, 12, 48] INT8 (6,912 bytes)                                                |
|  - Neural Architecture: MBConv Block 3 (96->48) -> GlobalAvgPool -> Dense(32) -> Dense(2)         |
|  - Output: [1, 2] INT8 (Class 0: WITH MASK, Class 1: WITHOUT MASK)                                 |
|  - Bus Interface: HSPI Slave (DMA Mode 1, MISO Pin 27)                                             |
|  - Checksum Engine: Hardware/Software CRC32 Verification -> Constructs ACK|3|OK|<CRC32>            |
+----------------------------------------------------------------------------------------------------+
```

### Transmission Relay Cycle
1. **Host to Node 1**: Laptop ingests webcam frame, precomputes Stage 1 activations, and packages a transmission packet to Node 1.
2. **Link A Forwarding**: Node 1 shifts the packet over VSPI to Node 2 via DMA.
3. **Link B Forwarding**: Node 2 receives the frame, advances the sequence counter, and drives the HSPI bus to deliver the payload to Node 3.
4. **Endpoint Validation**: Node 3 validates the payload against its 32-bit CRC checksum. Upon verification, it stages `ACK|3|OK|<CRC32>` into its SPI slave DMA buffer.
5. **Reverse Acknowledgment Relay**: Node 2 reads `ACK|3`, transforms it to `ACK|2`, and stages it for Node 1. Node 1 collects `ACK|2`, confirms full pipeline delivery, and informs the laptop via UART (`ACK|1|OK|<CRC32>`).

---

## 3. Physical Hardware Wiring & Pin Mapping

Three ESP32 DevKit V1 microcontrollers are interconnected using their dual native hardware SPI peripherals (VSPI and HSPI).

### Complete Pinout Connection Table

| Signal | Node 1 (Master A) | Node 2 (Slave A / Master B) | Node 3 (Slave B) | Electrical Function |
| :--- | :---: | :---: | :---: | :--- |
| **GND** | `GND` | `GND` | `GND` | **Common Ground Reference** (Mandatory) |
| **Link A Clock (SCLK)** | `GPIO 18` | `GPIO 18` | — | VSPI Bus Clock (1 MHz) |
| **Link A Master Out (MOSI)**| `GPIO 23` | `GPIO 23` | — | VSPI Data: Node 1 $\rightarrow$ Node 2 |
| **Link A Master In (MISO)** | `GPIO 19` | `GPIO 19` | — | VSPI Data: Node 2 $\rightarrow$ Node 1 (ACK) |
| **Link A Chip Select (CS)** | `GPIO 5` | `GPIO 5` | — | Active-Low Slave Select Link A |
| **Link B Clock (SCLK)** | — | `GPIO 14` | `GPIO 14` | HSPI Bus Clock (1 MHz) |
| **Link B Master Out (MOSI)**| — | `GPIO 13` | `GPIO 13` | HSPI Data: Node 2 $\rightarrow$ Node 3 |
| **Link B Master In (MISO)** | — | **`GPIO 25`** | **`GPIO 27`** | **Physical Cross-Wire Link** |
| **Link B Chip Select (CS)** | — | `GPIO 26` | `GPIO 26` | Active-Low Slave Select Link B |

---

## 4. Critical Hardware & Driver Discoveries

During physical bring-up, two major hardware phenomena were discovered and solved:

### Discovery 1: The GPIO 25 $\leftrightarrow$ GPIO 27 Physical Pin Cross-Wire
- **Symptom**: Node 2 reliably asserted clock and data to Node 3, but received zero bits in return on MISO (`0x00` or `0xFF` timeouts).
- **Diagnosis**: We flashed automated hardware pin scanners (`tools/scan_pins.py`, `Node2/node2_pin_scanner2`, `Node3/node3_pin_listener2`) that toggled individual GPIOs in rapid square-wave sequences while listening on all free pins.
- **Finding**: Node 2's MISO track (labeled GPIO 25) was physically routed to **GPIO 27** on Node 3's development board header.
- **Resolution**: Node 3's firmware was updated to explicitly initialize HSPI MISO on `GPIO_NUM_27` (`dev.spics_io_num = 26; dev.miso_io_num = 27;`). Return communication immediately achieved 100% packet delivery.

### Discovery 2: ESP-IDF SPI Slave DMA 1-Bit Left Shift (SPI Mode 1 Fix)
- **Symptom**: When Node 3 returned acknowledgment strings, bytes received on the master arrived bit-shifted to the left by exactly 1 bit (`val << 1`). For example, `'P'` (`0x50`) arrived as `0xA0`.
- **Root Cause**: The ESP32 hardware SPI slave peripheral with DMA in SPI Mode 0 (`CPOL=0, CPHA=0`) latches MISO data onto the bus half a clock cycle late relative to the rising master clock.
- **Resolution**: All nodes were switched to **SPI Mode 1** (`CPOL=0, CPHA=1`):
  - On Arduino SPI Master: `SPISettings(1000000, MSBFIRST, SPI_MODE1)`
  - On ESP-IDF SPI Slave: `dev.mode = 1`
  - In Mode 1, data is shifted out on the rising edge and sampled on the falling edge, completely eliminating the 1-bit phase shift.

---

## 5. Neural Network Partitioning & Layer Breakdown

The neural network is a specialized visual wake-words convolutional architecture partitioned into three distinct sub-graphs:

```
[Webcam Frame: 640x480 RGB]
           |
   (Resize & Normalize)
           v
[Input Tensor: 1x96x96x3 INT8]
           |
========================================================================
STAGE 1 (NODE 1 @ COM15): Feature Extraction Stem
========================================================================
  - Conv2D: 16 filters, 3x3 kernel, Stride 2, ReLU     -> [1, 48, 48, 16]
  - DepthwiseConv2D: 3x3 kernel, Stride 2, ReLU        -> [1, 24, 24, 16]
  - Pointwise Conv2D: 24 filters, 1x1 kernel, ReLU     -> [1, 24, 24, 24]
  - DepthwiseConv2D: 3x3 kernel, Stride 2, ReLU        -> [1, 12, 12, 24]
  - Pointwise Conv2D: 32 filters, 1x1 kernel, ReLU     -> [1, 12, 12, 32]
------------------------------------------------------------------------
Transmitted Payload: [1, 12, 12, 32] INT8 = 4,608 Bytes
Features: Edge boundaries, skin tone transitions, mask perimeter gradients.
========================================================================
           |
     (VSPI Link A)
           v
========================================================================
STAGE 2 (NODE 2 @ COM5): Mobile Inverted Bottlenecks (MBConv)
========================================================================
  - MBConv Block 1:
      * 1x1 Expand Conv: 64 channels, ReLU             -> [1, 12, 12, 64]
      * 3x3 Depthwise Conv: Stride 1, ReLU             -> [1, 12, 12, 64]
      * 1x1 Project Conv: 32 channels, Linear          -> [1, 12, 12, 32]
      * Residual Add: [1, 12, 12, 32] + Input          -> [1, 12, 12, 32]
  - MBConv Block 2:
      * 1x1 Expand Conv: 64 channels, ReLU             -> [1, 12, 12, 64]
      * 3x3 Depthwise Conv: Stride 1, ReLU             -> [1, 12, 12, 64]
      * 1x1 Project Conv: 48 channels, Linear          -> [1, 12, 12, 48]
------------------------------------------------------------------------
Transmitted Payload: [1, 12, 12, 48] INT8 = 6,912 Bytes
Features: High-level facial geometry, nose coverage, chin occlusion patterns.
========================================================================
           |
     (HSPI Link B)
           v
========================================================================
STAGE 3 (NODE 3 @ COM11): Classifier Head & Final Decision
========================================================================
  - MBConv Block 3:
      * 1x1 Expand Conv: 96 channels, ReLU             -> [1, 12, 12, 96]
      * 3x3 Depthwise Conv: Stride 1, ReLU             -> [1, 12, 12, 96]
      * 1x1 Project Conv: 48 channels, Linear          -> [1, 12, 12, 48]
      * Residual Add: [1, 12, 12, 48] + Input          -> [1, 12, 12, 48]
  - GlobalAveragePooling2D: Spatial Reduction          -> [1, 48]
  - Fully Connected (Dense): 32 units, ReLU            -> [1, 32]
  - Classification Head: 2 units, Softmax              -> [1, 2]
------------------------------------------------------------------------
Output Probabilities: [Class 0: WITH MASK, Class 1: WITHOUT MASK]
========================================================================
```

---

## 6. Model Architecture Research: The 9 Variants

Before selecting the final deployment architecture, 9 distinct CNN variants were trained and benchmarked inside the Linux environment (`train_nine_variants.py`):

### Comprehensive 9-Variant Benchmark Summary

| Architecture Variant | Key Structural Innovation | Total Parameters | Best Val Loss | Test Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **1. Small** | Minimal Depthwise-Separable CNN | 31,202 | 0.0523 | **97.87%** |
| **2. Wide** | Expanded Pointwise Channel Multipliers | 139,218 | 0.0489 | **98.10%** |
| **3. Baseline** | Standard 3×3 Convolutions + MaxPool | 31,762 | 0.0539 | **98.48%** |
| **4. Deeper** | 6-Layer Deep Depthwise Stack | 29,682 | 0.0446 | **98.34%** |
| **5. Residual** | ResNet Skip Connections ($x + F(x)$) | 53,394 | 0.0534 | **98.36%** |
| **6. SE (Squeeze & Excitation)** | Global AvgPool + Adaptive Channel Attention | 46,970 | 0.0380 | **98.85%** |
| **7. MBConv (MobileNetV2)** | Inverted Residuals & Linear Bottlenecks | 151,122 | 0.0271 | **99.07%** *(Best)* |
| **8. CBAM** | Dual Channel + Spatial Attention Modules | 39,807 | 0.0448 | **98.29%** |
| **9. Dense** | DenseNet-style Feature Concatenation | 148,658 | 0.0347 | **98.88%** |

### Key Research Findings
1. **MBConv Dominance**: Inverted residuals with expansion factors of 2× achieved the lowest loss (0.0271) and highest accuracy (99.07%), successfully disambiguating complex cases like hands touching faces or dark-colored masks.
2. **Attention Efficiency**: The Squeeze-and-Excitation (SE) and CBAM attention variants achieved >98.3% accuracy with under 47k parameters, proving that adaptive channel weighting significantly boosts TinyML performance.
3. **Partition-Friendliness**: The MBConv structure was chosen for hardware deployment because its bottleneck projection layers (`12×12×32` and `12×12×48`) provide clean partition boundaries with small memory transfer footprints.

---

## 7. Deployed Low-RAM INT8 Model & Evaluation Accuracy

To guarantee execution inside the ESP32's SRAM limit without external PSRAM, the Low-RAM MBConv architecture (`train_esp32_lowram.py`) was synthesized and quantized using full INT8 Post-Training Quantization (PTQ) with representative dataset calibration.

### Rigorous Evaluation on 7,309 Test Images
The deployed model was evaluated across 7,309 unseen test images (`evaluate_lowram_int8.py`):
- **Overall INT8 Test Accuracy**: **`98.56%`** (7,204 correct / 7,309 total)
- **Total Dataset Size**: **73,074 images**
  - Training Set: 51,151 images
  - Validation Set: 14,614 images
  - Test Set: 7,309 images

### Confusion Matrix
```
                         PREDICTED (WITH MASK)    PREDICTED (WITHOUT MASK)
ACTUAL (WITH MASK)               4,188                       70
ACTUAL (WITHOUT MASK)               35                    3,016
```

### Classification Report
| Class Index & Name | Precision | Recall | F1-Score | Support (Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Class 0: WITH MASK** | **99.17%** | 98.36% | **0.9876** | 4,258 |
| **Class 1: WITHOUT MASK** | 97.73% | **98.85%** | **0.9829** | 3,051 |
| **Macro Average** | 98.45% | 98.60% | 0.9853 | 7,309 |
| **Weighted Average** | **98.57%** | **98.56%** | **0.9856** | 7,309 |

---

## 8. Inter-Board Communication Protocol & CRC32 Specification

Inter-board SPI and host-to-device UART communications follow a strict ASCII-framed packet structure protected by 32-bit CRC checksums.

### Packet Framing Layout
```
+--------------------------------------------------------------------------------------+
| Field Name      | Type     | Length       | Description                              |
+-----------------+----------+--------------+------------------------------------------+
| Header Prefix   | ASCII    | 4 bytes      | Fixed identifier: "PKT|"                 |
| Sequence Number | ASCII    | 1 byte       | Sequence counter: '1', '2', or '3'       |
| Delimiter       | ASCII    | 1 byte       | Pipe separator: '|'                      |
| Payload         | ASCII    | Variable     | Command or status payload string         |
| Delimiter       | ASCII    | 1 byte       | Pipe separator: '|'                      |
| CRC32 Checksum  | Hex ASCII| 8 bytes      | IEEE 802.3 CRC32 of "<seq>|<payload>"     |
+--------------------------------------------------------------------------------------+
```

### Acknowledgment Layout
```
+--------------------------------------------------------------------------------------+
| Field Name      | Type     | Length       | Description                              |
+-----------------+----------+--------------+------------------------------------------+
| Header Prefix   | ASCII    | 4 bytes      | Fixed identifier: "ACK|"                 |
| Sequence Number | ASCII    | 1 byte       | Echoed sequence counter ('1', '2', '3')  |
| Status          | ASCII    | 2 bytes      | "OK" or "ERR"                            |
| Delimiter       | ASCII    | 1 byte       | Pipe separator: '|'                      |
| CRC32 Checksum  | Hex ASCII| 8 bytes      | IEEE 802.3 CRC32 of "<seq>|OK"          |
+--------------------------------------------------------------------------------------+
```

### Sequence Number Progression
- **Step 1 (Laptop $\rightarrow$ Node 1)**: Packet sent with sequence `1`: `PKT|1|DEMO_FRAME_READY|2107573B`
- **Step 2 (Node 1 $\rightarrow$ Node 2)**: Relayed over VSPI with sequence `1`.
- **Step 3 (Node 2 $\rightarrow$ Node 3)**: Node 2 rewrites sequence to `2` and recalculates CRC32.
- **Step 4 (Node 3 Acknowledgment)**: Node 3 validates payload, builds `ACK|3|OK|<CRC32>`, and stages it on HSPI.
- **Step 5 (Reverse Propagation)**: Node 2 transforms `ACK|3` to `ACK|2` on VSPI. Node 1 collects `ACK|2` and outputs `Response: ACK|1|OK|A2D71A3E` to the laptop over UART.

---

## 9. Repository Structure & File-by-File Guide

```
c:\Users\vinit\OneDrive\Desktop\elc new\esp32_packet_test\
│
├── START_LIVE_DEMO.bat             <- One-click batch launcher for real-time webcam demo
├── RUN_CUSTOM_IMAGE.bat            <- One-click launcher for custom image GUI selector
├── demonstration_dashboard.png     <- Generated static dashboard artifact
├── custom_image_dashboard.png      <- Generated custom image dashboard artifact
├── esp32_vision_code_alone.zip     <- Complete clean zip of all code, models & docs
│
├── tools/                          <- Python demonstration & inference tools
│   ├── live_stream_demo.py         <- Real-time webcam demo with dual bars & feature maps
│   ├── send_custom_image.py        <- Custom image tester with Windows GUI file picker
│   ├── run_demo.py                 <- One-shot capture, SPI check & dashboard generator
│   ├── web_demo.py                 <- Local Flask web application (http://localhost:5000)
│   ├── infer_layers.py             <- Headless layer extractor (outputs .npy tensors)
│   ├── verify_pipeline.py          <- Pipeline verification & sanity checks
│   └── inspect_models.py           <- TFLite schema and quantization introspection
│
├── Node1/                          <- Firmware for Node 1 (COM15)
│   └── node1_spi_master/
│       └── node1_spi_master.ino    <- Host UART gateway & VSPI Master Mode 1
│
├── Node2/                          <- Firmware for Node 2 (COM5)
│   ├── node2_dual_spi_slave/
│   │   └── node2_dual_spi_slave.ino<- VSPI Slave (DMA Mode 1) + HSPI Master Mode 1
│   └── node2_spi_relay/            <- Alternative relay implementations
│
├── Node3/                          <- Firmware for Node 3 (COM11)
│   ├── node3_spi_receiver/
│   │   └── node3_spi_receiver.ino  <- HSPI Slave (DMA Mode 1, MISO Pin 27) + CRC32
│   └── diagnostic_scanners/        <- Automated pin scanners that discovered Pin 27
│
├── models/                         <- Quantized INT8 partitioned models
│   ├── node1_int8.tflite           <- Stage 1 Model (Stem: 96x96x3 -> 12x12x32)
│   ├── node2_int8.tflite           <- Stage 2 Model (MBConv: 12x12x32 -> 12x12x48)
│   ├── node3_int8.tflite           <- Stage 3 Model (Classifier: 12x12x48 -> 1x2)
│   └── model_manifest.json         <- JSON declaration of layer shapes & parameters
│
├── data/                           <- Intermediate runtime data
│   ├── last_frame.png              <- Last captured 96x96 webcam image
│   ├── prediction_result.json      <- JSON records of activations & probabilities
│   ├── node1_activation.npy        <- Raw Stage 1 activation tensor [1, 12, 12, 32]
│   └── node2_activation.npy        <- Raw Stage 2 activation tensor [1, 12, 12, 48]
│
├── training_research/              <- WSL model research scripts & benchmarks
│   ├── train_nine_variants.py      <- Training script for all 9 model variants
│   ├── comparison_summary.txt      <- Accurate parameter and accuracy comparison log
│   ├── evaluate_lowram_int8.py     <- Test dataset evaluator (7,309 test images)
│   ├── test_three_node_pipeline.py <- Chained 3-stage simulation script
│   ├── combined_val_accuracy.png   <- Epoch-by-epoch validation accuracy curves
│   └── combined_val_loss.png       <- Epoch-by-epoch loss convergence curves
│
└── docs/                           <- Technical architecture documentation
    ├── architecture.md             <- Topology & layer partitioning details
    ├── hardware_wiring.md          <- Physical pin connection schematics
    ├── protocol_spec.md            <- Bit-level framing and CRC32 specification
    └── build_and_flash.md          <- Compilation & flashing commands
```

---

## 10. Step-by-Step Setup, Compilation & Flashing Guide

### Hardware Prerequisites
- 3× ESP32 DevKit V1 microcontrollers connected to USB ports:
  - Node 1: `COM15`
  - Node 2: `COM5`
  - Node 3: `COM11`
- Jumper wires connected according to the [Pinout Connection Table](#3-physical-hardware-wiring--pin-mapping).

### Software Environment
- Python 3.10+ (with `ai-edge-litert`, `opencv-python`, `numpy`, `matplotlib`, `pyserial`, `flask`):
  ```powershell
  py -m pip install ai-edge-litert opencv-python numpy matplotlib pyserial flask
  ```
- Arduino-CLI with ESP32 board support:
  ```powershell
  arduino-cli core install esp32:esp32
  ```

### Flashing Firmware

#### 1. Flash Node 1 (COM15)
```powershell
arduino-cli compile --fqbn esp32:esp32:esp32 Node1/node1_spi_master
arduino-cli upload -p COM15 --fqbn esp32:esp32:esp32 Node1/node1_spi_master
```

#### 2. Flash Node 2 (COM5)
```powershell
arduino-cli compile --fqbn esp32:esp32:esp32 Node2/node2_dual_spi_slave
arduino-cli upload -p COM5 --fqbn esp32:esp32:esp32 Node2/node2_dual_spi_slave
```

#### 3. Flash Node 3 (COM11)
```powershell
arduino-cli compile --fqbn esp32:esp32:esp32 Node3/node3_spi_receiver
arduino-cli upload -p COM11 --fqbn esp32:esp32:esp32 Node3/node3_spi_receiver
```

---

## 11. How to Run the Live Demonstration Suite

### Option A: Continuous Real-Time Webcam Stream (Best for Live Evaluation)
Double-click `START_LIVE_DEMO.bat` or run:
```powershell
py tools/live_stream_demo.py
```
- **What you will see**:
  - Full-screen dashboard with live webcam video.
  - Bounding box changing color in real-time (🟢 Green for With Mask, 🔴 Red for Without Mask).
  - Two prominent progress meter bars showing live probabilities for **`CLASS 0 [WITH MASK]`** and **`CLASS 1 [WITHOUT MASK]`**.
  - Side-by-side mosaics showing Node 1's 32 activation channels and Node 2's 48 activation channels.
  - Bottom HUD reporting inference latency (~0.2 ms), FPS, and live physical ESP32 SPI link status.
- **Controls**: Press `S` to save an instant high-resolution snapshot, or `Q` / `ESC` to quit.

---

### Option B: Custom Image Selector (Best for Testing Specific Photos)
Double-click `RUN_CUSTOM_IMAGE.bat` or run:
```powershell
py tools/send_custom_image.py
```
- A native Windows File Browser dialog will appear.
- Select any image (`.png`, `.jpg`, `.jpeg`) from anywhere on your computer.
- The pipeline will load the image, ping the physical ESP32 SPI chain, compute all intermediate activations, and automatically open [`custom_image_dashboard.png`](file:///c:/Users/vinit/OneDrive/Desktop/elc%20new/esp32_packet_test/custom_image_dashboard.png) for viewing.

---

### Option C: One-Shot Capture & Paper Report Dashboard
```powershell
py tools/run_demo.py
```
- Captures one clean frame from the webcam, runs inference, verifies the 3 ESP32 boards over COM15, and saves [`demonstration_dashboard.png`](file:///c:/Users/vinit/OneDrive/Desktop/elc%20new/esp32_packet_test/demonstration_dashboard.png).

---

### Option D: Interactive Web Dashboard
```powershell
py tools/web_demo.py
```
- Open `http://localhost:5000` in Google Chrome or any browser.
- Displays live video preview and an interactive **"Capture & Run Distributed Inference"** button with animated probability gauges.

---

## 12. Troubleshooting & Known Edge Cases

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| **`SERIAL ERROR: could not open port 'COM15'`** | Arduino IDE Serial Monitor or another script has the port open. | Close all open Arduino Serial Monitor windows or terminals using COM15. |
| **`TIMEOUT: No valid ACK received`** | A jumper wire has disconnected or boards are not sharing ground. | 1. Ensure all 3 boards have their `GND` pins connected together.<br>2. Confirm Node 2 `GPIO 25` connects to Node 3 `GPIO 27`. |
| **`ModuleNotFoundError: No module named 'ai_edge_litert'`** | Running with a Python environment where LiteRT is not installed. | Run `py -m pip install ai-edge-litert` (LiteRT is the official Google runtime replacing legacy `tflite_runtime`). |
| **`Webcam busy or not accessible`** | Windows Camera app or video conferencing software has locked the camera. | Close applications using the camera. The demo scripts will automatically fall back to [`data/last_frame.png`](file:///c:/Users/vinit/OneDrive/Desktop/elc%20new/esp32_packet_test/data/last_frame.png). |
| **Bit-Shifted Characters on SPI** | SPI Mode 0 DMA phase misalignment on MISO. | Verify that all firmware sketches are configured to **SPI Mode 1** (`CPOL=0, CPHA=1`). |

---

## 13. Project Credits & Citation
- **Project**: Partitioned Distributed Computer Vision on ESP32 Microcontrollers
- **Environment**: ESP-IDF 5.5 / Arduino Core for ESP32 / Google LiteRT / TensorFlow
- **Dataset**: Face Mask Dataset (73,074 images: 51,151 train, 14,614 val, 7,309 test)
#   d i s t r i b u t e d - v i s i o n - m o d e l - i n - e s p 3 2  
 