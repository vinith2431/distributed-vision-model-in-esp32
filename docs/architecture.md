# System Architecture

## Distributed Vision Inference on Three ESP32 Boards

```
+-------------------------------------------------------------+
|                      Host Laptop                            |
|  - Python 3.11 webcam capture (OpenCV: 96x96 RGB888)        |
|  - Packet & frame sender (USB-UART / 115200 baud)           |
+------------------------------+------------------------------+
                               | USB Serial
                               v
+-------------------------------------------------------------+
|                     Node 1 (ESP32)                          |
|  - USB Serial Endpoint                                      |
|  - SPI Master on Bus A (VSPI, SPI_MODE1, 1 MHz)             |
|  - Model Stage 1: node1_int8.tflite                         |
|    Input: [1, 96, 96, 3] INT8 (27,648 bytes image)          |
|    Output: [1, 12, 12, 32] INT8 (4,608 bytes feature map)   |
+------------------------------+------------------------------+
                               | SPI Link A (VSPI)
                               v
+-------------------------------------------------------------+
|                     Node 2 (ESP32)                          |
|  - Dual SPI Slave:                                          |
|    * Bus A: Slave on SPI3_HOST (from Node 1)                |
|    * Bus B: Slave on SPI2_HOST (to Node 3)                  |
|  - Model Stage 2: node2_int8.tflite                         |
|    Input: [1, 12, 12, 32] INT8 (4,608 bytes)                |
|    Output: [1, 12, 12, 48] INT8 (6,912 bytes feature map)   |
+------------------------------+------------------------------+
                               | SPI Link B (HSPI)
                               v
+-------------------------------------------------------------+
|                     Node 3 (ESP32)                          |
|  - SPI Master on Bus B (HSPI, SPI_MODE1, 1 MHz)             |
|  - Model Stage 3: node3_int8.tflite                         |
|    Input: [1, 12, 12, 48] INT8 (6,912 bytes)                |
|    Output: [1, 2] INT8 (Binary Classification Probability)  |
|  - Output: Class 0 (Background) vs Class 1 (Object/Person)   |
|  - CRC32 verification and return ACK/NACK generation        |
+-------------------------------------------------------------+
```
