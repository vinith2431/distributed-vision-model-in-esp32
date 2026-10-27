# Build and Flash Guide

## Tools Used
- **Compiler:** `arduino-cli` (esp32:esp32:esp32 v3.3.12)
  `C:\Users\vinit\AppData\Local\Programs\arduino-ide\resources\app\lib\backend\resources\arduino-cli.exe`
- **Flasher:** `esptool.py` (via ESP-IDF 5.5 Python environment)
  `C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe -m esptool`
- **Python:** Python 3.11 (Windows) & Python 3.12 (WSL with `ai-edge-litert`)

---

## Node 1: Compile & Flash (COM15)
```powershell
$cli = "C:\Users\vinit\AppData\Local\Programs\arduino-ide\resources\app\lib\backend\resources\arduino-cli.exe"
$n1 = "C:\Users\vinit\OneDrive\Desktop\elc new\esp32_packet_test\Node1\node1_spi_master"
& $cli compile -b esp32:esp32:esp32 --output-dir "$n1\build" $n1
C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe -m esptool --chip esp32 --port COM15 --baud 460800 write_flash 0x10000 "$n1\build\node1_spi_master.ino.bin"
```

## Node 2: Compile & Flash (COM5)
```powershell
$n2 = "C:\Users\vinit\OneDrive\Desktop\elc new\esp32_packet_test\Node2\node2_dual_spi_slave"
& $cli compile -b esp32:esp32:esp32 --output-dir "$n2\build" $n2
C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe -m esptool --chip esp32 --port COM5 --baud 460800 write_flash 0x10000 "$n2\build\node2_dual_spi_slave.ino.bin"
```

## Node 3: Compile & Flash (COM11)
```powershell
$n3 = "C:\Users\vinit\OneDrive\Desktop\elc new\esp32_packet_test\Node3\node3_spi_master"
& $cli compile -b esp32:esp32:esp32 --output-dir "$n3\build" $n3
C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe -m esptool --chip esp32 --port COM11 --baud 460800 write_flash 0x10000 "$n3\build\node3_spi_master.ino.bin"
```

---

## Verification Commands
```powershell
# 1. End-to-end packet test
py Laptop/laptop_sender.py --port COM15

# 2. Live monitor across all 3 nodes
py Laptop/verbose_test.py

# 3. Model pipeline test (WSL)
wsl python3 "/mnt/c/Users/vinit/OneDrive/Desktop/elc new/esp32_packet_test/tools/verify_pipeline.py"
```
