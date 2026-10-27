@echo off
title Distributed ESP32 Vision - Live Demonstration
echo ================================================================
echo   STARTING DISTRIBUTED ESP32 VISION LIVE DEMONSTRATION
echo   Pipeline: Laptop Camera -> Node 1 -> Node 2 -> Node 3
echo ================================================================
echo.
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py "%~dp0tools\live_stream_demo.py"
) else (
    C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe "%~dp0tools\live_stream_demo.py"
)
pause
