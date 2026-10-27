@echo off
title Distributed ESP32 Vision - Send Custom Image
echo ================================================================
echo   DISTRIBUTED ESP32 VISION: CUSTOM IMAGE SENDER
echo   Select any image from your laptop to test through the pipeline
echo ================================================================
echo.
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py "%~dp0tools\send_custom_image.py" %*
) else (
    C:\Espressif\python_env\idf5.5_py3.11_env\Scripts\python.exe "%~dp0tools\send_custom_image.py" %*
)
echo.
pause
