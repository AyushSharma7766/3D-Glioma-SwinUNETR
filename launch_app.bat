@echo off
title 3D Swin-UNETR Neurosurgical Suite Web App
color 0B
cls
echo ==============================================================================
echo        3D Swin-UNETR Neurosurgical Suite - Full-Stack Web Application
echo ==============================================================================
echo.
echo [*] Starting Flask Backend & AI Server...
echo [*] Loading 3D Swin-UNETR (62.2M Parameters) into memory...
echo.
echo Server running at: http://localhost:5000
echo.
start "" http://localhost:5000
python app.py
pause

