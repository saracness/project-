@echo off
REM ========================================
REM  MICRO-LIFE SIMULATION - WINDOWS LAUNCHER
REM  Double-click this file to start!
REM ========================================

echo.
echo ========================================
echo  MICRO-LIFE ML PROJECT
echo  One-Click Launcher for Windows
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://www.python.org/
    pause
    exit /b 1
)

REM Run the launcher
python START_SIMULATION.py

REM Pause so user can see the output
echo.
pause
