#!/bin/bash
# ========================================
#  MICRO-LIFE SIMULATION - LINUX/MAC LAUNCHER
#  Run this file to start the simulation!
# ========================================

echo ""
echo "========================================"
echo "  MICRO-LIFE ML PROJECT"
echo "  One-Click Launcher for Linux/Mac"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.7+ from your package manager"
    exit 1
fi

# Run the launcher
python3 START_SIMULATION.py

echo ""
echo "Press Enter to exit..."
read
