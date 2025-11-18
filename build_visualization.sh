#!/bin/bash
# Build C++ Visualization and Simulation

echo "=================================================="
echo "Building Neuron Visualization & Fast Simulation"
echo "=================================================="

# Check if SFML is installed
if ! command -v pkg-config &> /dev/null || ! pkg-config --exists sfml-graphics; then
    echo ""
    echo "⚠️  SFML not found!"
    echo ""
    echo "Please install SFML:"
    echo "  Ubuntu/Debian: sudo apt-get install libsfml-dev"
    echo "  Fedora:        sudo dnf install SFML-devel"
    echo "  macOS:         brew install sfml"
    echo "  Arch:          sudo pacman -S sfml"
    echo ""
    echo "After installing, run this script again."
    exit 1
fi

# Create build directory
mkdir -p build
cd cpp_visualization

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "CMake not found. Building manually with g++..."
    echo ""

    # Build visualizer
    echo "Building neuron_visualizer..."
    g++ -std=c++17 neuron_visualizer.cpp -o ../build/neuron_visualizer \
        -lsfml-graphics -lsfml-window -lsfml-system -O3

    # Build fast simulator
    echo "Building neuron_learning_fast..."
    g++ -std=c++17 neuron_learning_fast.cpp -o ../build/neuron_learning_fast \
        -lsfml-graphics -lsfml-window -lsfml-system -pthread -O3

    if [ $? -eq 0 ]; then
        echo ""
        echo "=================================================="
        echo "✓ Build successful!"
        echo "=================================================="
        echo ""
        echo "Executables created:"
        echo "  1. neuron_visualizer      (data viewer)"
        echo "  2. neuron_learning_fast   (fast simulation)"
        echo ""
        echo "Run fast simulation:"
        echo "  python run_neuron_learning.py"
        echo ""
        echo "Or directly:"
        echo "  ./build/neuron_learning_fast --neurons 100 --fps 60"
        echo ""
        exit 0
    else
        echo "✗ Build failed!"
        exit 1
    fi
fi

# Build with CMake
echo "Building with CMake..."
cd ..
mkdir -p build
cd build
cmake ../cpp_visualization
make

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Build successful!"
    echo "=================================================="
    echo ""
    echo "Executables created:"
    echo "  1. neuron_visualizer      (data viewer)"
    echo "  2. neuron_learning_fast   (fast simulation)"
    echo ""
    echo "Quick start:"
    echo "  python run_neuron_learning.py"
    echo ""
    echo "Advanced options:"
    echo "  python run_neuron_learning.py --neurons 200 --fps 120"
    echo ""
    echo "Direct C++ execution:"
    echo "  ./build/neuron_learning_fast --help"
    echo ""
else
    echo "✗ Build failed!"
    exit 1
fi
