#!/bin/bash
# Build C++ Visualization

echo "=================================="
echo "Building Neuron Visualizer"
echo "=================================="

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
    echo "CMake not found. Installing CMake..."
    # Try to build without CMake
    echo ""
    echo "Building manually with g++..."
    g++ -std=c++17 neuron_visualizer.cpp -o ../build/neuron_visualizer \
        -lsfml-graphics -lsfml-window -lsfml-system -O3

    if [ $? -eq 0 ]; then
        echo "✓ Build successful!"
        echo ""
        echo "Run with: ./build/neuron_visualizer"
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
    echo "=================================="
    echo "✓ Build successful!"
    echo "=================================="
    echo ""
    echo "Run visualization:"
    echo "  ./build/neuron_visualizer"
    echo ""
    echo "Or with data file:"
    echo "  ./build/neuron_visualizer visualization_data/neuron_state.json"
    echo ""
else
    echo "✗ Build failed!"
    exit 1
fi
