#!/usr/bin/env python3
"""
Python Wrapper for Fast C++ Neuron Learning Simulation

Easy launcher for the C++ neuron simulator with real-time visualization.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_executable():
    """Check if C++ executable exists"""
    exe_path = Path("build/neuron_learning_fast")

    if not exe_path.exists():
        print("❌ C++ executable not found!")
        print("\n🔧 Building C++ simulator...")
        print("=" * 60)

        # Try to build
        build_script = Path("build_visualization.sh")
        if build_script.exists():
            result = subprocess.run(["bash", str(build_script)],
                                   capture_output=False)
            if result.returncode != 0:
                print("\n❌ Build failed!")
                print("\nManual build:")
                print("  cd cpp_visualization")
                print("  g++ -std=c++17 neuron_learning_fast.cpp -o ../build/neuron_learning_fast \\")
                print("      -lsfml-graphics -lsfml-window -lsfml-system -pthread -O3")
                sys.exit(1)
        else:
            print("\n❌ Build script not found!")
            print("\nManual build:")
            print("  cd cpp_visualization")
            print("  g++ -std=c++17 neuron_learning_fast.cpp -o ../build/neuron_learning_fast \\")
            print("      -lsfml-graphics -lsfml-window -lsfml-system -pthread -O3")
            sys.exit(1)

    return exe_path

def main():
    print("🧠 Fast Neuron Learning Simulation Launcher")
    print("=" * 60)

    # Check and build if needed
    exe_path = check_executable()

    # Parse arguments or use defaults
    import argparse

    parser = argparse.ArgumentParser(
        description='Launch fast C++ neuron learning simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default: 60 neurons, 60 FPS
  %(prog)s --neurons 100                # 100 neurons
  %(prog)s --neurons 200 --fps 120      # 200 neurons at 120 FPS
  %(prog)s --neurons 500 --fps 60       # Large network

Controls (during simulation):
  SPACE - Pause/Resume
  S     - Print statistics
  ESC   - Exit

Learning Graphs (real-time):
  - Accuracy: Task performance
  - Reward: Average reward signal
  - Synapses: Connection count
  - Activity: Firing rate
        """
    )

    parser.add_argument('--neurons', type=int, default=60,
                       help='Number of neurons (default: 60)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Target FPS (default: 60)')
    parser.add_argument('--task', type=str, default='xor',
                       help='Learning task (default: xor)')

    args = parser.parse_args()

    # Validate
    if args.neurons < 10:
        print("⚠️  Warning: Too few neurons (<10), setting to 10")
        args.neurons = 10
    elif args.neurons > 1000:
        print("⚠️  Warning: Very large network (>1000), may be slow")

    if args.fps < 10:
        print("⚠️  Warning: Very low FPS (<10), setting to 10")
        args.fps = 10
    elif args.fps > 120:
        print("⚠️  Warning: Very high FPS (>120), setting to 120")
        args.fps = 120

    # Build command
    cmd = [
        str(exe_path),
        '--neurons', str(args.neurons),
        '--fps', str(args.fps),
        '--task', args.task
    ]

    print("\n🚀 Launching simulation...")
    print(f"   Neurons: {args.neurons}")
    print(f"   FPS: {args.fps}")
    print(f"   Task: {args.task}")
    print("=" * 60)
    print()

    # Run
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        sys.exit(1)

    print("\n✅ Simulation completed!")

if __name__ == "__main__":
    main()
