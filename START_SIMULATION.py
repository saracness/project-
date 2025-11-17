#!/usr/bin/env python3
"""
🦠 MICRO-LIFE SIMULATION - ONE-CLICK LAUNCHER
Just run this file to start the simulation!
"""
import sys
import subprocess
import os

def print_banner():
    """Display welcome banner."""
    print("=" * 70)
    print("🦠 MICRO-LIFE ML PROJECT - ONE-CLICK LAUNCHER")
    print("=" * 70)
    print()

def check_python_version():
    """Check if Python version is adequate."""
    print("[1/4] Checking Python version...")
    if sys.version_info < (3, 7):
        print("❌ ERROR: Python 3.7 or higher required!")
        print(f"   You have: Python {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")

def install_dependencies():
    """Install required packages."""
    print("\n[2/4] Checking dependencies...")

    required_packages = {
        'matplotlib': 'matplotlib',
        'pandas': 'pandas'
    }

    missing_packages = []

    for package_name, pip_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"   ✓ {package_name} already installed")
        except ImportError:
            missing_packages.append(pip_name)
            print(f"   ⚠ {package_name} not found")

    if missing_packages:
        print(f"\n   Installing missing packages: {', '.join(missing_packages)}")
        print("   This may take a minute...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet"
            ] + missing_packages)
            print("   ✓ All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to install dependencies automatically")
            print("   Please run manually: pip install matplotlib pandas")
            sys.exit(1)
    else:
        print("   ✓ All dependencies already installed")

def check_files():
    """Verify simulation files exist."""
    print("\n[3/4] Checking simulation files...")

    required_files = [
        'microlife/simulation/organism.py',
        'microlife/simulation/environment.py',
        'microlife/visualization/simple_renderer.py',
        'microlife/data/logger.py',
        'demo_phase2.py'
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path} NOT FOUND")

    if missing_files:
        print("\n   ❌ ERROR: Some simulation files are missing!")
        print("   Make sure you're in the project directory")
        print("   Or switch to branch: claude/microlife-ml-guide-011CUnQgJvemd2JyKLX8AkWK")
        sys.exit(1)

    print("   ✓ All simulation files present")

def run_simulation():
    """Launch the simulation."""
    print("\n[4/4] Starting simulation...")
    print()
    print("=" * 70)
    print("🚀 LAUNCHING MICRO-LIFE SIMULATION")
    print("=" * 70)
    print("\n🎯 CONTROLS:")
    print("   • Watch organisms seek food (green dots)")
    print("   • Red zones = HOT (drains energy)")
    print("   • Blue zones = COLD (drains energy)")
    print("   • Gray blocks = OBSTACLES")
    print("   • Close window to stop and save data")
    print("\n" + "=" * 70)
    print()

    try:
        # Run the Phase 2 demo
        subprocess.run([sys.executable, "demo_phase2.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Simulation stopped by user")
    except Exception as e:
        print(f"\n❌ Error running simulation: {e}")
        sys.exit(1)

def main():
    """Main launcher function."""
    print_banner()

    try:
        check_python_version()
        install_dependencies()
        check_files()
        run_simulation()

        print("\n" + "=" * 70)
        print("✅ SIMULATION COMPLETE!")
        print("=" * 70)
        print("\n📊 Data saved to: microlife/data/logs/")
        print("📖 Next steps: Check MICROLIFE_ML_GUIDE.md for Phase 3\n")

    except KeyboardInterrupt:
        print("\n\n⚠ Launcher interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
