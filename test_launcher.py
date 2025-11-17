"""
Quick test of the launcher logic (no GUI)
"""
import sys
import os

print("Testing launcher components...\n")

# Test 1: Python version
print("[1/4] Python Version Check")
print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Test 2: Check for matplotlib and pandas
print("\n[2/4] Dependency Check")
try:
    import matplotlib
    print(f"   ✓ matplotlib {matplotlib.__version__}")
except ImportError:
    print("   ⚠ matplotlib not installed (launcher will auto-install)")

try:
    import pandas
    print(f"   ✓ pandas {pandas.__version__}")
except ImportError:
    print("   ⚠ pandas not installed (launcher will auto-install)")

# Test 3: Check files
print("\n[3/4] File Check")
files = [
    'microlife/simulation/organism.py',
    'microlife/simulation/environment.py',
    'demo_phase2.py',
    'START_SIMULATION.py'
]

for f in files:
    if os.path.exists(f):
        print(f"   ✓ {f}")
    else:
        print(f"   ❌ {f}")

print("\n[4/4] Launcher Status")
if os.path.exists('START_SIMULATION.py'):
    print("   ✓ Launcher ready to use!")
    print("\n" + "=" * 60)
    print("TO START THE SIMULATION:")
    print("=" * 60)
    print("\n  Windows:   Double-click START_SIMULATION.bat")
    print("  Linux/Mac: Double-click START_SIMULATION.sh")
    print("  Any:       python START_SIMULATION.py")
    print("\n" + "=" * 60)
else:
    print("   ❌ Launcher not found")

print("\n✅ Launcher test complete!")
