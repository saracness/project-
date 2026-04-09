#!/usr/bin/env python3
"""
MicroLife -- single-command launcher.

  python START_SIMULATION.py
  python START_SIMULATION.py --steps 1000 --organisms 30 --seed 42
  python START_SIMULATION.py --no-logs      # skip writing log files
  python START_SIMULATION.py --help

What this script does:
  1. Checks that Python >= 3.8 is installed.
  2. Auto-installs any missing packages from requirements.txt.
  3. Verifies core simulation files are present.
  4. Runs a headless simulation and writes logs to ./logs/<timestamp>/
     (the logs/ directory is already in .gitignore).
"""
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Step 1 -- Python version
# ---------------------------------------------------------------------------
if sys.version_info < (3, 8):
    print(
        f"ERROR: Python 3.8 or higher is required.\n"
        f"       You have {sys.version_info.major}.{sys.version_info.minor}.\n"
        f"       Download a newer Python from https://python.org"
    )
    sys.exit(1)

print(f"[1/3] Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} -- OK")

# ---------------------------------------------------------------------------
# Step 2 -- Dependencies
# ---------------------------------------------------------------------------
_PROBE_IMPORTS = ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn"]


def _missing():
    absent = []
    for pkg in _PROBE_IMPORTS:
        try:
            __import__(pkg)
        except ImportError:
            absent.append(pkg)
    return absent


_absent = _missing()
if _absent:
    _req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print(f"[2/3] Installing missing packages: {', '.join(_absent)} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", _req, "--quiet"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("      ERROR: pip install failed.")
        print("      Try manually:  pip install -r requirements.txt")
        sys.exit(1)
    _still_absent = _missing()
    if _still_absent:
        print(f"      ERROR: still missing after install: {', '.join(_still_absent)}")
        sys.exit(1)
    print("[2/3] Dependencies installed -- OK")
else:
    print("[2/3] Dependencies -- OK")

# ---------------------------------------------------------------------------
# Step 3 -- Core file check
# ---------------------------------------------------------------------------
_CORE = [
    "microlife/simulation/organism.py",
    "microlife/simulation/environment.py",
    "microlife/ml/brain_rl.py",
    "microlife/logger.py",
]
_root = os.path.dirname(os.path.abspath(__file__))
_bad  = [f for f in _CORE if not os.path.exists(os.path.join(_root, f))]

if _bad:
    print("[3/3] ERROR -- missing files:")
    for f in _bad:
        print(f"        {f}")
    print("      Make sure you are in the project root directory.")
    sys.exit(1)

print("[3/3] Core files -- OK\n")

# ---------------------------------------------------------------------------
# Step 4 -- Run simulation
# (imports happen here so pip has had a chance to install everything above)
# ---------------------------------------------------------------------------
import argparse  # noqa: E402
import random    # noqa: E402

from microlife.simulation.environment import Environment   # noqa: E402
from microlife.logger import SimulationLogger              # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="MicroLife headless simulation with file logging"
    )
    parser.add_argument("--steps",     type=int,            default=500,
                        help="Simulation steps to run (default: 500)")
    parser.add_argument("--organisms", type=int,            default=20,
                        help="Initial organism count (default: 20)")
    parser.add_argument("--food",      type=int,            default=50,
                        help="Initial food particle count (default: 50)")
    parser.add_argument("--seed",      type=int,            default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-logs",   action="store_true",
                        help="Disable writing log files to ./logs/")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    log_base = os.path.join(_root, "logs")
    logger   = SimulationLogger(base_dir=log_base, enabled=not args.no_logs)

    # Header
    print("=" * 58)
    print("  MicroLife Simulation")
    print("=" * 58)
    print(f"  Steps     : {args.steps}")
    print(f"  Organisms : {args.organisms}")
    print(f"  Food      : {args.food}")
    if args.seed is not None:
        print(f"  Seed      : {args.seed}")
    if not args.no_logs:
        print(f"  Logs      : {logger.run_dir}")
    print("=" * 58 + "\n")

    logger.log_event("Simulation started", {
        "steps": args.steps, "organisms": args.organisms,
        "food": args.food,   "seed": args.seed,
    })

    # Build environment
    env = Environment(width=500, height=500)
    for _ in range(args.food):
        env.add_food()
    for _ in range(args.organisms):
        env.add_organism()

    print(f"{'Step':>6}  {'Pop':>5}  {'Food':>5}  {'AvgEnergy':>10}  {'AvgAge':>8}")
    print("-" * 44)

    max_pop  = 0
    prev_pop = args.organisms

    for _ in range(args.steps):
        env.update()
        s   = env.get_statistics()
        pop = s["population"]

        # One CSV row per step
        logger.log_step(s)

        # Console output every 50 steps
        if s["timestep"] % 50 == 0 or s["timestep"] == args.steps:
            print(
                f"{s['timestep']:>6}  {pop:>5}  "
                f"{s['food_count']:>5}  {s['avg_energy']:>10.1f}  "
                f"{s['avg_age']:>8.1f}"
            )

        # Notable event detection
        if pop > max_pop:
            max_pop = pop
            if max_pop % 10 == 0 or max_pop - prev_pop >= 5:
                logger.log_event("Population record",
                                 {"pop": max_pop, "step": s["timestep"]})

        if pop == 0:
            logger.log_event("Population collapsed", {"step": s["timestep"]})
            print(f"\n  All organisms died at step {s['timestep']}.")
            break

        if pop <= 3 and prev_pop > 3:
            logger.log_event("Critical population",
                             {"pop": pop, "step": s["timestep"]})

        prev_pop = pop

    final = env.get_statistics()
    logger.log_event("Simulation ended", {
        "final_pop":  final["population"],
        "final_step": final["timestep"],
        "max_pop":    max_pop,
    })
    logger.close()

    # Footer
    print("\n" + "=" * 58)
    print("  Simulation complete")
    print(f"  Final population : {final['population']}")
    print(f"  Max population   : {max_pop}")
    if not args.no_logs:
        print(f"  Log directory    : {logger.run_dir}")
        print(f"    stats.csv    -- per-step metrics (open in Excel / pandas)")
        print(f"    events.log   -- notable events timeline")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
