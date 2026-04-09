"""
Headless basic simulation runner.

Usage:
    python -m microlife.simulation.run_basic
    python -m microlife.simulation.run_basic --steps 1000 --organisms 30 --seed 42
"""
import argparse
import random

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism


def main():
    parser = argparse.ArgumentParser(
        description="Run a basic MicroLife simulation (headless, prints stats to stdout)"
    )
    parser.add_argument("--steps",     type=int, default=500,  help="Simulation steps")
    parser.add_argument("--organisms", type=int, default=20,   help="Initial organism count")
    parser.add_argument("--food",      type=int, default=50,   help="Initial food particles")
    parser.add_argument("--seed",      type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    env = Environment(width=500, height=500)

    for _ in range(args.food):
        env.add_food()
    for _ in range(args.organisms):
        env.add_organism()

    print(
        f"Starting: {args.organisms} organisms, {args.food} food, "
        f"{args.steps} steps"
        + (f", seed={args.seed}" if args.seed is not None else "")
    )
    print(f"{'Step':>6}  {'Pop':>5}  {'Food':>6}  {'Avg energy':>12}  {'Avg age':>9}")
    print("-" * 46)

    for _ in range(args.steps):
        env.update()
        step = env.timestep
        if step % 50 == 0 or step == args.steps:
            s = env.get_statistics()
            print(
                f"{s['timestep']:>6}  {s['population']:>5}  "
                f"{s['food_count']:>6}  {s['avg_energy']:>12.1f}  "
                f"{s['avg_age']:>9.1f}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
