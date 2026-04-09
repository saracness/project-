"""
Benchmark: movement strategies compared by organism survival time.

Conditions tested:
  random       -- pure random movement (environment built-in, no Brain)
  greedy       -- always moves toward nearest visible food (heuristic)
  random-brain -- RandomBrain through _move_with_ai (same code path as RL)
  q-learn      -- tabular Q-Learning brain (online, fresh per trial)

The 'random-brain' condition exists to verify that the overhead of going
through the Brain / _move_with_ai code path does not itself affect survival,
i.e. that 'random' and 'random-brain' produce equivalent results.

Usage:
    python experiments/benchmark_rl_vs_random.py
    python experiments/benchmark_rl_vs_random.py --trials 10 --steps 2000
    python experiments/benchmark_rl_vs_random.py --trials 5 --csv results.csv
"""
import sys
import os
import csv
import random
import statistics
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.ml.brain_rl import QLearningBrain
from microlife.ml.brain_random import RandomBrain

N_ORGANISMS  = 15
FOOD_DENSITY = 40
ENV_SIZE     = 400


def run_trial(mode, max_steps, seed):
    """
    Run one trial and return organism ages.

    Args:
        mode (str): 'random' | 'greedy' | 'random-brain' | 'q-learn'
        max_steps (int): Simulation steps limit.
        seed (int): RNG seed.

    Returns:
        list[int]: Age of each original organism at end of trial.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    use_intelligent = (mode == "greedy")
    env = Environment(
        width=ENV_SIZE, height=ENV_SIZE,
        use_intelligent_movement=use_intelligent,
    )
    for _ in range(FOOD_DENSITY):
        env.add_food()

    original_orgs = []
    for _ in range(N_ORGANISMS):
        org = Organism(
            x=random.uniform(0, ENV_SIZE),
            y=random.uniform(0, ENV_SIZE),
        )
        if mode == "q-learn":
            org.brain = QLearningBrain()
        elif mode == "random-brain":
            org.brain = RandomBrain()
        original_orgs.append(org)
        env.add_organism(organism=org)

    for _ in range(max_steps):
        env.update()
        if not any(o.alive for o in original_orgs):
            break

    return [o.age for o in original_orgs]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark organism movement strategies by survival time"
    )
    parser.add_argument("--trials", type=int, default=5,
                        help="Trials per condition (default: 5)")
    parser.add_argument("--steps",  type=int, default=1500,
                        help="Max simulation steps per trial (default: 1500)")
    parser.add_argument("--csv",    type=str, default=None,
                        help="Export raw per-organism results to this CSV file")
    args = parser.parse_args()

    conditions = ["random", "greedy", "random-brain", "q-learn"]
    results    = {}  # mode -> list[int] of all ages across all trials
    csv_rows   = []  # for optional CSV export

    print(
        f"\nBenchmark: {args.trials} trials x {N_ORGANISMS} organisms x {args.steps} steps"
    )
    print(f"Environment: {ENV_SIZE}x{ENV_SIZE}, {FOOD_DENSITY} food particles\n")

    for mode in conditions:
        ages_all = []
        for trial in range(args.trials):
            ages = run_trial(mode=mode, max_steps=args.steps, seed=trial * 42)
            ages_all.extend(ages)
            survived = sum(1 for a in ages if a >= args.steps)
            mean_t   = statistics.mean(ages) if ages else 0
            print(
                f"  {mode:<14}  trial {trial + 1}/{args.trials}  "
                f"mean_age={mean_t:6.0f}  "
                f"survived_full={survived}/{N_ORGANISMS}"
            )
            if args.csv:
                for i, age in enumerate(ages):
                    csv_rows.append({"condition": mode, "trial": trial + 1,
                                     "organism": i + 1, "age": age})
        results[mode] = ages_all

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    baseline_mean = statistics.mean(results["random"]) if results["random"] else 1.0

    print(f"\n{'=' * 66}")
    print(f"{'Condition':<16} {'Mean':>8} {'Median':>8} {'+/-Std':>8} {'vs random':>10}")
    print(f"{'-' * 66}")
    for mode in conditions:
        ages  = results[mode]
        mean  = statistics.mean(ages)   if ages else 0
        med   = statistics.median(ages) if ages else 0
        std   = statistics.stdev(ages)  if len(ages) > 1 else 0
        ratio = mean / baseline_mean    if baseline_mean > 0 else 0
        print(f"{mode:<16} {mean:>8.1f} {med:>8.1f} {std:>8.1f} {ratio:>10.2f}x")
    print(f"{'=' * 66}\n")

    # ------------------------------------------------------------------
    # Optional CSV export
    # ------------------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["condition", "trial", "organism", "age"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Per-organism results written to: {args.csv}\n")


if __name__ == "__main__":
    main()
