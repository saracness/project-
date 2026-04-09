"""
Benchmark: movement strategies compared by organism survival time.

Conditions tested:
  random   -- organisms move randomly each step
  greedy   -- organisms always move toward nearest visible food (heuristic)
  q-learn  -- organisms learn using tabular Q-Learning (online, per-trial)

Usage:
    python experiments/benchmark_rl_vs_random.py
    python experiments/benchmark_rl_vs_random.py --trials 10 --steps 2000

Each trial uses a fixed seed for reproducibility. A fresh brain is created
per organism per trial (online learning from scratch).
"""
import sys
import os
import random
import statistics
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.ml.brain_rl import QLearningBrain

N_ORGANISMS = 15
FOOD_DENSITY = 40
ENV_SIZE = 400


def run_trial(mode, max_steps, seed):
    """
    Run a single trial and return a list of organism ages at end-of-trial.

    Args:
        mode (str): 'random', 'greedy', or 'q-learn'
        max_steps (int): Maximum simulation steps
        seed (int): Random seed for reproducibility

    Returns:
        list[int]: Age of each original organism at end of trial
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    use_intelligent = (mode == "greedy")
    env = Environment(
        width=ENV_SIZE,
        height=ENV_SIZE,
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
        original_orgs.append(org)
        env.add_organism(organism=org)

    for _ in range(max_steps):
        env.update()
        if not any(o.alive for o in original_orgs):
            break

    return [o.age for o in original_orgs]


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark movement strategies by organism survival time"
    )
    parser.add_argument(
        "--trials", type=int, default=5,
        help="Number of independent trials per condition (default: 5)"
    )
    parser.add_argument(
        "--steps", type=int, default=1500,
        help="Max simulation steps per trial (default: 1500)"
    )
    args = parser.parse_args()

    conditions = ["random", "greedy", "q-learn"]
    results = {}

    print(
        f"\nBenchmark: {args.trials} trials x {N_ORGANISMS} organisms x {args.steps} steps"
    )
    print(f"Environment: {ENV_SIZE}x{ENV_SIZE}, {FOOD_DENSITY} food particles\n")

    for mode in conditions:
        ages_all = []
        for trial in range(args.trials):
            trial_ages = run_trial(mode=mode, max_steps=args.steps, seed=trial * 42)
            ages_all.extend(trial_ages)
            survived = sum(1 for a in trial_ages if a >= args.steps)
            mean_age = statistics.mean(trial_ages) if trial_ages else 0
            print(
                f"  {mode:<10}  trial {trial + 1}/{args.trials}  "
                f"mean_age={mean_age:6.0f}  "
                f"survived_full={survived}/{N_ORGANISMS}"
            )
        results[mode] = ages_all

    baseline_mean = statistics.mean(results["random"]) if results["random"] else 1

    print(f"\n{'─' * 58}")
    print(f"{'Condition':<12} {'Mean age':>10} {'Median age':>12} {'vs random':>10}")
    print(f"{'─' * 58}")
    for mode in conditions:
        ages = results[mode]
        mean = statistics.mean(ages) if ages else 0
        median = statistics.median(ages) if ages else 0
        ratio = mean / baseline_mean if baseline_mean > 0 else 0
        print(f"{mode:<12} {mean:>10.1f} {median:>12.1f} {ratio:>10.2f}x")
    print(f"{'─' * 58}\n")

    # Return exit code 0 regardless — ratios are observations, not pass/fail.
    return 0


if __name__ == "__main__":
    sys.exit(main())
