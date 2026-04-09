"""
Train a Q-Learning agent, checkpoint it, then evaluate in pure-exploitation mode.

This script separates the two phases that are often conflated in RL experiments:

  Phase 1 -- Training
    Epsilon-greedy exploration over N training episodes.
    The brain accumulates Q-table entries across all episodes (shared brain).

  Phase 2 -- Evaluation
    The trained brain is saved and reloaded.
    Epsilon is set to 0 so only the learnt policy is used.
    Performance is compared against a RandomBrain baseline run under
    identical conditions.

Usage:
    python experiments/train_and_evaluate.py
    python experiments/train_and_evaluate.py --train-episodes 10 --eval-trials 10
    python experiments/train_and_evaluate.py --save-path /tmp/my_brain.pkl
"""
import sys
import os
import random
import statistics
import argparse
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.ml.brain_rl import QLearningBrain
from microlife.ml.brain_random import RandomBrain

ENV_SIZE     = 400
N_ORGANISMS  = 15
FOOD_DENSITY = 40


def run_episode(brain, max_steps, seed, epsilon_override=None):
    """
    Run one simulation episode with *brain* shared across all organisms.

    Args:
        brain: Brain instance (modified in-place during training).
        max_steps (int): Maximum simulation steps.
        seed (int): RNG seed for reproducibility.
        epsilon_override (float | None): If set, temporarily replaces
            brain.epsilon for the duration of this episode.

    Returns:
        list[int]: Age of each original organism at end of episode.
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    env = Environment(
        width=ENV_SIZE, height=ENV_SIZE,
        use_intelligent_movement=False,  # brain handles movement
    )
    for _ in range(FOOD_DENSITY):
        env.add_food()

    original_orgs = []
    for _ in range(N_ORGANISMS):
        org = Organism(
            x=random.uniform(0, ENV_SIZE),
            y=random.uniform(0, ENV_SIZE),
        )
        org.brain = brain
        original_orgs.append(org)
        env.add_organism(organism=org)

    saved_epsilon = None
    if epsilon_override is not None and hasattr(brain, "epsilon"):
        saved_epsilon = brain.epsilon
        brain.epsilon = epsilon_override

    for _ in range(max_steps):
        env.update()
        if not any(o.alive for o in original_orgs):
            break

    if saved_epsilon is not None:
        brain.epsilon = saved_epsilon

    return [o.age for o in original_orgs]


def main():
    parser = argparse.ArgumentParser(
        description="Train Q-Learning agent then evaluate in exploitation mode"
    )
    parser.add_argument("--train-episodes", type=int, default=5,
                        help="Training episodes (default: 5)")
    parser.add_argument("--train-steps",    type=int, default=3000,
                        help="Max steps per training episode (default: 3000)")
    parser.add_argument("--eval-trials",    type=int, default=5,
                        help="Evaluation trials per condition (default: 5)")
    parser.add_argument("--eval-steps",     type=int, default=1500,
                        help="Max steps per evaluation trial (default: 1500)")
    parser.add_argument("--save-path",      type=str, default=None,
                        help="Path for brain checkpoint (default: system temp dir)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Phase 1: Training
    # ------------------------------------------------------------------
    print(f"\nPhase 1: Training ({args.train_episodes} episodes x {args.train_steps} steps)")
    print("-" * 60)

    brain = QLearningBrain(epsilon=0.4)
    train_ages_all = []

    for ep in range(args.train_episodes):
        ages = run_episode(brain, args.train_steps, seed=ep * 7)
        train_ages_all.extend(ages)
        print(
            f"  ep {ep + 1:>2}/{args.train_episodes}  "
            f"mean_age={statistics.mean(ages):6.0f}  "
            f"epsilon={brain.epsilon:.4f}  "
            f"q_states={len(brain.q_table):>5}"
        )

    print(f"\n  Training overall mean age: {statistics.mean(train_ages_all):.1f}")

    # ------------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------------
    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(tempfile.gettempdir(), "qlearning_brain.pkl")

    brain.save_model(save_path)
    print(f"  Checkpoint saved: {save_path}")

    # ------------------------------------------------------------------
    # Phase 2: Evaluation
    # ------------------------------------------------------------------
    print(f"\nPhase 2: Evaluation ({args.eval_trials} trials x {args.eval_steps} steps, epsilon=0)")
    print("-" * 60)

    # Reload from checkpoint
    eval_brain = QLearningBrain()
    eval_brain.load_model(save_path)

    random_ages, ql_ages = [], []

    for trial in range(args.eval_trials):
        seed = 1000 + trial * 13
        r_ages = run_episode(RandomBrain(),  args.eval_steps, seed=seed)
        q_ages = run_episode(eval_brain,     args.eval_steps, seed=seed,
                             epsilon_override=0.0)
        random_ages.extend(r_ages)
        ql_ages.extend(q_ages)
        print(
            f"  trial {trial + 1:>2}/{args.eval_trials}  "
            f"random={statistics.mean(r_ages):6.0f}  "
            f"q-learn={statistics.mean(q_ages):6.0f}"
        )

    rand_mean = statistics.mean(random_ages)
    ql_mean   = statistics.mean(ql_ages)
    ratio     = ql_mean / rand_mean if rand_mean > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {'Condition':<14} {'Mean age':>10} {'Median':>10} {'Std dev':>10}")
    print(f"  {'-' * 46}")
    print(
        f"  {'Random':<14} {rand_mean:>10.1f} "
        f"{statistics.median(random_ages):>10.1f} "
        f"{statistics.stdev(random_ages) if len(random_ages) > 1 else 0:>10.1f}"
    )
    print(
        f"  {'Q-Learning':<14} {ql_mean:>10.1f} "
        f"{statistics.median(ql_ages):>10.1f} "
        f"{statistics.stdev(ql_ages) if len(ql_ages) > 1 else 0:>10.1f}"
    )
    print(f"  {'-' * 46}")
    print(f"  Improvement over random: {ratio:.2f}x")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
