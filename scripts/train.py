"""
Training pipeline — connects MicrolifeEnv with brain_rl brains.

Usage
-----
    python scripts/train.py --brain dqn --episodes 500 --out outputs/run1

The script trains a single organism, logs every-episode metrics to
outputs/<run>/training_log.csv  and saves the best-reward brain weights
to  outputs/<run>/best_brain.npz.

Supported brains: qlearning | dqn | ddqn
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Make project root importable when script is called directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microlife.gym_env import MicrolifeEnv
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain


# ---------------------------------------------------------------------------
# Brain factory
# ---------------------------------------------------------------------------

def make_brain(name: str):
    name = name.lower()
    if name == "qlearning":
        return QLearningBrain(learning_rate=0.1, discount_factor=0.95, epsilon=0.5)
    if name == "dqn":
        return DQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    if name == "ddqn":
        return DoubleDQNBrain(state_size=8, hidden_size=64, learning_rate=0.001)
    raise ValueError(f"Unknown brain '{name}'. Choose: qlearning, dqn, ddqn")


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train(
    brain_name: str = "dqn",
    episodes: int = 300,
    max_steps: int = 500,
    out_dir: str = "outputs/run",
    seed: int | None = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    Run full training loop.

    Returns list of per-episode metric dicts (also written to CSV).
    """
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "training_log.csv")

    env = MicrolifeEnv(
        width=300, height=300,
        n_food=20, n_temp_zones=3,
        max_steps=max_steps,
        seed=seed,
    )
    brain = make_brain(brain_name)

    fieldnames = [
        "episode", "steps", "total_reward", "avg_reward_per_step",
        "final_energy", "epsilon", "elapsed_s",
    ]

    best_reward = float("-inf")
    log: list[dict] = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ep in range(1, episodes + 1):
            t0 = time.time()
            obs = env.reset()
            state = env.obs_to_brain_state(obs)
            ep_reward = 0.0
            steps = 0

            while True:
                # Brain decides action (returns dict with move_direction, etc.)
                action_dict = brain.decide_action(state)
                action_idx = action_dict.get("_action_idx", 0)

                obs_next, reward, done, info = env.step(action_idx)
                next_state = env.obs_to_brain_state(obs_next)

                brain.learn(state, action_dict, reward, next_state, done)

                ep_reward += reward
                steps += 1
                state = next_state

                if done:
                    break

            elapsed = time.time() - t0
            epsilon = getattr(brain, "epsilon", float("nan"))
            final_energy = info.get("energy", 0.0)

            row = {
                "episode":              ep,
                "steps":                steps,
                "total_reward":         round(ep_reward, 4),
                "avg_reward_per_step":  round(ep_reward / max(steps, 1), 4),
                "final_energy":         round(final_energy, 2),
                "epsilon":              round(epsilon, 4),
                "elapsed_s":            round(elapsed, 3),
            }
            writer.writerow(row)
            f.flush()
            log.append(row)

            # Save best model weights
            if ep_reward > best_reward:
                best_reward = ep_reward
                _save_brain(brain, os.path.join(out_dir, "best_brain.npz"))

            if verbose and (ep % 50 == 0 or ep == 1):
                print(
                    f"Ep {ep:4d}/{episodes} | "
                    f"steps={steps:4d} | "
                    f"reward={ep_reward:7.2f} | "
                    f"energy={final_energy:6.1f} | "
                    f"ε={epsilon:.3f}"
                )

    print(f"\nTraining complete. Log → {csv_path}")
    print(f"Best episode reward: {best_reward:.2f}")
    return log


# ---------------------------------------------------------------------------
# Weight persistence
# ---------------------------------------------------------------------------

def _save_brain(brain, path: str):
    """Save numpy weight arrays that exist on the brain to .npz."""
    arrays = {}
    for attr in ("w1", "b1", "w2", "b2"):
        val = getattr(brain, attr, None)
        if val is not None:
            arrays[attr] = val
    if arrays:
        np.savez(path, **arrays)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train a MicrolifeEnv RL agent")
    p.add_argument("--brain",    default="dqn",
                   choices=["qlearning", "dqn", "ddqn"],
                   help="Brain type to train")
    p.add_argument("--episodes", type=int, default=300,
                   help="Number of training episodes")
    p.add_argument("--steps",    type=int, default=500,
                   help="Max steps per episode")
    p.add_argument("--out",      default="outputs/run",
                   help="Output directory for logs and weights")
    p.add_argument("--seed",     type=int, default=42,
                   help="Random seed (use -1 for no seed)")
    p.add_argument("--quiet",    action="store_true",
                   help="Suppress per-episode output")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = None if args.seed == -1 else args.seed
    train(
        brain_name=args.brain,
        episodes=args.episodes,
        max_steps=args.steps,
        out_dir=args.out,
        seed=seed,
        verbose=not args.quiet,
    )
