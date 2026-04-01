"""
Post-training analysis — reads training_log.csv and produces publication-quality plots.

Usage
-----
  python scripts/analyze.py --runs outputs/qlearning outputs/test_run outputs/ddqn
  python scripts/analyze.py --runs outputs/test_run --out outputs/test_run/analysis
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_run(run_dir: str) -> pd.DataFrame | None:
    path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(path):
        print(f"  [skip] no training_log.csv in {run_dir}")
        return None
    df = pd.read_csv(path)
    # Infer brain name from directory name
    df["run"] = os.path.basename(run_dir.rstrip("/"))
    return df


def rolling(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Plot 1: Learning curves — reward over episodes for all runs
# ---------------------------------------------------------------------------

def plot_learning_curves(dfs: list[pd.DataFrame], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_raw, ax_roll = axes
    palette = plt.cm.tab10.colors

    for i, df in enumerate(dfs):
        col  = palette[i % len(palette)]
        name = df["run"].iloc[0]
        eps  = df["episode"]
        rew  = df["total_reward"]

        ax_raw.plot(eps, rew, alpha=0.25, color=col, linewidth=0.8)
        ax_roll.plot(eps, rolling(rew), color=col, linewidth=2, label=name)

    for ax, title in zip((ax_raw, ax_roll),
                         ("Raw episode reward", "20-episode rolling mean")):
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title(title)
        ax.grid(alpha=0.3)

    ax_roll.legend()
    fig.suptitle("RL Agent Learning Curves — MicrolifeEnv", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Energy and survival — is the agent actually staying alive longer?
# ---------------------------------------------------------------------------

def plot_survival(dfs: list[pd.DataFrame], out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_energy, ax_steps = axes
    palette = plt.cm.tab10.colors

    for i, df in enumerate(dfs):
        col  = palette[i % len(palette)]
        name = df["run"].iloc[0]
        eps  = df["episode"]

        ax_energy.plot(eps, rolling(df["final_energy"]),
                       color=col, linewidth=2, label=name)
        ax_steps.plot(eps, rolling(df["steps"]),
                      color=col, linewidth=2, label=name)

    ax_energy.axhline(200, color="gray", linestyle="--", alpha=0.4, label="Max energy")
    ax_energy.set_ylabel("Final energy (rolling mean)")
    ax_energy.set_xlabel("Episode")
    ax_energy.set_title("Final energy per episode")
    ax_energy.legend()
    ax_energy.grid(alpha=0.3)

    ax_steps.set_ylabel("Steps survived (rolling mean)")
    ax_steps.set_xlabel("Episode")
    ax_steps.set_title("Steps survived per episode")
    ax_steps.legend()
    ax_steps.grid(alpha=0.3)

    fig.suptitle("Survival & Energy — MicrolifeEnv", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Epsilon decay — confirms exploration → exploitation shift
# ---------------------------------------------------------------------------

def plot_epsilon(dfs: list[pd.DataFrame], out_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    palette = plt.cm.tab10.colors

    for i, df in enumerate(dfs):
        if "epsilon" not in df.columns:
            continue
        col  = palette[i % len(palette)]
        name = df["run"].iloc[0]
        ax.plot(df["episode"], df["epsilon"],
                color=col, linewidth=1.5, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (exploration rate)")
    ax.set_title("Epsilon decay — exploration → exploitation")
    ax.set_ylim(0, 0.6)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 4: Summary table — best / mean / std reward per brain
# ---------------------------------------------------------------------------

def print_summary(dfs: list[pd.DataFrame]):
    print("\n{'─'*60}")
    print(f"  {'Brain':<16} {'Episodes':>8} {'Best':>8} {'Mean':>8} {'Std':>7}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    for df in dfs:
        name = df["run"].iloc[0]
        rew  = df["total_reward"]
        print(f"  {name:<16} {len(df):>8} "
              f"{rew.max():>8.2f} {rew.mean():>8.2f} {rew.std():>7.2f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analyze MicrolifeEnv training runs")
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run directories (must contain training_log.csv)")
    p.add_argument("--out",  default=None,
                   help="Output directory for plots (default: first run dir)")
    return p.parse_args()


def main():
    args = parse_args()
    dfs = [df for r in args.runs if (df := load_run(r)) is not None]
    if not dfs:
        print("No valid runs found.")
        sys.exit(1)

    out_dir = args.out or args.runs[0]
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nAnalyzing {len(dfs)} run(s) → {out_dir}\n")

    plot_learning_curves(dfs, os.path.join(out_dir, "learning_curves.png"))
    plot_survival(dfs,        os.path.join(out_dir, "survival.png"))
    plot_epsilon(dfs,         os.path.join(out_dir, "epsilon_decay.png"))
    print_summary(dfs)


if __name__ == "__main__":
    main()
