"""
Tests for scripts/train_visual.py and scripts/analyze.py.

These run headless (no display required) and verify that:
- Headless training actually runs RL steps (not a stub)
- Screenshots are saved
- Q-values are extracted correctly from each brain type
- analyze.py produces real plots from real CSV data
"""

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless SDL before importing pygame via train_visual
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from microlife.gym_env import MicrolifeEnv, N_ACTIONS
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain


# ---------------------------------------------------------------------------
# Import helpers from scripts (without running __main__)
# ---------------------------------------------------------------------------

def _import_train_visual():
    import importlib.util, types
    spec = importlib.util.spec_from_file_location(
        "train_visual",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train_visual.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_analyze():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analyze",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# get_q_values — the function that proves the brain IS driving the animation
# ---------------------------------------------------------------------------

class TestGetQValues:
    """
    get_q_values() is the bridge: it reads the brain's actual internal state
    and returns the values that the visualizer draws as bars.
    """

    def setup_method(self):
        self.tv = _import_train_visual()

    def _state(self):
        return {
            "energy": 80.0, "nearest_food_distance": 50.0,
            "nearest_food_angle": 0.5, "in_temperature_zone": False,
            "near_obstacle": False, "age": 10, "speed": 1.0, "food_count": 12.0,
        }

    def test_dqn_returns_9_values(self):
        brain = DQNBrain(state_size=8)
        q = self.tv.get_q_values(brain, self._state())
        assert q.shape == (N_ACTIONS,)

    def test_ddqn_returns_9_values(self):
        brain = DoubleDQNBrain(state_size=8)
        q = self.tv.get_q_values(brain, self._state())
        assert q.shape == (N_ACTIONS,)

    def test_qlearning_returns_9_values(self):
        brain = QLearningBrain()
        q = self.tv.get_q_values(brain, self._state())
        assert q.shape == (N_ACTIONS,)

    def test_qlearning_zeros_before_learning(self):
        brain = QLearningBrain()
        q = self.tv.get_q_values(brain, self._state())
        assert np.allclose(q, 0.0)

    def test_dqn_values_change_after_learning(self):
        """After training, Q-values must differ from initial — proves weights updated."""
        brain = DQNBrain(state_size=8, hidden_size=16)
        state = self._state()
        q_before = self.tv.get_q_values(brain, state).copy()

        env = MicrolifeEnv(width=100, height=100, n_food=5, max_steps=50, seed=0)
        obs = env.reset()
        for _ in range(brain.batch_size + 10):
            s = env.obs_to_brain_state(obs)
            a = brain.decide_action(s)
            obs2, r, done, _ = env.step(a["_action_idx"])
            brain.learn(s, a, r, env.obs_to_brain_state(obs2), done)
            obs = obs2 if not done else env.reset()

        q_after = self.tv.get_q_values(brain, state)
        assert not np.allclose(q_before, q_after), \
            "Q-values unchanged after training — brain is not learning"

    def test_best_action_is_consistent_with_decide_action(self):
        """
        The action the brain decides must be argmax of Q-values.
        This proves the visualizer shows the SAME decision as the agent makes.
        """
        brain = DQNBrain(state_size=8)
        state = self._state()

        # Force greedy (no exploration)
        brain.epsilon = 0.0
        action_dict = brain.decide_action(state)
        q = self.tv.get_q_values(brain, state)

        assert action_dict["_action_idx"] == int(np.argmax(q)), \
            "Displayed Q-values disagree with brain's actual choice"


# ---------------------------------------------------------------------------
# Headless training run — end-to-end
# ---------------------------------------------------------------------------

class TestHeadlessRun:
    def test_run_produces_screenshots_and_curve(self):
        tv = _import_train_visual()
        with tempfile.TemporaryDirectory() as tmp:
            rewards = tv.run(
                brain_name="dqn",
                total_episodes=12,
                max_steps=50,
                target_fps=60,
                seed=0,
                headless=True,
                out_dir=tmp,
            )
            # Screenshot for episode 1 and 10
            assert os.path.exists(os.path.join(tmp, "ep_0001.png"))
            assert os.path.exists(os.path.join(tmp, "ep_0010.png"))
            # Learning curve saved
            assert os.path.exists(os.path.join(tmp, "learning_curve.png"))
            # Returns one reward per episode
            assert len(rewards) == 12

    def test_reward_list_is_real_numbers(self):
        tv = _import_train_visual()
        with tempfile.TemporaryDirectory() as tmp:
            rewards = tv.run(
                brain_name="dqn",
                total_episodes=5,
                max_steps=50,
                seed=1,
                headless=True,
                out_dir=tmp,
            )
        assert all(isinstance(r, float) for r in rewards)

    def test_learning_happens_over_episodes(self):
        """Later episodes should outperform early ones after sufficient training."""
        tv = _import_train_visual()
        with tempfile.TemporaryDirectory() as tmp:
            rewards = tv.run(
                brain_name="dqn",
                total_episodes=60,
                max_steps=200,
                seed=42,
                headless=True,
                out_dir=tmp,
            )
        first_10  = np.mean(rewards[:10])
        last_10   = np.mean(rewards[-10:])
        # Not a strict guarantee (stochastic), but should hold with seed=42
        assert last_10 > first_10, \
            f"Agent did not improve: first_10={first_10:.2f}, last_10={last_10:.2f}"

    @pytest.mark.parametrize("brain_name", ["qlearning", "dqn", "ddqn"])
    def test_all_brains_run_headless(self, brain_name):
        tv = _import_train_visual()
        with tempfile.TemporaryDirectory() as tmp:
            rewards = tv.run(
                brain_name=brain_name,
                total_episodes=3,
                max_steps=30,
                seed=0,
                headless=True,
                out_dir=tmp,
            )
        assert len(rewards) == 3


# ---------------------------------------------------------------------------
# analyze.py — plots from CSV
# ---------------------------------------------------------------------------

class TestAnalyze:
    def _make_csv(self, path: str, n: int = 50):
        """Write a synthetic training_log.csv."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "training_log.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "episode", "steps", "total_reward",
                "avg_reward_per_step", "final_energy", "epsilon", "elapsed_s",
            ])
            w.writeheader()
            for i in range(1, n + 1):
                w.writerow({
                    "episode": i, "steps": 200,
                    "total_reward": i * 0.5 + np.random.randn() * 2,
                    "avg_reward_per_step": 0.002 * i,
                    "final_energy": 80 + i * 0.3,
                    "epsilon": max(0.01, 0.5 * (0.995 ** i)),
                    "elapsed_s": 0.3,
                })

    def test_load_run(self):
        analyze = _import_analyze()
        with tempfile.TemporaryDirectory() as tmp:
            self._make_csv(tmp)
            df = analyze.load_run(tmp)
            assert df is not None
            assert len(df) == 50
            assert "total_reward" in df.columns

    def test_load_run_missing_returns_none(self):
        analyze = _import_analyze()
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze.load_run(tmp)
            assert result is None

    def test_plots_are_created(self):
        analyze = _import_analyze()
        with tempfile.TemporaryDirectory() as tmp:
            run_a = os.path.join(tmp, "run_a")
            run_b = os.path.join(tmp, "run_b")
            self._make_csv(run_a)
            self._make_csv(run_b)

            df_a = analyze.load_run(run_a)
            df_b = analyze.load_run(run_b)
            out  = os.path.join(tmp, "analysis")
            os.makedirs(out)

            analyze.plot_learning_curves([df_a, df_b],
                                         os.path.join(out, "lc.png"))
            analyze.plot_survival([df_a, df_b],
                                  os.path.join(out, "surv.png"))
            analyze.plot_epsilon([df_a, df_b],
                                 os.path.join(out, "eps.png"))

            assert os.path.exists(os.path.join(out, "lc.png"))
            assert os.path.exists(os.path.join(out, "surv.png"))
            assert os.path.exists(os.path.join(out, "eps.png"))
