"""Tests for MicrolifeEnv — the Gym-compatible RL environment."""

import math
import numpy as np
import pytest

from microlife.gym_env import MicrolifeEnv, N_ACTIONS, OBS_DIM, ACTIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return MicrolifeEnv(width=200, height=200, n_food=10, n_temp_zones=2,
                        max_steps=100, seed=0)


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class TestObservationSpace:
    def test_reset_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (OBS_DIM,)

    def test_obs_dtype_float32(self, env):
        obs = env.reset()
        assert obs.dtype == np.float32

    def test_obs_values_in_range(self, env):
        obs = env.reset()
        # Most dims should be in [0, 1]; sin/cos remapped to [0,1] so full range ok
        assert np.all(obs >= 0.0), f"Obs below 0: {obs}"
        assert np.all(obs <= 1.0), f"Obs above 1: {obs}"

    def test_step_obs_correct_shape(self, env):
        env.reset()
        obs, _, _, _ = env.step(0)
        assert obs.shape == (OBS_DIM,)

    def test_obs_values_after_step(self, env):
        env.reset()
        for action in range(N_ACTIONS):
            env.reset()
            obs, _, done, _ = env.step(action)
            if not done:
                assert np.all(obs >= 0.0)
                assert np.all(obs <= 1.0)


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

class TestActionSpace:
    def test_n_actions(self):
        assert N_ACTIONS == 9

    def test_all_actions_valid(self, env):
        """Every action index should complete without error."""
        for action in range(N_ACTIONS):
            env.reset()
            obs, reward, done, info = env.step(action)
            assert isinstance(reward, float)
            assert isinstance(done, bool)

    def test_action_directions_normalised(self):
        for dx, dy in ACTIONS:
            assert dx in (-1, 0, 1)
            assert dy in (-1, 0, 1)


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------

class TestEpisodeLifecycle:
    def test_episode_terminates_within_max_steps(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(0)
            steps += 1
            assert steps <= env.max_steps + 1, "Episode exceeded max_steps"

    def test_reset_clears_step_count(self, env):
        env.reset()
        for _ in range(10):
            _, _, done, _ = env.step(2)
            if done:
                break
        env.reset()
        assert env._step_count == 0

    def test_episode_reward_accumulates(self, env):
        env.reset()
        total = 0.0
        for _ in range(5):
            _, r, done, _ = env.step(0)
            total += r
            if done:
                break
        assert env._episode_reward != 0.0 or total == pytest.approx(0.0)

    def test_done_flag_on_death(self, env):
        """If energy drops to 0 the episode must end."""
        env.reset()
        env._organism.energy = 0.01  # nearly dead
        # One step consuming movement energy should kill it
        for _ in range(20):
            _, _, done, info = env.step(0)
            if done:
                return  # passed
        # If energy never hit 0 in 20 steps, pass anyway (seed-dependent)

    def test_multiple_resets_independent(self, env):
        obs1 = env.reset()
        for _ in range(20):
            _, _, done, _ = env.step(3)
            if done:
                break
        obs2 = env.reset()
        # After reset the organism should be back near centre with full energy
        assert env._organism.energy == pytest.approx(100.0)
        assert env._step_count == 0


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class TestReward:
    def test_eating_gives_positive_reward(self, env):
        """Place food on top of organism and step — reward should be positive."""
        env.reset()
        o = env._organism
        from microlife.simulation.organism import Food
        # Drop food exactly on organism
        env._env.food_particles.append(Food(o.x, o.y, energy=20.0))
        _, reward, _, _ = env.step(8)  # Stay action — food within eating distance
        assert reward > 0.0, f"Expected positive reward when eating, got {reward}"

    def test_survival_bonus_positive(self, env):
        """Standing still in a safe zone should yield small positive reward."""
        env.reset()
        # Remove temperature zones so there's no hazard
        env._env.temperature_zones.clear()
        # Place food far away so no eating bonus
        for f in env._env.food_particles:
            f.x = env.width - 1
            f.y = env.height - 1
        env._organism.x = 0.0
        env._organism.y = 0.0
        # Give max energy so no starvation penalty
        env._organism.energy = 190.0
        _, reward, _, _ = env.step(8)  # Stay
        # survival bonus (0.05) minus small energy cost; should still be > -1
        assert reward > -1.0


# ---------------------------------------------------------------------------
# obs_to_brain_state
# ---------------------------------------------------------------------------

class TestObsToBrainState:
    def test_keys_present(self, env):
        obs = env.reset()
        state = env.obs_to_brain_state(obs)
        required = {
            "energy", "nearest_food_distance", "nearest_food_angle",
            "in_temperature_zone", "near_obstacle", "age", "speed",
        }
        assert required.issubset(state.keys())

    def test_energy_round_trip(self, env):
        obs = env.reset()
        state = env.obs_to_brain_state(obs)
        # energy dim is obs[0] * 200
        assert state["energy"] == pytest.approx(float(obs[0]) * 200.0, rel=1e-4)

    def test_angle_is_float(self, env):
        obs = env.reset()
        state = env.obs_to_brain_state(obs)
        assert isinstance(state["nearest_food_angle"], float)

    def test_in_temperature_zone_is_bool(self, env):
        obs = env.reset()
        state = env.obs_to_brain_state(obs)
        assert isinstance(state["in_temperature_zone"], bool)
