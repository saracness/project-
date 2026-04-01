"""Tests for RL brain classes — QLearningBrain, DQNBrain, DoubleDQNBrain."""

import numpy as np
import pytest

from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain
from microlife.ml.brain_base import Brain


# ---------------------------------------------------------------------------
# Shared state dict fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_state():
    return {
        "energy": 80.0,
        "nearest_food_distance": 50.0,
        "nearest_food_angle": 0.5,
        "in_temperature_zone": False,
        "near_obstacle": False,
        "age": 10,
        "speed": 1.0,
    }


@pytest.fixture
def next_state():
    return {
        "energy": 90.0,
        "nearest_food_distance": 30.0,
        "nearest_food_angle": 0.3,
        "in_temperature_zone": False,
        "near_obstacle": False,
        "age": 11,
        "speed": 1.0,
    }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class TestBrainInterface:
    def test_qlearning_is_brain(self):
        assert isinstance(QLearningBrain(), Brain)

    def test_dqn_is_brain(self):
        assert isinstance(DQNBrain(state_size=7), Brain)

    def test_ddqn_is_brain(self):
        assert isinstance(DoubleDQNBrain(state_size=7), Brain)


# ---------------------------------------------------------------------------
# decide_action
# ---------------------------------------------------------------------------

class TestDecideAction:
    @pytest.mark.parametrize("BrainClass, kwargs", [
        (QLearningBrain, {}),
        (DQNBrain,       {"state_size": 7}),
        (DoubleDQNBrain, {"state_size": 7}),
    ])
    def test_returns_dict(self, BrainClass, kwargs, sample_state):
        brain = BrainClass(**kwargs)
        action = brain.decide_action(sample_state)
        assert isinstance(action, dict)

    @pytest.mark.parametrize("BrainClass, kwargs", [
        (QLearningBrain, {}),
        (DQNBrain,       {"state_size": 7}),
        (DoubleDQNBrain, {"state_size": 7}),
    ])
    def test_has_move_direction(self, BrainClass, kwargs, sample_state):
        brain = BrainClass(**kwargs)
        action = brain.decide_action(sample_state)
        assert "move_direction" in action
        dx, dy = action["move_direction"]
        assert dx in (-1, 0, 1)
        assert dy in (-1, 0, 1)

    @pytest.mark.parametrize("BrainClass, kwargs", [
        (QLearningBrain, {}),
        (DQNBrain,       {"state_size": 7}),
        (DoubleDQNBrain, {"state_size": 7}),
    ])
    def test_has_action_idx(self, BrainClass, kwargs, sample_state):
        brain = BrainClass(**kwargs)
        action = brain.decide_action(sample_state)
        assert "_action_idx" in action
        assert 0 <= action["_action_idx"] <= 8

    def test_decision_count_increments(self, sample_state):
        brain = DQNBrain(state_size=7)
        for i in range(5):
            brain.decide_action(sample_state)
        assert brain.decision_count == 5


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

class TestLearn:
    def test_qlearning_updates_q_table(self, sample_state, next_state):
        brain = QLearningBrain()
        action = brain.decide_action(sample_state)
        brain.learn(sample_state, action, reward=1.0, next_state=next_state, done=False)
        assert len(brain.q_table) > 0

    def test_dqn_stores_experience(self, sample_state, next_state):
        brain = DQNBrain(state_size=7)
        action = brain.decide_action(sample_state)
        brain.learn(sample_state, action, reward=1.0, next_state=next_state, done=False)
        assert len(brain.memory) == 1

    def test_dqn_replays_when_buffer_full(self, sample_state, next_state):
        brain = DQNBrain(state_size=7, hidden_size=16, learning_rate=0.01)
        # Fill buffer past batch_size
        for _ in range(brain.batch_size + 5):
            action = brain.decide_action(sample_state)
            brain.learn(sample_state, action, reward=0.5, next_state=next_state, done=False)
        # Weights should have been updated — check they differ from initial zeros
        assert not np.allclose(brain.w1, 0.0)

    def test_ddqn_target_network_updates(self, sample_state, next_state):
        brain = DoubleDQNBrain(state_size=7, hidden_size=16)
        initial_target_w1 = brain.target_w1.copy()
        # Run enough steps to trigger target update
        for _ in range(brain.update_target_every + brain.batch_size + 5):
            action = brain.decide_action(sample_state)
            brain.learn(sample_state, action, reward=0.5, next_state=next_state, done=False)
        # Target network should now differ from its initial state
        assert not np.allclose(brain.target_w1, initial_target_w1)

    def test_epsilon_decays_over_time(self, sample_state, next_state):
        brain = DQNBrain(state_size=7)
        initial_epsilon = brain.epsilon
        for _ in range(200):
            action = brain.decide_action(sample_state)
            brain.learn(sample_state, action, reward=0.0, next_state=next_state, done=False)
        assert brain.epsilon < initial_epsilon

    def test_epsilon_does_not_go_below_min(self, sample_state, next_state):
        brain = DQNBrain(state_size=7)
        brain.epsilon = brain.epsilon_min  # force to minimum
        for _ in range(10):
            action = brain.decide_action(sample_state)
            brain.learn(sample_state, action, reward=0.0, next_state=next_state, done=False)
        assert brain.epsilon >= brain.epsilon_min


# ---------------------------------------------------------------------------
# get_state_vector / calculate_reward
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_state_vector_shape(self, sample_state):
        brain = DQNBrain(state_size=7)
        vec = brain.get_state_vector(sample_state)
        assert vec.shape == (7,)

    def test_state_vector_normalised(self, sample_state):
        brain = DQNBrain(state_size=7)
        vec = brain.get_state_vector(sample_state)
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)

    def test_reward_eating_positive(self):
        brain = DQNBrain(state_size=7)
        old = {"energy": 80.0, "nearest_food_distance": 20.0}
        new = {"energy": 100.0, "nearest_food_distance": 5.0}  # ate food
        reward = brain.calculate_reward(old, new, {})
        assert reward > 0

    def test_get_stats(self, sample_state, next_state):
        brain = QLearningBrain()
        action = brain.decide_action(sample_state)
        brain.learn(sample_state, action, 1.0, next_state, False)
        stats = brain.get_stats()
        assert stats["type"] == "Q-Learning"
        assert stats["decisions"] >= 1
