"""
Reinforcement Learning Brains:
1. QLearningBrain  -- tabular Q-Learning
2. DQNBrain        -- Deep Q-Network (numpy, no external RL library)
3. DoubleDQNBrain  -- Double DQN (reduces overestimation bias)

References:
  Watkins & Dayan (1992): Q-Learning
  Mnih et al. (2015): DQN with experience replay
  Van Hasselt et al. (2016): Double DQN
"""
import os
import pickle
import random
from collections import deque

import numpy as np

from .brain_base import Brain


# ---------------------------------------------------------------------------
# Action space shared by all RL brains
# ---------------------------------------------------------------------------
_ACTIONS = [
    (0,  1),   # N
    (1,  1),   # NE
    (1,  0),   # E
    (1, -1),   # SE
    (0, -1),   # S
    (-1, -1),  # SW
    (-1,  0),  # W
    (-1,  1),  # NW
    (0,  0),   # stay
]


# ---------------------------------------------------------------------------
# Q-Learning (tabular)
# ---------------------------------------------------------------------------
class QLearningBrain(Brain):
    """
    Tabular Q-Learning.
    State is discretised into bins; suitable for the organism's low-dimensional
    observation (energy, food distance, food angle, temperature zone flag).
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        super().__init__(brain_type="Q-Learning")
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}          # discrete_state -> np.ndarray(n_actions,)
        self.actions = _ACTIONS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _discretize_state(self, state):
        energy_bin    = int(state.get("energy", 0) / 20)          # 0-10
        food_dist_bin = min(int(state.get("nearest_food_distance", 500) / 50), 10)
        angle_bin     = int(state.get("nearest_food_angle", 0) / (np.pi / 4))  # 0-7
        in_temp       = 1 if state.get("in_temperature_zone", False) else 0
        return (energy_bin, food_dist_bin, angle_bin, in_temp)

    def _q_row(self, discrete_state):
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        return self.q_table[discrete_state]

    # ------------------------------------------------------------------
    # Brain interface
    # ------------------------------------------------------------------
    def decide_action(self, state):
        self.decision_count += 1
        ds = self._discretize_state(state)

        if random.random() < self.epsilon:
            action_idx = random.randrange(len(self.actions))
        else:
            action_idx = int(np.argmax(self._q_row(ds)))

        return {
            "move_direction": self.actions[action_idx],
            "should_reproduce": state.get("energy", 0) > 150,
            "speed_multiplier": 1.0,
            "_action_idx": action_idx,
        }

    def learn(self, state, action, reward, next_state, done):
        ds      = self._discretize_state(state)
        ds_next = self._discretize_state(next_state)
        idx     = action.get("_action_idx", 0)

        current_q  = self._q_row(ds)[idx]
        max_next_q = 0.0 if done else float(np.max(self._q_row(ds_next)))
        target_q   = reward + self.gamma * max_next_q

        self._q_row(ds)[idx] = current_q + self.lr * (target_q - current_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_reward += reward

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, filepath):
        """Persist Q-table and training state to *filepath* (pickle)."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as fh:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "total_reward": self.total_reward,
                "decision_count": self.decision_count,
            }, fh)

    def load_model(self, filepath):
        """Restore Q-table and training state from *filepath*."""
        with open(filepath, "rb") as fh:
            data = pickle.load(fh)
        self.q_table       = data["q_table"]
        self.epsilon       = data.get("epsilon", self.epsilon_min)
        self.total_reward  = data.get("total_reward", 0.0)
        self.decision_count = data.get("decision_count", 0)


# ---------------------------------------------------------------------------
# DQN (Deep Q-Network, numpy)
# ---------------------------------------------------------------------------
class DQNBrain(Brain):
    """
    Single-hidden-layer feed-forward Q-network implemented in pure numpy
    with experience replay.

    Reference: Mnih et al. (2015) -- no external RL library required.
    """

    def __init__(self, state_size=7, hidden_size=32, learning_rate=0.001):
        super().__init__(brain_type="DQN")
        self.state_size  = state_size
        self.hidden_size = hidden_size
        self.action_size = len(_ACTIONS)
        self.lr          = learning_rate

        # Network weights (Xavier-ish initialisation)
        self.w1 = np.random.randn(state_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, self.action_size) * 0.1
        self.b2 = np.zeros(self.action_size)

        self.memory     = deque(maxlen=2000)
        self.batch_size = 32
        self.epsilon      = 0.5
        self.epsilon_min  = 0.01
        self.epsilon_decay = 0.995
        self.gamma        = 0.95
        self.actions      = _ACTIONS

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def _forward(self, x):
        h = np.maximum(0, x @ self.w1 + self.b1)   # ReLU hidden
        return h @ self.w2 + self.b2                 # linear output

    # ------------------------------------------------------------------
    # Brain interface
    # ------------------------------------------------------------------
    def decide_action(self, state):
        self.decision_count += 1
        sv = self.get_state_vector(state)

        if random.random() < self.epsilon:
            idx = random.randrange(self.action_size)
        else:
            idx = int(np.argmax(self._forward(sv)))

        return {
            "move_direction":  self.actions[idx],
            "should_reproduce": state.get("energy", 0) > 150,
            "speed_multiplier": 1.0,
            "_action_idx":    idx,
            "_state_vector":  sv,
        }

    def learn(self, state, action, reward, next_state, done):
        sv      = action.get("_state_vector") if action.get("_state_vector") is not None \
                  else self.get_state_vector(state)
        sv_next = self.get_state_vector(next_state)
        idx     = action.get("_action_idx", 0)

        self.memory.append((sv, idx, reward, sv_next, done))

        if len(self.memory) >= self.batch_size:
            self._replay()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_reward += reward

    def _replay(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for sv, idx, reward, sv_next, done in batch:
            q_pred = self._forward(sv)
            target = reward if done else reward + self.gamma * float(np.max(self._forward(sv_next)))
            q_pred[idx] = target  # only update the taken action

            # Backprop (gradient descent, one step)
            h           = np.maximum(0, sv @ self.w1 + self.b1)
            out_err     = q_pred - (h @ self.w2 + self.b2)
            self.w2    += self.lr * np.outer(h, out_err)
            self.b2    += self.lr * out_err
            hidden_err  = (out_err @ self.w2.T) * (h > 0)
            self.w1    += self.lr * np.outer(sv, hidden_err)
            self.b1    += self.lr * hidden_err

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_model(self, filepath):
        """Save network weights and epsilon to *filepath*.npz."""
        path = filepath if filepath.endswith(".npz") else filepath + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            w1=self.w1, b1=self.b1,
            w2=self.w2, b2=self.b2,
            meta=np.array([self.epsilon, self.total_reward, self.decision_count]),
        )

    def load_model(self, filepath):
        """Load network weights and epsilon from *filepath* (.npz)."""
        path = filepath if filepath.endswith(".npz") else filepath + ".npz"
        d = np.load(path)
        self.w1, self.b1 = d["w1"], d["b1"]
        self.w2, self.b2 = d["w2"], d["b2"]
        if "meta" in d:
            self.epsilon, self.total_reward, dc = d["meta"]
            self.decision_count = int(dc)


# ---------------------------------------------------------------------------
# Double DQN
# ---------------------------------------------------------------------------
class DoubleDQNBrain(DQNBrain):
    """
    Double DQN: decouples action selection from value estimation to reduce
    Q-value overestimation.

    Reference: Van Hasselt et al. (2016).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brain_type = "Double-DQN"

        # Target network (periodically synced from main network)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

        self.update_target_every = 100
        self.update_counter      = 0

    def _forward_target(self, x):
        h = np.maximum(0, x @ self.target_w1 + self.target_b1)
        return h @ self.target_w2 + self.target_b2

    def _replay(self):
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        for sv, idx, reward, sv_next, done in batch:
            q_pred = self._forward(sv)
            if done:
                target = reward
            else:
                # Main net selects action, target net evaluates it
                best_action = int(np.argmax(self._forward(sv_next)))
                target = reward + self.gamma * float(self._forward_target(sv_next)[best_action])
            q_pred[idx] = target

            h          = np.maximum(0, sv @ self.w1 + self.b1)
            out_err    = q_pred - (h @ self.w2 + self.b2)
            self.w2   += self.lr * np.outer(h, out_err)
            self.b2   += self.lr * out_err
            hidden_err = (out_err @ self.w2.T) * (h > 0)
            self.w1   += self.lr * np.outer(sv, hidden_err)
            self.b1   += self.lr * hidden_err

        self.update_counter += 1
        if self.update_counter >= self.update_target_every:
            self.target_w1 = self.w1.copy()
            self.target_b1 = self.b1.copy()
            self.target_w2 = self.w2.copy()
            self.target_b2 = self.b2.copy()
            self.update_counter = 0

    # ------------------------------------------------------------------
    # Persistence (target network included)
    # ------------------------------------------------------------------
    def save_model(self, filepath):
        path = filepath if filepath.endswith(".npz") else filepath + ".npz"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
            target_w1=self.target_w1, target_b1=self.target_b1,
            target_w2=self.target_w2, target_b2=self.target_b2,
            meta=np.array([self.epsilon, self.total_reward, self.decision_count]),
        )

    def load_model(self, filepath):
        path = filepath if filepath.endswith(".npz") else filepath + ".npz"
        d = np.load(path)
        self.w1, self.b1 = d["w1"], d["b1"]
        self.w2, self.b2 = d["w2"], d["b2"]
        self.target_w1 = d.get("target_w1", self.w1.copy())
        self.target_b1 = d.get("target_b1", self.b1.copy())
        self.target_w2 = d.get("target_w2", self.w2.copy())
        self.target_b2 = d.get("target_b2", self.b2.copy())
        if "meta" in d:
            self.epsilon, self.total_reward, dc = d["meta"]
            self.decision_count = int(dc)
