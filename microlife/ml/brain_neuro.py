"""
NeuroBrain — DQN modulated by neuron personality types.

Each personality maps a distinct biological mechanism onto a concrete RL change:

  dopaminergic  → RPE-scaled learning rate  (Schultz 1997; surprise → LTP boost)
  serotonergic  → patience modulation       (higher γ, survival bonus, slow ε decay)
  cholinergic   → attentional gain          (adaptive lr + food-reward amplification)
  fast_spiking  → impulse control           (rapid ε decay, action-repetition penalty)
  place_cell    → spatial exploration bonus (novel grid cells rewarded)

Scientific references
---------------------
Schultz W (1997). A neural substrate of prediction and reward.
  Science 275(5306):1593–9.
Jacobs BL & Azmitia EC (1992). Structure and function of the brain serotonin system.
  Physiol Rev 72(1):165–229.
Hasselmo ME (2006). The role of acetylcholine in learning and memory.
  Curr Opin Neurobiol 16(6):710–5.
Markram H et al. (2004). Interneurons of the neocortical inhibitory system.
  Nat Rev Neurosci 5(10):793–807.
O'Keefe J & Dostrovsky J (1971). The hippocampus as a spatial map.
  Brain Res 34(1):171–5.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np

from .brain_rl import DQNBrain
from ..simulation.neuron_personalities import NEURON_PERSONALITIES, PersonalityTraits

# Personalities with full RL modulation implemented
SUPPORTED_PERSONALITIES = {
    "dopaminergic", "serotonergic", "cholinergic", "fast_spiking", "place_cell"
}


class NeuroBrain(DQNBrain):
    """
    DQN brain whose learning dynamics are shaped by a neuron personality.

    Parameters
    ----------
    personality_type : str
        One of SUPPORTED_PERSONALITIES.
    state_size : int
        Observation dimension (must match gym_env OBS_DIM = 8).
    hidden_size : int
        Hidden layer width.
    learning_rate : float
        Base learning rate (personality may override it dynamically).
    world_size : tuple[int, int]
        (width, height) of the simulation world — used by place_cell.
    """

    PLACE_GRID = 20   # place-cell grid resolution (20×20 cells)

    def __init__(
        self,
        personality_type: str = "dopaminergic",
        state_size: int = 8,
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        world_size: tuple[int, int] = (300, 300),
    ):
        if personality_type not in SUPPORTED_PERSONALITIES:
            raise ValueError(
                f"Personality '{personality_type}' not supported for RL modulation. "
                f"Available: {sorted(SUPPORTED_PERSONALITIES)}"
            )

        super().__init__(
            state_size=state_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
        )
        self.brain_type = f"NeuroBrain[{personality_type}]"
        self.personality_type  = personality_type
        self.personality: PersonalityTraits = NEURON_PERSONALITIES[personality_type]
        self.world_size = world_size

        # State needed by specific personalities
        self._last_action_idx: Optional[int] = None   # fast_spiking
        self._visited_cells: set = set()              # place_cell
        self._last_q_estimate: Optional[float] = None # dopaminergic

        self._reconfigure_for_personality()

    # ------------------------------------------------------------------
    # Public: change personality at runtime (live editor support)
    # ------------------------------------------------------------------

    def set_personality(self, personality_type: str) -> None:
        """Switch personality mid-run (live parameter editor)."""
        if personality_type not in SUPPORTED_PERSONALITIES:
            raise ValueError(
                f"'{personality_type}' not in {sorted(SUPPORTED_PERSONALITIES)}"
            )
        self.personality_type = personality_type
        self.personality = NEURON_PERSONALITIES[personality_type]
        self._last_action_idx = None
        self._visited_cells.clear()
        self._last_q_estimate = None
        self._reconfigure_for_personality()

    def _reconfigure_for_personality(self) -> None:
        """Set base hyperparameters according to personality traits."""
        p = self.personality_type

        if p == "dopaminergic":
            # Standard epsilon, but lr will be boosted dynamically by RPE
            self.epsilon_decay = 0.995
            self.gamma         = 0.95

        elif p == "serotonergic":
            # Patient, long-horizon, cautious exploration
            # Jacobs & Azmitia (1992): slow, sustained 5-HT activity → patience
            self.epsilon_decay = 0.999   # very slow decay → stays exploratory longer
            self.gamma         = 0.99    # higher discount → values future rewards more

        elif p == "cholinergic":
            # Attention-dependent: lr and reward sensitivity change with food proximity
            # Hasselmo (2006): ACh boosts cortical gain during attentional states
            self.epsilon_decay = 0.995
            self.gamma         = 0.95

        elif p == "fast_spiking":
            # GABAergic inhibition → sharp, decisive learning; low impulsivity
            # Markram et al. (2004): fast-spiking interneurons control motor output
            self.epsilon_decay = 0.980   # rapid decay → exploits early
            self.gamma         = 0.93    # slightly shorter horizon

        elif p == "place_cell":
            # Spatial curiosity: explores to build a cognitive map
            self.epsilon_decay = 0.997
            self.gamma         = 0.96

    # ------------------------------------------------------------------
    # Core RL overrides
    # ------------------------------------------------------------------

    def decide_action(self, state: dict) -> dict:
        """
        Personality modulates the exploration strategy before action selection.
        """
        self.decision_count += 1
        state_vector = self.get_state_vector(state)

        # ── Dopaminergic: UCB-style optimistic exploration ──────────────
        if self.personality_type == "dopaminergic" and random.random() >= self.epsilon:
            q = self._forward(state_vector)
            # Add small exploration noise proportional to dopamine sensitivity
            da_noise = (
                self.personality.dopamine_sensitivity
                * 0.1
                * np.random.randn(self.action_size)
            )
            action_idx = int(np.argmax(q + da_noise))
        else:
            # Standard epsilon-greedy for other personalities (or when exploring)
            if random.random() < self.epsilon:
                action_idx = random.randint(0, self.action_size - 1)
            else:
                action_idx = int(np.argmax(self._forward(state_vector)))

        direction = self.actions[action_idx]
        return {
            "move_direction":   direction,
            "should_reproduce": state.get("energy", 0) > 150,
            "speed_multiplier": 1.0,
            "_action_idx":      action_idx,
            "_state_vector":    state_vector,
        }

    def learn(
        self,
        state: dict,
        action: dict,
        reward: float,
        next_state: dict,
        done: bool,
    ) -> None:
        """
        Shape reward according to personality, then call parent learn().
        """
        shaped = self._shape_reward(reward, state, action, next_state, done)

        # Temporarily override lr for dopaminergic / cholinergic
        original_lr = self.lr
        self.lr = self._effective_lr(reward, state, next_state)

        super().learn(state, action, shaped, next_state, done)

        self.lr = original_lr
        self._last_action_idx = action.get("_action_idx")

        # place_cell: register visited cell
        if self.personality_type == "place_cell":
            cx = int(state.get("energy", 0))   # use energy as proxy; real x not in state
            # We mark the discretised food_distance bucket as a "region"
            bucket = int(
                min(state.get("nearest_food_distance", 300), 300)
                / (300 / self.PLACE_GRID)
            )
            angle_bucket = int(
                (state.get("nearest_food_angle", 0) % (2 * 3.14159))
                / (2 * 3.14159 / self.PLACE_GRID)
            )
            self._visited_cells.add((bucket, angle_bucket))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _shape_reward(
        self,
        reward: float,
        state: dict,
        action: dict,
        next_state: dict,
        done: bool,
    ) -> float:
        """Return reward after personality-specific shaping."""
        p = self.personality_type

        if p == "serotonergic":
            # Survival bonus every step (patience = value staying alive)
            # Scaled by serotonin sensitivity from personality traits
            shaped = reward + self.personality.serotonin_sensitivity * 0.15
            return shaped

        elif p == "cholinergic":
            # Amplify food-reward (attention directs organism towards food)
            # Hasselmo (2006): ACh increases signal-to-noise for attended stimuli
            if reward > 1.0:   # ate food
                return reward * (1.0 + self.personality.acetylcholine_sensitivity)
            return reward

        elif p == "fast_spiking":
            # Penalise action repetition (inhibitory control)
            # Repeating "Stay" especially is anti-productive
            if (
                self._last_action_idx is not None
                and self._last_action_idx == action.get("_action_idx")
                and self._last_action_idx == 8   # Stay action
            ):
                return reward - 0.15
            return reward

        elif p == "place_cell":
            # Novelty bonus for entering a new spatial region
            bucket = int(
                min(state.get("nearest_food_distance", 300), 300)
                / (300 / self.PLACE_GRID)
            )
            angle_bucket = int(
                (state.get("nearest_food_angle", 0) % (2 * 3.14159))
                / (2 * 3.14159 / self.PLACE_GRID)
            )
            cell = (bucket, angle_bucket)
            if cell not in self._visited_cells:
                return reward + 0.4   # novelty bonus
            return reward

        # dopaminergic: reward unchanged here; lr boost is in _effective_lr
        return reward

    def _effective_lr(
        self,
        reward: float,
        state: dict,
        next_state: dict,
    ) -> float:
        """Compute the effective learning rate for this step."""
        p = self.personality_type

        MAX_LR = 0.004   # hard cap — prevents weight explosion in numpy DQN

        if p == "dopaminergic":
            # RPE = TD error proxy: reward + γ·V(s') - V(s)
            # Use raw reward magnitude as a fast approximation of RPE
            # Large |RPE| → dopamine burst → stronger LTP (Schultz 1997)
            rpe = abs(reward)
            boost = 1.0 + self.personality.dopamine_sensitivity * min(rpe, 3.0)
            return min(self.lr * boost, MAX_LR)

        elif p == "cholinergic":
            # Adaptive lr: higher when food is nearby (arousal/attention state)
            # Hasselmo (2006): ACh increases learning rate for attended information
            food_prox = 1.0 - min(
                state.get("nearest_food_distance", 300) / 300.0, 1.0
            )
            boost = 1.0 + self.personality.acetylcholine_sensitivity * food_prox
            return min(self.lr * boost, MAX_LR)

        return self.lr

    # ------------------------------------------------------------------
    # Info for visualizer
    # ------------------------------------------------------------------

    def get_neuro_info(self) -> dict:
        """
        Returns personality-specific modulation state for the live panel.
        Purely informational — nothing in the simulation reads this.
        """
        p = self.personality_type
        info = {
            "personality":    p,
            "da_sensitivity": self.personality.dopamine_sensitivity,
            "5ht_sensitivity": self.personality.serotonin_sensitivity,
            "ach_sensitivity": self.personality.acetylcholine_sensitivity,
        }
        if p == "place_cell":
            info["novel_cells"] = len(self._visited_cells)
        return info
