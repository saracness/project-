"""
RandomBrain: a Brain implementation that selects actions uniformly at random.

Used as a baseline in benchmarks. Running through the Brain interface
(and therefore through Environment._move_with_ai) keeps the evaluation
code path identical to that of RL agents, giving a fair comparison.
"""
import random
from .brain_base import Brain


class RandomBrain(Brain):
    """
    Chooses a uniformly random direction at every step.
    Does not learn from experience.

    Use this instead of mode='random' in Environment when you want all
    conditions to share the same _move_with_ai code path.
    """

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

    def __init__(self):
        super().__init__(brain_type="Random")

    def decide_action(self, state):
        return {
            "move_direction": random.choice(self._ACTIONS),
            "should_reproduce": False,
            "speed_multiplier": 1.0,
        }

    def learn(self, state, action, reward, next_state, done):
        pass  # stateless

    def save_model(self, filepath):
        pass  # nothing to save

    def load_model(self, filepath):
        pass
