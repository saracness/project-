"""
Machine Learning module.

Provides Brain implementations for organism agents:
  RandomBrain    -- uniform random action selection (baseline)
  QLearningBrain -- tabular Q-Learning
  DQNBrain       -- Deep Q-Network (numpy, no external RL library)
  DoubleDQNBrain -- Double DQN
"""
from .brain_base import Brain
from .brain_random import RandomBrain
from .brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain

__all__ = [
    "Brain",
    "RandomBrain",
    "QLearningBrain",
    "DQNBrain",
    "DoubleDQNBrain",
]
