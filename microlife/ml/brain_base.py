"""
Base Brain Interface for AI-powered organisms
Different AI models will implement this interface
"""
import numpy as np
from abc import ABC, abstractmethod


class Brain(ABC):
    """
    Abstract base class for organism brains.
    Each AI model (RL, CNN, GA, etc.) will implement this.
    """

    def __init__(self, brain_type="base"):
        """
        Initialize brain.

        Args:
            brain_type (str): Type of brain (RL, CNN, GA, etc.)
        """
        self.brain_type = brain_type
        self.decision_count = 0
        self.survival_time = 0
        self.total_reward = 0.0

    @abstractmethod
    def decide_action(self, state):
        """
        Decide what action to take based on current state.

        Args:
            state (dict): Current organism state with environment info
                {
                    'energy': float,
                    'nearest_food_distance': float,
                    'nearest_food_angle': float,
                    'in_temperature_zone': bool,
                    'near_obstacle': bool,
                    'age': int,
                    ...
                }

        Returns:
            dict: Action to take
                {
                    'move_direction': (dx, dy),  # Movement vector
                    'should_reproduce': bool,
                    'speed_multiplier': float
                }
        """
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done):
        """
        Learn from experience (for learning-based models).

        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: New state after action
            done: Whether episode ended (organism died)
        """
        pass

    def calculate_reward(self, old_state, new_state, action):
        """
        Calculate reward for an action.

        Returns:
            float: Reward value
        """
        reward = 0.0

        # Reward for surviving
        reward += 0.1

        # Reward for gaining energy
        energy_gain = new_state.get('energy', 0) - old_state.get('energy', 0)
        reward += energy_gain * 0.5

        # Reward for getting closer to food
        old_food_dist = old_state.get('nearest_food_distance', float('inf'))
        new_food_dist = new_state.get('nearest_food_distance', float('inf'))
        if new_food_dist < old_food_dist:
            reward += 1.0

        # Penalty for being in temperature zone
        if new_state.get('in_temperature_zone', False):
            reward -= 0.5

        # Penalty for low energy (danger!)
        if new_state.get('energy', 100) < 30:
            reward -= 1.0

        # Big reward for eating food
        if energy_gain > 15:  # Ate food
            reward += 10.0

        # Reward for reproduction (survival strategy)
        if action.get('should_reproduce', False):
            reward += 5.0

        return reward

    def get_state_vector(self, state):
        """
        Convert state dict to numpy vector for neural networks.

        Returns:
            np.ndarray: State as vector
        """
        return np.array([
            state.get('energy', 0) / 200.0,  # Normalize
            min(state.get('nearest_food_distance', 500) / 500.0, 1.0),
            state.get('nearest_food_angle', 0) / (2 * np.pi),
            1.0 if state.get('in_temperature_zone', False) else 0.0,
            1.0 if state.get('near_obstacle', False) else 0.0,
            min(state.get('age', 0) / 1000.0, 1.0),
            state.get('speed', 1.0) / 2.0
        ])

    def save_model(self, filepath):
        """Save brain model to file."""
        pass

    def load_model(self, filepath):
        """Load brain model from file."""
        pass

    def get_stats(self):
        """Get brain statistics."""
        return {
            'type': self.brain_type,
            'decisions': self.decision_count,
            'survival_time': self.survival_time,
            'total_reward': self.total_reward
        }
