"""
MICROLIFE Reinforcement Learning Environment
============================================

A biological RL environment where an organism learns to survive.

The agent controls a single organism in a simplified MICROLIFE world:
    - Must find food (energy sources)
    - Avoid predators
    - Survive as long as possible

This demonstrates RL in a continuous, biological setting.

State Space (8 dimensions):
    - Own energy level (0-1)
    - Own position (x, y) normalized
    - Nearest food direction (dx, dy) normalized
    - Nearest food distance (0-1)
    - Nearest predator direction (dx, dy) normalized
    - Nearest predator distance (0-1)

Action Space (4 actions):
    - 0: Move UP
    - 1: Move DOWN
    - 2: Move LEFT
    - 3: Move RIGHT

Rewards:
    - +10: Eat food
    - -20: Eaten by predator (terminal)
    - -0.01: Energy cost per step
    - +0.1: Still alive (survival bonus)

Episode ends when:
    - Agent runs out of energy (death)
    - Agent eaten by predator
    - Max steps reached (1000)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple, Optional
import sys


class MicroLifeEnv:
    """
    RL environment for MICROLIFE organism survival.

    An organism learns to:
        1. Find food (green particles)
        2. Avoid predators (red circles)
        3. Manage energy efficiently
    """

    def __init__(self,
                 world_size: float = 100.0,
                 n_food: int = 5,
                 n_predators: int = 2,
                 food_reward: float = 10.0,
                 death_penalty: float = -20.0,
                 step_penalty: float = -0.01,
                 survival_reward: float = 0.1,
                 max_steps: int = 1000):
        """
        Initialize MICROLIFE RL environment.

        Args:
            world_size: Size of the world (world_size x world_size)
            n_food: Number of food particles
            n_predators: Number of predators
            food_reward: Reward for eating food
            death_penalty: Penalty for being eaten
            step_penalty: Energy cost per step
            survival_reward: Reward for staying alive
            max_steps: Maximum steps per episode
        """
        self.world_size = world_size
        self.n_food = n_food
        self.n_predators = n_predators

        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.step_penalty = step_penalty
        self.survival_reward = survival_reward
        self.max_steps = max_steps

        # State and action spaces
        self.state_dim = 9  # See docstring for state space
        self.n_actions = 4  # UP, DOWN, LEFT, RIGHT

        # Game state
        self.agent_pos = np.array([50.0, 50.0])
        self.agent_energy = 1.0
        self.food_positions = []
        self.predator_positions = []
        self.predator_velocities = []

        self.steps = 0
        self.total_reward = 0
        self.done = False

        # Constants
        self.agent_speed = 2.0
        self.predator_speed = 1.5
        self.food_size = 2.0
        self.predator_size = 4.0
        self.agent_size = 3.0

        # For rendering
        self.fig = None
        self.ax = None

        # Statistics
        self.episode_lengths = []
        self.episode_rewards = []
        self.food_collected = []

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            state: Initial observation
        """
        # Reset agent
        self.agent_pos = np.array([
            np.random.uniform(20, self.world_size - 20),
            np.random.uniform(20, self.world_size - 20)
        ])
        self.agent_energy = 1.0

        # Reset food
        self.food_positions = []
        for _ in range(self.n_food):
            pos = np.array([
                np.random.uniform(0, self.world_size),
                np.random.uniform(0, self.world_size)
            ])
            self.food_positions.append(pos)

        # Reset predators
        self.predator_positions = []
        self.predator_velocities = []
        for _ in range(self.n_predators):
            pos = np.array([
                np.random.uniform(0, self.world_size),
                np.random.uniform(0, self.world_size)
            ])
            vel = np.random.randn(2)
            vel = vel / np.linalg.norm(vel) * self.predator_speed

            self.predator_positions.append(pos)
            self.predator_velocities.append(vel)

        self.steps = 0
        self.total_reward = 0
        self.done = False
        self.foods_eaten = 0

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take action in environment.

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

        Returns:
            observation: New state
            reward: Reward received
            done: Whether episode finished
            info: Additional info
        """
        if self.done:
            raise ValueError("Episode done. Call reset().")

        # Move agent
        if action == 0:  # UP
            self.agent_pos[1] -= self.agent_speed
        elif action == 1:  # DOWN
            self.agent_pos[1] += self.agent_speed
        elif action == 2:  # LEFT
            self.agent_pos[0] -= self.agent_speed
        elif action == 3:  # RIGHT
            self.agent_pos[0] += self.agent_speed

        # Clip to boundaries
        self.agent_pos = np.clip(self.agent_pos, 0, self.world_size)

        # Move predators (simple random walk with momentum)
        for i in range(len(self.predator_positions)):
            # Add some randomness to movement
            self.predator_velocities[i] += np.random.randn(2) * 0.5
            # Normalize to constant speed
            vel_norm = np.linalg.norm(self.predator_velocities[i])
            if vel_norm > 0:
                self.predator_velocities[i] = (self.predator_velocities[i] / vel_norm) * self.predator_speed

            # Move
            self.predator_positions[i] += self.predator_velocities[i]

            # Bounce off walls
            if self.predator_positions[i][0] < 0 or self.predator_positions[i][0] > self.world_size:
                self.predator_velocities[i][0] *= -1
            if self.predator_positions[i][1] < 0 or self.predator_positions[i][1] > self.world_size:
                self.predator_velocities[i][1] *= -1

            self.predator_positions[i] = np.clip(self.predator_positions[i], 0, self.world_size)

        # Initialize reward
        reward = self.step_penalty + self.survival_reward

        # Check food collection
        for food_pos in self.food_positions[:]:
            dist = np.linalg.norm(self.agent_pos - food_pos)
            if dist < (self.agent_size + self.food_size):
                # Eat food
                reward += self.food_reward
                self.agent_energy = min(1.0, self.agent_energy + 0.3)
                self.food_positions.remove(food_pos)
                self.foods_eaten += 1

                # Respawn food
                new_food = np.array([
                    np.random.uniform(0, self.world_size),
                    np.random.uniform(0, self.world_size)
                ])
                self.food_positions.append(new_food)

        # Check predator collision
        for pred_pos in self.predator_positions:
            dist = np.linalg.norm(self.agent_pos - pred_pos)
            if dist < (self.agent_size + self.predator_size):
                # Eaten by predator
                reward += self.death_penalty
                self.done = True
                break

        # Energy decay
        self.agent_energy -= 0.001

        # Check energy depletion
        if self.agent_energy <= 0:
            self.done = True
            reward += self.death_penalty

        # Check max steps
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True

        self.total_reward += reward

        obs = self._get_observation()
        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'energy': self.agent_energy,
            'food_collected': self.foods_eaten
        }

        if self.done:
            self.episode_lengths.append(self.steps)
            self.episode_rewards.append(self.total_reward)
            self.food_collected.append(self.foods_eaten)

        return obs, reward, self.done, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation vector.

        Returns:
            state: [energy, pos_x, pos_y, food_dx, food_dy, food_dist,
                   pred_dx, pred_dy, pred_dist]
        """
        # Agent state
        energy = self.agent_energy
        pos_x = self.agent_pos[0] / self.world_size
        pos_y = self.agent_pos[1] / self.world_size

        # Nearest food
        if len(self.food_positions) > 0:
            food_dists = [np.linalg.norm(self.agent_pos - f) for f in self.food_positions]
            nearest_food_idx = np.argmin(food_dists)
            nearest_food = self.food_positions[nearest_food_idx]

            food_vec = nearest_food - self.agent_pos
            food_dist = np.linalg.norm(food_vec)
            if food_dist > 0:
                food_vec = food_vec / food_dist  # Normalize direction

            food_dx = food_vec[0]
            food_dy = food_vec[1]
            food_dist_norm = min(food_dist / self.world_size, 1.0)
        else:
            food_dx = 0
            food_dy = 0
            food_dist_norm = 1.0

        # Nearest predator
        if len(self.predator_positions) > 0:
            pred_dists = [np.linalg.norm(self.agent_pos - p) for p in self.predator_positions]
            nearest_pred_idx = np.argmin(pred_dists)
            nearest_pred = self.predator_positions[nearest_pred_idx]

            pred_vec = nearest_pred - self.agent_pos
            pred_dist = np.linalg.norm(pred_vec)
            if pred_dist > 0:
                pred_vec = pred_vec / pred_dist

            pred_dx = pred_vec[0]
            pred_dy = pred_vec[1]
            pred_dist_norm = min(pred_dist / self.world_size, 1.0)
        else:
            pred_dx = 0
            pred_dy = 0
            pred_dist_norm = 1.0

        obs = np.array([
            energy,
            pos_x,
            pos_y,
            food_dx,
            food_dy,
            food_dist_norm,
            pred_dx,
            pred_dy,
            pred_dist_norm
        ], dtype=np.float32)

        return obs

    def render(self, mode='human'):
        """Visualize the environment."""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()
        self.ax.set_xlim(0, self.world_size)
        self.ax.set_ylim(0, self.world_size)
        self.ax.set_aspect('equal')

        # Draw food (green)
        for food_pos in self.food_positions:
            circle = Circle(food_pos, self.food_size, color='green', alpha=0.7)
            self.ax.add_patch(circle)

        # Draw predators (red)
        for pred_pos in self.predator_positions:
            circle = Circle(pred_pos, self.predator_size, color='red', alpha=0.7)
            self.ax.add_patch(circle)

        # Draw agent (blue)
        circle = Circle(self.agent_pos, self.agent_size, color='blue', alpha=0.9)
        self.ax.add_patch(circle)

        # Energy bar
        energy_bar_len = 20
        self.ax.plot([5, 5 + energy_bar_len * self.agent_energy],
                    [95, 95], linewidth=10, color='yellow')
        self.ax.text(5, 97, f'Energy: {self.agent_energy:.2f}', fontsize=10)

        # Stats
        self.ax.set_title(f'MICROLIFE RL - Steps: {self.steps} | '
                         f'Reward: {self.total_reward:.1f} | '
                         f'Food: {self.foods_eaten}',
                         fontsize=12)

        plt.tight_layout()

        if mode == 'human':
            plt.pause(0.001)

        return self.fig

    def close(self):
        """Close rendering."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None


# Simple testing
if __name__ == '__main__':
    print("MICROLIFE RL Environment Demo")
    print("=" * 50)

    env = MicroLifeEnv()

    print(f"State dimension: {env.state_dim}")
    print(f"Action space: {env.n_actions}")
    print()

    # Random policy
    print("Testing random policy (50 steps)...")
    obs = env.reset()
    env.render()

    for step in range(50):
        action = np.random.randint(env.n_actions)
        obs, reward, done, info = env.step(action)

        print(f"Step {step}: Reward={reward:6.2f}, Energy={info['energy']:.2f}, "
              f"Food={info['food_collected']}")

        env.render()

        if done:
            print(f"\nEpisode ended: {info}")
            break

    plt.show()
    env.close()
