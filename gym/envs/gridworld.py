"""
GridWorld Environment for Reinforcement Learning
================================================

A simple grid environment where an agent learns to navigate to a goal.

Environment:
    - Grid size: 5x5 (configurable)
    - Agent starts at (0, 0)
    - Goal at (4, 4)
    - Walls/obstacles (optional)

States:
    - (x, y) positions → flattened to single integer

Actions:
    - 0: UP
    - 1: DOWN
    - 2: LEFT
    - 3: RIGHT

Rewards:
    - +10: Reach goal
    - -1: Hit wall
    - -0.1: Each step (encourage efficiency)

This is a Markov Decision Process (MDP):
    - Deterministic transitions
    - Observable state
    - Episodic (resets when goal reached)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Tuple, Optional


class GridWorld:
    """
    Simple GridWorld environment for RL.

    Perfect for learning tabular methods (Q-Learning, SARSA).
    """

    # Action mapping
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    ACTION_EFFECTS = {
        UP: (-1, 0),
        DOWN: (1, 0),
        LEFT: (0, -1),
        RIGHT: (0, 1)
    }

    def __init__(self,
                 size: int = 5,
                 start_pos: Tuple[int, int] = (0, 0),
                 goal_pos: Tuple[int, int] = (4, 4),
                 walls: Optional[list] = None,
                 goal_reward: float = 10.0,
                 wall_penalty: float = -1.0,
                 step_penalty: float = -0.1):
        """
        Initialize GridWorld.

        Args:
            size: Grid size (size x size)
            start_pos: Starting position (row, col)
            goal_pos: Goal position (row, col)
            walls: List of wall positions [(r1,c1), (r2,c2), ...]
            goal_reward: Reward for reaching goal
            wall_penalty: Penalty for hitting wall
            step_penalty: Penalty for each step
        """
        self.size = size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.walls = walls if walls else []

        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.step_penalty = step_penalty

        # Current state
        self.agent_pos = list(start_pos)
        self.done = False
        self.total_reward = 0
        self.steps = 0

        # State/action spaces
        self.n_states = size * size
        self.n_actions = 4

        # For rendering
        self.fig = None
        self.ax = None

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            Initial state (flattened position)
        """
        self.agent_pos = list(self.start_pos)
        self.done = False
        self.total_reward = 0
        self.steps = 0
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take action in environment.

        Args:
            action: Action to take (0-3)

        Returns:
            next_state: New state
            reward: Reward received
            done: Whether episode is finished
            info: Additional information
        """
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        # Calculate new position
        dr, dc = self.ACTION_EFFECTS[action]
        new_pos = [self.agent_pos[0] + dr, self.agent_pos[1] + dc]

        # Check boundaries
        if not self._is_valid_pos(new_pos):
            # Hit wall or boundary - stay in place
            reward = self.wall_penalty
        else:
            # Valid move
            self.agent_pos = new_pos
            reward = self.step_penalty

            # Check if reached goal
            if tuple(self.agent_pos) == self.goal_pos:
                reward = self.goal_reward
                self.done = True

        self.steps += 1
        self.total_reward += reward

        state = self._pos_to_state(self.agent_pos)

        info = {
            'steps': self.steps,
            'total_reward': self.total_reward,
            'position': tuple(self.agent_pos)
        }

        return state, reward, self.done, info

    def _is_valid_pos(self, pos: list) -> bool:
        """Check if position is valid (in bounds and not a wall)."""
        r, c = pos

        # Check boundaries
        if r < 0 or r >= self.size or c < 0 or c >= self.size:
            return False

        # Check walls
        if tuple(pos) in self.walls:
            return False

        return True

    def _pos_to_state(self, pos: list) -> int:
        """Convert (row, col) to flattened state index."""
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert flattened state to (row, col)."""
        return (state // self.size, state % self.size)

    def render(self, mode='human', q_values: Optional[np.ndarray] = None):
        """
        Visualize the environment.

        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            q_values: Optional Q-values to display (shape: [n_states, n_actions])
        """
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.size - 0.5)
        self.ax.set_ylim(-0.5, self.size - 0.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Row 0 at top

        # Draw grid
        for i in range(self.size + 1):
            self.ax.axhline(i - 0.5, color='black', linewidth=1)
            self.ax.axvline(i - 0.5, color='black', linewidth=1)

        # Draw walls
        for wall in self.walls:
            rect = Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1,
                           facecolor='gray', edgecolor='black')
            self.ax.add_patch(rect)

        # Draw goal
        rect = Rectangle((self.goal_pos[1] - 0.5, self.goal_pos[0] - 0.5), 1, 1,
                        facecolor='gold', edgecolor='black', linewidth=2)
        self.ax.add_patch(rect)
        self.ax.text(self.goal_pos[1], self.goal_pos[0], 'G',
                    ha='center', va='center', fontsize=20, fontweight='bold')

        # Draw Q-values (arrows showing best action)
        if q_values is not None:
            for state in range(self.n_states):
                pos = self._state_to_pos(state)
                if tuple(pos) in self.walls or tuple(pos) == self.goal_pos:
                    continue

                # Get best action
                best_action = np.argmax(q_values[state])
                q_val = q_values[state, best_action]

                # Arrow direction
                dr, dc = self.ACTION_EFFECTS[best_action]

                # Draw arrow
                self.ax.arrow(pos[1], pos[0], dc * 0.3, dr * 0.3,
                            head_width=0.15, head_length=0.1,
                            fc='blue', ec='blue', alpha=0.7)

                # Draw Q-value
                self.ax.text(pos[1], pos[0] + 0.35, f'{q_val:.1f}',
                           ha='center', va='center', fontsize=8, color='blue')

        # Draw agent
        agent_circle = plt.Circle((self.agent_pos[1], self.agent_pos[0]), 0.3,
                                 color='red', zorder=10)
        self.ax.add_patch(agent_circle)
        self.ax.text(self.agent_pos[1], self.agent_pos[0], 'A',
                    ha='center', va='center', fontsize=16,
                    fontweight='bold', color='white')

        # Title
        self.ax.set_title(f'GridWorld - Steps: {self.steps} | Reward: {self.total_reward:.1f}',
                         fontsize=14)

        # Remove tick labels
        self.ax.set_xticks(range(self.size))
        self.ax.set_yticks(range(self.size))

        plt.tight_layout()

        if mode == 'human':
            plt.pause(0.01)

        return self.fig

    def close(self):
        """Close rendering window."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


class GridWorldWithObstacles(GridWorld):
    """
    GridWorld with predefined obstacle patterns.

    Good for testing exploration strategies.
    """

    def __init__(self, pattern: str = 'maze', **kwargs):
        """
        Initialize GridWorld with obstacle pattern.

        Args:
            pattern: 'maze', 'rooms', 'corridor', or 'empty'
        """
        size = kwargs.get('size', 7)

        # Define wall patterns
        if pattern == 'maze':
            walls = [
                (1, 1), (1, 2), (1, 3),
                (3, 1), (3, 3), (3, 4), (3, 5),
                (5, 2), (5, 3), (5, 4)
            ]
        elif pattern == 'rooms':
            walls = [
                (2, 0), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6),
                (4, 0), (4, 1), (4, 2), (4, 3), (4, 5), (4, 6)
            ]
        elif pattern == 'corridor':
            walls = [
                (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
                (4, 1), (4, 2), (4, 3), (4, 4), (4, 5)
            ]
        else:  # empty
            walls = []

        kwargs['walls'] = walls
        kwargs['size'] = size
        kwargs['start_pos'] = (0, 0)
        kwargs['goal_pos'] = (size - 1, size - 1)

        super().__init__(**kwargs)


# Example usage
if __name__ == '__main__':
    print("GridWorld Environment Demo")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=5)

    print(f"State space: {env.n_states} states")
    print(f"Action space: {env.n_actions} actions")
    print(f"Actions: {env.ACTION_NAMES}")
    print()

    # Random policy test
    print("Testing with random policy...")
    state = env.reset()
    env.render()

    for step in range(50):
        action = np.random.randint(env.n_actions)
        next_state, reward, done, info = env.step(action)

        print(f"Step {step}: Action={env.ACTION_NAMES[action]}, "
              f"State={next_state}, Reward={reward:.2f}, Done={done}")

        env.render()

        if done:
            print(f"\nReached goal in {info['steps']} steps!")
            print(f"Total reward: {info['total_reward']:.2f}")
            break

    plt.show()
    env.close()
