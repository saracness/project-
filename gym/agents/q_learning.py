"""
Q-Learning Agent
================

Implementation of the Q-Learning algorithm (Watkins, 1989).

Algorithm:
    Initialize Q(s,a) arbitrarily
    For each episode:
        Initialize s
        For each step:
            Choose a from s using ε-greedy policy
            Take action a, observe r, s'
            Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            s ← s'

Key Features:
    - Off-policy: Learns optimal policy while following ε-greedy
    - Model-free: Doesn't require environment model
    - Tabular: Uses Q-table (requires discrete states/actions)

Hyperparameters:
    - α (alpha): Learning rate (0.1-0.5)
    - γ (gamma): Discount factor (0.9-0.99)
    - ε (epsilon): Exploration rate (0.1-0.3)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent with ε-greedy exploration.

    Suitable for discrete state and action spaces.
    """

    def __init__(self,
                 n_states: int,
                 n_actions: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 0.1,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Initialize Q-Learning agent.

        Args:
            n_states: Number of states
            n_actions: Number of actions
            learning_rate: α - step size for Q-updates
            discount_factor: γ - importance of future rewards
            epsilon: ε - probability of random action
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions

        # Q-table: Q(s,a) = expected return from (s,a)
        self.Q = np.zeros((n_states, n_actions))

        # Hyperparameters
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        With probability ε: random action (exploration)
        With probability 1-ε: best action (exploitation)

        Args:
            state: Current state
            training: If False, always exploit (for evaluation)

        Returns:
            action: Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.Q[state])

    def learn(self, state: int, action: int, reward: float,
              next_state: int, done: bool):
        """
        Update Q-value using Q-Learning rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        This is the Bellman optimality equation in update form.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Current Q-value
        current_q = self.Q[state, action]

        # TD target: r + γ max_a' Q(s',a')
        if done:
            # Terminal state has value 0
            target = reward
        else:
            # Bellman optimality: use max over actions
            target = reward + self.gamma * np.max(self.Q[next_state])

        # TD error: δ = target - current
        td_error = target - current_q

        # Q-update: Q(s,a) ← Q(s,a) + α * δ
        self.Q[state, action] += self.alpha * td_error

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_greedy_policy(self) -> np.ndarray:
        """
        Extract greedy policy from Q-table.

        π(s) = argmax_a Q(s,a)

        Returns:
            policy: Array of shape [n_states] with best action for each state
        """
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """
        Extract value function from Q-table.

        V(s) = max_a Q(s,a)

        Returns:
            values: Array of shape [n_states] with state values
        """
        return np.max(self.Q, axis=1)

    def save_statistics(self, episode_reward: float, episode_length: int):
        """Save episode statistics for plotting."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Track average Q-value
        avg_q = np.mean(np.max(self.Q, axis=1))
        self.q_value_history.append(avg_q)

    def plot_training_progress(self, window: int = 100):
        """
        Plot training progress: rewards, lengths, Q-values.

        Args:
            window: Moving average window size
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))

        # Plot 1: Episode rewards
        axes[0].plot(self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= window:
            moving_avg = np.convolve(self.episode_rewards,
                                    np.ones(window)/window, mode='valid')
            axes[0].plot(range(window-1, len(self.episode_rewards)),
                        moving_avg, label=f'{window}-episode MA', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Learning Curve: Episode Rewards')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        axes[1].plot(self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= window:
            moving_avg = np.convolve(self.episode_lengths,
                                    np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(self.episode_lengths)),
                        moving_avg, label=f'{window}-episode MA', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps to Goal')
        axes[1].set_title('Efficiency: Steps per Episode')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Average Q-values
        axes[2].plot(self.q_value_history)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Average Max Q-value')
        axes[2].set_title('Value Function Growth')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def save(self, filename: str):
        """Save Q-table to file."""
        np.save(filename, self.Q)
        print(f"✓ Q-table saved to {filename}")

    def load(self, filename: str):
        """Load Q-table from file."""
        self.Q = np.load(filename)
        print(f"✓ Q-table loaded from {filename}")


class SARSAAgent(QLearningAgent):
    """
    SARSA agent (on-policy TD control).

    Difference from Q-Learning:
        - Q-Learning (off-policy): Q(s,a) ← ... + γ max_a' Q(s',a') ...
        - SARSA (on-policy): Q(s,a) ← ... + γ Q(s',a') ...

    SARSA uses the action actually taken (a'), not the max.
    This makes it safer but potentially slower to converge.
    """

    def learn(self, state: int, action: int, reward: float,
              next_state: int, next_action: int, done: bool):
        """
        Update Q-value using SARSA rule.

        Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]

        Args:
            next_action: Action taken in next state (key difference!)
        """
        current_q = self.Q[state, action]

        if done:
            target = reward
        else:
            # SARSA: use Q(s', a') where a' is the action taken
            target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error


class ExpectedSARSAAgent(QLearningAgent):
    """
    Expected SARSA agent (combines Q-Learning and SARSA).

    Update rule:
        Q(s,a) ← Q(s,a) + α[r + γ E_π[Q(s',a')] - Q(s,a)]

    Where E_π[Q(s',a')] = Σ_a' π(a'|s') Q(s',a')

    Often performs better than both Q-Learning and SARSA.
    """

    def learn(self, state: int, action: int, reward: float,
              next_state: int, done: bool):
        """
        Update Q-value using Expected SARSA rule.

        Takes expectation over actions according to policy.
        """
        current_q = self.Q[state, action]

        if done:
            target = reward
        else:
            # Compute expected value under ε-greedy policy
            # E[Q] = ε * (average Q) + (1-ε) * (max Q)
            avg_q = np.mean(self.Q[next_state])
            max_q = np.max(self.Q[next_state])
            expected_q = self.epsilon * avg_q + (1 - self.epsilon) * max_q

            target = reward + self.gamma * expected_q

        td_error = target - current_q
        self.Q[state, action] += self.alpha * td_error


# Comparison function
def compare_algorithms(env, episodes: int = 1000, runs: int = 10):
    """
    Compare Q-Learning, SARSA, and Expected SARSA.

    Args:
        env: Environment to train on
        episodes: Episodes per run
        runs: Number of runs (for averaging)

    Returns:
        dict with results for each algorithm
    """
    results = {
        'Q-Learning': {'rewards': [], 'lengths': []},
        'SARSA': {'rewards': [], 'lengths': []},
        'Expected SARSA': {'rewards': [], 'lengths': []}
    }

    print("Comparing algorithms...")
    print(f"Episodes: {episodes}, Runs: {runs}")
    print()

    for run in range(runs):
        print(f"Run {run + 1}/{runs}")

        # Create agents
        agents = {
            'Q-Learning': QLearningAgent(env.n_states, env.n_actions),
            'SARSA': SARSAAgent(env.n_states, env.n_actions),
            'Expected SARSA': ExpectedSARSAAgent(env.n_states, env.n_actions)
        }

        for name, agent in agents.items():
            rewards = []
            lengths = []

            for episode in range(episodes):
                state = env.reset()
                episode_reward = 0
                episode_length = 0

                while True:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)

                    if isinstance(agent, SARSAAgent) and not done:
                        next_action = agent.select_action(next_state)
                        agent.learn(state, action, reward, next_state, next_action, done)
                    else:
                        agent.learn(state, action, reward, next_state, done)

                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if done:
                        break

                rewards.append(episode_reward)
                lengths.append(episode_length)
                agent.decay_epsilon()

            results[name]['rewards'].append(rewards)
            results[name]['lengths'].append(lengths)

    # Average over runs
    for name in results:
        results[name]['avg_rewards'] = np.mean(results[name]['rewards'], axis=0)
        results[name]['avg_lengths'] = np.mean(results[name]['lengths'], axis=0)

    return results


# Example usage
if __name__ == '__main__':
    from gym.envs.gridworld import GridWorld

    print("Q-Learning Agent Demo")
    print("=" * 50)

    # Create environment
    env = GridWorld(size=5)

    # Create agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.3
    )

    print(f"Training for 500 episodes...")

    # Training
    for episode in range(500):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        agent.save_statistics(episode_reward, episode_length)
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Avg Length = {avg_length:.1f}, ε = {agent.epsilon:.3f}")

    # Plot results
    agent.plot_training_progress()

    # Visualize learned policy
    env.reset()
    env.render(q_values=agent.Q)
    plt.savefig('q_learning_policy.png', dpi=150, bbox_inches='tight')
    print("\n✓ Learned policy saved to q_learning_policy.png")

    plt.show()
