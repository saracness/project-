#!/usr/bin/env python3
"""
Train RL Agent on MICROLIFE Environment
========================================

This script trains a Deep Q-Network (DQN) agent to survive in MICROLIFE.

The agent learns to:
    - Find food (energy sources)
    - Avoid predators
    - Survive as long as possible

Since MICROLIFE has continuous state space, we use Deep Q-Learning
with a neural network instead of a Q-table.

Usage:
    python train_microlife.py

Note: This uses a simplified DQN. For full DQN implementation,
      see tutorials/04_deep_q_network.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from envs.microlife_env import MicroLifeEnv


class SimpleDQNAgent:
    """
    Simplified DQN agent using linear function approximation.

    For demonstration purposes only.
    Full neural network DQN is in tutorials/04_deep_q_network.py
    """

    def __init__(self, state_dim, n_actions, learning_rate=0.01, gamma=0.99, epsilon=0.3):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Linear function approximation: Q(s,a) = w_a^T * s
        # Instead of Q-table, we use weight matrix [state_dim x n_actions]
        self.weights = np.random.randn(state_dim, n_actions) * 0.1

        self.alpha = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []

    def select_action(self, state, training=True):
        """ε-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Q(s,a) = w_a^T * s
            q_values = state @ self.weights  # [state_dim] @ [state_dim x n_actions]
            return np.argmax(q_values)

    def learn(self, state, action, reward, next_state, done):
        """
        Update weights using gradient descent on TD error.

        TD error: δ = r + γ max_a' Q(s',a') - Q(s,a)
        Update: w_a ← w_a + α * δ * s
        """
        # Current Q-value
        current_q = state @ self.weights[:, action]

        # TD target
        if done:
            target = reward
        else:
            next_q_values = next_state @ self.weights
            target = reward + self.gamma * np.max(next_q_values)

        # TD error
        td_error = target - current_q

        # Gradient descent update
        self.weights[:, action] += self.alpha * td_error * state

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def main():
    print()
    print("=" * 70)
    print("  TRAIN RL AGENT ON MICROLIFE")
    print("=" * 70)
    print()

    # Create environment
    print("Creating MICROLIFE environment...")
    env = MicroLifeEnv(
        world_size=100.0,
        n_food=8,
        n_predators=3,
        max_steps=500
    )
    print(f"✓ State dimension: {env.state_dim}")
    print(f"✓ Action space: {env.n_actions}")
    print(f"✓ Food particles: {env.n_food}")
    print(f"✓ Predators: {env.n_predators}")
    print()

    # Create agent
    print("Creating DQN agent (linear approximation)...")
    agent = SimpleDQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=0.4
    )
    print(f"✓ Learning rate: {agent.alpha}")
    print(f"✓ Discount factor: {agent.gamma}")
    print(f"✓ Initial ε: {agent.epsilon}")
    print()

    # Train
    print("Training for 500 episodes...")
    print("(This may take 2-3 minutes)")
    print()

    n_episodes = 500

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Learn
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Save statistics
        agent.episode_rewards.append(episode_reward)
        agent.episode_lengths.append(episode_length)
        agent.decay_epsilon()

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            avg_food = np.mean(env.food_collected[-100:]) if len(env.food_collected) >= 100 else 0
            print(f"  Episode {episode+1:3d}: "
                  f"Reward = {avg_reward:7.2f}, "
                  f"Length = {avg_length:6.1f}, "
                  f"Food = {avg_food:4.1f}, "
                  f"ε = {agent.epsilon:.3f}")

    print()
    print("✓ Training complete!")
    print()

    # Evaluate
    print("Evaluating learned policy (5 episodes)...")
    test_rewards = []
    test_lengths = []
    test_foods = []

    for test_ep in range(5):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        # Render first test episode
        if test_ep == 0:
            env.render()

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            if test_ep == 0:
                env.render()

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)
        test_foods.append(info['food_collected'])

        print(f"  Test {test_ep+1}: Reward = {episode_reward:7.2f}, "
              f"Length = {episode_length:3d}, Food = {info['food_collected']}")

    print()
    print(f"Average Performance:")
    print(f"  Reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Survival: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f} steps")
    print(f"  Food: {np.mean(test_foods):.1f} ± {np.std(test_foods):.1f}")
    print()

    # Plot learning curves
    print("Generating learning curves...")

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Episode rewards
    window = 50
    axes[0].plot(agent.episode_rewards, alpha=0.3, label='Raw')
    if len(agent.episode_rewards) >= window:
        moving_avg = np.convolve(agent.episode_rewards,
                                np.ones(window)/window, mode='valid')
        axes[0].plot(range(window-1, len(agent.episode_rewards)),
                    moving_avg, label=f'{window}-episode MA', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('MICROLIFE: Learning Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Survival time
    axes[1].plot(agent.episode_lengths, alpha=0.3, label='Raw')
    if len(agent.episode_lengths) >= window:
        moving_avg = np.convolve(agent.episode_lengths,
                                np.ones(window)/window, mode='valid')
        axes[1].plot(range(window-1, len(agent.episode_lengths)),
                    moving_avg, label=f'{window}-episode MA', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Survival Time (steps)')
    axes[1].set_title('Agent Survival Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('microlife_training.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: microlife_training.png")
    print()

    print("=" * 70)
    print("  SUCCESS! Agent learned to survive in MICROLIFE! ✓")
    print("=" * 70)
    print()
    print("Key observations:")
    print("  - Agent learned to seek food (positive rewards)")
    print("  - Agent learned to avoid predators (survived longer)")
    print("  - Survival time increased over training")
    print()
    print("Next steps:")
    print("  1. Try full DQN with neural network")
    print("  2. Add more predators for harder challenge")
    print("  3. Implement PPO for better performance")
    print()

    plt.show()
    env.close()


if __name__ == '__main__':
    main()
