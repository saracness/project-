#!/usr/bin/env python3
"""
Quick Start: Train Q-Learning Agent on GridWorld
================================================

This script demonstrates the basics in under 2 minutes:
    1. Create GridWorld environment
    2. Create Q-Learning agent
    3. Train for 200 episodes
    4. Visualize learned policy

Perfect for testing your installation!

Usage:
    python quick_start.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from envs.gridworld import GridWorld
from agents.q_learning import QLearningAgent


def main():
    print()
    print("=" * 60)
    print("  QUICK START: Q-LEARNING ON GRIDWORLD")
    print("=" * 60)
    print()

    # Create environment
    print("Creating 5x5 GridWorld...")
    env = GridWorld(size=5)
    print(f"✓ State space: {env.n_states} states")
    print(f"✓ Action space: {env.n_actions} actions")
    print()

    # Create agent
    print("Creating Q-Learning agent...")
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.2
    )
    print(f"✓ Learning rate: {agent.alpha}")
    print(f"✓ Discount factor: {agent.gamma}")
    print(f"✓ Exploration rate: {agent.epsilon}")
    print()

    # Train
    print("Training for 200 episodes...")
    n_episodes = 200

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Learn
            agent.learn(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Save stats
        agent.save_statistics(episode_reward, episode_length)
        agent.decay_epsilon()

        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(agent.episode_rewards[-50:])
            avg_length = np.mean(agent.episode_lengths[-50:])
            print(f"  Episode {episode+1:3d}: "
                  f"Reward = {avg_reward:6.2f}, "
                  f"Length = {avg_length:5.1f}, "
                  f"ε = {agent.epsilon:.3f}")

    print()
    print("✓ Training complete!")
    print()

    # Evaluate
    print("Evaluating policy (5 test episodes)...")
    test_rewards = []
    for _ in range(5):
        state = env.reset()
        episode_reward = 0
        steps = 0

        while steps < 50:
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            steps += 1
            if done:
                break

        test_rewards.append(episode_reward)

    print(f"✓ Average reward: {np.mean(test_rewards):.2f}")
    print(f"✓ Success rate: {sum(r > 0 for r in test_rewards) / len(test_rewards) * 100:.0f}%")
    print()

    # Visualize
    print("Generating visualizations...")

    # Learning curves
    fig = agent.plot_training_progress(window=20)
    plt.savefig('quick_start_training.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: quick_start_training.png")

    # Policy
    env.reset()
    env.render(q_values=agent.Q)
    plt.savefig('quick_start_policy.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: quick_start_policy.png")
    print()

    print("=" * 60)
    print("  SUCCESS! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Check the generated PNG files")
    print("  2. Run full tutorial: python tutorials/01_q_learning_gridworld.py")
    print("  3. Try MICROLIFE environment: python examples/train_microlife.py")
    print()

    plt.show()


if __name__ == '__main__':
    main()
