#!/usr/bin/env python3
"""
Tutorial 1: Q-Learning on GridWorld
====================================

This tutorial teaches you Q-Learning from scratch using a simple GridWorld.

What You'll Learn:
    1. What is Q-Learning?
    2. How does the Q-table work?
    3. ε-greedy exploration
    4. Bellman equation in practice
    5. Visualizing the learned policy

Time: 10-15 minutes

Prerequisites: Basic Python, NumPy

Author: RL Gym Tutorial Series
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from envs.gridworld import GridWorld
from agents.q_learning import QLearningAgent


def print_section(title):
    """Pretty print section headers."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def tutorial_part_1_theory():
    """Part 1: Understand Q-Learning Theory"""
    print_section("PART 1: Q-LEARNING THEORY")

    print("""
Q-Learning is a **model-free** reinforcement learning algorithm that learns
the value of actions in states.

KEY IDEA:
    Learn Q(s,a) = expected total reward from taking action a in state s

ALGORITHM:
    1. Initialize Q(s,a) = 0 for all states s and actions a
    2. For each episode:
        a. Start in initial state s
        b. While not done:
            - Choose action a using ε-greedy policy (explore vs exploit)
            - Take action a, observe reward r and next state s'
            - Update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            - Set s ← s'

BELLMAN EQUATION:
    Q*(s,a) = E[r + γ max_a' Q*(s',a')]

    The optimal Q-value equals:
        - Immediate reward r
        - Plus discounted best future value γ max_a' Q*(s',a')

HYPERPARAMETERS:
    α (alpha) = learning rate (0.1-0.5)
        - How much to update Q-values each step
        - Too high → unstable, too low → slow learning

    γ (gamma) = discount factor (0.9-0.99)
        - Importance of future rewards
        - 0 = only immediate reward, 1 = far future matters

    ε (epsilon) = exploration rate (0.1-0.3)
        - Probability of random action
        - Balance exploration (new actions) vs exploitation (best known action)

Press Enter to continue...
    """)
    input()


def tutorial_part_2_environment():
    """Part 2: Explore the Environment"""
    print_section("PART 2: GRIDWORLD ENVIRONMENT")

    print("""
Our environment is a 5x5 grid:
    - Agent starts at (0, 0) - top left
    - Goal is at (4, 4) - bottom right
    - Agent can move UP, DOWN, LEFT, RIGHT

REWARDS:
    +10.0  : Reach goal
    -1.0   : Hit wall/boundary
    -0.1   : Each step (encourages efficiency)

This is a Markov Decision Process (MDP):
    - States: 25 positions (5x5 grid)
    - Actions: 4 directions (UP, DOWN, LEFT, RIGHT)
    - Transitions: Deterministic (action always succeeds if valid)
    - Rewards: As defined above

Let's create the environment and take random actions...
    """)

    # Create environment
    env = GridWorld(size=5)

    print(f"State space: {env.n_states} states")
    print(f"Action space: {env.n_actions} actions")
    print(f"Actions: {env.ACTION_NAMES}")
    print()

    # Take random actions
    print("Taking 10 random actions...")
    state = env.reset()

    for step in range(10):
        action = np.random.randint(env.n_actions)
        next_state, reward, done, info = env.step(action)

        print(f"  Step {step+1}: Action={env.ACTION_NAMES[action]:5s} → "
              f"State={next_state:2d}, Reward={reward:5.1f}, "
              f"Position={info['position']}")

        if done:
            print(f"\n  Reached goal by chance! (very lucky)")
            break

        state = next_state

    print("\nPress Enter to continue...")
    input()


def tutorial_part_3_q_table():
    """Part 3: Understanding the Q-Table"""
    print_section("PART 3: THE Q-TABLE")

    print("""
The Q-table stores Q(s,a) for every state-action pair.

SHAPE: [n_states x n_actions]
    - Rows: states (25 in 5x5 grid)
    - Columns: actions (4 directions)

INTERPRETATION:
    Q[s, a] = expected total reward starting from state s,
              taking action a, then following optimal policy

EXAMPLE Q-table (random initialization):
    """)

    # Create dummy Q-table
    n_states = 25
    n_actions = 4
    Q = np.random.randn(n_states, n_actions) * 0.1

    print("State |  UP    DOWN   LEFT   RIGHT")
    print("----- | " + "-" * 30)
    for s in range(5):  # Show first 5 states
        print(f"  {s:2d}  | ", end="")
        for a in range(n_actions):
            print(f"{Q[s, a]:6.2f} ", end="")
        print()

    print("\nINTUITION:")
    print("  - Initially: Q-values are random (agent knows nothing)")
    print("  - After training: Q-values reflect true expected returns")
    print("  - Policy: π(s) = argmax_a Q(s, a)  (choose best action)")

    print("\nPress Enter to continue...")
    input()


def tutorial_part_4_training():
    """Part 4: Train Q-Learning Agent"""
    print_section("PART 4: TRAINING Q-LEARNING AGENT")

    print("""
Now let's train an agent using Q-Learning!

We'll train for 500 episodes and watch it learn.
    """)

    # Create environment and agent
    env = GridWorld(size=5)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,      # α
        discount_factor=0.99,   # γ
        epsilon=0.3,            # ε (start with high exploration)
        epsilon_decay=0.995,    # Decay ε over time
        epsilon_min=0.01        # Minimum ε
    )

    print(f"Hyperparameters:")
    print(f"  Learning rate (α): {agent.alpha}")
    print(f"  Discount factor (γ): {agent.gamma}")
    print(f"  Initial epsilon (ε): {agent.epsilon}")
    print()

    print("Training...")

    # Training loop
    n_episodes = 500

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Select action (ε-greedy)
            action = agent.select_action(state, training=True)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Learn (Q-Learning update)
            agent.learn(state, action, reward, next_state, done)

            # Update statistics
            state = next_state
            episode_reward += reward
            episode_length += 1

            if done:
                break

        # Save statistics
        agent.save_statistics(episode_reward, episode_length)
        agent.decay_epsilon()

        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(agent.episode_rewards[-100:])
            avg_length = np.mean(agent.episode_lengths[-100:])
            print(f"  Episode {episode+1:3d}: "
                  f"Reward={avg_reward:6.2f}, "
                  f"Length={avg_length:5.1f}, "
                  f"ε={agent.epsilon:.3f}")

    print("\n✓ Training complete!")

    # Show final Q-table statistics
    print("\nFinal Q-Table Statistics:")
    print(f"  Min Q-value: {np.min(agent.Q):.2f}")
    print(f"  Max Q-value: {np.max(agent.Q):.2f}")
    print(f"  Mean Q-value: {np.mean(agent.Q):.2f}")

    print("\nPress Enter to see learning curves...")
    input()

    # Plot learning curves
    agent.plot_training_progress(window=50)
    plt.savefig('q_learning_training.png', dpi=150, bbox_inches='tight')
    print("\n✓ Learning curves saved to: q_learning_training.png")

    return env, agent


def tutorial_part_5_evaluation():
    """Part 5: Evaluate Learned Policy"""
    print_section("PART 5: EVALUATE LEARNED POLICY")

    print("""
Let's test the learned policy!

We'll run 10 episodes with ε=0 (no exploration, pure exploitation).
    """)

    # Get trained agent from previous part
    print("Re-training agent for evaluation...")
    env = GridWorld(size=5)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.3
    )

    # Quick training
    for episode in range(500):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

    print("\nEvaluating learned policy (10 episodes)...")

    test_rewards = []
    test_lengths = []

    for test_ep in range(10):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        path = [env._state_to_pos(state)]

        while episode_length < 50:  # Max 50 steps
            # Greedy action (no exploration)
            action = agent.select_action(state, training=False)
            next_state, reward, done, _ = env.step(action)

            path.append(env._state_to_pos(next_state))
            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                break

        test_rewards.append(episode_reward)
        test_lengths.append(episode_length)

        print(f"  Test {test_ep+1:2d}: Reward={episode_reward:6.2f}, "
              f"Steps={episode_length:2d}, Success={'✓' if done else '✗'}")

    print(f"\nAverage Performance:")
    print(f"  Mean reward: {np.mean(test_rewards):.2f} ± {np.std(test_rewards):.2f}")
    print(f"  Mean steps: {np.mean(test_lengths):.1f} ± {np.std(test_lengths):.1f}")
    print(f"  Success rate: {sum(r > 0 for r in test_rewards) / len(test_rewards) * 100:.0f}%")

    print("\nPress Enter to visualize policy...")
    input()

    # Visualize policy
    env.reset()
    env.render(q_values=agent.Q)
    plt.savefig('q_learning_policy.png', dpi=150, bbox_inches='tight')
    print("\n✓ Policy visualization saved to: q_learning_policy.png")

    print("""
INTERPRETATION:
    - Blue arrows show the best action for each state
    - Numbers show Q-value of best action
    - Agent learned to navigate to goal!
    """)

    return agent


def tutorial_part_6_analysis():
    """Part 6: Analyze What Agent Learned"""
    print_section("PART 6: DEEP DIVE - WHAT DID THE AGENT LEARN?")

    print("""
Let's examine specific states and their Q-values to understand
what the agent learned.
    """)

    # Re-create trained agent
    env = GridWorld(size=5)
    agent = QLearningAgent(env.n_states, env.n_actions,
                          learning_rate=0.1, discount_factor=0.99, epsilon=0.3)

    # Quick training
    for _ in range(500):
        state = env.reset()
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

    # Analyze specific states
    print("Q-Values for Key States:")
    print()

    # Start state (0, 0)
    start_state = env._pos_to_state([0, 0])
    print(f"START STATE (0,0) - State {start_state}:")
    for a in range(env.n_actions):
        print(f"  {env.ACTION_NAMES[a]:5s}: Q = {agent.Q[start_state, a]:7.2f}")
    best_action = np.argmax(agent.Q[start_state])
    print(f"  → Best action: {env.ACTION_NAMES[best_action]}")
    print()

    # Middle state (2, 2)
    middle_state = env._pos_to_state([2, 2])
    print(f"MIDDLE STATE (2,2) - State {middle_state}:")
    for a in range(env.n_actions):
        print(f"  {env.ACTION_NAMES[a]:5s}: Q = {agent.Q[middle_state, a]:7.2f}")
    best_action = np.argmax(agent.Q[middle_state])
    print(f"  → Best action: {env.ACTION_NAMES[best_action]}")
    print()

    # Near goal (4, 3)
    near_goal = env._pos_to_state([4, 3])
    print(f"NEAR GOAL (4,3) - State {near_goal}:")
    for a in range(env.n_actions):
        print(f"  {env.ACTION_NAMES[a]:5s}: Q = {agent.Q[near_goal, a]:7.2f}")
    best_action = np.argmax(agent.Q[near_goal])
    print(f"  → Best action: {env.ACTION_NAMES[best_action]} (should be RIGHT)")
    print()

    print("OBSERVATIONS:")
    print("  1. Q-values increase as agent gets closer to goal")
    print("  2. Best actions point toward the goal")
    print("  3. Agent learned value of different paths")

    # Value function heatmap
    print("\nPress Enter to see value function heatmap...")
    input()

    V = agent.get_value_function()
    V_grid = V.reshape(5, 5)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(V_grid, cmap='viridis', interpolation='nearest')

    # Add value text
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{V_grid[i, j]:.1f}',
                          ha="center", va="center", color="white",
                          fontsize=12, fontweight='bold')

    ax.set_title('Value Function V(s) = max_a Q(s,a)', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    plt.colorbar(im, ax=ax, label='State Value')

    plt.tight_layout()
    plt.savefig('q_learning_values.png', dpi=150, bbox_inches='tight')
    print("\n✓ Value function saved to: q_learning_values.png")


def tutorial_part_7_exercises():
    """Part 7: Exercises for You"""
    print_section("PART 7: EXERCISES")

    print("""
Now it's your turn! Try these exercises to deepen your understanding:

EXERCISE 1: Hyperparameter Tuning
    - Try different learning rates: α = 0.01, 0.1, 0.5
    - Try different discount factors: γ = 0.9, 0.95, 0.99
    - Try different exploration: ε = 0.05, 0.1, 0.3
    - Which works best? Why?

EXERCISE 2: Larger Grid
    - Change GridWorld size to 10x10
    - How does training time change?
    - Does the agent still learn?

EXERCISE 3: Add Obstacles
    - Use GridWorldWithObstacles(pattern='maze')
    - Does Q-Learning still work?
    - How many episodes needed?

EXERCISE 4: Compare Algorithms
    - Implement SARSA (it's in agents/q_learning.py!)
    - Compare Q-Learning vs SARSA
    - Which learns faster?

EXERCISE 5: Visualize Learning
    - Create animation of agent's policy improving over time
    - Show Q-values changing episode by episode

BONUS: Implement Double Q-Learning
    - Reduces overestimation bias
    - Use two Q-tables
    - See if it improves performance

Press Enter to finish tutorial...
    """)
    input()


def main():
    """Run complete tutorial."""
    print()
    print("=" * 70)
    print("  TUTORIAL 1: Q-LEARNING ON GRIDWORLD")
    print("=" * 70)
    print()
    print("This interactive tutorial teaches Q-Learning step by step.")
    print("Follow along and press Enter to continue through sections.")
    print()
    input("Press Enter to begin...")

    # Run tutorial parts
    tutorial_part_1_theory()
    tutorial_part_2_environment()
    tutorial_part_3_q_table()
    env, agent = tutorial_part_4_training()
    agent = tutorial_part_5_evaluation()
    tutorial_part_6_analysis()
    tutorial_part_7_exercises()

    # Show all plots
    plt.show()

    # Summary
    print_section("TUTORIAL COMPLETE! 🎉")
    print("""
Congratulations! You've learned:
    ✓ Q-Learning algorithm
    ✓ Q-table representation
    ✓ ε-greedy exploration
    ✓ Bellman equation in practice
    ✓ Training and evaluation
    ✓ Policy visualization

Next Steps:
    1. Complete the exercises above
    2. Try Tutorial 2: SARSA
    3. Move on to Deep Q-Networks (DQN)

Generated Files:
    - q_learning_training.png (learning curves)
    - q_learning_policy.png (learned policy)
    - q_learning_values.png (value function)

Happy learning! 🚀
    """)


if __name__ == '__main__':
    main()
