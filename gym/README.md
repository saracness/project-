# 🎓 Reinforcement Learning Gym
## A Textbook Approach to RL: From Theory to Practice

Welcome to the RL Gym! This is a **hands-on textbook** that teaches you Reinforcement Learning from scratch, with both theory and working code.

---

## 📚 Table of Contents

### Part I: Foundations
1. [What is Reinforcement Learning?](#what-is-rl)
2. [Key Concepts: MDP, Rewards, Policies](#key-concepts)
3. [The RL Framework](#rl-framework)

### Part II: Tabular Methods
4. [Chapter 1: Q-Learning](tutorials/01_q_learning_gridworld.md)
5. [Chapter 2: SARSA and Expected SARSA](tutorials/02_sarsa.md)
6. [Chapter 3: Monte Carlo Methods](tutorials/03_monte_carlo.md)

### Part III: Deep Reinforcement Learning
7. [Chapter 4: Deep Q-Networks (DQN)](tutorials/04_deep_q_network.md)
8. [Chapter 5: Policy Gradients (REINFORCE)](tutorials/05_policy_gradient.md)
9. [Chapter 6: Actor-Critic Methods (A2C/A3C)](tutorials/06_actor_critic.md)
10. [Chapter 7: Proximal Policy Optimization (PPO)](tutorials/07_ppo.md)

### Part IV: Custom Environments
11. [Chapter 8: Building Custom Envs](tutorials/08_custom_environments.md)
12. [Chapter 9: MICROLIFE RL Environment](tutorials/09_microlife_rl.md)
13. [Chapter 10: Multi-Agent RL](tutorials/10_multi_agent.md)

### Part V: Advanced Topics
14. [Chapter 11: Model-Based RL](tutorials/11_model_based.md)
15. [Chapter 12: Curiosity & Exploration](tutorials/12_exploration.md)
16. [Chapter 13: Hierarchical RL](tutorials/13_hierarchical_rl.md)

---

## 🎯 What is RL?

**Reinforcement Learning** is learning what to do—how to map situations to actions—to maximize a numerical reward signal.

### The Core Problem

An **agent** interacts with an **environment**:
1. Agent observes **state** s
2. Agent takes **action** a
3. Environment returns **reward** r and new **state** s'
4. Repeat

The agent's goal: Find a **policy** π that maximizes **cumulative reward**.

### Example: Teaching a Robot to Walk

```
State (s):     Joint angles, velocities
Action (a):    Torques to apply to joints
Reward (r):    +1 for forward motion, -1 for falling
Policy (π):    Function that maps states → actions
Goal:          Learn π that makes robot walk forward
```

---

## 🧠 Key Concepts

### 1. Markov Decision Process (MDP)

An MDP is a tuple (S, A, P, R, γ):
- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probabilities P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor (0 ≤ γ ≤ 1)

### 2. Policy

A policy π maps states to actions:
- **Deterministic**: π(s) = a
- **Stochastic**: π(a|s) = probability of action a in state s

### 3. Value Functions

**State Value Function V^π(s)**:
```
V^π(s) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | S_t = s, π]
```
Expected return starting from state s, following policy π.

**Action Value Function Q^π(s,a)**:
```
Q^π(s,a) = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | S_t = s, A_t = a, π]
```
Expected return from taking action a in state s, then following π.

### 4. Bellman Equations

**Bellman Expectation Equation**:
```
V^π(s) = Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γV^π(s')]
```

**Bellman Optimality Equation**:
```
V*(s) = max_a Σ_{s',r} P(s',r|s,a) [r + γV*(s')]
Q*(s,a) = Σ_{s',r} P(s',r|s,a) [r + γ max_{a'} Q*(s',a')]
```

### 5. Exploration vs. Exploitation

**Exploration**: Try new actions to discover their rewards
**Exploitation**: Choose actions known to yield high rewards

**ε-greedy**: With probability ε, explore (random action); otherwise exploit (best action)

---

## 🔧 The RL Framework

### Standard Environment Interface

```python
class Environment:
    def reset(self):
        """Reset environment, return initial state"""
        return state

    def step(self, action):
        """Take action, return (next_state, reward, done, info)"""
        return next_state, reward, done, info

    def render(self):
        """Visualize current state"""
        pass
```

### Standard Agent Interface

```python
class Agent:
    def select_action(self, state):
        """Choose action based on current state"""
        return action

    def learn(self, state, action, reward, next_state, done):
        """Update policy based on experience"""
        pass
```

---

## 🚀 Quick Start

### Example 1: Q-Learning on GridWorld (5 minutes)

```bash
cd gym/tutorials
python 01_q_learning_gridworld.py
```

**What you'll learn:**
- States, actions, rewards
- Q-table updates
- ε-greedy exploration
- Convergence to optimal policy

### Example 2: DQN on CartPole (15 minutes)

```bash
python 04_deep_q_network.py
```

**What you'll learn:**
- Neural network function approximation
- Experience replay
- Target networks
- Training stability

### Example 3: RL on MICROLIFE (30 minutes)

```bash
python 09_microlife_rl.py
```

**What you'll learn:**
- Custom environments
- Continuous action spaces
- Multi-objective rewards
- Biological realism

---

## 📊 Learning Path

### Beginner Track (Start Here!)
1. ✅ **GridWorld Q-Learning** → Learn basics
2. ✅ **SARSA** → Compare with Q-Learning
3. ✅ **CartPole DQN** → Neural networks

### Intermediate Track
4. ✅ **Policy Gradients** → Different approach
5. ✅ **Actor-Critic** → Best of both worlds
6. ✅ **PPO** → State-of-the-art

### Advanced Track
7. ✅ **Custom MICROLIFE Env** → Real-world problems
8. ✅ **Multi-Agent RL** → Competition/cooperation
9. ✅ **Model-Based RL** → Planning

---

## 🏆 Algorithms Implemented

### Tabular Methods (Discrete State/Action)
| Algorithm | Type | Complexity | Use Case |
|-----------|------|------------|----------|
| Q-Learning | Off-policy TD | Simple | Small discrete problems |
| SARSA | On-policy TD | Simple | Safe exploration |
| Monte Carlo | Episode-based | Medium | Episodic tasks |

### Deep RL (Continuous State)
| Algorithm | Type | Complexity | Use Case |
|-----------|------|------------|----------|
| DQN | Value-based | Medium | Discrete actions |
| REINFORCE | Policy gradient | Medium | Stochastic policies |
| A2C/A3C | Actor-critic | High | General purpose |
| PPO | Actor-critic | High | Stable training |
| DDPG | Actor-critic | High | Continuous control |
| SAC | Actor-critic | High | Robust performance |

---

## 📁 Directory Structure

```
gym/
├── README.md                    # This file
├── envs/                        # Environment implementations
│   ├── gridworld.py            # Simple GridWorld
│   ├── cartpole.py             # Cartpole (OpenAI Gym compatible)
│   ├── microlife_env.py        # MICROLIFE RL environment
│   └── base_env.py             # Base environment class
├── agents/                      # RL agent implementations
│   ├── q_learning.py           # Q-Learning agent
│   ├── dqn.py                  # Deep Q-Network
│   ├── ppo.py                  # Proximal Policy Optimization
│   └── base_agent.py           # Base agent class
├── tutorials/                   # Step-by-step tutorials
│   ├── 01_q_learning_gridworld.py
│   ├── 04_deep_q_network.py
│   └── 09_microlife_rl.py
├── examples/                    # Complete working examples
│   ├── train_cartpole.py
│   ├── train_microlife.py
│   └── compare_algorithms.py
├── visualizations/              # Visualization tools
│   ├── plot_training.py
│   ├── render_policy.py
│   └── animate_episode.py
└── cpp/                         # C++ implementations (performance)
    ├── q_learning.cpp
    ├── dqn.cpp
    └── microlife_rl.cpp
```

---

## 🎓 Textbook Style Learning

Each tutorial follows this structure:

### 1. Theory Section
- Mathematical formulation
- Intuitive explanation
- Pseudocode

### 2. Implementation
- Python code (readable)
- C++ code (fast)
- Line-by-line explanation

### 3. Experiments
- Train agent
- Visualize learning curves
- Compare hyperparameters

### 4. Exercises
- Modify code
- Test understanding
- Explore variations

---

## 🔬 Example: Q-Learning

### Theory

**Q-Learning Algorithm**:
```
Initialize Q(s,a) arbitrarily
For each episode:
    Initialize state s
    For each step:
        Choose action a using ε-greedy policy from Q
        Take action a, observe r, s'
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        s ← s'
```

### Python Implementation

```python
class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def select_action(self, state):
        # ε-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        # Q-Learning update
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
```

### Training Loop

```python
agent = QLearningAgent(n_states=100, n_actions=4)
env = GridWorld()

for episode in range(1000):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {episode}: Reward = {total_reward}")
```

---

## 🎯 Learning Objectives

By completing this gym, you will:

### Understand Theory
- ✅ Markov Decision Processes
- ✅ Bellman equations
- ✅ Policy vs. value iteration
- ✅ Temporal difference learning
- ✅ Function approximation

### Implement Algorithms
- ✅ Q-Learning
- ✅ Deep Q-Networks (DQN)
- ✅ Policy gradients
- ✅ Actor-critic methods
- ✅ PPO

### Apply to Real Problems
- ✅ Custom environments
- ✅ Biological systems (MICROLIFE)
- ✅ Multi-agent scenarios
- ✅ Continuous control

### Master Best Practices
- ✅ Hyperparameter tuning
- ✅ Debugging RL
- ✅ Visualization
- ✅ Reproducibility

---

## 📚 Recommended Reading

### Textbooks
1. **Sutton & Barto** - "Reinforcement Learning: An Introduction" (2nd ed.)
   - THE classic RL textbook
   - Free online: http://incompleteideas.net/book/the-book-2nd.html

2. **Bertsekas** - "Reinforcement Learning and Optimal Control"
   - More mathematical approach

### Papers
1. **DQN**: Mnih et al. (2015) - "Human-level control through deep RL"
2. **PPO**: Schulman et al. (2017) - "Proximal Policy Optimization"
3. **AlphaGo**: Silver et al. (2016) - "Mastering the game of Go with deep RL"

### Online Courses
1. **DeepMind x UCL**: RL Lecture Series (YouTube)
2. **Berkeley CS285**: Deep Reinforcement Learning
3. **Stanford CS234**: Reinforcement Learning

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
# Python
pip install numpy matplotlib torch gym

# C++ (for high performance)
# SFML already installed for visualizations
```

### 2. Run Your First Tutorial

```bash
cd gym/tutorials
python 01_q_learning_gridworld.py
```

### 3. Follow the Textbook

Start with Chapter 1 and work your way through!

---

## 🎮 Environments Available

### Classic Control
- **GridWorld** - Learn navigation
- **CartPole** - Balance a pole
- **MountainCar** - Climb a hill
- **Pendulum** - Swing up

### Custom Biological
- **MICROLIFE Survival** - Organism learns to survive
- **MICROLIFE Predator** - Learn to hunt
- **MICROLIFE Ecosystem** - Multi-agent dynamics

### Research
- **Multi-Agent** - Competition/cooperation
- **Sparse Rewards** - Hard exploration
- **Continuous Control** - Real-world robotics

---

## 📊 Performance Comparison

| Environment | Q-Learning | DQN | PPO | Training Time |
|-------------|------------|-----|-----|---------------|
| GridWorld 10x10 | ✅ 1 min | N/A | N/A | - |
| CartPole | N/A | ✅ 5 min | ✅ 3 min | Python |
| MICROLIFE | N/A | ✅ 30 min | ✅ 15 min | Python |
| MICROLIFE | N/A | ✅ 5 min | ✅ 3 min | C++ |

*C++ implementations are 6-10x faster than Python*

---

## 🐛 Debugging RL

Common issues and solutions:

### 1. Agent Not Learning
- Check reward signal (is it too sparse?)
- Verify Q-updates are happening
- Plot Q-values over time
- Try lower learning rate

### 2. Unstable Training
- Reduce learning rate
- Increase batch size (DQN)
- Use target network (DQN)
- Normalize observations

### 3. Slow Convergence
- Tune exploration (ε)
- Adjust discount factor (γ)
- Use reward shaping
- Increase network capacity (DQN)

---

## 🎯 Next Steps

1. **Complete Tutorial 1** - Get your hands dirty with Q-Learning
2. **Implement from Scratch** - Don't just copy-paste!
3. **Experiment** - Change hyperparameters, see what happens
4. **Build Custom Environment** - Apply to your own problem
5. **Read Papers** - Understand state-of-the-art
6. **Contribute** - Share your implementations!

---

## 🤝 Contributing

This is a living textbook! Contributions welcome:
- New algorithms
- Better explanations
- Bug fixes
- More examples

---

## 📄 License

Educational use - feel free to learn, modify, and share!

---

**Ready to learn RL? Start with [Tutorial 1: Q-Learning](tutorials/01_q_learning_gridworld.md)!** 🚀

---

*"The reward is the signal." - Richard Sutton*
