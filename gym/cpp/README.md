# C++ Reinforcement Learning Implementations

High-performance C++ implementations of RL algorithms with real-time visualization.

## 🎬 RL Learning Animation

**Watch Q-Learning learn in real-time!**

This is a **live visualization** of the Q-Learning algorithm learning to solve GridWorld.

### What You'll See

<img src="https://via.placeholder.com/800x600/1e1e2e/ffffff?text=Q-Learning+Animation+Preview" alt="Animation Preview" width="600"/>

**Visual Elements:**
- 🔴 **Red Circle**: Agent exploring the grid
- 🎨 **Color Heatmap**: Q-values changing (blue = low, green = medium, red = high)
- 🎯 **Blue Arrows**: Policy forming (best action per state)
- 🏆 **Gold Square**: Goal position
- 📊 **Stats Panel**: Real-time metrics (rewards, epsilon, Q-values)
- 📈 **Learning Curve**: Episode rewards over time

**Learning Process:**
1. **Early Episodes (1-50)**: Random exploration, Q-values near zero, no clear policy
2. **Middle Episodes (50-200)**: Q-values growing, policy arrows forming near goal
3. **Late Episodes (200+)**: Optimal policy learned, agent goes straight to goal

---

## 🚀 Quick Start

### Compile

```bash
cd gym/cpp
make
```

**Output:**
```
Compiling RL Learning Animation...
g++ -std=c++17 -O3 -march=native -Wall -Wextra rl_learning_animation.cpp ...
✓ Build complete!
```

### Run

```bash
./rl_learning_animation
```

**Expected:**
- Window opens showing 5x5 GridWorld
- Agent (red circle) starts at top-left
- Goal (gold square) at bottom-right
- Stats panel on right showing metrics
- Training happens automatically at 120 FPS

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume training |
| **+** or **=** | Speed up (1.5x faster) |
| **-** | Slow down (1.5x slower) |
| **S** | Skip to next episode |
| **R** | Reset learning (start over) |
| **Q** or **ESC** | Quit |

### Tips

**To watch learning slowly:**
- Press `-` several times to slow down to 0.1x speed
- You'll see each individual step the agent takes

**To train quickly:**
- Press `+` several times to speed up to 10x+ speed
- Episodes fly by, agent learns faster

**To analyze specific episodes:**
- Press `SPACE` to pause
- Examine Q-values (cell colors)
- Check policy arrows
- Press `SPACE` again to resume

---

## 📊 Understanding the Visualization

### Color Heatmap (Q-values)

The background color of each cell shows the **maximum Q-value** for that state:

- **Deep Blue**: Very low Q-value (bad state, far from goal)
- **Cyan/Green**: Medium Q-value (getting closer)
- **Yellow/Red**: High Q-value (near goal, good state)

As training progresses, you'll see:
1. All cells start dark (Q-values ≈ 0)
2. Goal area turns yellow/red first
3. Color spreads backward toward start
4. Eventually entire path from start to goal is colored

This is the **value function** forming!

### Policy Arrows

Blue arrows show the **best action** for each state (argmax Q(s,a)).

Watch them evolve:
1. **Episode 1-10**: No arrows (random policy)
2. **Episode 10-50**: Arrows appear near goal
3. **Episode 50-200**: Arrows propagate toward start
4. **Episode 200+**: Clear path from start to goal

### Stats Panel

**Episode Info:**
- Current episode number
- Steps taken this episode
- Cumulative reward

**Parameters:**
- Alpha (α): Learning rate = 0.1
- Gamma (γ): Discount factor = 0.99
- Epsilon (ε): Exploration rate (decays over time)

**Statistics:**
- Avg Reward: Moving average over last N episodes
- Avg Length: Average steps to reach goal
- Avg Q-value: Average maximum Q-value across all states

**Learning Curve:**
- Mini graph showing episode rewards
- Should trend upward as agent learns

---

## 🧠 What's Happening Inside?

### Q-Learning Algorithm

```cpp
// For each step:
Action action = agent.selectAction(state);  // ε-greedy
auto [next_state, reward] = env.step(action);

// Q-update (Bellman equation):
Q[s][a] += α * (reward + γ * max(Q[s'][a']) - Q[s][a])
          //     └─────────┬────────────┘
          //           TD target
```

**Key Concepts:**

1. **ε-greedy Exploration**:
   - With probability ε: random action (explore)
   - With probability 1-ε: best action (exploit)
   - ε decays from 0.3 → 0.01 over training

2. **Temporal Difference (TD) Learning**:
   - Update Q-values using immediate reward + estimate of future
   - Don't wait for episode to finish (unlike Monte Carlo)

3. **Bellman Optimality**:
   - Q*(s,a) = E[r + γ max Q*(s',a')]
   - Optimal Q-value = reward + discounted best future value

---

## 🎯 Expected Learning Curve

### Episode Progression

**Episodes 1-50** (Exploration Phase):
- Reward: -10 to +5 (mostly negative)
- Length: 50-100 steps (wandering)
- Behavior: Random exploration, hitting walls
- Q-values: Growing slowly near goal

**Episodes 50-150** (Learning Phase):
- Reward: +5 to +8
- Length: 20-40 steps (getting better)
- Behavior: Some good paths, still exploring
- Q-values: Spreading from goal to start

**Episodes 150-300** (Convergence Phase):
- Reward: +8 to +9.5
- Length: 8-12 steps (near optimal)
- Behavior: Mostly direct path to goal
- Q-values: Stable, path clearly defined

**Episodes 300+** (Optimal):
- Reward: +9 to +10 (optimal = +9.2)
- Length: 8 steps (optimal for 5x5 grid)
- Behavior: Straight to goal every time
- Q-values: Converged, no longer changing

---

## 🔧 Customization

### Change Grid Size

In `rl_learning_animation.cpp`:

```cpp
const int GRID_SIZE = 5;  // Change to 7, 10, etc.
```

Larger grids take longer to learn!

### Add More Obstacles

In `GridWorld` constructor:

```cpp
walls.push_back({3, 3});
walls.push_back({3, 4});
// Add more walls to make it harder
```

### Adjust Learning Parameters

```cpp
const float LEARNING_RATE = 0.1f;      // Higher = faster updates
const float DISCOUNT_FACTOR = 0.99f;   // Higher = values future more
const float INITIAL_EPSILON = 0.3f;    // Higher = more exploration
```

### Change Rewards

```cpp
const float GOAL_REWARD = 10.0f;       // Reward for reaching goal
const float WALL_PENALTY = -1.0f;      // Penalty for hitting wall
const float STEP_PENALTY = -0.1f;      // Cost per step
```

---

## 📈 Performance

**Training Speed:**
- 120 FPS visualization
- ~100-200 episodes/second at max speed
- Complete learning (300 episodes) in 1-3 seconds

**Comparison:**
- Python version: ~20 FPS, 15-30 seconds
- C++ version: ~120 FPS, 1-3 seconds
- **6-10x faster!**

---

## 🎓 Educational Use

This visualization is perfect for:

### Teaching RL Concepts

1. **Show students live:**
   - How Q-values propagate backward from goal
   - Exploration vs. exploitation trade-off
   - Policy emergence from value function

2. **Interactive experiments:**
   - Pause and ask: "What should the agent do here?"
   - Change parameters live: "What if we increase ε?"
   - Add obstacles: "How does the policy adapt?"

### Understanding Algorithms

**Compare algorithms visually:**
- Run Q-Learning (this code)
- Implement SARSA (change one line!)
- See different learning speeds

**Debug RL problems:**
- Watch if Q-values diverge
- Check if policy makes sense
- Verify convergence

---

## 🐛 Troubleshooting

### Window doesn't open

**Problem:** X11 display error

**Solution:**
```bash
# If running on server, enable X11 forwarding
ssh -X user@server

# Or compile without visualization (headless mode)
# (modify code to remove rendering)
```

### Agent not learning

**Symptoms:**
- Q-values stay near 0
- No policy arrows appear
- Reward doesn't improve

**Possible causes:**
1. Learning rate too low (increase α)
2. Epsilon too high (agent explores too much)
3. Discount factor wrong (check γ)

### Training too slow/fast

**Too slow:**
- Press `+` multiple times
- Or modify `animation_speed` default

**Too fast:**
- Press `-` to slow down
- Or reduce FPS limit in code

---

## 🚀 Next Steps

### Extend the Code

**1. Add SARSA:**
```cpp
// In learn(), change from:
max_q_next = *std::max_element(Q[s_next].begin(), Q[s_next].end());

// To (SARSA):
next_action = selectAction(next_state);
next_q = Q[s_next][next_action];
```

**2. Add Experience Replay (DQN):**
```cpp
struct Experience {
    State state;
    Action action;
    float reward;
    State next_state;
    bool done;
};

std::deque<Experience> replay_buffer;
```

**3. Add Multiple Agents:**
```cpp
std::vector<QLearningAgent> agents;
// Multi-agent RL!
```

### Compare Algorithms

**Create comparison mode:**
- Run Q-Learning and SARSA side-by-side
- Show both policies
- Compare learning curves

---

## 📚 References

**Textbooks:**
- Sutton & Barto - "Reinforcement Learning: An Introduction"
  - Chapter 6: Temporal-Difference Learning
  - Section 6.5: Q-Learning

**Papers:**
- Watkins (1989) - "Learning from Delayed Rewards" (Original Q-Learning)
- Rummery & Niranjan (1994) - SARSA

**Online:**
- Our Python tutorial: `../tutorials/01_q_learning_gridworld.py`
- DeepMind RL Course: https://www.deepmind.com/learning-resources

---

## 💬 Feedback

This visualization helps you understand RL intuitively!

**Did you notice:**
- How value function propagates backward?
- How epsilon decay affects exploration?
- How quickly convergence happens?

Experiment and learn! 🚀

---

*"The only way to learn reinforcement learning is to implement it and watch it learn."*
