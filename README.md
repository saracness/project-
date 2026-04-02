# Microlife — Agent-Based RL + Computational Neuroscience

Micro-organism simulations that combine **biologically grounded neuroscience** with **reinforcement learning**.  
Organisms live, eat, and die in a 2-D world; their brains learn through experience.

---

## Project state

| Component | Status |
|---|---|
| Python simulation core (`microlife/simulation/`) | Stable |
| 9-type neuron personality system with literature refs | Stable |
| RL brains — Q-Learning, DQN, Double-DQN (`microlife/ml/`) | Stable |
| Gym-compatible RL environment (`microlife/gym_env.py`) | Active |
| Training pipeline (`scripts/train.py`) | Active |
| C++ SFML visualiser — 120 FPS, 4 variants | Stable — build from `cpp/` |
| Test suite (150 tests, 0 failures) | Active |
| CI — GitHub Actions on every push | Active |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a DQN agent (300 episodes, logs to outputs/run/)
python scripts/train.py --brain dqn --episodes 300 --out outputs/run

# 3. Compare brain types
python scripts/train.py --brain qlearning --episodes 300 --out outputs/qlearning
python scripts/train.py --brain ddqn     --episodes 300 --out outputs/ddqn

# 4. Run tests
pytest tests/ -v
```

---

## Architecture

```
microlife/
├── gym_env.py              ← Gym-style env: reset() / step(action) / obs_to_brain_state()
├── simulation/
│   ├── environment.py      ← 2-D world, temperature zones, obstacles
│   ├── organism.py         ← Organism: energy, morphology, reproduction
│   ├── neuron.py           ← Biophysical neuron (membrane potential, STDP, Ca²⁺)
│   ├── neuron_personalities.py  ← 9 neuron types (dopaminergic → grid cells)
│   └── neuron_learning.py  ← LTP/LTD, intrinsic + structural plasticity
├── ml/
│   ├── brain_base.py       ← Abstract Brain interface
│   ├── brain_rl.py         ← QLearningBrain, DQNBrain, DoubleDQNBrain
│   ├── brain_cnn.py        ← Visual/grid-based CNN brains
│   ├── brain_gpu.py        ← PyTorch GPU-accelerated DQN
│   └── brain_evolutionary.py  ← Genetic algorithm brain
├── neat/                   ← NEAT (NeuroEvolution of Augmenting Topologies)
└── analytics/              ← SQLite logging, matplotlib plots, reports

cpp/                        ← C++ SFML visualisers (build locally, not tracked in git)
scripts/
└── train.py                ← Training pipeline — CLI entry point
tests/                      ← 150 pytest tests
docs/                       ← All project documentation
```

---

## Gym environment

`MicrolifeEnv` wraps the Python simulation into a standard RL interface:

```python
from microlife.gym_env import MicrolifeEnv
from microlife.ml.brain_rl import DQNBrain

env = MicrolifeEnv(width=300, height=300, n_food=20, max_steps=500, seed=42)
brain = DQNBrain(state_size=8, hidden_size=64)

obs = env.reset()
done = False
while not done:
    state = env.obs_to_brain_state(obs)
    action_dict = brain.decide_action(state)
    obs, reward, done, info = env.step(action_dict["_action_idx"])
    brain.learn(state, action_dict, reward, env.obs_to_brain_state(obs), done)
```

**Observation (8 dims, all in [0, 1]):**  
`energy | food_dist | sin(food_angle) | cos(food_angle) | in_temp_zone | near_obstacle | age | food_count`

**Actions (9 discrete):** 8 cardinal/diagonal directions + Stay

---

## Neuron personality system

9 biologically validated neuron types based on peer-reviewed literature:

| Type | Role | Firing pattern | Key property |
|---|---|---|---|
| Dopaminergic VTA | Reward coding | Burst | Pacemaker, reward prediction error |
| Serotonergic Raphe | Mood regulation | Regular | Stable, non-plastic |
| Cholinergic BF | Attention | Irregular | Arousal-dependent |
| Hippocampal Place cell | Spatial navigation | Burst | Place field encoding |
| Entorhinal Grid cell | Spatial metric | Regular | Grid-phase geometry |
| Mirror neuron | Action understanding | Adaptive | Observed action mirroring |
| Von Economo (VEN) | Social cognition | Burst/pacemaker | Social salience |
| Fast-spiking interneuron | Timing/inhibition | Fast spiking | >100 Hz capable |
| Chattering neuron | Pattern recognition | Chattering | High-freq burst trains |

---

## Building the C++ visualiser

```bash
# SFML required: sudo apt install libsfml-dev
make -f cpp/Makefile.microlife          # MICROLIFE_ULTIMATE (ecosystem)
make -f cpp/Makefile.nature.ultimate    # ONLY_FOR_NATURE_ULTIMATE (neural)
```

Compiled binaries are not tracked in git — always build from source.

---

## Running tests

```bash
pytest tests/ -v                            # all 150 tests
pytest tests/test_gym_env.py               # environment only
pytest tests/test_brains.py                # RL brains only
pytest tests/test_neuron_personalities.py  # neuroscience layer
pytest tests/test_simulation.py            # simulation core
```

---

## Research directions

1. **Agent-Based Modeling + RL** — organisms learn via DQN/NEAT in a physically grounded world
2. **Computational Neuroscience** — 9-type neuron system with STDP, homeostasis, neuromodulation
3. **Evolutionary Simulation** — genetic algorithm brains, natural selection, population dynamics

---

## License

See `LICENSE.txt`.
