# MicroLife: Agent-Based Micro-Organism Simulation

A simulation platform for studying emergent behaviors in artificial micro-organisms.
Combines agent-based modeling with tabular and deep reinforcement learning.

## Implementation Status

| Component | Status |
|-----------|--------|
| Organism: movement, energy, morphology, reproduction | Complete |
| Environment: food, temperature zones, obstacles | Complete |
| Greedy food-seeking behavior | Complete |
| Tabular Q-Learning brain | Complete |
| DQN / Double-DQN (numpy, no external RL library required) | Complete |
| Evolutionary and CNN brains | Complete |
| Neural tissue model: neuron life-cycle, Hebbian plasticity | Complete |
| Neurovascular coupling simulation | Complete |
| Multi-species ecosystem | In progress |
| GPU-accelerated batch rollouts | Experimental |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Compare RL agents vs random/greedy baselines
python experiments/benchmark_rl_vs_random.py

# Neuron plasticity demo
python demo_neuron_learning.py

# Multi-agent AI battle
python demo_ai_battle.py

# Headless basic simulation
python -m microlife.simulation.run_basic
```

## Project Structure

```
microlife/
├── simulation/
│   ├── organism.py            # Agent: energy, movement, morphology, reproduction
│   ├── environment.py         # World: food, temperature zones, obstacles
│   ├── morphology.py          # Physical trait system with mutation
│   ├── neuron.py              # Neuron life-cycle (neurogenesis -> apoptosis)
│   ├── neuron_morphology.py   # Morphological neuron types
│   ├── neural_environment.py  # Tissue: neurons + blood vessels + astrocytes
│   └── neuron_learning.py     # Synaptic plasticity (LTP/LTD/STDP)
├── ml/
│   ├── brain_base.py          # Abstract Brain interface
│   ├── brain_rl.py            # Q-Learning, DQN, Double-DQN (numpy)
│   ├── brain_evolutionary.py  # Genetic algorithm brain
│   └── brain_cnn.py           # CNN-based perception
experiments/
└── benchmark_rl_vs_random.py  # Reproducible survival comparison
```

## Running the Benchmark

```bash
python experiments/benchmark_rl_vs_random.py --trials 10 --steps 2000
```

Tests whether Q-Learning agents outlive random-walking agents and greedy
food-seekers across repeated trials with fixed seeds. Results are printed
as a comparison table with a ratio relative to the random baseline.

## Scientific Basis

The neural tissue simulation (`microlife/simulation/neuron.py`) is grounded
in peer-reviewed neuroscience:

- Hebbian learning / LTP / LTD: Bliss & Collingridge (1993); Malenka & Bear (2004)
- Neuron life-cycle (neurogenesis to apoptosis): Kempermann et al. (2015); Yuan & Yankner (2000)
- Neurovascular coupling: Attwell et al. (2010); Iadecola (2017)
- Astrocyte-neuron lactate shuttle: Pellerin & Magistretti (1994)
- STDP: Markram et al. (1997)

See [NEURON_BIOLOGY.md](NEURON_BIOLOGY.md) for the complete reference list.

The RL implementation follows:

- Q-Learning: Watkins & Dayan (1992)
- DQN with experience replay: Mnih et al. (2015)
- Double DQN: Van Hasselt et al. (2016)

## Known Limitations

- Electrophysiology uses a simplified continuous model, not full Hodgkin-Huxley
  compartmental dynamics.
- RL brains train from scratch each episode; no persistent model checkpointing
  by default.
- No Gym-compatible environment wrapper yet, so standard RL benchmarking
  pipelines cannot be applied directly.
- C++ visualization components (MICROLIFE_ULTIMATE.cpp, ONLY_FOR_NATURE.cpp)
  require SFML and separate compilation via the provided Makefiles.

## License

AGPL-3.0 — see [LICENSE.txt](LICENSE.txt)
