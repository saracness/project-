# MicroLife: Agent-Based Micro-Organism Simulation

A simulation platform for studying emergent behaviors in artificial micro-organisms.
Combines agent-based modeling with tabular and deep reinforcement learning.

## Quick Start

```bash
git clone https://github.com/saracness/project-
cd project-
python START_SIMULATION.py
```

That's it. The launcher checks your Python version, installs any missing
dependencies from `requirements.txt`, runs the simulation, and writes logs
to `./logs/<timestamp>/`.

```
logs/2026-04-09_14-30-00/
    stats.csv      <-- per-step metrics (population, energy, age, food)
    events.log     <-- notable events (start, milestones, collapse, end)
```

Optional flags:

```bash
python START_SIMULATION.py --steps 1000 --organisms 30
python START_SIMULATION.py --seed 42        # reproducible run
python START_SIMULATION.py --no-logs        # console only
```

## Other Entry Points

```bash
# Compare RL agents vs random/greedy baselines
python experiments/benchmark_rl_vs_random.py

# Train Q-Learning, save checkpoint, evaluate (epsilon=0)
python experiments/train_and_evaluate.py

# Neuron plasticity demo (Hebbian learning)
python demo_neuron_learning.py

# Minimal headless runner (no logging)
python -m microlife.simulation.run_basic
```

## Implementation Status

| Component | Status |
|-----------|--------|
| Organism: movement, energy, morphology, reproduction | Complete |
| Environment: food, temperature zones, obstacles | Complete |
| Greedy food-seeking behavior | Complete |
| Tabular Q-Learning brain | Complete |
| DQN / Double-DQN (numpy, no external RL library required) | Complete |
| Brain model save / load (pickle / numpy.savez) | Complete |
| Evolutionary and CNN brains | Complete |
| Neural tissue model: neuron life-cycle, Hebbian plasticity | Complete |
| Neurovascular coupling simulation | Complete |
| File logging (stats.csv + events.log per run) | Complete |
| Multi-species ecosystem | In progress |
| GPU-accelerated batch rollouts | Experimental |

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
│   ├── brain_random.py        # RandomBrain (uniform random, benchmark baseline)
│   ├── brain_rl.py            # Q-Learning, DQN, Double-DQN (numpy)
│   ├── brain_evolutionary.py  # Genetic algorithm brain
│   └── brain_cnn.py           # CNN-based perception
└── logger.py                  # SimulationLogger (stats.csv + events.log)
experiments/
├── benchmark_rl_vs_random.py  # Reproducible survival comparison (4 conditions)
└── train_and_evaluate.py      # Train -> checkpoint -> eval pipeline
logs/                          # Auto-created, gitignored
```

## Log Format

`stats.csv` columns:
```
timestep, population, total_organisms, food_count,
avg_energy, avg_age, seeking_count, wandering_count,
temperature_zones, obstacles
```

Load in Python:
```python
import pandas as pd
df = pd.read_csv("logs/2026-04-09_14-30-00/stats.csv")
df.plot(x="timestep", y=["population", "avg_energy"])
```

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
- RL brains train from scratch each episode unless `save_model` / `load_model`
  is used (see `experiments/train_and_evaluate.py`).
- No Gym-compatible environment wrapper yet.
- C++ visualization components require SFML and separate compilation
  (see the provided Makefiles).

## License

AGPL-3.0 -- see [LICENSE.txt](LICENSE.txt)
