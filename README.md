# 🦠 Micro-Life ML Project

A machine learning-powered simulation of micro-organism behaviors, from simple random movement to complex ecosystem dynamics with reinforcement learning.

## 🌟 Overview

This project simulates artificial micro-organisms that:
- Move and interact in a 2D/3D environment
- Learn behaviors through machine learning algorithms
- Evolve and adapt using reinforcement learning
- Form complex ecosystems with emergent behaviors

## 📚 Documentation

**→ See [MICROLIFE_ML_GUIDE.md](./MICROLIFE_ML_GUIDE.md) for the complete step-by-step development roadmap**

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Create virtual environment
python -m venv microlife_env
source microlife_env/bin/activate  # Windows: microlife_env\Scripts\activate
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Run Your First Simulation
```bash
# Coming soon - Phase 1
python -m microlife.simulation.run_basic
```

## 🗂️ Project Structure

```
microlife/
├── simulation/     # Core simulation engine
│   ├── organism.py
│   ├── environment.py
│   └── physics.py
├── ml/            # Machine learning models
│   ├── clustering.py
│   ├── prediction.py
│   └── reinforcement.py
├── visualization/ # Graphics and animation
│   ├── renderer.py
│   └── dashboard.py
└── data/          # Logged simulation data
    └── logs/
```

## 🎯 Development Phases

1. **Phase 1:** Simple random movement ✅ (Starting here!)
2. **Phase 2:** Behaviors & data collection
3. **Phase 3:** Pattern recognition (K-Means, Decision Trees)
4. **Phase 4:** Behavior prediction (LSTM, Random Forest)
5. **Phase 5:** Reinforcement learning (Q-Learning, DQN)
6. **Phase 6:** Advanced visualization
7. **Phase 7:** Complex multi-species ecosystem

## 🧬 Features (Planned)

- [x] Project setup
- [ ] Basic organism simulation
- [ ] Food seeking behavior
- [ ] Energy & reproduction system
- [ ] ML behavior clustering
- [ ] Predictive movement models
- [ ] Reinforcement learning agents
- [ ] Interactive visualization
- [ ] Multi-species ecosystem

## 🛠️ Technologies

- **Python 3.8+**
- **NumPy & SciPy** - Scientific computing
- **Scikit-learn** - Machine learning
- **TensorFlow/PyTorch** - Deep learning
- **Matplotlib/Pygame** - Visualization
- **Pandas** - Data analysis

## 📖 Learning Goals

- Understand agent-based modeling
- Apply various ML algorithms to behavioral data
- Implement reinforcement learning from scratch
- Create emergent artificial life systems

## 🤝 Contributing

This is a learning project! Feel free to:
- Experiment with parameters
- Add new organism behaviors
- Try different ML algorithms
- Improve visualizations

## 📄 License

See LICENSE.txt

## 🎓 Resources

- [Complete Development Guide](./MICROLIFE_ML_GUIDE.md)
- Nature of Code by Daniel Shiffman
- Reinforcement Learning: An Introduction

---

**Current Status:** Phase 1 - Project Setup ✅

*Let's create artificial life! 🧬*
