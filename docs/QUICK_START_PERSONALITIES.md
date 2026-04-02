# Quick Start Guide: Neuron Personalities

A practical guide to using the 9 neuron personality types in your simulations.

---

## Basic Usage Pattern

```python
from simulation.neuron import Neuron
from simulation.neuron_personalities import create_personalized_neuron

# Step 1: Create base neuron
base = Neuron(x=100, y=100, z=50, neuron_type="pyramidal")
base.stage = "mature"
base.energy = 200.0

# Step 2: Add personality
personalized = create_personalized_neuron(base, "personality_type")

# Step 3: Update with context
context = {...}  # Depends on personality type
personalized.update(dt=0.01, time=0.0, context=context)

# Step 4: Check behavior
print(f"Firing rate: {personalized.neuron.firing_rate} Hz")
```

---

## 1. Dopaminergic Neurons - Reward Learning

**Use case**: Reward-based learning, reinforcement learning signals

```python
# Create dopaminergic neuron
da_neuron = create_personalized_neuron(base, "dopaminergic")

# Positive reward prediction error (unexpected reward)
if reward > expected:
    da_neuron.neuron.membrane_potential += 50.0  # Burst!
    da_neuron.update(dt, time)
    # Result: Firing rate increases to ~20 Hz

# Use dopamine signal to modulate learning
dopamine_level = da_neuron.neuron.firing_rate / 20.0  # Normalize
for synapse in learning_synapses:
    synapse.weight += learning_rate * dopamine_level * hebbian_term
```

**What you get**:
- Burst firing (4 → 20 Hz) for reward prediction errors
- Learning signal for synaptic plasticity
- Motivational salience encoding

---

## 2. Place Cells - Spatial Navigation

**Use case**: Spatial memory, navigation tasks, cognitive maps

```python
# Create place cell
place_cell = create_personalized_neuron(base, "place_cell")

# Simulate movement through space
for x, y in path:
    context = {'position': [x, y, 50.0]}
    place_cell.update(dt, time, context)

    # Place cell fires strongly at specific location
    if place_cell.neuron.firing_rate > 30.0:
        print(f"At place field center! Position: ({x}, {y})")
```

**What you get**:
- Gaussian place field (peaks at ~40 Hz in field center)
- Location-specific firing
- Spatial memory encoding

**Create a complete place cell map**:
```python
# Network of 20 place cells covering environment
place_cells = []
for i in range(20):
    base = Neuron(x=random.uniform(0, 300), y=random.uniform(0, 300), z=50)
    base.stage = "mature"
    pc = create_personalized_neuron(base, "place_cell")
    place_cells.append(pc)

# Each cell develops random place field
# Together they tile the environment
```

---

## 3. Grid Cells - Metric Encoding

**Use case**: Path integration, distance/direction encoding

```python
# Create grid cell
grid_cell = create_personalized_neuron(base, "grid_cell")

# Move through environment
trajectory = [(x, y, 50) for x in range(0, 300, 5) for y in range(0, 300, 5)]

firing_map = []
for position in trajectory:
    context = {'position': position}
    grid_cell.update(dt, time, context)
    firing_map.append(grid_cell.neuron.firing_rate)

# Result: Hexagonal firing pattern across space
```

**What you get**:
- Hexagonal grid pattern (100 μm spacing)
- Periodic spatial firing
- Metric for distance and direction

---

## 4. Mirror Neurons - Action Learning

**Use case**: Imitation learning, action understanding

```python
# Create mirror neuron tuned to specific action
mirror_neuron = create_personalized_neuron(base, "mirror_neuron")
mirror_neuron.observed_action = "grasp"  # Tune to grasping action

# Scenario 1: Execute action yourself
context = {'self_action': 'grasp'}
mirror_neuron.update(dt, time, context)
# Result: Fires at ~50 Hz

# Scenario 2: Observe someone else grasping
context = {'observed_action': 'grasp'}
mirror_neuron.update(dt, time, context)
# Result: Also fires at ~50 Hz (action understanding!)

# Scenario 3: Different action
context = {'observed_action': 'reach'}
mirror_neuron.update(dt, time, context)
# Result: Firing decays (not the tuned action)
```

**What you get**:
- Action-observation matching
- Imitation learning capability
- Intention understanding

---

## 5. Fast-Spiking Interneurons - Timing & Rhythm

**Use case**: Gamma oscillations, precise timing, inhibitory control

```python
# Create fast-spiking interneuron
fs_interneuron = create_personalized_neuron(base, "fast_spiking")

# These neurons can fire extremely fast
fs_interneuron.neuron.membrane_potential = -50.0  # Slightly depolarized

for step in range(100):
    fs_interneuron.update(dt=0.001, time=step*0.001)

    # Can reach 200 Hz!
    if fs_interneuron.neuron.firing_rate > 100:
        print(f"Ultra-fast firing: {fs_interneuron.neuron.firing_rate} Hz")

# Use for gamma oscillations (40 Hz)
for pyramidal_neuron in pyramidal_cells:
    if fs_interneuron.neuron.firing_rate > 40:
        # Inhibit pyramidal cells rhythmically
        pyramidal_neuron.membrane_potential -= 15.0
```

**What you get**:
- Ultra-fast firing (10-200 Hz)
- Non-adapting (stable inhibition)
- Precise timing for network synchronization

---

## 6. Serotonergic Neurons - Mood Regulation

**Use case**: Long-term state modulation, mood, sleep-wake cycles

```python
# Create serotonergic neuron
serotonin_neuron = create_personalized_neuron(base, "serotonergic")

# Very slow, regular firing (1-5 Hz)
serotonin_neuron.update(dt, time)

# Modulates entire network state
serotonin_level = serotonin_neuron.neuron.firing_rate / 5.0  # Normalize

for neuron in all_neurons:
    # High serotonin: stabilizes activity, reduces impulsivity
    if serotonin_level > 0.5:
        neuron.firing_threshold += 2.0  # Harder to fire
        neuron.noise_level *= 0.5       # Less noisy
    else:
        # Low serotonin: more reactive, impulsive
        neuron.firing_threshold -= 2.0
        neuron.noise_level *= 1.5
```

**What you get**:
- Slow, stable firing (1-5 Hz)
- Widespread influence
- State regulation (mood, anxiety, impulse control)

---

## 7. Cholinergic Neurons - Attention

**Use case**: Attention enhancement, sensory gating, memory encoding

```python
# Create cholinergic neuron
ach_neuron = create_personalized_neuron(base, "cholinergic")

# During attention/arousal, fires irregularly at 5-30 Hz
# Simulate attention task
attention_on = True

for step in range(1000):
    if attention_on:
        # Irregular, enhanced firing
        ach_neuron.neuron.membrane_potential += random.uniform(5, 25)

    ach_neuron.update(dt, time)

    # Use ACh signal to gate sensory input
    ach_level = ach_neuron.neuron.firing_rate / 30.0

    for sensory_neuron in sensory_neurons:
        if ach_level > 0.3:
            # High ACh: enhance sensory signals
            sensory_neuron.response_gain *= 2.0
        else:
            # Low ACh: suppress background
            sensory_neuron.response_gain *= 0.5
```

**What you get**:
- Irregular, state-dependent firing (5-30 Hz)
- Attention enhancement
- Sensory gating and memory encoding

---

## 8. Von Economo Neurons - Social Cognition

**Use case**: Social awareness, rapid social processing

```python
# Create von Economo neuron
ven = create_personalized_neuron(base, "von_economo")

# These neurons are fast-conducting for rapid social responses
social_cue_detected = True

if social_cue_detected:
    # Very fast response (15 ms latency)
    ven.neuron.membrane_potential += 40.0
    ven.update(dt, time)

    # Fast-spiking (8-80 Hz)
    if ven.neuron.firing_rate > 60:
        print("Social awareness activated!")

        # Trigger social cognition network
        for neuron in social_network:
            neuron.membrane_potential += 20.0
```

**What you get**:
- Very fast firing (8-80 Hz)
- Rapid social information processing
- Self-awareness and empathy signals

---

## 9. Chattering Neurons - Pattern Recognition

**Use case**: Feature binding, pattern detection, attention

```python
# Create chattering neuron
chatter = create_personalized_neuron(base, "chattering")

# Detects patterns via high-frequency bursts
input_pattern = [1, 0, 1, 0, 1]  # Pattern to detect

for timestep, bit in enumerate(input_pattern):
    if bit == 1:
        chatter.neuron.membrane_potential += 30.0

    chatter.update(dt=0.001, time=timestep*0.001)

    # When pattern detected, generates 200-600 Hz burst
    if chatter.neuron.firing_rate > 80:
        print(f"Pattern detected! Burst at {chatter.neuron.firing_rate} Hz")

        # Bind features together
        for feature_neuron in feature_neurons:
            feature_neuron.membrane_potential += 15.0  # Synchronize
```

**What you get**:
- High-frequency bursts (2-100 Hz)
- Pattern detection
- Feature binding via synchronization

---

## Complete Example: Spatial Learning with Multiple Personalities

```python
from simulation.neuron import Neuron
from simulation.neuron_personalities import create_personalized_neuron
import numpy as np

# Create heterogeneous network
neurons = {
    'place_cells': [],
    'grid_cells': [],
    'dopamine': [],
    'fast_spiking': []
}

# 10 place cells
for i in range(10):
    base = Neuron(x=np.random.uniform(50, 250), y=np.random.uniform(50, 250), z=50)
    base.stage = "mature"
    base.energy = 200.0
    neurons['place_cells'].append(create_personalized_neuron(base, "place_cell"))

# 5 grid cells
for i in range(5):
    base = Neuron(x=np.random.uniform(50, 250), y=np.random.uniform(50, 250), z=50)
    base.stage = "mature"
    base.energy = 200.0
    neurons['grid_cells'].append(create_personalized_neuron(base, "grid_cell"))

# 2 dopamine neurons
for i in range(2):
    base = Neuron(x=150, y=150, z=50)
    base.stage = "mature"
    base.energy = 200.0
    neurons['dopamine'].append(create_personalized_neuron(base, "dopaminergic"))

# 5 fast-spiking interneurons
for i in range(5):
    base = Neuron(x=np.random.uniform(50, 250), y=np.random.uniform(50, 250), z=50)
    base.stage = "mature"
    base.energy = 200.0
    neurons['fast_spiking'].append(create_personalized_neuron(base, "fast_spiking"))

# Simulate navigation task
path = [(x, y, 50) for x in range(0, 300, 10) for y in range(0, 300, 10)]
rewards = [1.0 if (x == 150 and y == 150) else 0.0 for x, y, z in path]

dt = 0.01
time = 0.0

for step, (position, reward) in enumerate(zip(path, rewards)):
    # Update spatial neurons
    context = {'position': position}

    for pc in neurons['place_cells']:
        pc.update(dt, time, context)

    for gc in neurons['grid_cells']:
        gc.update(dt, time, context)

    # Dopamine signal for reward
    if reward > 0:
        for da in neurons['dopamine']:
            da.neuron.membrane_potential += 50.0  # Reward burst!

    for da in neurons['dopamine']:
        da.update(dt, time)

    # Fast-spiking for timing
    for fs in neurons['fast_spiking']:
        fs.update(dt, time)

    # Learning: use dopamine to modulate place cell connections
    dopamine_level = np.mean([da.neuron.firing_rate / 20.0 for da in neurons['dopamine']])

    # (Connect place cells to grid cells with dopamine modulation)
    # ... synaptic plasticity here ...

    time += dt

    if step % 100 == 0:
        print(f"Step {step}: Position {position}, Reward {reward}, DA level {dopamine_level:.2f}")

print("\nSpatial learning complete!")
print(f"Place cells active: {sum(1 for pc in neurons['place_cells'] if pc.neuron.firing_rate > 20)}")
print(f"Grid cells active: {sum(1 for gc in neurons['grid_cells'] if gc.neuron.firing_rate > 15)}")
```

---

## Tips for Integration

### 1. Match Personality to Task

- **Spatial tasks** → place_cell, grid_cell
- **Reward learning** → dopaminergic
- **Timing tasks** → fast_spiking
- **Pattern recognition** → chattering
- **Imitation** → mirror_neuron

### 2. Use Context Dictionary

Different personalities need different context:

```python
context = {
    'position': [x, y, z],           # For place/grid cells
    'observed_action': 'grasp',      # For mirror neurons
    'reward': 1.0,                   # For dopamine neurons
    'social_cue': True,              # For von Economo neurons
}
```

### 3. Combine Personalities

Real brains use multiple neuron types together:

```python
# Reward-modulated spatial learning
place_cells = [create_personalized_neuron(..., "place_cell") for _ in range(20)]
da_neurons = [create_personalized_neuron(..., "dopaminergic") for _ in range(5)]

# DA signal modulates place cell plasticity
dopamine = da_neurons[0].neuron.firing_rate / 20.0
for pc in place_cells:
    pc.neuron.morphology.learning_rate = 0.01 * (1 + dopamine)
```

---

## Troubleshooting

### Neuron not firing?

```python
# Check energy and stage
assert base.stage == "mature"
assert base.energy > 10.0
assert base.glucose_level > 0.1
assert base.oxygen_level > 0.1

# Provide input
base.membrane_potential += 20.0
```

### Place field not working?

```python
# Ensure context has position
context = {'position': [x, y, z]}  # Must be list/array
place_cell.update(dt, time, context)
```

### Want to see personality info?

```python
print(personalized.get_description())
```

---

## Next Steps

1. **Run the demo**: `python demo_neuron_personalities.py`
2. **Read full docs**: `NEURON_PERSONALITIES.md`
3. **Integrate with learning**: Use with `NeuralLearningEnvironment`
4. **Create custom tasks**: Leverage personality-specific functions

---

**Remember**: All existing code is unchanged! Personalities are added via wrapper pattern.

Enjoy your biologically-realistic neuron simulations! 🧠✨
