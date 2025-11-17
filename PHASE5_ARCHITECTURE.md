# Phase 5: Advanced Visualization & GPU Acceleration
## Architectural Design Document

**Version:** 5.0
**Date:** 2025-11-17
**Features:** AI Training Visualization + Advanced Rendering + GPU Acceleration

---

## 1. AI Training Visualization System

### 1.1 Components

#### AIMetricsTracker (`microlife/visualization/ai_metrics.py`)
- **Purpose:** Collect and aggregate AI training metrics
- **Features:**
  - Track reward per timestep
  - Track loss (DQN/CNN)
  - Track Q-values distribution
  - Track action distribution
  - Track epsilon/learning rate decay
  - Moving averages (window=100)
  - Per-AI-type statistics

#### TrainingVisualizer (`microlife/visualization/training_visualizer.py`)
- **Purpose:** Real-time training graphs
- **Features:**
  - Reward curve (with moving average)
  - Loss curve (DQN/CNN/DoubleDQN)
  - Q-value distribution histogram
  - Action distribution pie chart
  - Epsilon decay curve
  - Survival time comparison
  - Multi-AI comparison mode

### 1.2 Data Structure

```python
class AIMetrics:
    - organism_id: str
    - brain_type: str
    - timesteps: List[int]
    - rewards: List[float]
    - losses: List[float]  # Neural network loss
    - q_values: List[float]
    - actions: List[int]
    - epsilon_history: List[float]
    - survival_time: int
    - total_reward: float
    - avg_reward: float (moving average)
```

### 1.3 Visualization Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION AREA                          │
│                   (Main Renderer)                           │
└─────────────────────────────────────────────────────────────┘
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Reward Curve │ Loss Curve   │ Q-Value Dist │ Action Dist  │
│ (Line)       │ (Line)       │ (Histogram)  │ (Pie Chart)  │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

---

## 2. Advanced Rendering System

### 2.1 Components

#### AdvancedRenderer (`microlife/visualization/advanced_renderer.py`)
- **Extends:** SimpleRenderer
- **Features:**
  - All SimpleRenderer features
  - Trail rendering
  - Particle effects
  - Heatmap overlay
  - Mini-map
  - Glow effects
  - Gradient coloring

#### TrailSystem (`microlife/visualization/effects/trails.py`)
- **Purpose:** Organism movement trails
- **Features:**
  - Fade-out effect (alpha decay)
  - Color based on organism type
  - Configurable length (default: 20 positions)
  - GPU-friendly line batch rendering
  - Performance: O(n) per frame

#### ParticleSystem (`microlife/visualization/effects/particles.py`)
- **Purpose:** Visual feedback for events
- **Events:**
  - Food consumption (green sparkles)
  - Death (red explosion)
  - Reproduction (blue burst)
  - Energy gain/loss (floating numbers)
- **Features:**
  - Physics (velocity, gravity, fade)
  - Batch rendering
  - Object pooling (max 1000 particles)

#### HeatmapGenerator (`microlife/visualization/effects/heatmap.py`)
- **Purpose:** Population density visualization
- **Features:**
  - 2D density grid (50x50)
  - Gaussian blur
  - Color gradient (blue → green → yellow → red)
  - Real-time update
  - Toggle on/off
  - Performance: GPU-accelerated if available

#### MiniMap (`microlife/visualization/effects/minimap.py`)
- **Purpose:** Overview of entire simulation
- **Features:**
  - 100x100 pixel mini-map
  - Shows all organisms (dots)
  - Shows food (smaller dots)
  - Shows current viewport (rectangle)
  - Corner positioning

### 2.2 Rendering Pipeline

```
Frame Start
    ↓
Clear Buffer
    ↓
Render Background
    ↓
Render Heatmap (if enabled)
    ↓
Render Trails (batch)
    ↓
Render Food Particles
    ↓
Render Organisms (with glow)
    ↓
Render Particles (batch)
    ↓
Render Mini-Map
    ↓
Render UI (stats, controls)
    ↓
Blit to Screen
```

### 2.3 Visual Effects Details

#### Trail Effect
```python
Trail:
    - positions: deque(maxlen=20)
    - colors: gradient from full color → transparent
    - alpha: 1.0 → 0.0 (linear decay)
    - width: 3px → 1px
```

#### Particle Types
```python
FoodParticle:
    - color: (0, 255, 0)  # Green
    - lifetime: 0.5s
    - velocity: random spray
    - size: 5px → 1px

DeathParticle:
    - color: (255, 0, 0)  # Red
    - lifetime: 1.0s
    - velocity: explosion pattern
    - size: 8px → 1px

ReproductionParticle:
    - color: (0, 150, 255)  # Blue
    - lifetime: 0.8s
    - velocity: burst pattern
    - size: 6px → 1px
```

---

## 3. GPU Acceleration System

### 3.1 Components

#### Config System (`microlife/config.py`)
```python
class SimulationConfig:
    - use_gpu: bool
    - gpu_device: str ('cuda:0', 'cpu')
    - batch_size: int (default: 32)
    - max_organisms: int (default: 1000)
    - enable_trails: bool
    - enable_particles: bool
    - enable_heatmap: bool
```

#### GPU Brain Base (`microlife/ml/brain_gpu.py`)
```python
class GPUBrain(Brain):
    - device: torch.device
    - batch_forward()  # Process multiple states
    - to_gpu()
    - to_cpu()
```

#### GPU-Accelerated Brains
- **GPUDQNBrain:** PyTorch neural network on GPU
- **GPUDoubleDQNBrain:** Double DQN on GPU
- **GPUCNNBrain:** Convolutional network on GPU

### 3.2 GPU Optimization Strategies

#### Batch Processing
```python
# Instead of processing one state at a time:
for organism in organisms:
    action = organism.brain.decide_action(state)

# Batch process all states:
states = [get_state(org) for org in organisms]
actions = brain.batch_decide_action(states)  # GPU parallel
```

#### Vectorized Environment
```python
# Instead of updating organisms one by one:
for organism in organisms:
    organism.update()

# Vectorized update:
positions = np.array([org.position for org in organisms])
velocities = np.array([org.velocity for org in organisms])
# Update all at once using NumPy vectorization
```

#### Memory Management
```python
- Preallocate tensors
- Reuse memory buffers
- Clear GPU cache periodically
- Monitor GPU memory usage
```

### 3.3 Performance Targets

| Organisms | CPU (FPS) | GPU (FPS) | Speedup |
|-----------|-----------|-----------|---------|
| 10        | 60        | 60        | 1x      |
| 100       | 40        | 60        | 1.5x    |
| 500       | 15        | 55        | 3.7x    |
| 1000      | 8         | 50        | 6.2x    |
| 2000      | 3         | 40        | 13x     |

### 3.4 PyTorch GPU Architecture

```python
class GPUDQNBrain:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)

    def batch_forward(self, states):
        # states: (batch_size, state_size)
        states_tensor = torch.FloatTensor(states).to(self.device)
        q_values = self.network(states_tensor)
        return q_values.cpu().numpy()
```

---

## 4. File Structure

```
microlife/
├── visualization/
│   ├── ai_metrics.py              # NEW - AI metrics tracking
│   ├── training_visualizer.py     # NEW - Training graphs
│   ├── advanced_renderer.py       # NEW - Advanced rendering
│   ├── effects/                   # NEW - Effect systems
│   │   ├── __init__.py
│   │   ├── trails.py              # Trail system
│   │   ├── particles.py           # Particle system
│   │   ├── heatmap.py             # Heatmap generator
│   │   └── minimap.py             # Mini-map
│   ├── simple_renderer.py         # EXISTING
│   └── interactive_panel.py       # EXISTING
├── ml/
│   ├── brain_gpu.py               # NEW - GPU-accelerated brains
│   ├── brain_rl.py                # EXISTING
│   ├── brain_cnn.py               # EXISTING
│   └── brain_evolution.py         # EXISTING
├── simulation/
│   ├── environment.py             # MODIFY - GPU optimization
│   ├── organism.py                # MODIFY - Trail tracking
│   └── ...
├── config.py                      # NEW - Configuration system
└── performance.py                 # NEW - Performance profiler

demo_advanced.py                   # NEW - Advanced demo
demo_gpu_benchmark.py              # NEW - GPU benchmark
```

---

## 5. Implementation Phases

### Phase 5.1: AI Training Visualization
1. Create AIMetricsTracker
2. Create TrainingVisualizer
3. Integrate with existing brains
4. Test with demo

### Phase 5.2: Advanced Rendering
1. Create effect systems (trails, particles, heatmap, minimap)
2. Create AdvancedRenderer
3. Integrate with simulation
4. Test performance

### Phase 5.3: GPU Acceleration
1. Create config system
2. Create GPU brain base class
3. Implement GPU-DQN, GPU-DoubleDQN, GPU-CNN
4. Vectorize environment operations
5. Benchmark and optimize

### Phase 5.4: Integration & Testing
1. Create demo_advanced.py
2. Create demo_gpu_benchmark.py
3. Write documentation
4. Performance testing
5. User guide

---

## 6. Configuration Examples

### Example 1: Full Features (GPU)
```python
config = SimulationConfig(
    use_gpu=True,
    gpu_device='cuda:0',
    batch_size=64,
    max_organisms=1000,
    enable_trails=True,
    enable_particles=True,
    enable_heatmap=True,
    enable_minimap=True
)
```

### Example 2: CPU Performance Mode
```python
config = SimulationConfig(
    use_gpu=False,
    max_organisms=100,
    enable_trails=True,
    enable_particles=False,
    enable_heatmap=False,
    enable_minimap=True
)
```

### Example 3: GPU Benchmark Mode
```python
config = SimulationConfig(
    use_gpu=True,
    batch_size=128,
    max_organisms=2000,
    enable_trails=False,
    enable_particles=False,
    enable_heatmap=False,
    enable_minimap=False
)
```

---

## 7. API Examples

### AI Metrics
```python
from microlife.visualization.ai_metrics import AIMetricsTracker
from microlife.visualization.training_visualizer import TrainingVisualizer

tracker = AIMetricsTracker()
visualizer = TrainingVisualizer(tracker)

# During simulation
for timestep in range(1000):
    for organism in env.organisms:
        if organism.brain:
            tracker.record(organism.id, organism.brain)

    # Update graphs every 10 timesteps
    if timestep % 10 == 0:
        visualizer.update()
```

### Advanced Rendering
```python
from microlife.visualization.advanced_renderer import AdvancedRenderer

renderer = AdvancedRenderer(
    env,
    enable_trails=True,
    enable_particles=True,
    enable_heatmap=True
)

# Render loop
while running:
    env.update()
    renderer.render()
```

### GPU Acceleration
```python
from microlife.ml.brain_gpu import GPUDQNBrain
from microlife.config import SimulationConfig

config = SimulationConfig(use_gpu=True)
brain = GPUDQNBrain(state_size=7, action_size=9)

# Batch processing
states = [get_state(org) for org in organisms]
actions = brain.batch_decide_action(states)  # GPU parallel
```

---

## 8. Performance Considerations

### Rendering Optimizations
- Use batch rendering (all trails in one call)
- Object pooling for particles
- Cull off-screen objects
- LOD (Level of Detail) for distant organisms
- Skip rendering if unchanged

### GPU Optimizations
- Minimize CPU↔GPU transfers
- Use pinned memory
- Batch operations
- Asynchronous data loading
- Mixed precision (FP16)

### Memory Management
- Limit trail history (20 positions)
- Limit particle count (1000 max)
- Clear dead particles
- Preallocate buffers
- Use generators for large datasets

---

## 9. Testing Strategy

### Unit Tests
- Trail system (fade, length limit)
- Particle system (pooling, physics)
- Heatmap generation (density calculation)
- GPU brain (forward pass, batch processing)
- Metrics tracking (recording, aggregation)

### Integration Tests
- Full rendering pipeline
- GPU vs CPU comparison
- Multi-organism simulation (10, 100, 1000)
- Memory leak detection
- FPS benchmarks

### Performance Tests
- 1000 organisms @ 30 FPS target
- GPU memory usage < 2GB
- CPU memory usage < 4GB
- Startup time < 5s

---

## 10. Success Criteria

✅ **AI Visualization:**
- Real-time reward curves displayed
- Loss curves for neural networks
- Q-value distribution visible
- Action distribution shown
- Multi-AI comparison works

✅ **Advanced Rendering:**
- Smooth trails (60 FPS with 100 organisms)
- Particle effects working (food, death, reproduction)
- Heatmap overlay functional
- Mini-map showing overview
- Visual quality improvement obvious

✅ **GPU Acceleration:**
- CUDA support detected
- GPU brains 3x+ faster than CPU
- 1000+ organisms running smoothly
- Batch processing working
- Memory management stable

---

**Status:** Architecture Designed ✅
**Next:** Begin Implementation
