# AI Training Visualization System

## 🎯 Overview

Professional-grade AI training visualization system for monitoring and analyzing neural network training in real-time.

## 🏗️ Architecture

### Components

1. **TrainingMetrics** - Metrics collection and storage
2. **MetricsTracker** - Real-time metric tracking with history
3. **TrainingVisualizer** - Real-time graph rendering
4. **NetworkVisualizer** - Neural network structure visualization
5. **DecisionBoundaryVisualizer** - Decision boundary plotting
6. **TrainingDashboard** - Integrated dashboard combining all visualizations

## 📊 Features

### 1. Training Metrics Tracking

- **Loss tracking** - Training and validation loss over time
- **Accuracy metrics** - Classification accuracy
- **Reward tracking** - Reinforcement learning rewards
- **Learning rate** - Adaptive learning rate monitoring
- **Gradient statistics** - Gradient norm, variance
- **Weight statistics** - Layer weight distributions

### 2. Real-Time Graphs

- **Line plots** - Loss, accuracy, rewards over episodes
- **Moving averages** - Smoothed metrics (window size configurable)
- **Multiple series** - Train vs validation comparison
- **Auto-scaling** - Dynamic Y-axis scaling
- **Time-series** - X-axis as training steps/episodes

### 3. Neural Network Visualization

- **Network topology** - Input → Hidden → Output layers
- **Activation visualization** - Real-time neuron activations
- **Weight heatmaps** - Connection strength visualization
- **Gradient flow** - Backprop gradient magnitude
- **Layer statistics** - Mean/std per layer

### 4. Decision Boundary Visualization

- **2D projection** - High-dimensional → 2D (PCA/t-SNE)
- **Classification regions** - Color-coded decision regions
- **Sample points** - Training data overlay
- **Confidence contours** - Prediction confidence levels

### 5. Performance Dashboard

- **Multi-panel layout** - Configurable grid layout
- **Live updates** - Real-time refresh (configurable FPS)
- **Export** - Save metrics to CSV/JSON
- **Checkpointing** - Save/load training state

## 🔧 Technical Design

### Data Flow

```
Training Loop
    ↓
MetricsTracker.record(loss, accuracy, ...)
    ↓
TrainingVisualizer.update(metrics)
    ↓
Render graphs (ModernGL or Matplotlib)
```

### Metrics Storage

```python
class MetricsTracker:
    metrics = {
        'loss': [0.5, 0.45, 0.42, ...],
        'accuracy': [0.6, 0.65, 0.68, ...],
        'reward': [10, 12, 15, ...],
        'episode': [1, 2, 3, ...],
    }
```

### Visualization Modes

1. **Real-time mode** - Update every N steps (low overhead)
2. **Batch mode** - Update after each epoch
3. **On-demand** - Manual refresh
4. **Post-training** - Analyze saved metrics

## 📁 File Structure

```
microlife/training/
├── metrics_tracker.py          # Metrics collection
├── training_visualizer.py      # Graph rendering
├── network_visualizer.py       # NN structure visualization
├── decision_boundary.py        # Decision boundary plots
└── training_dashboard.py       # Integrated dashboard

microlife/ml/
├── training_loop.py            # Enhanced training loop with metrics
└── callbacks.py                # Training callbacks

demos/
└── demo_training_viz.py        # Training visualization demo
```

## 🎨 Visualization Examples

### Loss Graph
```
Loss over Time
│
│ ╭─────────╮
│ │         ╰──────
│ │
│ │
│ ╰
└─────────────────→ Episodes
```

### Neural Network
```
Input Layer    Hidden Layer    Output Layer
   ○              ○──┐            ○
   ○──┐        ┌──○  │         ┌──○
   ○──┼────────┤  ○──┼─────────┤
   ○──┘        └──○  │         └──○
   ○              ○──┘
```

### Decision Boundary
```
   ┌─────────────────┐
   │ ████▓▓▓░░░░     │
   │ ████▓▓░░░░○○    │
   │ ███▓▓░░░○○○○    │
   │ ██▓▓░░××××      │
   │ █▓░░××××××      │
   └─────────────────┘
   Blue = Class 0
   Red  = Class 1
```

## 🚀 Usage

### Basic Usage

```python
from microlife.training import MetricsTracker, TrainingVisualizer

# Create tracker
tracker = MetricsTracker()

# Training loop
for episode in range(1000):
    loss = train_step()
    accuracy = evaluate()

    # Record metrics
    tracker.record(
        episode=episode,
        loss=loss,
        accuracy=accuracy,
        reward=total_reward
    )

    # Visualize every 10 episodes
    if episode % 10 == 0:
        visualizer.update(tracker.get_metrics())
```

### Advanced Usage

```python
from microlife.training import TrainingDashboard

# Create dashboard
dashboard = TrainingDashboard(
    window_size=(1920, 1080),
    update_interval=10,  # Update every 10 episodes
    smooth_window=20     # Moving average window
)

# Configure panels
dashboard.add_panel('loss', position=(0, 0), size=(600, 400))
dashboard.add_panel('network', position=(600, 0), size=(600, 400))
dashboard.add_panel('boundary', position=(0, 400), size=(1200, 680))

# Run training with dashboard
dashboard.train(training_function, episodes=1000)
```

## 📊 Metrics API

### Record Metrics

```python
tracker.record(
    episode=1,
    loss=0.5,
    accuracy=0.8,
    reward=15.0,
    learning_rate=0.001,
    gradient_norm=2.5
)
```

### Query Metrics

```python
# Get all metrics
metrics = tracker.get_metrics()

# Get specific metric
loss_history = tracker.get('loss')

# Get recent N values
recent_loss = tracker.get('loss', last=100)

# Get statistics
stats = tracker.get_statistics('loss')  # mean, std, min, max
```

### Export Metrics

```python
# Export to CSV
tracker.export_csv('training_metrics.csv')

# Export to JSON
tracker.export_json('training_metrics.json')

# Save checkpoint
tracker.save_checkpoint('checkpoint.pkl')
```

## 🎮 Integration with ModernGL

The training visualizer can integrate with the ModernGL renderer for unified visualization:

```python
class TrainingGLRenderer(GLRenderer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_viz = TrainingVisualizer(self.ctx, self.shader_manager)

    def render(self, time_elapsed, frame_time):
        # Render simulation
        super().render(time_elapsed, frame_time)

        # Render training metrics overlay
        self.training_viz.render(metrics)
```

## 🔍 Technical Details

### Smoothing Algorithms

**Exponential Moving Average (EMA):**
```python
smooth_value = alpha * new_value + (1 - alpha) * smooth_value
```

**Simple Moving Average (SMA):**
```python
smooth_value = sum(last_N_values) / N
```

### Graph Rendering

**ModernGL Mode:**
- GPU-accelerated line rendering
- Dynamic vertex buffers
- Real-time updates (60 FPS)

**Matplotlib Mode:**
- High-quality publication plots
- More chart types
- Slower updates (~10 FPS)

## 📈 Performance Targets

- **Update overhead**: <5ms per update
- **Render time**: <10ms per frame
- **Memory usage**: <100MB for 10k episodes
- **Export time**: <1s for CSV/JSON

## 🛠️ Configuration

```python
config = {
    'update_interval': 10,      # Update every N episodes
    'smooth_window': 20,        # Moving average window
    'max_history': 10000,       # Max stored episodes
    'auto_save': True,          # Auto-save checkpoints
    'save_interval': 100,       # Save every N episodes
    'graph_style': 'dark',      # 'dark' or 'light'
    'line_width': 2.0,          # Graph line width
    'font_size': 12,            # Text size
}
```

## 🎯 Use Cases

1. **Research** - Monitor training progress, compare algorithms
2. **Debugging** - Identify training issues (divergence, overfitting)
3. **Optimization** - Tune hyperparameters visually
4. **Demonstration** - Show AI learning in real-time
5. **Education** - Teach ML concepts with visualization

## 🔬 Advanced Features

### Gradient Visualization

```python
visualizer.add_gradient_histogram(layer_name='hidden1')
```

### Weight Distribution

```python
visualizer.add_weight_heatmap(layer_name='hidden1')
```

### Activation Maps

```python
visualizer.add_activation_map(layer_name='hidden1', input_sample=x)
```

### Attention Visualization

```python
visualizer.add_attention_map(attention_weights)
```

## 📚 References

- TensorBoard visualization
- Weights & Biases
- MLflow tracking
- PyTorch Lightning metrics

## 🎓 Best Practices

1. **Record frequently** - Every training step
2. **Visualize periodically** - Every N episodes (reduce overhead)
3. **Save checkpoints** - Regular backups
4. **Use smoothing** - Reduce noise in graphs
5. **Monitor multiple metrics** - Loss alone is not enough

---

**Next Steps:**
1. Implement MetricsTracker
2. Implement TrainingVisualizer
3. Implement NetworkVisualizer
4. Create TrainingDashboard
5. Build demo
