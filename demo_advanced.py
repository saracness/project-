"""
Advanced Demo - Phase 5
Showcases all advanced features:
- AI Training Visualization
- Advanced Rendering (trails, particles, heatmap, minimap)
- GPU Acceleration
- Real-time metrics
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import random
import matplotlib.pyplot as plt

# Core simulation
from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species

# GPU Brains
from microlife.ml.brain_gpu import GPUDQNBrain, GPUDoubleDQNBrain, GPUCNNBrain

# CPU Brains (for comparison)
from microlife.ml.brain_rl import QLearningBrain, DQNBrain

# Visualization
from microlife.visualization.advanced_renderer import AdvancedRenderer
from microlife.visualization.ai_metrics import AIMetricsTracker
from microlife.visualization.training_visualizer import TrainingVisualizer

# Configuration
from microlife.config import SimulationConfig, get_auto_config

print("=" * 70)
print("🚀 MICRO-LIFE ADVANCED DEMO - PHASE 5")
print("=" * 70)

# Configuration selection
print("\nKonfigürasyon Seç:")
print("1. 🎨 Quality (Tüm efektler, orta hız)")
print("2. ⚡ Performance (Minimal efektler, yüksek hız)")
print("3. ⚙️  Balanced (Dengeli)")
print("4. 🖥️  CPU Only (GPU yok)")
print("5. 🤖 Auto (Otomatik donanım tespiti)")

choice = input("\nSeçim (1-5) [Enter=Auto]: ").strip()

if choice == '1':
    from microlife.config import get_quality_config
    config = get_quality_config()
elif choice == '2':
    from microlife.config import get_performance_config
    config = get_performance_config()
elif choice == '3':
    from microlife.config import get_balanced_config
    config = get_balanced_config()
elif choice == '4':
    from microlife.config import get_cpu_config
    config = get_cpu_config()
else:
    config = get_auto_config()

print("\n" + config.get_info())

# Create environment
env_width = 800
env_height = 600

env = Environment(
    width=env_width,
    height=env_height,
    use_intelligent_movement=True
)

print(f"\n✅ Environment created: {env_width}x{env_height}")

# Add food
num_food = min(config.max_food, 100)
for _ in range(num_food):
    env.add_food(
        x=random.uniform(50, env_width - 50),
        y=random.uniform(50, env_height - 50),
        energy=25
    )

print(f"✅ Added {num_food} food particles")

# Create AI metrics tracker
metrics_tracker = AIMetricsTracker(window_size=100)

# Add organisms with different AI types
print("\n🧠 Adding AI organisms:")

species_list = ['Euglena', 'Paramecium', 'Amoeba', 'Spirillum', 'Stentor']

# GPU brains (if available)
if config.use_gpu:
    print("\n🎮 GPU Brains:")

    # GPU-DQN
    for i in range(2):
        morph = get_species(random.choice(species_list))
        org = Organism(
            x=random.uniform(100, 300),
            y=random.uniform(100, 300),
            energy=150,
            morphology=morph
        )
        org.brain = GPUDQNBrain(device=config.device, batch_size=config.batch_size)
        env.add_organism(org)
        metrics_tracker.register_organism(id(org), 'GPU-DQN')
        print(f"  ✓ {morph.species_name} + GPU-DQN")

    # GPU-DoubleDQN
    for i in range(2):
        morph = get_species(random.choice(species_list))
        org = Organism(
            x=random.uniform(300, 500),
            y=random.uniform(100, 300),
            energy=150,
            morphology=morph
        )
        org.brain = GPUDoubleDQNBrain(device=config.device, batch_size=config.batch_size)
        env.add_organism(org)
        metrics_tracker.register_organism(id(org), 'GPU-Double-DQN')
        print(f"  ✓ {morph.species_name} + GPU-DoubleDQN")

    # GPU-CNN
    for i in range(2):
        morph = get_species(random.choice(species_list))
        org = Organism(
            x=random.uniform(500, 700),
            y=random.uniform(100, 300),
            energy=150,
            morphology=morph
        )
        org.brain = GPUCNNBrain(device=config.device, batch_size=config.batch_size)
        env.add_organism(org)
        metrics_tracker.register_organism(id(org), 'GPU-CNN')
        print(f"  ✓ {morph.species_name} + GPU-CNN")

# CPU brains (for comparison)
print("\n💻 CPU Brains (Comparison):")

# Q-Learning
for i in range(2):
    morph = get_species(random.choice(species_list))
    org = Organism(
        x=random.uniform(100, 300),
        y=random.uniform(300, 500),
        energy=150,
        morphology=morph
    )
    org.brain = QLearningBrain()
    env.add_organism(org)
    metrics_tracker.register_organism(id(org), 'Q-Learning')
    print(f"  ✓ {morph.species_name} + Q-Learning (CPU)")

# CPU DQN
for i in range(2):
    morph = get_species(random.choice(species_list))
    org = Organism(
        x=random.uniform(300, 500),
        y=random.uniform(300, 500),
        energy=150,
        morphology=morph
    )
    org.brain = DQNBrain()
    env.add_organism(org)
    metrics_tracker.register_organism(id(org), 'DQN')
    print(f"  ✓ {morph.species_name} + DQN (CPU)")

# No AI (control group)
print("\n🧬 Control Group (No AI):")
for i in range(3):
    morph = get_species(random.choice(species_list))
    org = Organism(
        x=random.uniform(500, 700),
        y=random.uniform(300, 500),
        energy=150,
        morphology=morph
    )
    env.add_organism(org)
    print(f"  ✓ {morph.species_name} (No AI)")

total_organisms = len(env.organisms)
ai_organisms = sum(1 for o in env.organisms if hasattr(o, 'brain') and o.brain)
print(f"\n📊 Total: {total_organisms} organisms ({ai_organisms} with AI)")

# Create advanced renderer
print("\n🎨 Initializing Advanced Renderer...")
renderer = AdvancedRenderer(env, config)
print(f"  ✓ Trails: {'ON' if config.enable_trails else 'OFF'}")
print(f"  ✓ Particles: {'ON' if config.enable_particles else 'OFF'}")
print(f"  ✓ Heatmap: {'ON' if config.enable_heatmap else 'OFF'}")
print(f"  ✓ MiniMap: {'ON' if config.enable_minimap else 'OFF'}")

# Create training visualizer
if config.enable_ai_metrics:
    print("\n📈 Initializing Training Visualizer...")
    training_viz = TrainingVisualizer(metrics_tracker, update_interval=20)
    training_viz.initialize()
    print("  ✓ Training graphs enabled")

# Simulation loop
print("\n▶️  Starting simulation...")
print("  Temel Kontroller:")
print("    Q: Quit")
print("    SPACE: Pause")
print("    T: Toggle Trails")
print("    P: Toggle Particles")
print("    H: Toggle Heatmap")
print("    M: Toggle MiniMap")
print("    S: Save screenshot")
print("\n" + "=" * 70)

running = True
paused = False
max_timesteps = config.max_timesteps or 1000

def on_key_press(event):
    global running, paused

    if event.key == 'q':
        running = False
    elif event.key == ' ':
        paused = not paused
        print(f"⏸️  {'Paused' if paused else 'Resumed'}")
    elif event.key == 't':
        renderer.toggle_trails()
    elif event.key == 'p':
        renderer.toggle_particles()
    elif event.key == 'h':
        renderer.toggle_heatmap()
    elif event.key == 'm':
        renderer.toggle_minimap()
    elif event.key == 's':
        # Save screenshot
        filename = f'microlife_advanced_t{env.timestep}.png'
        renderer.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📸 Screenshot saved: {filename}")

# Connect keyboard handler
renderer.fig.canvas.mpl_connect('key_press_event', on_key_press)

plt.ion()
plt.show()

try:
    while running and env.timestep < max_timesteps:
        if not paused:
            # Update environment
            env.update()

            # Track AI metrics
            if config.enable_ai_metrics:
                metrics_tracker.update_timestep(env.timestep)
                for org in env.organisms:
                    if org.alive and hasattr(org, 'brain') and org.brain:
                        metrics_tracker.record(id(org), org.brain, env.timestep)

            # Spawn new food periodically
            if env.timestep % 20 == 0 and len(env.food_particles) < num_food:
                env.add_food(
                    x=random.uniform(50, env_width - 50),
                    y=random.uniform(50, env_height - 50),
                    energy=25
                )

            # Update training visualizer
            if config.enable_ai_metrics and env.timestep % config.metrics_update_interval == 0:
                training_viz.update(env.timestep)

        # Render frame
        renderer.render_frame()
        plt.pause(0.001)

        # Progress update
        if env.timestep % 100 == 0:
            alive = len([o for o in env.organisms if o.alive])
            perf = renderer.get_performance_stats()
            print(f"T={env.timestep:4d} | Alive: {alive:2d} | FPS: {perf['fps']:.1f} | "
                  f"Trails: {perf['trail_count']:2d} | Particles: {perf['particle_count']:3d}")

except KeyboardInterrupt:
    print("\n\n⏹️  Simulation interrupted by user")

# Final stats
print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)

print(f"\nFinal Timestep: {env.timestep}")
print(f"Final Alive: {len([o for o in env.organisms if o.alive])}/{total_organisms}")

if config.enable_ai_metrics:
    print("\n" + metrics_tracker.get_summary())

    # Save training metrics
    training_viz.save('training_metrics_advanced.png')

# Save final screenshot
renderer.fig.savefig('microlife_advanced_final.png', dpi=150, bbox_inches='tight')
print("\n📸 Final screenshot saved: microlife_advanced_final.png")

# Performance stats
perf = renderer.get_performance_stats()
print(f"\n⚡ Performance Statistics:")
print(f"  Average FPS: {perf['fps']:.1f}")
print(f"  Avg Render Time: {perf['avg_render_time_ms']:.2f} ms")
print(f"  Total Frames: {perf['frame_count']}")

print("\n✅ Demo complete!")
print("=" * 70)

plt.ioff()
plt.show()
