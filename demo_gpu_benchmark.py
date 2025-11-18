"""
GPU Benchmark Demo
Compares GPU vs CPU performance with varying organism counts
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch

# Core simulation
from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species

# Brains
from microlife.ml.brain_gpu import GPUDQNBrain, GPUDoubleDQNBrain
from microlife.ml.brain_rl import DQNBrain

# Configuration
from microlife.config import SimulationConfig

print("=" * 70)
print("⚡ GPU PERFORMANCE BENCHMARK")
print("=" * 70)

# Check GPU availability
print("\n🖥️  Hardware Detection:")
if torch.cuda.is_available():
    print(f"  ✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"  📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    use_gpu = True
else:
    print("  ❌ CUDA not available - CPU only mode")
    use_gpu = False

# Benchmark parameters
organism_counts = [10, 25, 50, 100, 200, 500] if use_gpu else [10, 25, 50, 100]
timesteps_per_test = 100

print(f"\n⚙️  Benchmark Configuration:")
print(f"  Organism counts: {organism_counts}")
print(f"  Timesteps per test: {timesteps_per_test}")

# Results storage
results = {
    'counts': organism_counts,
    'gpu_dqn': [],
    'cpu_dqn': [],
    'gpu_double_dqn': [],
}

def run_benchmark(brain_type, organism_count, use_gpu_device=False):
    """
    Run benchmark for a specific configuration.

    Returns:
        avg_time_per_step (ms), fps
    """
    # Create environment
    env = Environment(width=800, height=600, use_intelligent_movement=True)

    # Add food
    for _ in range(50):
        env.add_food(x=random.uniform(50, 750), y=random.uniform(50, 550), energy=25)

    # Add organisms
    species_list = ['Euglena', 'Paramecium', 'Amoeba', 'Spirillum']

    for i in range(organism_count):
        morph = get_species(random.choice(species_list))
        org = Organism(
            x=random.uniform(100, 700),
            y=random.uniform(100, 500),
            energy=150,
            morphology=morph
        )

        # Assign brain
        if brain_type == 'GPU-DQN' and use_gpu_device:
            org.brain = GPUDQNBrain(device='cuda', batch_size=32)
        elif brain_type == 'GPU-DoubleDQN' and use_gpu_device:
            org.brain = GPUDoubleDQNBrain(device='cuda', batch_size=32)
        elif brain_type == 'CPU-DQN':
            org.brain = DQNBrain()

        env.add_organism(org)

    # Warm-up
    for _ in range(10):
        env.update()

    # Benchmark
    start_time = time.time()

    for step in range(timesteps_per_test):
        env.update()

        # Spawn food occasionally
        if step % 20 == 0:
            env.add_food(x=random.uniform(50, 750), y=random.uniform(50, 550), energy=25)

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = (total_time / timesteps_per_test) * 1000  # ms
    fps = timesteps_per_test / total_time

    return avg_time_per_step, fps

# Run benchmarks
print("\n" + "=" * 70)
print("RUNNING BENCHMARKS")
print("=" * 70)

for count in organism_counts:
    print(f"\n🧬 Testing with {count} organisms:")

    # GPU-DQN
    if use_gpu:
        print(f"  🎮 GPU-DQN...", end='', flush=True)
        avg_time, fps = run_benchmark('GPU-DQN', count, use_gpu_device=True)
        results['gpu_dqn'].append((avg_time, fps))
        print(f" ✓ {avg_time:.2f} ms/step ({fps:.1f} FPS)")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # CPU-DQN
    print(f"  💻 CPU-DQN...", end='', flush=True)
    avg_time, fps = run_benchmark('CPU-DQN', count, use_gpu_device=False)
    results['cpu_dqn'].append((avg_time, fps))
    print(f" ✓ {avg_time:.2f} ms/step ({fps:.1f} FPS)")

    # GPU-DoubleDQN
    if use_gpu:
        print(f"  🎮 GPU-DoubleDQN...", end='', flush=True)
        avg_time, fps = run_benchmark('GPU-DoubleDQN', count, use_gpu_device=True)
        results['gpu_double_dqn'].append((avg_time, fps))
        print(f" ✓ {avg_time:.2f} ms/step ({fps:.1f} FPS)")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate speedup
    if use_gpu:
        speedup_dqn = results['cpu_dqn'][-1][1] / results['gpu_dqn'][-1][1]
        speedup_double = results['cpu_dqn'][-1][1] / results['gpu_double_dqn'][-1][1]
        print(f"  ⚡ GPU Speedup: DQN={speedup_dqn:.2f}x, DoubleDQN={speedup_double:.2f}x")

# Visualization
print("\n" + "=" * 70)
print("GENERATING RESULTS")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('GPU vs CPU Performance Benchmark', fontsize=16, fontweight='bold')

# Plot 1: Time per step
ax1.set_title('Average Time per Step', fontweight='bold')
ax1.set_xlabel('Number of Organisms')
ax1.set_ylabel('Time (ms)')
ax1.grid(True, alpha=0.3)

cpu_times = [r[0] for r in results['cpu_dqn']]
ax1.plot(organism_counts, cpu_times, marker='o', label='CPU-DQN', linewidth=2, color='#E74C3C')

if use_gpu:
    gpu_dqn_times = [r[0] for r in results['gpu_dqn']]
    gpu_double_times = [r[0] for r in results['gpu_double_dqn']]
    ax1.plot(organism_counts, gpu_dqn_times, marker='s', label='GPU-DQN', linewidth=2, color='#3498DB')
    ax1.plot(organism_counts, gpu_double_times, marker='^', label='GPU-DoubleDQN', linewidth=2, color='#9B59B6')

ax1.legend()
ax1.set_yscale('log')

# Plot 2: FPS
ax2.set_title('Frames Per Second (Higher is Better)', fontweight='bold')
ax2.set_xlabel('Number of Organisms')
ax2.set_ylabel('FPS')
ax2.grid(True, alpha=0.3)

cpu_fps = [r[1] for r in results['cpu_dqn']]
ax2.plot(organism_counts, cpu_fps, marker='o', label='CPU-DQN', linewidth=2, color='#E74C3C')

if use_gpu:
    gpu_dqn_fps = [r[1] for r in results['gpu_dqn']]
    gpu_double_fps = [r[1] for r in results['gpu_double_dqn']]
    ax2.plot(organism_counts, gpu_dqn_fps, marker='s', label='GPU-DQN', linewidth=2, color='#3498DB')
    ax2.plot(organism_counts, gpu_double_fps, marker='^', label='GPU-DoubleDQN', linewidth=2, color='#9B59B6')

ax2.legend()

plt.tight_layout()
plt.savefig('gpu_benchmark_results.png', dpi=150, bbox_inches='tight')
print("📊 Results saved: gpu_benchmark_results.png")

# Summary table
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print("\n{:<15} {:<12} {:<12} {:<12}".format("Organisms", "CPU-DQN", "GPU-DQN", "Speedup"))
print("-" * 55)

for i, count in enumerate(organism_counts):
    cpu_fps_val = results['cpu_dqn'][i][1]

    if use_gpu:
        gpu_fps_val = results['gpu_dqn'][i][1]
        speedup = gpu_fps_val / cpu_fps_val
        print("{:<15} {:<12.1f} {:<12.1f} {:<12.2f}x".format(
            count, cpu_fps_val, gpu_fps_val, speedup
        ))
    else:
        print("{:<15} {:<12.1f} {:<12} {:<12}".format(
            count, cpu_fps_val, "N/A", "N/A"
        ))

# Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

if use_gpu:
    avg_speedup = np.mean([
        results['gpu_dqn'][i][1] / results['cpu_dqn'][i][1]
        for i in range(len(organism_counts))
    ])

    print(f"\n✨ Average GPU Speedup: {avg_speedup:.2f}x")

    if avg_speedup >= 3.0:
        print("\n🚀 EXCELLENT GPU Performance!")
        print("   Recommended: Use GPU for simulations with 100+ organisms")
    elif avg_speedup >= 2.0:
        print("\n⚡ GOOD GPU Performance!")
        print("   Recommended: Use GPU for simulations with 50+ organisms")
    elif avg_speedup >= 1.5:
        print("\n👍 MODERATE GPU Performance")
        print("   Recommended: Use GPU for simulations with 200+ organisms")
    else:
        print("\n💻 Limited GPU Benefit")
        print("   CPU may be sufficient for most use cases")

    # Memory recommendation
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem_gb >= 6:
            print(f"\n💾 GPU Memory: {gpu_mem_gb:.1f} GB - Can handle 1000+ organisms")
        elif gpu_mem_gb >= 4:
            print(f"\n💾 GPU Memory: {gpu_mem_gb:.1f} GB - Can handle 500-1000 organisms")
        else:
            print(f"\n💾 GPU Memory: {gpu_mem_gb:.1f} GB - Recommended max: 500 organisms")

else:
    print("\n💻 CPU-only mode")
    print("   Install CUDA-enabled PyTorch for GPU acceleration")
    print("   Visit: https://pytorch.org/get-started/locally/")

print("\n" + "=" * 70)
print("✅ Benchmark Complete!")
print("=" * 70)

plt.show()
