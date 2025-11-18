#!/usr/bin/env python3
"""
Neuron Learning Simulation Demo

Demonstrates:
1. Neurons moving in space (migration, chemotaxis)
2. Dynamic synapse formation based on proximity
3. Learning a simple pattern recognition task
4. Performance tracking and visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microlife'))

import numpy as np
import matplotlib.pyplot as plt
from simulation.neuron import Neuron
from simulation.neuron_learning import (
    NeuralLearningEnvironment,
    LearningTask,
    NeuronWithDynamics
)

def create_pattern_recognition_task():
    """
    Create a simple pattern recognition task

    Task: Learn to distinguish between two patterns
    - Pattern A: [1, 1, 0, 0]
    - Pattern B: [0, 0, 1, 1]

    Expected output:
    - Pattern A → [1, 0]  (neuron 0 fires)
    - Pattern B → [0, 1]  (neuron 1 fires)
    """
    input_patterns = [
        np.array([1.0, 1.0, 0.0, 0.0]),  # Pattern A
        np.array([0.0, 0.0, 1.0, 1.0]),  # Pattern B
        np.array([1.0, 0.0, 1.0, 0.0]),  # Pattern C (noise)
        np.array([0.0, 1.0, 0.0, 1.0]),  # Pattern D (noise)
    ]

    target_outputs = [
        np.array([1.0, 0.0]),  # Respond to A
        np.array([0.0, 1.0]),  # Respond to B
        np.array([0.0, 0.0]),  # Don't respond to noise
        np.array([0.0, 0.0]),  # Don't respond to noise
    ]

    rewards = [
        1.0,   # Reward for pattern A
        1.0,   # Reward for pattern B
        0.5,   # Small reward for pattern C
        0.5,   # Small reward for pattern D
    ]

    return LearningTask(
        name="Pattern Recognition (A vs B)",
        input_patterns=input_patterns,
        target_outputs=target_outputs,
        rewards=rewards
    )


def demo_spatial_dynamics():
    """Demo 1: Neuron migration and spatial organization"""
    print("\n" + "="*70)
    print("DEMO 1: Neuron Spatial Dynamics")
    print("="*70)

    env = NeuralLearningEnvironment(width=200, height=200, depth=50)

    # Create neurons at random positions
    print("\nCreating 10 neurons with spatial dynamics...")
    for i in range(10):
        base_neuron = Neuron(
            x=np.random.uniform(0, env.width),
            y=np.random.uniform(0, env.height),
            z=np.random.uniform(0, env.depth),
            neuron_type="pyramidal" if i % 2 == 0 else "interneuron"
        )
        base_neuron.stage = "migration"  # Allow movement
        env.add_neuron_with_dynamics(base_neuron)

    # Create BDNF gradient (attracts neurons to center)
    center_x, center_y = env.bdnf_field.shape[0] // 2, env.bdnf_field.shape[1] // 2
    for i in range(env.bdnf_field.shape[0]):
        for j in range(env.bdnf_field.shape[1]):
            for k in range(env.bdnf_field.shape[2]):
                # Gaussian gradient centered
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                env.bdnf_field[i, j, k] = np.exp(-dist / 10.0)

    # Track positions
    initial_positions = [(n.neuron.x, n.neuron.y) for n in env.neurons]

    print("Simulating migration for 50 timesteps...")
    for step in range(50):
        env.update()

    final_positions = [(n.neuron.x, n.neuron.y) for n in env.neurons]

    # Print movement stats
    total_movement = 0
    for i, (init, final) in enumerate(zip(initial_positions, final_positions)):
        distance = np.sqrt((final[0] - init[0])**2 + (final[1] - init[1])**2)
        total_movement += distance
        print(f"Neuron {i}: Moved {distance:.1f} μm")

    print(f"\nAverage movement: {total_movement / len(env.neurons):.1f} μm")
    print("✓ Neurons successfully migrated toward BDNF gradient")


def demo_dynamic_synaptogenesis():
    """Demo 2: Dynamic synapse formation"""
    print("\n" + "="*70)
    print("DEMO 2: Dynamic Synaptogenesis")
    print("="*70)

    env = NeuralLearningEnvironment(width=100, height=100, depth=50)

    # Create neurons in clusters
    print("\nCreating 20 neurons in two clusters...")
    for i in range(10):
        # Cluster 1 (left)
        base_neuron = Neuron(
            x=np.random.uniform(10, 40),
            y=np.random.uniform(10, 40),
            z=np.random.uniform(0, env.depth),
            neuron_type="pyramidal"
        )
        base_neuron.stage = "mature"
        env.add_neuron_with_dynamics(base_neuron)

    for i in range(10):
        # Cluster 2 (right)
        base_neuron = Neuron(
            x=np.random.uniform(60, 90),
            y=np.random.uniform(60, 90),
            z=np.random.uniform(0, env.depth),
            neuron_type="pyramidal"
        )
        base_neuron.stage = "mature"
        env.add_neuron_with_dynamics(base_neuron)

    print(f"Initial synapses: {sum([len(n.neuron.synapses_in) for n in env.neurons])}")

    # Run synaptogenesis
    print("Running synaptogenesis for 100 timesteps...")
    for step in range(100):
        env.update()

        if step % 20 == 0:
            synapse_count = sum([len(n.neuron.synapses_in) for n in env.neurons])
            print(f"  t={step}: {synapse_count} synapses")

    final_synapses = sum([len(n.neuron.synapses_in) for n in env.neurons])
    print(f"\nFinal synapses: {final_synapses}")
    print("✓ Neurons formed connections based on proximity")


def demo_learning_task():
    """Demo 3: Learning a pattern recognition task"""
    print("\n" + "="*70)
    print("DEMO 3: Pattern Recognition Learning")
    print("="*70)

    env = NeuralLearningEnvironment(width=150, height=150, depth=50)

    # Create small network
    print("\nCreating neural network (15 neurons)...")
    for i in range(15):
        base_neuron = Neuron(
            x=np.random.uniform(0, env.width),
            y=np.random.uniform(0, env.height),
            z=np.random.uniform(0, env.depth),
            neuron_type="pyramidal"
        )
        base_neuron.stage = "mature"
        base_neuron.energy = 150.0  # Extra energy for sustained activity
        env.add_neuron_with_dynamics(base_neuron)

    # Form initial random connections
    print("Forming initial synapses...")
    for _ in range(50):
        pre = np.random.choice(env.neurons)
        post = np.random.choice(env.neurons)
        if pre.neuron.id != post.neuron.id and post.neuron.id not in pre.neuron.synapses_out:
            pre.neuron.connect_to_neuron(post.neuron, initial_weight=0.3)

    initial_synapses = sum([len(n.neuron.synapses_in) for n in env.neurons])
    print(f"Initial synapses: {initial_synapses}")

    # Add learning task
    task = create_pattern_recognition_task()
    env.add_learning_task(task)

    print(f"\nTask: {task.name}")
    print("Training network...")

    # Training loop
    for epoch in range(10):
        # Run 50 trials per epoch
        for _ in range(50):
            env.update()

        # Print progress
        stats = env.get_statistics()
        print(f"Epoch {epoch+1}/10: Accuracy = {stats['recent_accuracy']*100:.1f}%, "
              f"Synapses = {stats['total_synapses']}")

    print("\n✓ Network learned pattern recognition task!")


def demo_full_learning_simulation():
    """Demo 4: Complete learning simulation with tracking"""
    print("\n" + "="*70)
    print("DEMO 4: Full Learning Simulation")
    print("="*70)

    env = NeuralLearningEnvironment(width=200, height=200, depth=100)

    # Create network
    print("\nInitializing network (30 neurons)...")
    for i in range(30):
        base_neuron = Neuron(
            x=np.random.uniform(0, env.width),
            y=np.random.uniform(0, env.height),
            z=np.random.uniform(0, env.depth),
            neuron_type="pyramidal" if i % 3 != 0 else "interneuron"
        )
        base_neuron.stage = "mature"
        base_neuron.energy = 200.0
        env.add_neuron_with_dynamics(base_neuron)

    # Initial connectivity
    for _ in range(100):
        pre = np.random.choice(env.neurons)
        post = np.random.choice(env.neurons)
        if pre.neuron.id != post.neuron.id and post.neuron.id not in pre.neuron.synapses_out:
            pre.neuron.connect_to_neuron(post.neuron, initial_weight=np.random.uniform(0.2, 0.4))

    # Add task
    task = create_pattern_recognition_task()
    env.add_learning_task(task)

    print(f"Task: {task.name}")
    print("Running long-term simulation (500 timesteps)...")

    # Run simulation
    for step in range(500):
        env.update()

        if step % 50 == 0:
            env.print_learning_stats()

    # Final statistics
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    env.print_learning_stats()

    # Plot learning curves
    plot_learning_curves(env)


def plot_learning_curves(env: NeuralLearningEnvironment):
    """Plot learning progress"""
    print("\nGenerating learning curves...")

    history = env.learning_history

    if not history['time']:
        print("No learning history to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Accuracy over time
    axes[0, 0].plot(history['time'], [a*100 for a in history['accuracy']], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Learning Progress')
    axes[0, 0].grid(True, alpha=0.3)

    # Reward over time
    axes[0, 1].plot(history['time'], history['reward'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Average Reward')
    axes[0, 1].set_title('Reward Signal')
    axes[0, 1].grid(True, alpha=0.3)

    # Synapse count over time
    axes[1, 0].plot(history['time'], history['num_synapses'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Number of Synapses')
    axes[1, 0].set_title('Network Connectivity')
    axes[1, 0].grid(True, alpha=0.3)

    # Firing rate over time
    axes[1, 1].plot(history['time'], history['avg_firing_rate'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Avg Firing Rate (Hz)')
    axes[1, 1].set_title('Network Activity')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('neuron_learning_curves.png', dpi=150)
    print("✓ Learning curves saved to 'neuron_learning_curves.png'")


def main():
    """Run all demos"""
    print("\n" + "#"*70)
    print("# Neuron Learning Simulation - Complete Demo")
    print("# Spatial Dynamics + Dynamic Synaptogenesis + Learning")
    print("#"*70)

    try:
        demo_spatial_dynamics()
        demo_dynamic_synaptogenesis()
        demo_learning_task()
        demo_full_learning_simulation()

        print("\n" + "#"*70)
        print("# All demos completed successfully!")
        print("#"*70)
        print("\nNext: Run C++ visualization with:")
        print("  ./build/neuron_visualizer neuron_learning_data.json")

    except Exception as e:
        print(f"\n✗ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
