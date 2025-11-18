#!/usr/bin/env python3
"""
Complete Learning Simulation with Visualization Export

Runs a neuron learning simulation and exports data for C++ visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microlife'))

import numpy as np
from simulation.neuron import Neuron
from simulation.neuron_learning import NeuralLearningEnvironment, LearningTask
from simulation.visualization_export import VisualizationExporter

def create_xor_task():
    """
    XOR problem - classic non-linearly separable task

    Requires network to learn XOR function:
    [0, 0] → 0
    [0, 1] → 1
    [1, 0] → 1
    [1, 1] → 0
    """
    input_patterns = [
        np.array([0.0, 0.0, 1.0]),  # [A, B, bias]
        np.array([0.0, 1.0, 1.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
    ]

    target_outputs = [
        np.array([0.0]),  # 0 XOR 0 = 0
        np.array([1.0]),  # 0 XOR 1 = 1
        np.array([1.0]),  # 1 XOR 0 = 1
        np.array([0.0]),  # 1 XOR 1 = 0
    ]

    rewards = [1.0, 1.0, 1.0, 1.0]  # Equal reward for all correct

    return LearningTask(
        name="XOR Problem",
        input_patterns=input_patterns,
        target_outputs=target_outputs,
        rewards=rewards
    )

def main():
    print("\n" + "="*70)
    print("Neuron Learning with Visualization Export")
    print("="*70)

    # Create environment
    env = NeuralLearningEnvironment(width=300, height=300, depth=150)

    # Create neurons
    print("\nCreating neural network (40 neurons)...")
    print("  - Input layer: 3 neurons")
    print("  - Hidden layer: 30 neurons")
    print("  - Output layer: 7 neurons")

    for i in range(40):
        neuron_type = "pyramidal" if i % 4 != 0 else "interneuron"

        base_neuron = Neuron(
            x=np.random.uniform(50, env.width - 50),
            y=np.random.uniform(50, env.height - 50),
            z=np.random.uniform(20, env.depth - 20),
            neuron_type=neuron_type
        )
        base_neuron.stage = "mature"
        base_neuron.energy = 200.0  # High energy for learning
        env.add_neuron_with_dynamics(base_neuron)

    # Form initial connections
    print("\nForming initial synaptic connections...")
    connection_count = 0
    for _ in range(150):
        pre = np.random.choice(env.neurons)
        post = np.random.choice(env.neurons)
        if (pre.neuron.id != post.neuron.id and
            post.neuron.id not in pre.neuron.synapses_out):
            pre.neuron.connect_to_neuron(post.neuron, initial_weight=np.random.uniform(0.3, 0.5))
            connection_count += 1

    print(f"✓ Created {connection_count} initial synapses")

    # Add learning task
    task = create_xor_task()
    env.add_learning_task(task)

    print(f"\nLearning task: {task.name}")
    print("Training for 1000 timesteps...")

    # Setup visualization export
    exporter = VisualizationExporter()

    # Training loop with periodic export
    export_interval = 50  # Export every 50 steps

    for step in range(1000):
        env.update()

        # Export frame for visualization
        if step % export_interval == 0:
            exporter.capture_frame(env, step // export_interval)

        # Print progress
        if (step + 1) % 100 == 0:
            stats = env.get_statistics()
            print(f"Step {step+1}/1000: "
                  f"Accuracy={stats['recent_accuracy']*100:.1f}%, "
                  f"Synapses={stats['total_synapses']}, "
                  f"AvgFiring={stats['avg_firing_rate']:.1f} Hz")

    # Final statistics
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    env.print_learning_stats()

    # Export all data
    print("\n" + "="*70)
    print("Exporting Visualization Data")
    print("="*70)

    # Export animation
    animation_file = exporter.save_animation("neuron_animation.json")

    # Export final state
    state_file = exporter.save_current_state(env, "neuron_state.json")

    # Export learning curves
    curves_file = exporter.export_learning_curves(env, "learning_curves.csv")

    print("\n" + "="*70)
    print("Visualization Files Created")
    print("="*70)
    print(f"1. Animation:      {animation_file}")
    print(f"2. Final state:    {state_file}")
    print(f"3. Learning data:  {curves_file}")
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("\n1. Build C++ visualizer:")
    print("   chmod +x build_visualization.sh")
    print("   ./build_visualization.sh")
    print("\n2. Run visualization:")
    print("   ./build/neuron_visualizer")
    print("\n3. View learning curves:")
    print("   python -c \"import pandas as pd; import matplotlib.pyplot as plt;")
    print("   df = pd.read_csv('visualization_data/learning_curves.csv');")
    print("   df.plot(x='Time'); plt.show()\"")
    print("\n" + "="*70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
