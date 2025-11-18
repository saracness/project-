#!/usr/bin/env python3
"""
Neuron Personalities Demonstration

Showcases different neuron personality types and their unique behaviors
without modifying any existing code.

Demonstrates:
1. All 9 personality types with distinct firing patterns
2. Special functions (place fields, reward coding, grid cells, mirror neurons)
3. Integration with existing learning environment
4. Visualization export showing personality-specific behaviors
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microlife'))

import numpy as np
# Configure matplotlib for non-GUI environment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulation.neuron import Neuron
from simulation.neuron_personalities import (
    create_personalized_neuron,
    NEURON_PERSONALITIES,
    PersonalizedNeuron
)


def demo_1_basic_personalities():
    """
    Demo 1: Show all personality types and their basic characteristics
    """
    print("\n" + "="*70)
    print("DEMO 1: Basic Neuron Personalities")
    print("="*70)

    personalities = []

    for personality_name, personality_traits in NEURON_PERSONALITIES.items():
        print(f"\n{personality_traits.name}")
        print("-" * 70)

        # Create base neuron
        base_neuron = Neuron(
            x=100.0, y=100.0, z=50.0,
            neuron_type="pyramidal"
        )
        base_neuron.stage = "mature"
        base_neuron.energy = 200.0

        # Add personality
        personalized = create_personalized_neuron(base_neuron, personality_name)
        personalities.append(personalized)

        # Print description
        print(f"  Role: {personality_traits.role.value}")
        print(f"  Firing pattern: {personality_traits.firing_pattern.value}")
        print(f"  Baseline rate: {personality_traits.baseline_firing_rate:.1f} Hz")
        print(f"  Max rate: {personality_traits.max_firing_rate:.1f} Hz")
        print(f"  Special functions:")
        if personality_traits.special_functions:
            for func_name in personality_traits.special_functions.keys():
                print(f"    - {func_name}")
        else:
            print("    - None")

    return personalities


def demo_2_firing_patterns():
    """
    Demo 2: Compare firing patterns across personality types
    """
    print("\n" + "="*70)
    print("DEMO 2: Firing Pattern Comparison")
    print("="*70)

    # Create neurons with different personalities
    personalities_to_test = [
        "dopaminergic",      # Burst firing
        "serotonergic",      # Slow regular
        "fast_spiking",      # Fast spiking
        "chattering",        # High-frequency bursts
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, personality_name in enumerate(personalities_to_test):
        print(f"\nTesting {personality_name}...")

        # Create neuron
        base_neuron = Neuron(x=100, y=100, z=50, neuron_type="pyramidal")
        base_neuron.stage = "mature"
        base_neuron.energy = 200.0

        personalized = create_personalized_neuron(base_neuron, personality_name)

        # Simulate for 1 second
        time_points = []
        firing_rates = []
        membrane_potentials = []

        dt = 0.001  # 1 ms timesteps
        time = 0.0

        for step in range(1000):
            personalized.update(dt, time)

            time_points.append(time)
            firing_rates.append(personalized.neuron.firing_rate)
            membrane_potentials.append(personalized.neuron.membrane_potential)

            time += dt

        # Plot
        ax = axes[idx]
        ax.plot(time_points, firing_rates, linewidth=1.5)
        ax.set_title(f"{NEURON_PERSONALITIES[personality_name].name}", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.grid(True, alpha=0.3)

        avg_rate = np.mean(firing_rates)
        max_rate = np.max(firing_rates)
        print(f"  Average firing rate: {avg_rate:.1f} Hz")
        print(f"  Maximum firing rate: {max_rate:.1f} Hz")

    plt.tight_layout()
    plt.savefig("neuron_personality_firing_patterns.png", dpi=150)
    print(f"\n✓ Saved firing pattern comparison to neuron_personality_firing_patterns.png")
    plt.close()


def demo_3_place_cells():
    """
    Demo 3: Demonstrate place cell spatial encoding
    """
    print("\n" + "="*70)
    print("DEMO 3: Place Cell Spatial Encoding")
    print("="*70)

    # Create multiple place cells
    num_cells = 5
    place_cells = []

    for i in range(num_cells):
        base_neuron = Neuron(x=100, y=100, z=50, neuron_type="pyramidal")
        base_neuron.stage = "mature"
        base_neuron.energy = 200.0

        place_cell = create_personalized_neuron(base_neuron, "place_cell")
        place_cells.append(place_cell)

    # Simulate movement through space
    print(f"\nCreated {num_cells} place cells")
    print("Simulating movement through 2D space...")

    # Create grid of positions
    x_range = np.linspace(0, 300, 50)
    y_range = np.linspace(0, 300, 50)

    # Initialize place field centers randomly
    for pc in place_cells:
        pc.place_field_center = [
            np.random.uniform(50, 250),
            np.random.uniform(50, 250),
            50.0
        ]

    # Create firing rate maps
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, place_cell in enumerate(place_cells):
        firing_map = np.zeros((len(y_range), len(x_range)))

        for i, y in enumerate(y_range):
            for j, x in enumerate(x_range):
                context = {'position': [x, y, 50.0]}
                place_cell._update_place_cell(context)
                firing_map[i, j] = place_cell.neuron.firing_rate

        # Plot
        ax = axes[idx]
        im = ax.imshow(firing_map, extent=[0, 300, 0, 300], origin='lower', cmap='hot')
        ax.set_title(f"Place Cell {idx+1}", fontsize=10)
        ax.set_xlabel("X position (μm)")
        ax.set_ylabel("Y position (μm)")

        # Mark place field center
        center = place_cell.place_field_center
        ax.plot(center[0], center[1], 'b*', markersize=15, label='Field center')
        ax.legend()

        plt.colorbar(im, ax=ax, label='Firing rate (Hz)')

    # Remove extra subplot
    axes[-1].remove()

    plt.tight_layout()
    plt.savefig("place_cell_fields.png", dpi=150)
    print(f"\n✓ Saved place cell fields to place_cell_fields.png")
    plt.close()


def demo_4_reward_coding():
    """
    Demo 4: Demonstrate dopaminergic reward prediction error coding
    """
    print("\n" + "="*70)
    print("DEMO 4: Dopaminergic Reward Prediction Error")
    print("="*70)

    # Create dopaminergic neuron
    base_neuron = Neuron(x=100, y=100, z=50, neuron_type="pyramidal")
    base_neuron.stage = "mature"
    base_neuron.energy = 200.0

    da_neuron = create_personalized_neuron(base_neuron, "dopaminergic")

    print("\nSimulating reward prediction scenarios...")

    # Scenario 1: Unexpected reward
    print("\n  Scenario 1: Unexpected reward (positive RPE)")
    da_neuron.neuron.firing_rate = NEURON_PERSONALITIES["dopaminergic"].baseline_firing_rate
    initial_rate = da_neuron.neuron.firing_rate

    # Apply reward signal
    da_neuron.neuron.membrane_potential += 50.0  # Large depolarization
    da_neuron.update(0.01, 0.0)

    print(f"    Baseline rate: {initial_rate:.1f} Hz")
    print(f"    After unexpected reward: {da_neuron.neuron.firing_rate:.1f} Hz")
    print(f"    Change: +{da_neuron.neuron.firing_rate - initial_rate:.1f} Hz (burst!)")

    # Scenario 2: Predicted reward (no change)
    print("\n  Scenario 2: Predicted reward (no RPE)")
    da_neuron.neuron.firing_rate = NEURON_PERSONALITIES["dopaminergic"].baseline_firing_rate
    initial_rate = da_neuron.neuron.firing_rate

    # No change in activity
    for _ in range(10):
        da_neuron.update(0.01, 0.0)

    print(f"    Baseline rate: {initial_rate:.1f} Hz")
    print(f"    After predicted reward: {da_neuron.neuron.firing_rate:.1f} Hz")
    print(f"    Change: {da_neuron.neuron.firing_rate - initial_rate:.1f} Hz (no change)")

    # Scenario 3: Reward omission
    print("\n  Scenario 3: Reward omission (negative RPE)")
    da_neuron.neuron.firing_rate = NEURON_PERSONALITIES["dopaminergic"].baseline_firing_rate
    initial_rate = da_neuron.neuron.firing_rate

    # Suppress activity
    da_neuron.neuron.membrane_potential -= 20.0  # Hyperpolarization
    for _ in range(10):
        da_neuron.update(0.01, 0.0)

    print(f"    Baseline rate: {initial_rate:.1f} Hz")
    print(f"    After reward omission: {da_neuron.neuron.firing_rate:.1f} Hz")
    print(f"    Change: {da_neuron.neuron.firing_rate - initial_rate:.1f} Hz (pause!)")


def demo_5_grid_cells():
    """
    Demo 5: Demonstrate entorhinal grid cell hexagonal firing pattern
    """
    print("\n" + "="*70)
    print("DEMO 5: Entorhinal Grid Cell Hexagonal Pattern")
    print("="*70)

    # Create grid cell
    base_neuron = Neuron(x=150, y=150, z=50, neuron_type="pyramidal")
    base_neuron.stage = "mature"
    base_neuron.energy = 200.0

    grid_cell = create_personalized_neuron(base_neuron, "grid_cell")

    print("\nSimulating grid cell as animal moves through space...")

    # Create spatial firing map
    x_range = np.linspace(0, 300, 100)
    y_range = np.linspace(0, 300, 100)

    firing_map = np.zeros((len(y_range), len(x_range)))

    for i, y in enumerate(y_range):
        for j, x in enumerate(x_range):
            context = {'position': [x, y, 50.0]}
            grid_cell._update_grid_cell(context)
            firing_map[i, j] = grid_cell.neuron.firing_rate

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(firing_map, extent=[0, 300, 0, 300], origin='lower', cmap='jet')
    plt.title("Grid Cell: Hexagonal Firing Pattern", fontsize=14)
    plt.xlabel("X position (μm)")
    plt.ylabel("Y position (μm)")
    plt.colorbar(im, label='Firing rate (Hz)')

    # Add text
    plt.text(10, 280,
             "Hexagonal grid pattern\n(simplified model)",
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)

    plt.tight_layout()
    plt.savefig("grid_cell_pattern.png", dpi=150)
    print(f"\n✓ Saved grid cell pattern to grid_cell_pattern.png")
    plt.close()


def demo_6_neuromodulation():
    """
    Demo 6: Compare neuromodulator sensitivity across neuron types
    """
    print("\n" + "="*70)
    print("DEMO 6: Neuromodulator Sensitivity")
    print("="*70)

    # Test different neuron types
    test_types = [
        "dopaminergic",
        "serotonergic",
        "cholinergic",
        "fast_spiking"
    ]

    modulators = ["dopamine", "serotonin", "acetylcholine"]
    sensitivity_matrix = []

    print("\nNeuromodulator sensitivity matrix:")
    print("-" * 70)
    print(f"{'Neuron Type':<30} {'Dopamine':<15} {'Serotonin':<15} {'ACh':<15}")
    print("-" * 70)

    for neuron_type in test_types:
        personality = NEURON_PERSONALITIES[neuron_type]
        sensitivities = [
            personality.dopamine_sensitivity,
            personality.serotonin_sensitivity,
            personality.acetylcholine_sensitivity
        ]
        sensitivity_matrix.append(sensitivities)

        print(f"{personality.name:<30} "
              f"{sensitivities[0]:<15.2f} "
              f"{sensitivities[1]:<15.2f} "
              f"{sensitivities[2]:<15.2f}")

    # Visualize as heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(modulators)))
    ax.set_xticklabels(modulators)
    ax.set_yticks(range(len(test_types)))
    ax.set_yticklabels([NEURON_PERSONALITIES[t].name for t in test_types])

    # Colorbar
    plt.colorbar(im, label='Sensitivity (0-1)')

    # Add values to cells
    for i in range(len(test_types)):
        for j in range(len(modulators)):
            text = ax.text(j, i, f'{sensitivity_matrix[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    plt.title("Neuromodulator Sensitivity by Neuron Type", fontsize=14)
    plt.tight_layout()
    plt.savefig("neuromodulator_sensitivity.png", dpi=150)
    print(f"\n✓ Saved neuromodulator sensitivity to neuromodulator_sensitivity.png")
    plt.close()


def demo_7_integrated_network():
    """
    Demo 7: Create a heterogeneous network with multiple personality types
    """
    print("\n" + "="*70)
    print("DEMO 7: Heterogeneous Neural Network")
    print("="*70)

    # Create mixed network
    network_composition = {
        "dopaminergic": 3,
        "serotonergic": 2,
        "cholinergic": 2,
        "place_cell": 5,
        "fast_spiking": 8,
        "chattering": 5,
        "mirror_neuron": 3,
        "von_economo": 2,
    }

    print("\nNetwork composition:")
    total_neurons = sum(network_composition.values())
    print(f"  Total neurons: {total_neurons}")

    neurons = []
    personality_counts = {}

    for personality_type, count in network_composition.items():
        print(f"  {personality_type}: {count} neurons")

        for i in range(count):
            # Create base neuron at random position
            base_neuron = Neuron(
                x=np.random.uniform(50, 250),
                y=np.random.uniform(50, 250),
                z=np.random.uniform(20, 80),
                neuron_type="pyramidal"
            )
            base_neuron.stage = "mature"
            base_neuron.energy = 200.0

            # Add personality
            personalized = create_personalized_neuron(base_neuron, personality_type)
            neurons.append(personalized)

            if personality_type not in personality_counts:
                personality_counts[personality_type] = 0
            personality_counts[personality_type] += 1

    # Simulate network for 100 timesteps
    print(f"\nSimulating network for 100 timesteps...")

    dt = 0.01
    time = 0.0

    # Track statistics
    firing_rates_over_time = {ptype: [] for ptype in network_composition.keys()}

    for step in range(100):
        # Update all neurons
        for neuron in neurons:
            # Create simple context
            context = {
                'position': [neuron.neuron.x, neuron.neuron.y, neuron.neuron.z]
            }
            neuron.update(dt, time, context)

        # Record firing rates by type
        if step % 10 == 0:
            for ptype in network_composition.keys():
                rates = [n.neuron.firing_rate for n in neurons
                        if n.personality == NEURON_PERSONALITIES[ptype]]
                if rates:
                    firing_rates_over_time[ptype].append(np.mean(rates))

        time += dt

    # Plot average firing rates by personality type
    plt.figure(figsize=(12, 6))

    for ptype, rates in firing_rates_over_time.items():
        if rates:
            plt.plot(rates, label=ptype, linewidth=2, alpha=0.7)

    plt.xlabel("Time (× 10 timesteps)")
    plt.ylabel("Average Firing Rate (Hz)")
    plt.title("Heterogeneous Network Activity by Neuron Type")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("heterogeneous_network_activity.png", dpi=150)
    print(f"\n✓ Saved network activity to heterogeneous_network_activity.png")
    plt.close()

    # Print final statistics
    print("\nFinal network statistics:")
    print("-" * 70)
    for ptype in network_composition.keys():
        rates = [n.neuron.firing_rate for n in neurons
                if n.personality == NEURON_PERSONALITIES[ptype]]
        if rates:
            print(f"  {ptype:<20} Avg: {np.mean(rates):6.1f} Hz  "
                  f"Range: {np.min(rates):6.1f}-{np.max(rates):6.1f} Hz")


def main():
    print("\n" + "="*70)
    print("NEURON PERSONALITIES DEMONSTRATION")
    print("="*70)
    print("\nDemonstrating 9 distinct neuron personality types based on")
    print("neuroscience literature, integrated with existing simulation.")
    print("\nIMPORTANT: No existing code was modified!")
    print("="*70)

    # Run all demos
    demos = [
        ("Basic Personalities", demo_1_basic_personalities),
        ("Firing Patterns", demo_2_firing_patterns),
        ("Place Cell Encoding", demo_3_place_cells),
        ("Reward Coding", demo_4_reward_coding),
        ("Grid Cell Pattern", demo_5_grid_cells),
        ("Neuromodulation", demo_6_neuromodulation),
        ("Integrated Network", demo_7_integrated_network),
    ]

    for demo_name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ Error in {demo_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  1. neuron_personality_firing_patterns.png - Firing pattern comparison")
    print("  2. place_cell_fields.png - Spatial place fields")
    print("  3. grid_cell_pattern.png - Hexagonal grid pattern")
    print("  4. neuromodulator_sensitivity.png - Sensitivity heatmap")
    print("  5. heterogeneous_network_activity.png - Mixed network dynamics")

    print("\n" + "="*70)
    print("Key Features Demonstrated:")
    print("="*70)
    print("  ✓ 9 distinct neuron personalities from literature")
    print("  ✓ Unique firing patterns (burst, regular, fast, chattering)")
    print("  ✓ Spatial encoding (place cells, grid cells)")
    print("  ✓ Reward prediction error (dopaminergic neurons)")
    print("  ✓ Neuromodulator sensitivity differences")
    print("  ✓ Heterogeneous network simulation")
    print("  ✓ Integration with existing neuron.py (no modifications!)")

    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("\n1. Integrate with learning simulation:")
    print("   - Use dopaminergic neurons for reward signals")
    print("   - Use place/grid cells for spatial learning tasks")
    print("   - Use fast-spiking interneurons for timing")

    print("\n2. Add to C++ fast simulator:")
    print("   - Port personality traits to neuron_learning_fast.cpp")
    print("   - Visualize different neuron types with color coding")

    print("\n3. Create learning tasks that leverage personalities:")
    print("   - Spatial navigation with place/grid cells")
    print("   - Reward-based learning with dopaminergic modulation")
    print("   - Action learning with mirror neurons")

    print("\n" + "="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
