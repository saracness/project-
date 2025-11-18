#!/usr/bin/env python3
"""
NEAT XOR Demo - Classic neuroevolution benchmark

Evolves neural networks to solve the XOR problem.
"""
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from microlife.neat import NEATPopulation, NEATConfig, NEATGenome
from microlife.neat.visualizer import NEATVisualizer


def activate_network(genome: NEATGenome, inputs: np.ndarray) -> np.ndarray:
    """
    Activate NEAT network with given inputs.

    Simple feedforward activation (no recurrence).

    Args:
        genome: NEAT genome
        inputs: Input values

    Returns:
        Output values
    """
    # Initialize node activations
    activations = {}

    # Set input activations
    input_nodes = [n for n, node in genome.nodes.items() if node.type == 'input']
    for i, node_id in enumerate(sorted(input_nodes)):
        if i < len(inputs):
            activations[node_id] = inputs[i]

    # Get all nodes sorted by layer (feedforward)
    nodes_by_layer = {}
    for node_id, node in genome.nodes.items():
        layer = node.layer
        if layer not in nodes_by_layer:
            nodes_by_layer[layer] = []
        nodes_by_layer[layer].append(node_id)

    # Activate layers in order
    for layer in sorted(nodes_by_layer.keys()):
        if layer == 0:  # Skip input layer (already set)
            continue

        for node_id in nodes_by_layer[layer]:
            # Sum incoming connections
            activation_sum = 0.0

            for conn in genome.connections.values():
                if conn.out_node == node_id and conn.enabled:
                    if conn.in_node in activations:
                        activation_sum += activations[conn.in_node] * conn.weight

            # Apply activation function (sigmoid)
            activations[node_id] = 1.0 / (1.0 + np.exp(-activation_sum))

    # Get output activations
    output_nodes = [n for n, node in genome.nodes.items() if node.type == 'output']
    outputs = []
    for node_id in sorted(output_nodes):
        outputs.append(activations.get(node_id, 0.0))

    return np.array(outputs)


def evaluate_xor(genome: NEATGenome) -> float:
    """
    Evaluate genome on XOR problem.

    Args:
        genome: NEAT genome

    Returns:
        Fitness (4.0 - error, higher is better)
    """
    # XOR truth table
    xor_inputs = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])

    xor_outputs = np.array([0.0, 1.0, 1.0, 0.0])

    # Evaluate network
    total_error = 0.0

    for inputs, expected in zip(xor_inputs, xor_outputs):
        output = activate_network(genome, inputs)[0]
        error = abs(output - expected)
        total_error += error

    # Fitness (4.0 is perfect)
    fitness = 4.0 - total_error

    return fitness


def main():
    """Run NEAT on XOR problem."""
    print("=" * 70)
    print("🧬 NEAT: NeuroEvolution of Augmenting Topologies")
    print("Problem: XOR (Classic Benchmark)")
    print("=" * 70)
    print()

    # Configuration
    config = NEATConfig(
        population_size=150,
        prob_add_node=0.03,
        prob_add_connection=0.05,
        compatibility_threshold=3.0,
        species_stagnation_threshold=15
    )

    print("Configuration:")
    print(config)
    print()

    # Create population
    population = NEATPopulation(
        config=config,
        num_inputs=2,  # XOR has 2 inputs
        num_outputs=1  # XOR has 1 output
    )

    print("🚀 Starting evolution...")
    print()

    # Evolve
    num_generations = 100

    try:
        population.evolve(
            fitness_function=evaluate_xor,
            num_generations=num_generations,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n⏸️  Evolution stopped by user")

    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)

    # Get best genome
    best = population.get_best_genome()

    if best:
        print(f"Best Genome: {best}")
        print()

        # Test best genome
        print("Testing Best Genome on XOR:")
        print("-" * 40)

        xor_inputs = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ]

        xor_expected = [0.0, 1.0, 1.0, 0.0]

        for inputs, expected in zip(xor_inputs, xor_expected):
            output = activate_network(best, np.array(inputs))[0]
            print(f"  {inputs} → {output:.4f} (expected: {expected:.1f})")

        print()

    # Summary statistics
    summary = population.get_stats_summary()
    print("Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    # Generate visualizations
    print("📈 Generating visualizations...")
    visualizer = NEATVisualizer(dpi=300)

    visualizer.generate_all_plots(population, output_dir='outputs/neat')

    if best:
        visualizer.visualize_network(
            best,
            save_path='outputs/neat/best_xor_network.png'
        )

    print()
    print("✅ Complete! Check outputs/neat/ for visualizations")
    print("=" * 70)


if __name__ == '__main__':
    main()
