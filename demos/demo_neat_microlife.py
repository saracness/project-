#!/usr/bin/env python3
"""
NEAT MicroLife Demo - Evolution of Organism Brains

Evolves neural network controllers for artificial life organisms.
True evolutionary dynamics with survival, reproduction, and adaptation.
"""
import sys
import os
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from microlife.core.environment import Environment
from microlife.core.organism import Organism
from microlife.neat import NEATPopulation, NEATConfig, NEATGenome
from microlife.neat.visualizer import NEATVisualizer
from microlife.neat.paper_generator import LaTeXPaperGenerator


class NEATBrain:
    """NEAT-evolved neural network brain for organisms."""

    def __init__(self, genome: NEATGenome):
        """
        Initialize brain from NEAT genome.

        Args:
            genome: NEAT genome encoding network
        """
        self.genome = genome

    def decide(self, state: dict) -> dict:
        """
        Make decision based on current state.

        Args:
            state: Dictionary with sensor inputs

        Returns:
            Action dictionary
        """
        # Extract inputs from state
        inputs = np.array([
            state.get('energy_normalized', 0.5),
            state.get('nearest_food_distance', 1.0),
            state.get('nearest_food_angle', 0.0),
            state.get('age_normalized', 0.0),
        ])

        # Activate network
        outputs = self._activate_network(inputs)

        # Convert outputs to actions
        action = {
            'move_x': outputs[0] * 2.0 - 1.0,  # -1 to 1
            'move_y': outputs[1] * 2.0 - 1.0,
            'reproduce': outputs[2] > 0.7,  # Threshold
        }

        return action

    def _activate_network(self, inputs: np.ndarray) -> np.ndarray:
        """Activate NEAT network (feedforward)."""
        # Initialize activations
        activations = {}

        # Set input activations
        input_nodes = sorted([
            n for n, node in self.genome.nodes.items()
            if node.type == 'input'
        ])

        for i, node_id in enumerate(input_nodes):
            if i < len(inputs):
                activations[node_id] = inputs[i]

        # Get nodes by layer
        nodes_by_layer = {}
        for node_id, node in self.genome.nodes.items():
            layer = node.layer
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(node_id)

        # Activate layers
        for layer in sorted(nodes_by_layer.keys()):
            if layer == 0:
                continue

            for node_id in nodes_by_layer[layer]:
                total_input = 0.0

                for conn in self.genome.connections.values():
                    if conn.out_node == node_id and conn.enabled:
                        if conn.in_node in activations:
                            total_input += activations[conn.in_node] * conn.weight

                # Sigmoid activation
                activations[node_id] = 1.0 / (1.0 + np.exp(-total_input))

        # Get outputs
        output_nodes = sorted([
            n for n, node in self.genome.nodes.items()
            if node.type == 'output'
        ])

        outputs = []
        for node_id in output_nodes:
            outputs.append(activations.get(node_id, 0.0))

        return np.array(outputs)


def evaluate_genome_in_environment(genome: NEATGenome,
                                  simulation_steps: int = 500) -> float:
    """
    Evaluate genome by running organism in simulation.

    Args:
        genome: NEAT genome to evaluate
        simulation_steps: Number of simulation steps

    Returns:
        Fitness score (survival + food collected + reproduction)
    """
    # Create environment
    env = Environment(width=800, height=600, food_count=50)

    # Create organism with NEAT brain
    organism = Organism(
        x=400,
        y=300,
        brain_type='neat',
        environment=env
    )

    # Replace brain with NEAT brain
    organism.brain = NEATBrain(genome)

    env.add_organism(organism)

    # Run simulation
    food_collected = 0
    survival_time = 0
    reproductions = 0

    for step in range(simulation_steps):
        # Update environment
        env.update(0.016)  # ~60 Hz

        # Check if organism alive
        if organism not in env.organisms:
            break

        survival_time = step + 1

        # Track food collected (approximate)
        if hasattr(organism, '_food_collected'):
            food_collected = organism._food_collected

        # Track reproductions
        if len(env.organisms) > 1:
            reproductions += len(env.organisms) - 1
            # Keep only original organism for fair evaluation
            env.organisms = [organism]

    # Calculate fitness
    fitness = (
        survival_time +                    # Survival bonus
        food_collected * 10.0 +           # Food collection bonus
        reproductions * 50.0               # Reproduction bonus
    )

    return fitness


def main():
    """Run NEAT evolution on MicroLife organisms."""
    print("=" * 70)
    print("🧬 NEAT: Evolving MicroLife Organism Brains")
    print("=" * 70)
    print()

    # Configuration
    config = NEATConfig(
        population_size=100,
        prob_add_node=0.03,
        prob_add_connection=0.05,
        compatibility_threshold=3.0,
        species_stagnation_threshold=15
    )

    print("Experiment Configuration:")
    print(config)
    print()

    # Create population
    # Inputs: energy, food_distance, food_angle, age
    # Outputs: move_x, move_y, reproduce
    population = NEATPopulation(
        config=config,
        num_inputs=4,
        num_outputs=3
    )

    print("🚀 Starting evolution...")
    print("   Organisms will learn to:")
    print("   - Survive as long as possible")
    print("   - Find and eat food")
    print("   - Reproduce when healthy")
    print()

    # Evolve
    num_generations = 50  # Reduced for demo (each gen is slower)

    start_time = time.time()

    try:
        # Custom evolution loop with progress tracking
        for gen in range(num_generations):
            # Evaluate fitness
            print(f"\nGen {gen:3d} - Evaluating {len(population.genomes)} genomes...")

            for i, genome in enumerate(population.genomes):
                genome.fitness = evaluate_genome_in_environment(genome, simulation_steps=300)

                if i % 20 == 0:
                    print(f"  Progress: {i}/{len(population.genomes)} genomes evaluated")

                # Update best
                if genome.fitness > population.best_fitness:
                    population.best_fitness = genome.fitness
                    population.best_genome = genome.copy()

            # Speciate
            population.speciate()

            # Collect stats
            population._collect_stats()

            # Print progress
            population._print_stats()

            # Reproduce
            population.reproduce()

    except KeyboardInterrupt:
        print("\n⏸️  Evolution stopped by user")

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    print(f"Evolution time: {elapsed:.1f} seconds")
    print()

    # Best genome
    best = population.get_best_genome()

    if best:
        print(f"Best Genome: {best}")
        print()

        # Test best genome in longer simulation
        print("Testing Best Genome (1000 steps):")
        print("-" * 40)

        test_fitness = evaluate_genome_in_environment(best, simulation_steps=1000)
        print(f"  Test Fitness: {test_fitness:.2f}")
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
            save_path='outputs/neat/best_microlife_brain.png'
        )

    print()

    # Generate LaTeX paper
    print("📄 Generating scientific paper...")
    paper_gen = LaTeXPaperGenerator(template='nature')
    paper_gen.generate_paper(
        population,
        experiment_name='microlife_evolution',
        output_path='outputs/neat/microlife_paper.tex'
    )

    print()
    print("✅ Complete! Check outputs/neat/ for:")
    print("   - Visualizations (PNG)")
    print("   - LaTeX paper (microlife_paper.tex)")
    print("   - Compile paper: cd outputs/neat && pdflatex microlife_paper.tex")
    print("=" * 70)


if __name__ == '__main__':
    main()
