"""
NEAT Visualization - Network topology and evolution trees

Publication-quality visualizations for research papers.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from .genome import NEATGenome
from .population import NEATPopulation


class NEATVisualizer:
    """
    Visualize NEAT networks and evolution.

    Features:
    - Network topology diagrams
    - Phylogenetic trees
    - Evolution statistics
    - Publication-quality output
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            dpi: Output resolution
        """
        self.dpi = dpi

        # Style settings
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.size'] = 10

    def visualize_network(self, genome: NEATGenome,
                         save_path: Optional[str] = None,
                         show: bool = False):
        """
        Visualize neural network topology.

        Args:
            genome: NEAT genome to visualize
            save_path: Output file path
            show: Show plot interactively
        """
        # Create directed graph
        G = nx.DiGraph()

        # Add nodes
        for node_id, node in genome.nodes.items():
            G.add_node(node_id, type=node.type, layer=node.layer)

        # Add edges (enabled connections only)
        for conn in genome.connections.values():
            if conn.enabled:
                G.add_edge(
                    conn.in_node,
                    conn.out_node,
                    weight=conn.weight
                )

        # Calculate node positions (layered layout)
        pos = self._calculate_positions(genome)

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw nodes by type
        input_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'input']
        hidden_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'hidden']
        output_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'output']

        nx.draw_networkx_nodes(G, pos, nodelist=input_nodes,
                              node_color='#3498db', node_size=800,
                              label='Input', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes,
                              node_color='#2ecc71', node_size=600,
                              label='Hidden', ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=output_nodes,
                              node_color='#e74c3c', node_size=800,
                              label='Output', ax=ax)

        # Draw edges with varying thickness based on weight
        weights = [d['weight'] for _, _, d in G.edges(data=True)]
        if weights:
            max_weight = max(abs(w) for w in weights)

            for (u, v, data) in G.edges(data=True):
                weight = data['weight']
                width = abs(weight) / max_weight * 3.0 + 0.5
                color = '#2c3e50' if weight > 0 else '#95a5a6'
                alpha = min(abs(weight) / max_weight, 0.8)

                nx.draw_networkx_edges(
                    G, pos, [(u, v)],
                    width=width,
                    edge_color=color,
                    alpha=alpha,
                    arrows=True,
                    arrowsize=15,
                    ax=ax
                )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

        # Title and legend
        nodes, conns = genome.size()
        ax.set_title(
            f'NEAT Network Topology\n'
            f'Nodes: {nodes} | Connections: {conns} | Fitness: {genome.fitness:.2f}',
            fontsize=14, fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved network topology to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _calculate_positions(self, genome: NEATGenome) -> Dict[int, Tuple[float, float]]:
        """
        Calculate node positions for layered layout.

        Args:
            genome: NEAT genome

        Returns:
            Dictionary mapping node_id to (x, y) position
        """
        # Group nodes by layer
        layers: Dict[int, List[int]] = {}
        for node_id, node in genome.nodes.items():
            layer = node.layer
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node_id)

        # Calculate positions
        pos = {}
        layer_keys = sorted(layers.keys())
        num_layers = len(layer_keys)

        for layer_idx, layer in enumerate(layer_keys):
            nodes_in_layer = layers[layer]
            num_nodes = len(nodes_in_layer)

            x = layer_idx / (num_layers - 1) if num_layers > 1 else 0.5

            for node_idx, node_id in enumerate(nodes_in_layer):
                y = (node_idx + 1) / (num_nodes + 1)
                pos[node_id] = (x, y)

        return pos

    def plot_evolution_stats(self, population: NEATPopulation,
                            save_path: Optional[str] = None,
                            show: bool = False):
        """
        Plot evolution statistics.

        Args:
            population: NEAT population
            save_path: Output file path
            show: Show plot interactively
        """
        if not population.generation_stats:
            print("⚠️  No statistics to plot")
            return

        stats = population.generation_stats

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NEAT Evolution Statistics', fontsize=16, fontweight='bold')

        # Plot 1: Fitness over generations
        ax1 = axes[0, 0]
        generations = [s['generation'] for s in stats]
        avg_fitness = [s['avg_fitness'] for s in stats]
        max_fitness = [s['max_fitness'] for s in stats]
        best_fitness = [s['best_fitness'] for s in stats]

        ax1.plot(generations, avg_fitness, label='Avg Fitness', linewidth=2, alpha=0.7)
        ax1.plot(generations, max_fitness, label='Max Fitness', linewidth=2, alpha=0.7)
        ax1.plot(generations, best_fitness, label='Best Ever', linewidth=2, linestyle='--')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Number of species
        ax2 = axes[0, 1]
        num_species = [s['num_species'] for s in stats]
        ax2.plot(generations, num_species, linewidth=2, color='#e74c3c')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Number of Species')
        ax2.set_title('Species Diversity')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average genome size
        ax3 = axes[1, 0]
        avg_size = [s['avg_size'] for s in stats]
        ax3.plot(generations, avg_size, linewidth=2, color='#2ecc71')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Average Genome Size')
        ax3.set_title('Complexity Over Time')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Fitness distribution (final generation)
        ax4 = axes[1, 1]
        final_fitnesses = [g.fitness for g in population.genomes]
        ax4.hist(final_fitnesses, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax4.axvline(population.best_fitness, color='r', linestyle='--',
                   linewidth=2, label='Best')
        ax4.set_xlabel('Fitness')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Final Generation Fitness Distribution (Gen {population.generation})')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved evolution stats to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_species_timeline(self, population: NEATPopulation,
                             save_path: Optional[str] = None,
                             show: bool = False):
        """
        Plot species evolution timeline.

        Args:
            population: NEAT population
            save_path: Output file path
            show: Show plot interactively
        """
        if not population.generation_stats:
            print("⚠️  No statistics to plot")
            return

        # Track species across generations (simplified version)
        fig, ax = plt.subplots(figsize=(14, 8))

        generations = [s['generation'] for s in population.generation_stats]
        num_species = [s['num_species'] for s in population.generation_stats]

        # Create stacked area plot (simplified - just show count)
        ax.fill_between(generations, num_species, alpha=0.6, color='#3498db')
        ax.plot(generations, num_species, linewidth=2, color='#2c3e50')

        ax.set_xlabel('Generation')
        ax.set_ylabel('Number of Species')
        ax.set_title('Species Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved species timeline to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_all_plots(self, population: NEATPopulation,
                          output_dir: str = 'outputs/neat'):
        """
        Generate all visualizations.

        Args:
            population: NEAT population
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📊 Generating NEAT visualizations...")

        # Best network topology
        if population.best_genome:
            self.visualize_network(
                population.best_genome,
                save_path=output_path / 'best_network_topology.png'
            )

        # Evolution statistics
        self.plot_evolution_stats(
            population,
            save_path=output_path / 'evolution_statistics.png'
        )

        # Species timeline
        self.plot_species_timeline(
            population,
            save_path=output_path / 'species_timeline.png'
        )

        print(f"✅ All plots generated in {output_dir}")
