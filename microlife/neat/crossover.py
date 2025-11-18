"""
NEAT Crossover - Structural recombination operator
"""
import numpy as np
from typing import Tuple
from .genome import NEATGenome, NodeGene, ConnectionGene


def crossover(parent1: NEATGenome, parent2: NEATGenome) -> NEATGenome:
    """
    Perform crossover between two genomes.

    Matching genes: randomly chosen from either parent
    Disjoint/Excess genes: inherited from fitter parent

    Args:
        parent1: First parent (should be fitter)
        parent2: Second parent

    Returns:
        Offspring genome
    """
    # Ensure parent1 is fitter
    if parent2.fitness > parent1.fitness:
        parent1, parent2 = parent2, parent1

    # Create offspring
    offspring = NEATGenome(parent1.num_inputs, parent1.num_outputs)

    # Inherit all nodes from fitter parent
    offspring.nodes = {k: v.copy() for k, v in parent1.nodes.items()}
    offspring.next_node_id = parent1.next_node_id

    # Get innovation numbers
    innovations1 = set(parent1.connections.keys())
    innovations2 = set(parent2.connections.keys())

    # Matching genes
    matching = innovations1 & innovations2

    # Disjoint and excess genes (from fitter parent only)
    disjoint_excess = innovations1 - innovations2

    # Inherit matching genes randomly
    for innov in matching:
        if np.random.random() < 0.5:
            conn = parent1.connections[innov].copy()
        else:
            conn = parent2.connections[innov].copy()

        offspring.connections[innov] = conn

        # Add nodes if not present
        _ensure_nodes_exist(offspring, conn, parent1, parent2)

    # Inherit disjoint/excess from fitter parent
    for innov in disjoint_excess:
        conn = parent1.connections[innov].copy()
        offspring.connections[innov] = conn

        _ensure_nodes_exist(offspring, conn, parent1, parent2)

    return offspring


def _ensure_nodes_exist(offspring: NEATGenome, conn: ConnectionGene,
                       parent1: NEATGenome, parent2: NEATGenome):
    """
    Ensure connection's nodes exist in offspring.

    Args:
        offspring: Offspring genome
        conn: Connection gene
        parent1: First parent
        parent2: Second parent
    """
    # Add in_node if not present
    if conn.in_node not in offspring.nodes:
        if conn.in_node in parent1.nodes:
            offspring.nodes[conn.in_node] = parent1.nodes[conn.in_node].copy()
        elif conn.in_node in parent2.nodes:
            offspring.nodes[conn.in_node] = parent2.nodes[conn.in_node].copy()

    # Add out_node if not present
    if conn.out_node not in offspring.nodes:
        if conn.out_node in parent1.nodes:
            offspring.nodes[conn.out_node] = parent1.nodes[conn.out_node].copy()
        elif conn.out_node in parent2.nodes:
            offspring.nodes[conn.out_node] = parent2.nodes[conn.out_node].copy()
