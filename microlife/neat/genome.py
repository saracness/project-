"""
NEAT Genome - Genetic representation of neural networks

Implements historical markings and structural encoding.
"""
import numpy as np
import copy
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class NodeGene:
    """
    Represents a neuron in the network.

    Attributes:
        id: Unique node identifier
        type: 'input', 'hidden', or 'output'
        activation: Activation function name
        bias: Neuron bias value
        layer: Network layer (for visualization)
    """
    id: int
    type: str  # 'input', 'hidden', 'output'
    activation: str = 'sigmoid'
    bias: float = 0.0
    layer: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, NodeGene):
            return False
        return self.id == other.id

    def copy(self):
        """Create deep copy of node."""
        return NodeGene(
            id=self.id,
            type=self.type,
            activation=self.activation,
            bias=self.bias,
            layer=self.layer
        )


@dataclass
class ConnectionGene:
    """
    Represents a synapse (connection) between neurons.

    Attributes:
        innovation: Global innovation number (historical marking)
        in_node: Source node ID
        out_node: Target node ID
        weight: Connection weight
        enabled: Whether connection is active
    """
    innovation: int
    in_node: int
    out_node: int
    weight: float
    enabled: bool = True

    def __hash__(self):
        return hash(self.innovation)

    def __eq__(self, other):
        if not isinstance(other, ConnectionGene):
            return False
        return self.innovation == other.innovation

    def copy(self):
        """Create deep copy of connection."""
        return ConnectionGene(
            innovation=self.innovation,
            in_node=self.in_node,
            out_node=self.out_node,
            weight=self.weight,
            enabled=self.enabled
        )


class InnovationDB:
    """
    Global innovation number tracker.

    Ensures same structural mutations get same innovation numbers
    across different genomes in same generation.
    """

    def __init__(self):
        self.innovations: Dict[Tuple[int, int], int] = {}
        self.next_innovation = 1

    def get_innovation(self, in_node: int, out_node: int) -> int:
        """
        Get innovation number for connection.

        Args:
            in_node: Source node ID
            out_node: Target node ID

        Returns:
            Innovation number (reused if connection existed before)
        """
        key = (in_node, out_node)
        if key in self.innovations:
            return self.innovations[key]
        else:
            innovation_id = self.next_innovation
            self.innovations[key] = innovation_id
            self.next_innovation += 1
            return innovation_id

    def reset(self):
        """Reset innovation database (new generation)."""
        self.innovations.clear()


class NEATGenome:
    """
    NEAT genome encoding neural network structure.

    Features:
    - Variable topology (nodes and connections evolve)
    - Historical markings (innovation numbers)
    - Mutation operators (add node, add connection, weight)
    - Crossover with topology alignment
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        """
        Initialize genome with minimal structure.

        Args:
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Genes
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, ConnectionGene] = {}

        # Fitness
        self.fitness: float = 0.0
        self.adjusted_fitness: float = 0.0

        # Species
        self.species_id: Optional[int] = None

        # Node ID counter
        self.next_node_id = 0

        # Initialize input and output nodes
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Create input and output nodes."""
        # Input nodes
        for i in range(self.num_inputs):
            self.nodes[self.next_node_id] = NodeGene(
                id=self.next_node_id,
                type='input',
                layer=0
            )
            self.next_node_id += 1

        # Output nodes
        for i in range(self.num_outputs):
            self.nodes[self.next_node_id] = NodeGene(
                id=self.next_node_id,
                type='output',
                layer=2  # Output layer
            )
            self.next_node_id += 1

    def add_node_mutation(self, innovation_db: InnovationDB) -> bool:
        """
        Add node mutation - splits existing connection.

        Before:  A ---w---> B
        After:   A --1.0--> N ---w---> B

        Args:
            innovation_db: Global innovation tracker

        Returns:
            True if mutation successful
        """
        # Get enabled connections
        enabled_conns = [c for c in self.connections.values() if c.enabled]

        if not enabled_conns:
            return False

        # Choose random connection to split
        conn = np.random.choice(enabled_conns)

        # Disable old connection
        conn.enabled = False

        # Create new node
        new_node = NodeGene(
            id=self.next_node_id,
            type='hidden',
            layer=1  # Hidden layer
        )
        self.nodes[self.next_node_id] = new_node
        new_node_id = self.next_node_id
        self.next_node_id += 1

        # Create two new connections
        # Connection 1: in_node -> new_node (weight = 1.0)
        innov1 = innovation_db.get_innovation(conn.in_node, new_node_id)
        conn1 = ConnectionGene(
            innovation=innov1,
            in_node=conn.in_node,
            out_node=new_node_id,
            weight=1.0,
            enabled=True
        )
        self.connections[innov1] = conn1

        # Connection 2: new_node -> out_node (weight = old weight)
        innov2 = innovation_db.get_innovation(new_node_id, conn.out_node)
        conn2 = ConnectionGene(
            innovation=innov2,
            in_node=new_node_id,
            out_node=conn.out_node,
            weight=conn.weight,
            enabled=True
        )
        self.connections[innov2] = conn2

        return True

    def add_connection_mutation(self, innovation_db: InnovationDB,
                               prob_recurrent: float = 0.0) -> bool:
        """
        Add connection mutation - creates new synapse.

        Args:
            innovation_db: Global innovation tracker
            prob_recurrent: Probability of recurrent connection

        Returns:
            True if mutation successful
        """
        # Get all possible connections
        possible_connections = []

        for node1_id, node1 in self.nodes.items():
            for node2_id, node2 in self.nodes.items():
                # Skip self-connections (unless recurrent allowed)
                if node1_id == node2_id and np.random.random() > prob_recurrent:
                    continue

                # Skip if connection already exists
                exists = any(
                    c.in_node == node1_id and c.out_node == node2_id
                    for c in self.connections.values()
                )

                if not exists:
                    possible_connections.append((node1_id, node2_id))

        if not possible_connections:
            return False

        # Choose random connection
        in_node, out_node = possible_connections[np.random.randint(len(possible_connections))]

        # Get innovation number
        innovation = innovation_db.get_innovation(in_node, out_node)

        # Create connection with random weight
        weight = np.random.randn() * 2.0  # N(0, 2)

        conn = ConnectionGene(
            innovation=innovation,
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            enabled=True
        )

        self.connections[innovation] = conn

        return True

    def mutate_weights(self, prob_uniform: float = 0.8,
                      prob_replace: float = 0.1,
                      power: float = 0.5):
        """
        Mutate connection weights.

        Args:
            prob_uniform: Probability of uniform perturbation
            prob_replace: Probability of complete replacement
            power: Perturbation strength
        """
        for conn in self.connections.values():
            if np.random.random() < prob_uniform:
                # Uniform perturbation
                if np.random.random() < prob_replace:
                    # Complete replacement
                    conn.weight = np.random.randn() * 2.0
                else:
                    # Small perturbation
                    conn.weight += np.random.randn() * power

                # Clamp weights
                conn.weight = np.clip(conn.weight, -10.0, 10.0)

    def mutate(self, config, innovation_db: InnovationDB):
        """
        Apply all mutations according to config probabilities.

        Args:
            config: NEAT configuration
            innovation_db: Global innovation tracker
        """
        # Add node mutation
        if np.random.random() < config.prob_add_node:
            self.add_node_mutation(innovation_db)

        # Add connection mutation
        if np.random.random() < config.prob_add_connection:
            self.add_connection_mutation(innovation_db)

        # Weight mutations
        if np.random.random() < config.prob_mutate_weight:
            self.mutate_weights(
                prob_uniform=config.prob_weight_uniform,
                prob_replace=config.prob_weight_replace,
                power=config.weight_mutation_power
            )

    def distance(self, other: 'NEATGenome', config) -> float:
        """
        Calculate compatibility distance between genomes.

        δ = c1 * E/N + c2 * D/N + c3 * W̄

        Args:
            other: Other genome
            config: NEAT configuration

        Returns:
            Compatibility distance
        """
        # Get innovation numbers
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())

        # Find matching, disjoint, and excess genes
        matching = innovations1 & innovations2
        disjoint = innovations1 ^ innovations2

        max_innov1 = max(innovations1) if innovations1 else 0
        max_innov2 = max(innovations2) if innovations2 else 0

        # Excess: innovations beyond the other genome's max
        if max_innov1 > max_innov2:
            excess = sum(1 for i in innovations1 if i > max_innov2)
        else:
            excess = sum(1 for i in innovations2 if i > max_innov1)

        disjoint_count = len(disjoint) - excess

        # Average weight difference of matching genes
        if matching:
            weight_diff = sum(
                abs(self.connections[i].weight - other.connections[i].weight)
                for i in matching
            ) / len(matching)
        else:
            weight_diff = 0.0

        # Normalizing factor (number of genes in larger genome)
        N = max(len(innovations1), len(innovations2))
        if N < 20:
            N = 1  # Don't normalize for small genomes

        # Compatibility distance
        distance = (
            config.c1 * excess / N +
            config.c2 * disjoint_count / N +
            config.c3 * weight_diff
        )

        return distance

    def copy(self) -> 'NEATGenome':
        """Create deep copy of genome."""
        new_genome = NEATGenome(self.num_inputs, self.num_outputs)

        # Copy nodes
        new_genome.nodes = {k: v.copy() for k, v in self.nodes.items()}

        # Copy connections
        new_genome.connections = {k: v.copy() for k, v in self.connections.items()}

        # Copy metadata
        new_genome.fitness = self.fitness
        new_genome.adjusted_fitness = self.adjusted_fitness
        new_genome.species_id = self.species_id
        new_genome.next_node_id = self.next_node_id

        return new_genome

    def size(self) -> Tuple[int, int]:
        """
        Get genome size.

        Returns:
            (num_nodes, num_connections)
        """
        return len(self.nodes), len(self.connections)

    def __repr__(self):
        nodes, conns = self.size()
        return f"NEATGenome(nodes={nodes}, connections={conns}, fitness={self.fitness:.2f})"
