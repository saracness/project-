"""
NEAT Species - Grouping genomes by compatibility

Protects innovations through explicit fitness sharing.
"""
import numpy as np
from typing import List, Optional
from .genome import NEATGenome


class Species:
    """
    A species groups similar genomes.

    Features:
    - Compatibility-based grouping
    - Explicit fitness sharing
    - Stagnation tracking
    - Champion preservation
    """

    def __init__(self, species_id: int, representative: NEATGenome):
        """
        Initialize species.

        Args:
            species_id: Unique species identifier
            representative: Exemplar genome
        """
        self.id = species_id
        self.representative = representative.copy()
        self.members: List[NEATGenome] = []

        # Statistics
        self.age = 0
        self.best_fitness = 0.0
        self.avg_fitness = 0.0
        self.stagnation_counter = 0

        # Champion (best genome)
        self.champion: Optional[NEATGenome] = None

    def add_member(self, genome: NEATGenome):
        """Add genome to species."""
        genome.species_id = self.id
        self.members.append(genome)

    def calculate_adjusted_fitness(self):
        """
        Calculate adjusted fitness with explicit fitness sharing.

        adjusted_fitness(i) = fitness(i) / |species|
        """
        if not self.members:
            return

        species_size = len(self.members)

        for genome in self.members:
            genome.adjusted_fitness = genome.fitness / species_size

    def calculate_stats(self):
        """Calculate species statistics."""
        if not self.members:
            self.avg_fitness = 0.0
            return

        # Average fitness
        self.avg_fitness = sum(g.fitness for g in self.members) / len(self.members)

        # Best fitness
        current_best = max(self.members, key=lambda g: g.fitness)
        current_best_fitness = current_best.fitness

        # Update champion
        if self.champion is None or current_best_fitness > self.best_fitness:
            self.champion = current_best.copy()
            self.best_fitness = current_best_fitness
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

    def get_num_offspring(self, total_adjusted_fitness: float,
                         total_offspring: int) -> int:
        """
        Calculate number of offspring for this species.

        Args:
            total_adjusted_fitness: Sum of all adjusted fitnesses
            total_offspring: Total offspring to allocate

        Returns:
            Number of offspring for species
        """
        if total_adjusted_fitness == 0:
            return 0

        species_adjusted_fitness = sum(g.adjusted_fitness for g in self.members)
        num_offspring = int(
            (species_adjusted_fitness / total_adjusted_fitness) * total_offspring
        )

        return max(0, num_offspring)

    def reproduce(self, num_offspring: int, config, innovation_db) -> List[NEATGenome]:
        """
        Produce offspring for next generation.

        Args:
            num_offspring: Number of offspring to produce
            config: NEAT configuration
            innovation_db: Global innovation tracker

        Returns:
            List of offspring genomes
        """
        from .crossover import crossover

        offspring = []

        if not self.members:
            return offspring

        # Sort members by fitness
        self.members.sort(key=lambda g: g.fitness, reverse=True)

        # Elitism: preserve champion
        if len(self.members) >= config.elitism_threshold and self.champion:
            offspring.append(self.champion.copy())
            num_offspring -= 1

        # Survival threshold: only top performers reproduce
        cutoff = max(1, int(len(self.members) * config.survival_threshold))
        parents = self.members[:cutoff]

        # Generate offspring
        for _ in range(num_offspring):
            if np.random.random() < config.crossover_prob and len(parents) >= 2:
                # Crossover
                parent1 = np.random.choice(parents)
                parent2 = np.random.choice(parents)

                child = crossover(parent1, parent2)
            else:
                # Mutation only
                child = np.random.choice(parents).copy()

            # Apply mutations
            child.mutate(config, innovation_db)

            offspring.append(child)

        return offspring

    def update_representative(self):
        """Update species representative (random member)."""
        if self.members:
            self.representative = np.random.choice(self.members).copy()

    def reset(self):
        """Reset species for new generation."""
        self.members.clear()
        self.age += 1

    def __repr__(self):
        return (
            f"Species(id={self.id}, size={len(self.members)}, "
            f"age={self.age}, best_fitness={self.best_fitness:.2f}, "
            f"stagnation={self.stagnation_counter})"
        )
