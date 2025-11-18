"""
NEAT Population - Main evolution controller

Orchestrates speciation, reproduction, and evolution.
"""
import numpy as np
from typing import List, Callable, Optional, Dict, Any
from .genome import NEATGenome, InnovationDB
from .species import Species
from .config import NEATConfig


class NEATPopulation:
    """
    NEAT population manager.

    Coordinates:
    - Speciation
    - Fitness evaluation
    - Reproduction
    - Evolution statistics
    """

    def __init__(self, config: NEATConfig, num_inputs: int, num_outputs: int):
        """
        Initialize population.

        Args:
            config: NEAT configuration
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
        """
        self.config = config
        self.config.validate()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        # Population
        self.genomes: List[NEATGenome] = []
        self.species_list: List[Species] = []

        # Innovation tracking
        self.innovation_db = InnovationDB()

        # Generation counter
        self.generation = 0

        # Species ID counter
        self.next_species_id = 0

        # Statistics
        self.best_genome: Optional[NEATGenome] = None
        self.best_fitness = -float('inf')
        self.generation_stats: List[Dict[str, Any]] = []

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population with minimal structure."""
        for _ in range(self.config.population_size):
            genome = NEATGenome(self.num_inputs, self.num_outputs)

            # Add initial connections (fully connected input to output)
            for in_node in range(self.num_inputs):
                for out_node in range(self.num_inputs, self.num_inputs + self.num_outputs):
                    innovation = self.innovation_db.get_innovation(in_node, out_node)

                    from .genome import ConnectionGene
                    conn = ConnectionGene(
                        innovation=innovation,
                        in_node=in_node,
                        out_node=out_node,
                        weight=np.random.randn() * 2.0,
                        enabled=True
                    )
                    genome.connections[innovation] = conn

            self.genomes.append(genome)

    def evaluate(self, fitness_function: Callable[[NEATGenome], float]):
        """
        Evaluate all genomes.

        Args:
            fitness_function: Function that takes genome and returns fitness
        """
        for genome in self.genomes:
            genome.fitness = fitness_function(genome)

            # Update best genome
            if genome.fitness > self.best_fitness:
                self.best_fitness = genome.fitness
                self.best_genome = genome.copy()

    def speciate(self):
        """
        Assign genomes to species based on compatibility.

        Uses compatibility distance threshold.
        """
        # Reset species membership
        for species in self.species_list:
            species.reset()

        # Remove extinct species
        self.species_list = [s for s in self.species_list if s.age > 0 or len(s.members) > 0]

        # Assign each genome to species
        for genome in self.genomes:
            found_species = False

            for species in self.species_list:
                distance = genome.distance(species.representative, self.config)

                if distance < self.config.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break

            if not found_species:
                # Create new species
                new_species = Species(self.next_species_id, genome)
                self.next_species_id += 1
                new_species.add_member(genome)
                self.species_list.append(new_species)

        # Calculate species statistics
        for species in self.species_list:
            species.calculate_adjusted_fitness()
            species.calculate_stats()

    def reproduce(self):
        """
        Create next generation through reproduction.

        Steps:
        1. Remove stagnant species
        2. Calculate offspring allocation
        3. Reproduce within species
        4. Update representatives
        """
        # Remove stagnant species
        self.species_list = [
            s for s in self.species_list
            if s.stagnation_counter < self.config.species_stagnation_threshold
        ]

        if not self.species_list:
            # Extinction! Restart from best genome
            print("⚠️  All species extinct! Restarting from best genome...")
            self._restart_from_best()
            return

        # Calculate total adjusted fitness
        total_adjusted_fitness = sum(
            sum(g.adjusted_fitness for g in species.members)
            for species in self.species_list
        )

        # Allocate offspring to species
        offspring_allocation = []
        total_allocated = 0

        for species in self.species_list:
            num_offspring = species.get_num_offspring(
                total_adjusted_fitness,
                self.config.population_size
            )
            offspring_allocation.append((species, num_offspring))
            total_allocated += num_offspring

        # Handle rounding errors
        if total_allocated < self.config.population_size:
            # Give extra offspring to best species
            best_species = max(self.species_list, key=lambda s: s.avg_fitness)
            for i, (species, num) in enumerate(offspring_allocation):
                if species == best_species:
                    offspring_allocation[i] = (species, num + (self.config.population_size - total_allocated))
                    break

        # Reset innovation database for new generation
        self.innovation_db.reset()

        # Reproduce
        new_genomes = []
        for species, num_offspring in offspring_allocation:
            offspring = species.reproduce(num_offspring, self.config, self.innovation_db)
            new_genomes.extend(offspring)

        # Update representatives
        for species in self.species_list:
            species.update_representative()

        # Replace population
        self.genomes = new_genomes[:self.config.population_size]  # Ensure exact size

        # Increment generation
        self.generation += 1

    def _restart_from_best(self):
        """Restart population from best genome (after extinction)."""
        if self.best_genome is None:
            # Complete restart
            self.genomes.clear()
            self.species_list.clear()
            self._initialize_population()
        else:
            # Clone best genome with mutations
            self.genomes = []
            for _ in range(self.config.population_size):
                genome = self.best_genome.copy()
                genome.mutate(self.config, self.innovation_db)
                self.genomes.append(genome)

            self.species_list.clear()

    def evolve(self, fitness_function: Callable[[NEATGenome], float],
               num_generations: int, verbose: bool = True):
        """
        Run evolution for multiple generations.

        Args:
            fitness_function: Function to evaluate fitness
            num_generations: Number of generations to evolve
            verbose: Print progress
        """
        for gen in range(num_generations):
            # Evaluate fitness
            self.evaluate(fitness_function)

            # Speciate
            self.speciate()

            # Collect statistics
            self._collect_stats()

            # Print progress
            if verbose and gen % 10 == 0:
                self._print_stats()

            # Reproduce
            self.reproduce()

    def _collect_stats(self):
        """Collect generation statistics."""
        avg_fitness = np.mean([g.fitness for g in self.genomes])
        max_fitness = np.max([g.fitness for g in self.genomes])
        avg_size = np.mean([sum(g.size()) for g in self.genomes])
        num_species = len(self.species_list)

        stats = {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'best_fitness': self.best_fitness,
            'avg_size': avg_size,
            'num_species': num_species,
            'population_size': len(self.genomes)
        }

        self.generation_stats.append(stats)

    def _print_stats(self):
        """Print current generation statistics."""
        if not self.generation_stats:
            return

        stats = self.generation_stats[-1]
        print(
            f"Gen {stats['generation']:4d} | "
            f"Species: {stats['num_species']:2d} | "
            f"Avg Fitness: {stats['avg_fitness']:7.2f} | "
            f"Max: {stats['max_fitness']:7.2f} | "
            f"Best Ever: {stats['best_fitness']:7.2f} | "
            f"Avg Size: {stats['avg_size']:5.1f}"
        )

    def get_best_genome(self) -> Optional[NEATGenome]:
        """Get best genome found so far."""
        return self.best_genome

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.generation_stats:
            return {}

        return {
            'total_generations': self.generation,
            'final_best_fitness': self.best_fitness,
            'final_num_species': self.generation_stats[-1]['num_species'],
            'final_avg_fitness': self.generation_stats[-1]['avg_fitness'],
            'best_genome_size': sum(self.best_genome.size()) if self.best_genome else 0
        }

    def __repr__(self):
        return (
            f"NEATPopulation(gen={self.generation}, "
            f"pop_size={len(self.genomes)}, "
            f"species={len(self.species_list)}, "
            f"best_fitness={self.best_fitness:.2f})"
        )
