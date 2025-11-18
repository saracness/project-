"""
NEAT Configuration - Hyperparameters and settings
"""
from dataclasses import dataclass


@dataclass
class NEATConfig:
    """
    NEAT hyperparameters.

    Based on original NEAT paper with modern tweaks.
    """

    # Population
    population_size: int = 150

    # Mutation probabilities
    prob_add_node: float = 0.03
    prob_add_connection: float = 0.05
    prob_mutate_weight: float = 0.8
    prob_weight_uniform: float = 0.9
    prob_weight_replace: float = 0.1
    weight_mutation_power: float = 0.5

    # Compatibility distance coefficients
    c1: float = 1.0  # Excess genes
    c2: float = 1.0  # Disjoint genes
    c3: float = 0.4  # Weight difference

    # Speciation
    compatibility_threshold: float = 3.0
    species_stagnation_threshold: int = 15

    # Reproduction
    survival_threshold: float = 0.2  # Top 20% reproduce
    elitism_threshold: int = 5  # Copy champion if species size > 5
    crossover_prob: float = 0.75  # Probability of crossover vs mutation

    # Activation functions
    activation_default: str = 'sigmoid'
    activation_options: list = None

    def __post_init__(self):
        if self.activation_options is None:
            self.activation_options = ['sigmoid', 'tanh', 'relu']

    def validate(self):
        """Validate configuration parameters."""
        assert 0 < self.prob_add_node < 1
        assert 0 < self.prob_add_connection < 1
        assert 0 <= self.prob_mutate_weight <= 1
        assert 0 < self.compatibility_threshold
        assert 0 < self.survival_threshold < 1
        assert self.population_size > 0
        assert self.species_stagnation_threshold > 0

    def __repr__(self):
        return (
            f"NEATConfig(\n"
            f"  population={self.population_size},\n"
            f"  add_node={self.prob_add_node},\n"
            f"  add_conn={self.prob_add_connection},\n"
            f"  compat_threshold={self.compatibility_threshold}\n"
            f")"
        )
