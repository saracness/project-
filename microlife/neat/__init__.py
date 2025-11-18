"""
NEAT: NeuroEvolution of Augmenting Topologies

Production-grade implementation for evolving neural network topologies.
"""
from .genome import NodeGene, ConnectionGene, NEATGenome
from .species import Species
from .population import NEATPopulation
from .config import NEATConfig

__all__ = [
    'NodeGene',
    'ConnectionGene',
    'NEATGenome',
    'Species',
    'NEATPopulation',
    'NEATConfig',
]
__version__ = '1.0.0'
