"""
Visual Effects System
"""
from .trails import TrailSystem
from .particles import ParticleSystem, Particle, ParticleType
from .heatmap import HeatmapGenerator
from .minimap import MiniMap

__all__ = [
    'TrailSystem',
    'ParticleSystem',
    'Particle',
    'ParticleType',
    'HeatmapGenerator',
    'MiniMap'
]
