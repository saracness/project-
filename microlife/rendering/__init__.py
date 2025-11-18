"""
ModernGL-based high-performance renderer
Production-grade OpenGL rendering for micro-life simulation

Performance Target: 100+ FPS with 1000+ organisms
"""
from .gl_renderer import GLRenderer
from .camera import Camera

__all__ = ['GLRenderer', 'Camera']
__version__ = '1.0.0'
