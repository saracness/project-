"""
Particle System
Visual feedback for events (food consumption, death, reproduction)
"""
import numpy as np
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple


class ParticleType(Enum):
    """Types of particles for different events."""
    FOOD_CONSUME = "food_consume"
    DEATH = "death"
    REPRODUCTION = "reproduction"
    ENERGY_GAIN = "energy_gain"
    ENERGY_LOSS = "energy_loss"


@dataclass
class Particle:
    """
    A single particle with physics.
    """
    x: float
    y: float
    vx: float  # Velocity X
    vy: float  # Velocity Y
    color: Tuple[float, float, float]
    size: float
    lifetime: float
    max_lifetime: float
    particle_type: ParticleType

    def update(self, dt=1.0):
        """Update particle physics."""
        # Apply velocity
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Apply gravity (slight downward pull)
        self.vy -= 0.5 * dt

        # Air resistance
        self.vx *= 0.95
        self.vy *= 0.95

        # Decrease lifetime
        self.lifetime -= dt

    def is_alive(self):
        """Check if particle is still alive."""
        return self.lifetime > 0

    def get_alpha(self):
        """Get alpha based on remaining lifetime."""
        return self.lifetime / self.max_lifetime

    def get_current_size(self):
        """Get size (shrinks over time)."""
        return self.size * self.get_alpha()


class ParticleSystem:
    """
    Manages all particles in the simulation.
    """

    def __init__(self, max_particles=1000, enabled=True):
        self.max_particles = max_particles
        self.enabled = enabled
        self.particles: List[Particle] = []

        # Particle configs
        self.configs = {
            ParticleType.FOOD_CONSUME: {
                'color': (0.0, 1.0, 0.0),  # Green
                'size': 5,
                'lifetime': 0.5,
                'count': 8,
                'speed': 3.0
            },
            ParticleType.DEATH: {
                'color': (1.0, 0.0, 0.0),  # Red
                'size': 8,
                'lifetime': 1.0,
                'count': 20,
                'speed': 5.0
            },
            ParticleType.REPRODUCTION: {
                'color': (0.0, 0.6, 1.0),  # Blue
                'size': 6,
                'lifetime': 0.8,
                'count': 12,
                'speed': 4.0
            },
            ParticleType.ENERGY_GAIN: {
                'color': (1.0, 1.0, 0.0),  # Yellow
                'size': 4,
                'lifetime': 0.4,
                'count': 5,
                'speed': 2.0
            },
            ParticleType.ENERGY_LOSS: {
                'color': (1.0, 0.5, 0.0),  # Orange
                'size': 3,
                'lifetime': 0.3,
                'count': 3,
                'speed': 1.5
            }
        }

    def emit(self, particle_type: ParticleType, x, y):
        """
        Emit particles at a location.

        Args:
            particle_type: Type of particle
            x, y: Location
        """
        if not self.enabled:
            return

        config = self.configs[particle_type]

        # Create particles
        for _ in range(config['count']):
            # Random velocity (explosion pattern)
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(0.5, 1.0) * config['speed']
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed

            # Add some randomness to position
            px = x + random.uniform(-2, 2)
            py = y + random.uniform(-2, 2)

            # Create particle
            particle = Particle(
                x=px,
                y=py,
                vx=vx,
                vy=vy,
                color=config['color'],
                size=config['size'],
                lifetime=config['lifetime'],
                max_lifetime=config['lifetime'],
                particle_type=particle_type
            )

            self.particles.append(particle)

        # Limit particle count
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

    def update(self, dt=1.0):
        """Update all particles."""
        if not self.enabled:
            return

        # Update all particles
        for particle in self.particles:
            particle.update(dt)

        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]

    def render(self, ax):
        """
        Render all particles on matplotlib axis.

        Args:
            ax: Matplotlib axis
        """
        if not self.enabled or not self.particles:
            return

        # Batch render particles
        for particle in self.particles:
            alpha = particle.get_alpha()
            size = particle.get_current_size()

            # Create RGBA color
            r, g, b = particle.color
            color = (r, g, b, alpha)

            # Draw circle
            circle = plt.Circle(
                (particle.x, particle.y),
                radius=size,
                color=color,
                zorder=10  # Draw on top
            )
            ax.add_patch(circle)

    def clear(self):
        """Clear all particles."""
        self.particles.clear()

    def get_particle_count(self):
        """Get current number of particles."""
        return len(self.particles)

    def set_enabled(self, enabled):
        """Enable or disable particle system."""
        self.enabled = enabled
        if not enabled:
            self.clear()

    def get_stats(self):
        """Get particle system statistics."""
        type_counts = {}
        for particle in self.particles:
            ptype = particle.particle_type.value
            type_counts[ptype] = type_counts.get(ptype, 0) + 1

        return {
            'total': len(self.particles),
            'by_type': type_counts,
            'max_particles': self.max_particles,
            'enabled': self.enabled
        }
