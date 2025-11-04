"""
Basic Organism class for Phase 1
Simple organism with random movement
"""
import random
import numpy as np


class Organism:
    """
    Represents a single micro-organism in the simulation.

    Attributes:
        x (float): X position in the environment
        y (float): Y position in the environment
        energy (float): Current energy level (dies at 0)
        speed (float): Movement speed
        size (float): Visual size/radius
        alive (bool): Whether organism is alive
    """

    def __init__(self, x, y, energy=100.0, speed=1.0, size=3.0):
        """
        Initialize a new organism.

        Args:
            x (float): Starting X position
            y (float): Starting Y position
            energy (float): Starting energy level
            speed (float): Movement speed
            size (float): Visual size
        """
        self.x = x
        self.y = y
        self.energy = energy
        self.speed = speed
        self.size = size
        self.alive = True
        self.age = 0

        # Movement history for visualization
        self.trail = [(x, y)]
        self.max_trail_length = 50

    def move_random(self, bounds=None):
        """
        Move in a random direction.

        Args:
            bounds (tuple): (width, height) to constrain movement
        """
        if not self.alive:
            return

        # Random movement
        dx = random.uniform(-self.speed, self.speed)
        dy = random.uniform(-self.speed, self.speed)

        self.x += dx
        self.y += dy

        # Apply boundaries if provided
        if bounds:
            width, height = bounds
            self.x = max(0, min(width, self.x))
            self.y = max(0, min(height, self.y))

        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        # Consume energy for movement
        self.energy -= 0.1
        self.age += 1

        # Check if still alive
        if self.energy <= 0:
            self.alive = False

    def eat(self, food_energy=20.0):
        """
        Consume food and gain energy.

        Args:
            food_energy (float): Amount of energy to gain
        """
        if self.alive:
            self.energy += food_energy
            self.energy = min(self.energy, 200.0)  # Max energy cap

    def can_reproduce(self, threshold=150.0):
        """
        Check if organism has enough energy to reproduce.

        Args:
            threshold (float): Energy required for reproduction

        Returns:
            bool: True if can reproduce
        """
        return self.alive and self.energy >= threshold

    def reproduce(self, energy_cost=50.0):
        """
        Create offspring, consuming energy.

        Args:
            energy_cost (float): Energy spent on reproduction

        Returns:
            Organism: New offspring organism
        """
        if not self.can_reproduce():
            return None

        # Consume energy
        self.energy -= energy_cost

        # Create offspring near parent with slight variation
        offset = 5.0
        child_x = self.x + random.uniform(-offset, offset)
        child_y = self.y + random.uniform(-offset, offset)
        child_speed = self.speed * random.uniform(0.8, 1.2)

        return Organism(child_x, child_y, energy=energy_cost, speed=child_speed)

    def get_state(self):
        """
        Get current state for data logging.

        Returns:
            dict: Current organism state
        """
        return {
            'x': self.x,
            'y': self.y,
            'energy': self.energy,
            'speed': self.speed,
            'size': self.size,
            'alive': self.alive,
            'age': self.age
        }

    def __repr__(self):
        status = "Alive" if self.alive else "Dead"
        return f"Organism(pos=({self.x:.1f}, {self.y:.1f}), energy={self.energy:.1f}, {status})"


class Food:
    """
    Food particle that organisms can consume.
    """

    def __init__(self, x, y, energy=20.0):
        """
        Initialize food particle.

        Args:
            x (float): X position
            y (float): Y position
            energy (float): Energy provided when consumed
        """
        self.x = x
        self.y = y
        self.energy = energy
        self.consumed = False

    def get_position(self):
        """Get food position as tuple."""
        return (self.x, self.y)

    def __repr__(self):
        return f"Food(pos=({self.x:.1f}, {self.y:.1f}), energy={self.energy})"
