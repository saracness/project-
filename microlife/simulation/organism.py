"""
Enhanced Organism class for Phase 2+
Organisms with intelligent food-seeking behavior and morphology
"""
import random
import math
from .morphology import Morphology, create_random_morphology


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

    def __init__(self, x, y, energy=100.0, speed=1.0, size=3.0, morphology=None):
        """
        Initialize a new organism.

        Args:
            x (float): Starting X position
            y (float): Starting Y position
            energy (float): Starting energy level
            speed (float): Movement speed
            size (float): Visual size
            morphology (Morphology): Physical characteristics (optional)
        """
        self.x = x
        self.y = y
        self.energy = energy
        self.alive = True
        self.age = 0

        # Morphology system (new!)
        if morphology is None:
            morphology = create_random_morphology()
        self.morphology = morphology

        # Apply morphology advantages
        self.speed = speed * self.morphology.speed_multiplier
        self.size = self.morphology.visual_size
        self.color = self.morphology.color

        # Movement history for visualization
        self.trail = [(x, y)]
        self.max_trail_length = 50

        # Phase 2: Behavior attributes
        self.hunger_threshold = 100.0  # Seek food when energy below this
        self.perception_radius = 100.0 * self.morphology.perception_multiplier
        self.behavior_mode = "seeking"  # seeking, fleeing, wandering

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

        self._update_after_movement()

    def move_towards(self, target_x, target_y, bounds=None):
        """
        Move towards a target position (Phase 2: Food seeking).

        Args:
            target_x (float): Target X position
            target_y (float): Target Y position
            bounds (tuple): (width, height) to constrain movement
        """
        if not self.alive:
            return

        # Calculate direction to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx**2 + dy**2)

        if distance > 0:
            # Normalize and scale by speed
            dx = (dx / distance) * self.speed
            dy = (dy / distance) * self.speed

            self.x += dx
            self.y += dy

        # Apply boundaries
        if bounds:
            width, height = bounds
            self.x = max(0, min(width, self.x))
            self.y = max(0, min(height, self.y))

        self._update_after_movement()

    def move_intelligent(self, food_list, bounds=None, obstacles=None):
        """
        Move intelligently based on environment (Phase 2).
        Seeks food when hungry, otherwise wanders.

        Args:
            food_list (list): List of Food objects
            bounds (tuple): (width, height) to constrain movement
            obstacles (list): List of obstacle objects
        """
        if not self.alive:
            return

        # Check if hungry and food is available
        if self.energy < self.hunger_threshold and food_list:
            nearest_food = self._find_nearest_food(food_list)
            if nearest_food:
                # Calculate distance to nearest food
                dx = nearest_food.x - self.x
                dy = nearest_food.y - self.y
                distance = math.sqrt(dx**2 + dy**2)

                # Only seek if within perception radius
                if distance <= self.perception_radius:
                    self.behavior_mode = "seeking"
                    self.move_towards(nearest_food.x, nearest_food.y, bounds)
                    return

        # Otherwise, wander randomly
        self.behavior_mode = "wandering"
        self.move_random(bounds)

    def _find_nearest_food(self, food_list):
        """
        Find the nearest food particle.

        Args:
            food_list (list): List of Food objects

        Returns:
            Food: Nearest food object or None
        """
        if not food_list:
            return None

        min_distance = float('inf')
        nearest_food = None

        for food in food_list:
            if food.consumed:
                continue

            dx = food.x - self.x
            dy = food.y - self.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance < min_distance:
                min_distance = distance
                nearest_food = food

        return nearest_food

    def _update_after_movement(self):
        """
        Update organism state after movement.
        """
        # Update trail
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

        # Consume energy for movement (affected by morphology)
        energy_cost = 0.1 / self.morphology.energy_efficiency
        self.energy -= energy_cost
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
        child_speed = 1.0  # Base speed (will be modified by morphology)

        # Inherit and mutate morphology
        from copy import deepcopy
        child_morphology = deepcopy(self.morphology)
        child_morphology.mutate(mutation_rate=0.15)

        return Organism(child_x, child_y, energy=energy_cost, speed=child_speed, morphology=child_morphology)

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
            'age': self.age,
            'behavior_mode': self.behavior_mode
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
