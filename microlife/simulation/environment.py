"""
Environment class for Phase 1
Manages the simulation world and organisms
"""
import random
from .organism import Organism, Food


class Environment:
    """
    The simulation environment containing organisms and food.

    Attributes:
        width (float): Environment width
        height (float): Environment height
        organisms (list): List of Organism instances
        food_particles (list): List of Food instances
        timestep (int): Current simulation timestep
    """

    def __init__(self, width=500, height=500):
        """
        Initialize the environment.

        Args:
            width (float): Environment width
            height (float): Environment height
        """
        self.width = width
        self.height = height
        self.organisms = []
        self.food_particles = []
        self.timestep = 0
        self.history = []

    def add_organism(self, organism=None, x=None, y=None):
        """
        Add an organism to the environment.

        Args:
            organism (Organism): Existing organism to add
            x (float): X position if creating new organism
            y (float): Y position if creating new organism
        """
        if organism:
            self.organisms.append(organism)
        else:
            if x is None:
                x = random.uniform(0, self.width)
            if y is None:
                y = random.uniform(0, self.height)
            self.organisms.append(Organism(x, y))

    def add_food(self, x=None, y=None, energy=20.0):
        """
        Add food to the environment.

        Args:
            x (float): X position (random if None)
            y (float): Y position (random if None)
            energy (float): Energy value of food
        """
        if x is None:
            x = random.uniform(0, self.width)
        if y is None:
            y = random.uniform(0, self.height)
        self.food_particles.append(Food(x, y, energy))

    def spawn_food(self, count=5):
        """
        Spawn multiple random food particles.

        Args:
            count (int): Number of food particles to spawn
        """
        for _ in range(count):
            self.add_food()

    def update(self):
        """
        Update simulation by one timestep.
        Moves organisms, handles eating, reproduction, and death.
        """
        self.timestep += 1
        bounds = (self.width, self.height)

        # Move all organisms
        for organism in self.organisms:
            if organism.alive:
                organism.move_random(bounds)

                # Check for food consumption
                self._check_food_consumption(organism)

                # Check for reproduction
                if organism.can_reproduce():
                    offspring = organism.reproduce()
                    if offspring:
                        self.organisms.append(offspring)

        # Remove dead organisms periodically
        if self.timestep % 100 == 0:
            self.organisms = [o for o in self.organisms if o.alive]

        # Remove consumed food
        self.food_particles = [f for f in self.food_particles if not f.consumed]

        # Spawn new food periodically
        if self.timestep % 20 == 0:
            self.spawn_food(count=2)

    def _check_food_consumption(self, organism):
        """
        Check if organism is near food and consume it.

        Args:
            organism (Organism): The organism to check
        """
        eating_distance = 5.0

        for food in self.food_particles:
            if food.consumed:
                continue

            # Calculate distance to food
            dx = organism.x - food.x
            dy = organism.y - food.y
            distance = (dx**2 + dy**2) ** 0.5

            if distance < eating_distance:
                organism.eat(food.energy)
                food.consumed = True
                break  # Only eat one food per timestep

    def get_statistics(self):
        """
        Get current simulation statistics.

        Returns:
            dict: Statistics about the simulation
        """
        alive_organisms = [o for o in self.organisms if o.alive]

        if alive_organisms:
            avg_energy = sum(o.energy for o in alive_organisms) / len(alive_organisms)
            avg_age = sum(o.age for o in alive_organisms) / len(alive_organisms)
        else:
            avg_energy = 0
            avg_age = 0

        return {
            'timestep': self.timestep,
            'population': len(alive_organisms),
            'total_organisms': len(self.organisms),
            'food_count': len([f for f in self.food_particles if not f.consumed]),
            'avg_energy': avg_energy,
            'avg_age': avg_age
        }

    def reset(self):
        """Reset the environment to initial state."""
        self.organisms = []
        self.food_particles = []
        self.timestep = 0
        self.history = []

    def __repr__(self):
        stats = self.get_statistics()
        return (f"Environment(size={self.width}x{self.height}, "
                f"population={stats['population']}, "
                f"timestep={self.timestep})")
