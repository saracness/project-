"""
Enhanced Environment class for Phase 2
Manages the simulation world with temperature zones and obstacles
"""
import random
import math
from .organism import Organism, Food


class TemperatureZone:
    """
    Temperature zone that affects organism energy (Phase 2).
    """
    def __init__(self, x, y, radius, temperature):
        """
        Args:
            x, y: Center position
            radius: Zone radius
            temperature: Hot (positive) or cold (negative)
        """
        self.x = x
        self.y = y
        self.radius = radius
        self.temperature = temperature  # +1 = hot, -1 = cold

    def affects(self, organism):
        """Check if organism is in this zone."""
        dx = organism.x - self.x
        dy = organism.y - self.y
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= self.radius

    def get_energy_effect(self):
        """Energy drain/gain from being in this zone."""
        return -0.05 * abs(self.temperature)  # Hot/cold zones drain energy


class Obstacle:
    """
    Obstacle/wall that blocks organism movement (Phase 2).
    """
    def __init__(self, x, y, width, height):
        """
        Args:
            x, y: Top-left corner
            width, height: Obstacle dimensions
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def collides_with(self, organism):
        """Check if organism collides with this obstacle."""
        return (self.x <= organism.x <= self.x + self.width and
                self.y <= organism.y <= self.y + self.height)

    def get_bounds(self):
        """Get obstacle boundaries."""
        return (self.x, self.y, self.x + self.width, self.y + self.height)


class Environment:
    """
    The simulation environment containing organisms and food.

    Attributes:
        width (float): Environment width
        height (float): Environment height
        organisms (list): List of Organism instances
        food_particles (list): List of Food instances
        temperature_zones (list): List of TemperatureZone instances
        obstacles (list): List of Obstacle instances
        timestep (int): Current simulation timestep
        use_intelligent_movement (bool): Use Phase 2 intelligent behavior
    """

    def __init__(self, width=500, height=500, use_intelligent_movement=True):
        """
        Initialize the environment.

        Args:
            width (float): Environment width
            height (float): Environment height
            use_intelligent_movement (bool): Enable Phase 2 behaviors
        """
        self.width = width
        self.height = height
        self.organisms = []
        self.food_particles = []
        self.temperature_zones = []
        self.obstacles = []
        self.timestep = 0
        self.history = []
        self.use_intelligent_movement = use_intelligent_movement

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

    def add_temperature_zone(self, x=None, y=None, radius=50, temperature=1):
        """
        Add a temperature zone to the environment.

        Args:
            x, y: Center position (random if None)
            radius: Zone radius
            temperature: +1 for hot, -1 for cold
        """
        if x is None:
            x = random.uniform(0, self.width)
        if y is None:
            y = random.uniform(0, self.height)
        self.temperature_zones.append(TemperatureZone(x, y, radius, temperature))

    def add_obstacle(self, x=None, y=None, width=50, height=50):
        """
        Add an obstacle to the environment.

        Args:
            x, y: Top-left corner (random if None)
            width, height: Obstacle dimensions
        """
        if x is None:
            x = random.uniform(0, self.width - width)
        if y is None:
            y = random.uniform(0, self.height - height)
        self.obstacles.append(Obstacle(x, y, width, height))

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
                # Use AI brain if available
                if hasattr(organism, 'brain') and organism.brain:
                    self._move_with_ai(organism)
                elif self.use_intelligent_movement:
                    organism.move_intelligent(self.food_particles, bounds, self.obstacles)
                else:
                    organism.move_random(bounds)

                # Check for obstacle collision
                self._handle_obstacle_collision(organism)

                # Apply temperature zone effects
                self._apply_temperature_effects(organism)

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

    def _move_with_ai(self, organism):
        """Move organism using AI brain."""
        import math

        # Get current state
        old_energy = organism.energy
        nearest_food = None
        min_dist = float('inf')

        for food in self.food_particles:
            if not food.consumed:
                dist = math.sqrt((food.x - organism.x)**2 + (food.y - organism.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_food = food

        # Calculate food angle
        food_angle = 0
        if nearest_food:
            food_angle = math.atan2(nearest_food.y - organism.y, nearest_food.x - organism.x)

        # Prepare state for AI
        state = {
            'energy': organism.energy,
            'nearest_food_distance': min_dist if nearest_food else 999,
            'nearest_food_angle': food_angle,
            'in_temperature_zone': any(z.affects(organism) for z in self.temperature_zones),
            'near_obstacle': any(o.collides_with(organism) for o in self.obstacles),
            'age': organism.age,
            'speed_multiplier': organism.morphology.speed_multiplier if hasattr(organism, 'morphology') else 1.0,
            'perception': organism.morphology.perception_multiplier if hasattr(organism, 'morphology') else 1.0
        }

        # Get action from AI
        action = organism.brain.decide_action(state)

        # Apply movement
        dx, dy = action.get('move_direction', (0, 0))
        speed = organism.speed * action.get('speed_multiplier', 1.0)
        organism.x += dx * speed
        organism.y += dy * speed

        # Keep in bounds
        organism.x = max(0, min(self.width, organism.x))
        organism.y = max(0, min(self.height, organism.y))

        # Update internal state
        organism._update_after_movement()

        # Update brain statistics
        organism.brain.survival_time += 1
        organism.brain.decision_count += 1

        # Calculate reward and learn
        new_state = state.copy()
        new_state['energy'] = organism.energy
        reward = organism.energy - old_energy  # Reward = energy gain
        if reward > 15:  # Found food
            reward += 5  # Bonus

        organism.brain.total_reward += reward
        organism.brain.learn(state, action, reward, new_state, not organism.alive)

    def _handle_obstacle_collision(self, organism):
        """
        Handle organism collision with obstacles.

        Args:
            organism (Organism): The organism to check
        """
        for obstacle in self.obstacles:
            if obstacle.collides_with(organism):
                # Push organism out of obstacle
                x1, y1, x2, y2 = obstacle.get_bounds()
                # Simple collision response: move to nearest edge
                if len(organism.trail) > 1:
                    prev_x, prev_y = organism.trail[-2]
                    organism.x = prev_x
                    organism.y = prev_y

    def _apply_temperature_effects(self, organism):
        """
        Apply temperature zone effects to organism.

        Args:
            organism (Organism): The organism to affect
        """
        for zone in self.temperature_zones:
            if zone.affects(organism):
                organism.energy += zone.get_energy_effect()

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
            # Count behaviors
            seeking_count = sum(1 for o in alive_organisms if o.behavior_mode == "seeking")
            wandering_count = sum(1 for o in alive_organisms if o.behavior_mode == "wandering")
        else:
            avg_energy = 0
            avg_age = 0
            seeking_count = 0
            wandering_count = 0

        return {
            'timestep': self.timestep,
            'population': len(alive_organisms),
            'total_organisms': len(self.organisms),
            'food_count': len([f for f in self.food_particles if not f.consumed]),
            'avg_energy': avg_energy,
            'avg_age': avg_age,
            'seeking_count': seeking_count,
            'wandering_count': wandering_count,
            'temperature_zones': len(self.temperature_zones),
            'obstacles': len(self.obstacles)
        }

    def reset(self):
        """Reset the environment to initial state."""
        self.organisms = []
        self.food_particles = []
        self.temperature_zones = []
        self.obstacles = []
        self.timestep = 0
        self.history = []

    def __repr__(self):
        stats = self.get_statistics()
        return (f"Environment(size={self.width}x{self.height}, "
                f"population={stats['population']}, "
                f"timestep={self.timestep})")
