"""
🏆 AI BATTLE ARENA - Yapay Zeka Savaşı Demo
8 farklı AI modeli aynı ortamda yarışıyor!
Hangisi en uzun süre hayatta kalacak?
"""
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.visualization.simple_renderer import SimpleRenderer
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain
from microlife.ml.brain_cnn import CNNBrain, ResidualCNNBrain
from microlife.ml.brain_evolutionary import GeneticAlgorithmBrain, NEATBrain, CMAESBrain
import matplotlib.pyplot as plt
import math


class AIOrganismWithBrain(Organism):
    """Extended organism that uses AI brain for decisions."""

    def __init__(self, x, y, brain, **kwargs):
        super().__init__(x, y, **kwargs)
        self.brain = brain
        self.brain_type = brain.brain_type
        self.old_state = None

    def move_with_brain(self, environment):
        """Move using AI brain decision."""
        if not self.alive:
            return

        # Build state for AI
        state = self._build_state(environment)

        # AI decides action
        action = self.brain.decide_action(state)

        # Execute action
        dx, dy = action.get('move_direction', (0, 0))
        speed_mult = action.get('speed_multiplier', 1.0)

        self.x += dx * self.speed * speed_mult
        self.y += dy * self.speed * speed_mult

        # Apply boundaries
        self.x = max(0, min(environment.width, self.x))
        self.y = max(0, min(environment.height, self.y))

        # Update after movement
        self._update_after_movement()

        # Learn from experience
        if self.old_state is not None:
            new_state = self._build_state(environment)
            reward = self.brain.calculate_reward(self.old_state, new_state, action)
            self.brain.learn(self.old_state, action, reward, new_state, not self.alive)

        self.old_state = state

        # Handle reproduction from AI decision
        if action.get('should_reproduce', False) and self.can_reproduce():
            return self.reproduce()

        return None

    def _build_state(self, environment):
        """Build state dict for AI brain."""
        # Find nearest food
        nearest_food_dist = float('inf')
        nearest_food_angle = 0.0

        for food in environment.food_particles:
            if food.consumed:
                continue
            dx = food.x - self.x
            dy = food.y - self.y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < nearest_food_dist:
                nearest_food_dist = dist
                nearest_food_angle = math.atan2(dy, dx)

        # Check temperature zone
        in_temp_zone = any(zone.affects(self) for zone in environment.temperature_zones)

        # Check obstacles
        near_obstacle = any(
            abs(self.x - obs.x) < 50 and abs(self.y - obs.y) < 50
            for obs in environment.obstacles
        )

        return {
            'energy': self.energy,
            'nearest_food_distance': nearest_food_dist,
            'nearest_food_angle': nearest_food_angle,
            'in_temperature_zone': in_temp_zone,
            'near_obstacle': near_obstacle,
            'age': self.age,
            'speed': self.speed,
            'x': self.x,
            'y': self.y
        }


def create_ai_organisms(environment):
    """Create organisms with different AI brains."""
    organisms = []
    brain_colors = {}

    # Positions for different AI types (spread around)
    positions = [
        (100, 100), (400, 100), (100, 400), (400, 400),
        (250, 100), (100, 250), (400, 250), (250, 400)
    ]

    print("\n🧠 Creating AI Organisms:")
    print("=" * 60)

    # 1. Q-Learning
    for i in range(2):
        brain = QLearningBrain(learning_rate=0.1, epsilon=0.3)
        org = AIOrganismWithBrain(positions[0][0] + i*20, positions[0][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#FF6B6B'  # Red
    print("✓ 2x Q-Learning (Red)")

    # 2. DQN
    for i in range(2):
        brain = DQNBrain(state_size=7, hidden_size=24)
        org = AIOrganismWithBrain(positions[1][0] + i*20, positions[1][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#4ECDC4'  # Cyan
    print("✓ 2x DQN (Cyan)")

    # 3. Double DQN
    for i in range(2):
        brain = DoubleDQNBrain(state_size=7, hidden_size=24)
        org = AIOrganismWithBrain(positions[2][0] + i*20, positions[2][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#95E1D3'  # Light Green
    print("✓ 2x Double-DQN (Light Green)")

    # 4. CNN
    for i in range(2):
        brain = CNNBrain(grid_size=15)
        org = AIOrganismWithBrain(positions[3][0] + i*20, positions[3][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#F38181'  # Pink
    print("✓ 2x CNN (Pink)")

    # 5. Genetic Algorithm
    for i in range(2):
        brain = GeneticAlgorithmBrain(genome_size=20, mutation_rate=0.1)
        org = AIOrganismWithBrain(positions[4][0] + i*20, positions[4][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#AA96DA'  # Purple
    print("✓ 2x Genetic Algorithm (Purple)")

    # 6. NEAT
    for i in range(2):
        brain = NEATBrain(input_size=7, output_size=9)
        org = AIOrganismWithBrain(positions[5][0] + i*20, positions[5][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#FCBAD3'  # Light Pink
    print("✓ 2x NEAT (Light Pink)")

    # 7. CMA-ES
    for i in range(2):
        brain = CMAESBrain(param_size=20)
        org = AIOrganismWithBrain(positions[6][0] + i*20, positions[6][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#FFFFD2'  # Light Yellow
    print("✓ 2x CMA-ES (Light Yellow)")

    # 8. ResNet-CNN
    for i in range(2):
        brain = ResidualCNNBrain(grid_size=15)
        org = AIOrganismWithBrain(positions[7][0] + i*20, positions[7][1] + i*20, brain, energy=120)
        organisms.append(org)
        brain_colors[id(org)] = '#A8D8EA'  # Light Blue
    print("✓ 2x ResNet-CNN (Light Blue)")

    print("=" * 60)
    print(f"Total: {len(organisms)} AI organisms created!\n")

    return organisms, brain_colors


class AIBattleRenderer(SimpleRenderer):
    """Custom renderer for AI battle with color coding."""

    def __init__(self, environment, brain_colors):
        super().__init__(environment)
        self.brain_colors = brain_colors

    def render_frame(self):
        """Render with AI brain colors."""
        self.ax.clear()
        self.setup_plot()

        # Draw environment
        for zone in self.env.temperature_zones:
            color = '#ff6b6b' if zone.temperature > 0 else '#4dabf7'
            from matplotlib.patches import Circle
            circle = Circle((zone.x, zone.y), zone.radius,
                          color=color, alpha=0.15, linestyle='--', fill=True)
            self.ax.add_patch(circle)

        for obstacle in self.env.obstacles:
            from matplotlib.patches import Rectangle
            rect = Rectangle((obstacle.x, obstacle.y),
                           obstacle.width, obstacle.height,
                           color='#555555', alpha=0.7)
            self.ax.add_patch(rect)

        # Draw food
        for food in self.env.food_particles:
            if not food.consumed:
                from matplotlib.patches import Circle
                circle = Circle((food.x, food.y), 2, color='#00ff00', alpha=0.6)
                self.ax.add_patch(circle)

        # Draw AI organisms with brain colors
        brain_counts = {}
        for organism in self.env.organisms:
            if organism.alive and hasattr(organism, 'brain_type'):
                # Use brain type color
                color = self.brain_colors.get(id(organism), '#FFFFFF')

                from matplotlib.patches import Circle
                circle = Circle((organism.x, organism.y),
                              organism.size,
                              color=color,
                              alpha=0.9,
                              edgecolor='black',
                              linewidth=0.5)
                self.ax.add_patch(circle)

                # Count brain types
                brain_type = organism.brain_type
                brain_counts[brain_type] = brain_counts.get(brain_type, 0) + 1

        # Statistics with brain counts
        stats = self.env.get_statistics()
        stats_text = f"Timestep: {stats['timestep']}\n"
        stats_text += f"Total Population: {stats['population']}\n"
        stats_text += f"Food: {stats['food_count']}\n"
        stats_text += "\n🧠 AI Brain Survivors:\n"

        for brain_type, count in sorted(brain_counts.items()):
            stats_text += f"{brain_type}: {count}\n"

        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='black',
                            alpha=0.8),
                    color='white',
                    family='monospace')

        self.ax.set_title('🏆 AI Battle Arena - Who Survives?',
                         color='white',
                         fontsize=14,
                         pad=20,
                         weight='bold')

        plt.tight_layout()


def main():
    """Run AI Battle Arena."""
    print("=" * 70)
    print("🏆 AI BATTLE ARENA - YAPAY ZEKA SAVAŞI")
    print("=" * 70)
    print("\n8 Farklı AI Modeli Hayatta Kalma Yarışında!")
    print("\n🧠 AI Models:")
    print("  🔴 Red         → Q-Learning")
    print("  🔵 Cyan        → DQN")
    print("  🟢 Light Green → Double-DQN")
    print("  🔴 Pink        → CNN")
    print("  🟣 Purple      → Genetic Algorithm")
    print("  🌸 Light Pink  → NEAT")
    print("  🟡 Light Yellow→ CMA-ES")
    print("  🔵 Light Blue  → ResNet-CNN")
    print("\n" + "=" * 70)

    # Create environment
    print("\n[1/4] Creating battle arena...")
    env = Environment(width=500, height=500, use_intelligent_movement=False)

    # Add challenges
    env.add_obstacle(x=200, y=200, width=100, height=20)
    env.add_obstacle(x=350, y=100, width=20, height=150)
    env.add_temperature_zone(x=100, y=100, radius=60, temperature=1)
    env.add_temperature_zone(x=400, y=400, radius=70, temperature=-1)
    env.spawn_food(count=50)

    # Create AI organisms
    print("\n[2/4] Creating AI organisms...")
    ai_organisms, brain_colors = create_ai_organisms(env)
    for org in ai_organisms:
        env.organisms.append(org)

    # Create renderer
    print("\n[3/4] Setting up visualization...")
    renderer = AIBattleRenderer(env, brain_colors)

    print("\n[4/4] Starting battle...")
    print("\n" + "=" * 70)
    print("⚔️  BATTLE STARTING!")
    print("=" * 70)
    print("\n🎯 Watch which AI survives the longest!")
    print("💡 Each color represents a different AI model")
    print("📊 Statistics show how many of each AI are still alive")
    print("\nClose window to end battle.\n")

    # Custom update function
    def update_battle(frame):
        # Move AI organisms with their brains
        new_organisms = []
        for organism in env.organisms[:]:
            if organism.alive and hasattr(organism, 'move_with_brain'):
                offspring = organism.move_with_brain(env)
                if offspring:
                    # Create AI version of offspring
                    ai_offspring = AIOrganismWithBrain(
                        offspring.x, offspring.y,
                        organism.brain,  # Use same brain
                        energy=offspring.energy,
                        speed=offspring.speed
                    )
                    new_organisms.append(ai_offspring)
                    brain_colors[id(ai_offspring)] = brain_colors.get(id(organism), '#FFFFFF')

        env.organisms.extend(new_organisms)

        # Standard environment updates
        for organism in env.organisms:
            if organism.alive:
                env._check_food_consumption(organism)
                env._apply_temperature_effects(organism)

        env.timestep += 1

        # Remove dead organisms periodically
        if env.timestep % 50 == 0:
            env.organisms = [o for o in env.organisms if o.alive]

        # Spawn food
        if env.timestep % 20 == 0:
            env.spawn_food(count=3)

        renderer.render_frame()
        return renderer.ax.patches

    # Run animation
    import matplotlib.animation as animation
    anim = animation.FuncAnimation(
        renderer.fig,
        update_battle,
        frames=2000,
        interval=50,
        blit=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n\nBattle interrupted!")

    # Final results
    print("\n" + "=" * 70)
    print("🏆 BATTLE RESULTS")
    print("=" * 70)

    brain_survivors = {}
    for org in env.organisms:
        if org.alive and hasattr(org, 'brain_type'):
            brain_type = org.brain_type
            brain_survivors[brain_type] = brain_survivors.get(brain_type, 0) + 1

    print("\n🥇 Survivors by AI Type:")
    for brain_type, count in sorted(brain_survivors.items(), key=lambda x: x[1], reverse=True):
        print(f"  {brain_type}: {count} survivors")

    if brain_survivors:
        winner = max(brain_survivors.items(), key=lambda x: x[1])
        print(f"\n👑 WINNER: {winner[0]} with {winner[1]} survivors!")
    else:
        print("\n💀 No survivors! Everyone died!")

    print("\n" + "=" * 70)
    print("✅ Battle Complete! Check which AI dominated!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
