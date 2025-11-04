"""
Simple matplotlib-based renderer for Phase 1
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import numpy as np


class SimpleRenderer:
    """
    Simple 2D visualization using matplotlib.
    """

    def __init__(self, environment):
        """
        Initialize renderer.

        Args:
            environment (Environment): The environment to visualize
        """
        self.env = environment
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_plot()

    def setup_plot(self):
        """Set up the plot area."""
        self.ax.set_xlim(0, self.env.width)
        self.ax.set_ylim(0, self.env.height)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#0a0a0a')
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.grid(True, alpha=0.2, color='#333333')
        self.ax.set_xlabel('X Position', color='white')
        self.ax.set_ylabel('Y Position', color='white')
        self.ax.tick_params(colors='white')

    def render_frame(self):
        """Render a single frame of the current environment state."""
        self.ax.clear()
        self.setup_plot()

        # Draw temperature zones (Phase 2)
        for zone in self.env.temperature_zones:
            color = '#ff6b6b' if zone.temperature > 0 else '#4dabf7'  # Red = hot, Blue = cold
            circle = Circle((zone.x, zone.y), zone.radius,
                          color=color, alpha=0.15, linestyle='--', fill=True)
            self.ax.add_patch(circle)

        # Draw obstacles (Phase 2)
        for obstacle in self.env.obstacles:
            from matplotlib.patches import Rectangle
            rect = Rectangle((obstacle.x, obstacle.y),
                           obstacle.width, obstacle.height,
                           color='#555555', alpha=0.7)
            self.ax.add_patch(rect)

        # Draw food
        for food in self.env.food_particles:
            if not food.consumed:
                circle = Circle((food.x, food.y), 2, color='#00ff00', alpha=0.6)
                self.ax.add_patch(circle)

        # Draw organisms
        for organism in self.env.organisms:
            if organism.alive:
                # Color based on energy (red = low, yellow = high)
                energy_ratio = min(organism.energy / 150.0, 1.0)
                color = plt.cm.plasma(energy_ratio)

                # Draw organism
                circle = Circle((organism.x, organism.y),
                              organism.size,
                              color=color,
                              alpha=0.8)
                self.ax.add_patch(circle)

                # Draw trail (optional)
                if len(organism.trail) > 1:
                    trail_x = [pos[0] for pos in organism.trail]
                    trail_y = [pos[1] for pos in organism.trail]
                    self.ax.plot(trail_x, trail_y,
                               color=color,
                               alpha=0.3,
                               linewidth=0.5)

        # Add statistics text (Phase 2 enhanced)
        stats = self.env.get_statistics()
        phase = "Phase 2" if self.env.use_intelligent_movement else "Phase 1"
        stats_text = (f"Timestep: {stats['timestep']}\n"
                     f"Population: {stats['population']}\n"
                     f"Food: {stats['food_count']}\n"
                     f"Avg Energy: {stats['avg_energy']:.1f}\n"
                     f"Seeking: {stats.get('seeking_count', 0)}\n"
                     f"Wandering: {stats.get('wandering_count', 0)}")

        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='black',
                            alpha=0.7),
                    color='white')

        self.ax.set_title(f'Micro-Life Simulation ({phase})',
                         color='white',
                         fontsize=14,
                         pad=20)

        plt.tight_layout()

    def show(self):
        """Display the current frame."""
        self.render_frame()
        plt.show()

    def animate(self, frames=500, interval=50):
        """
        Create an animation of the simulation.

        Args:
            frames (int): Number of frames to animate
            interval (int): Milliseconds between frames
        """
        def update(frame):
            self.env.update()
            self.render_frame()
            return self.ax.patches

        anim = animation.FuncAnimation(self.fig,
                                      update,
                                      frames=frames,
                                      interval=interval,
                                      blit=False)
        plt.show()
        return anim

    def save_snapshot(self, filename='simulation_snapshot.png'):
        """
        Save current frame as image.

        Args:
            filename (str): Output filename
        """
        self.render_frame()
        plt.savefig(filename, dpi=150, facecolor=self.fig.get_facecolor())
        print(f"Snapshot saved to {filename}")
