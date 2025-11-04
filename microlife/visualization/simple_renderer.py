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

        # Add statistics text
        stats = self.env.get_statistics()
        stats_text = (f"Timestep: {stats['timestep']}\n"
                     f"Population: {stats['population']}\n"
                     f"Food: {stats['food_count']}\n"
                     f"Avg Energy: {stats['avg_energy']:.1f}")

        self.ax.text(0.02, 0.98, stats_text,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round',
                            facecolor='black',
                            alpha=0.7),
                    color='white')

        self.ax.set_title('Micro-Life Simulation (Phase 1)',
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
