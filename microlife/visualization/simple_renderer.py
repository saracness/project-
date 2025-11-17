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

        # Draw organisms with morphology
        for organism in self.env.organisms:
            if organism.alive:
                # Use organism's morphology color if available
                if hasattr(organism, 'color'):
                    color = organism.color
                else:
                    # Fallback: Color based on energy
                    energy_ratio = min(organism.energy / 150.0, 1.0)
                    color = plt.cm.plasma(energy_ratio)

                # Draw flagella (tail) if present
                if hasattr(organism, 'morphology') and organism.morphology.flagella_length > 0.1:
                    tail_length = organism.morphology.flagella_length * 15
                    # Tail points backward from movement direction
                    if len(organism.trail) >= 2:
                        dx = organism.trail[-1][0] - organism.trail[-2][0]
                        dy = organism.trail[-1][1] - organism.trail[-2][1]
                        if dx != 0 or dy != 0:
                            angle = np.arctan2(dy, dx)
                            tail_x = organism.x - np.cos(angle) * tail_length
                            tail_y = organism.y - np.sin(angle) * tail_length
                            self.ax.plot([organism.x, tail_x], [organism.y, tail_y],
                                       color=color, linewidth=2, alpha=0.7)

                # Draw organism body
                circle = Circle((organism.x, organism.y),
                              organism.size,
                              color=color,
                              alpha=0.8,
                              edgecolor='white',
                              linewidth=0.5)
                self.ax.add_patch(circle)

                # Draw cilia (short lines around body) if present
                if hasattr(organism, 'morphology') and organism.morphology.cilia_density > 0.3:
                    num_cilia = int(organism.morphology.cilia_density * 12)
                    for i in range(num_cilia):
                        angle = (i / num_cilia) * 2 * np.pi
                        cilia_start_x = organism.x + np.cos(angle) * organism.size
                        cilia_start_y = organism.y + np.sin(angle) * organism.size
                        cilia_end_x = cilia_start_x + np.cos(angle) * 3
                        cilia_end_y = cilia_start_y + np.sin(angle) * 3
                        self.ax.plot([cilia_start_x, cilia_end_x],
                                   [cilia_start_y, cilia_end_y],
                                   color=color, linewidth=1, alpha=0.5)

                # Draw trail (optional)
                if len(organism.trail) > 1:
                    trail_x = [pos[0] for pos in organism.trail]
                    trail_y = [pos[1] for pos in organism.trail]
                    self.ax.plot(trail_x, trail_y,
                               color=color,
                               alpha=0.2,
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
