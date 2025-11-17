"""
Advanced Renderer
High-quality visualization with trails, particles, heatmap, and minimap
"""
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Circle
from .simple_renderer import SimpleRenderer
from .effects import TrailSystem, ParticleSystem, ParticleType, HeatmapGenerator, MiniMap
from ..config import SimulationConfig


class AdvancedRenderer(SimpleRenderer):
    """
    Advanced renderer with visual effects and optimizations.
    Extends SimpleRenderer with:
    - Trail effects
    - Particle system
    - Heatmap overlay
    - Mini-map
    - Glow effects
    - Performance optimizations
    """

    def __init__(self, environment, config: SimulationConfig = None):
        """
        Initialize advanced renderer.

        Args:
            environment: The environment to visualize
            config: Simulation configuration
        """
        # Configuration - MUST be set BEFORE super().__init__()
        # because parent calls setup_plot() which needs self.config
        self.config = config or SimulationConfig()

        # Store environment reference before parent init
        self.env = environment

        # Effect systems - Initialize BEFORE parent constructor
        self.trail_system = TrailSystem(
            max_length=self.config.trail_length,
            fade=self.config.trail_fade,
            enabled=self.config.enable_trails
        )

        self.particle_system = ParticleSystem(
            max_particles=self.config.max_particles,
            enabled=self.config.enable_particles
        )

        self.heatmap = HeatmapGenerator(
            width=environment.width,
            height=environment.height,
            resolution=self.config.heatmap_resolution,
            blur=self.config.heatmap_blur,
            enabled=self.config.enable_heatmap
        )

        self.minimap = MiniMap(
            env_width=environment.width,
            env_height=environment.height,
            size=100,
            position='top-right',
            enabled=self.config.enable_minimap
        )

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.render_times = []

        # Event tracking for particles
        self.tracked_events = set()

        # NOW call parent constructor (which calls setup_plot())
        super().__init__(environment)

    def setup_plot(self):
        """Set up the plot area with mini-map."""
        super().setup_plot()

        # Initialize mini-map
        if self.config.enable_minimap:
            self.minimap.initialize(self.fig, self.ax)

    def render_frame(self):
        """Render a single frame with all effects."""
        start_time = time.time()

        # Clear axis
        self.ax.clear()
        self.setup_plot()

        # Update heatmap
        if self.config.enable_heatmap:
            self.heatmap.update(self.env.organisms)

        # Render layers (bottom to top)
        self._render_heatmap()
        self._render_environment()
        self._render_trails()
        self._render_food()
        self._render_organisms()
        self._render_particles()
        self._render_minimap()
        self._render_stats()

        # Update particles
        if self.config.enable_particles:
            self.particle_system.update()

        # Track events for particles
        self._track_events()

        # Update FPS
        self._update_fps(start_time)

        self.frame_count += 1

    def _render_heatmap(self):
        """Render population density heatmap."""
        if self.config.enable_heatmap:
            self.heatmap.render(self.ax)

    def _render_environment(self):
        """Render environment (temperature zones, obstacles)."""
        # Temperature zones
        for zone in self.env.temperature_zones:
            color = '#ff6b6b' if zone.temperature > 0 else '#4dabf7'
            circle = Circle((zone.x, zone.y), zone.radius,
                          color=color, alpha=0.15, linestyle='--', fill=True)
            self.ax.add_patch(circle)

        # Obstacles
        for obstacle in self.env.obstacles:
            from matplotlib.patches import Rectangle
            rect = Rectangle((obstacle.x, obstacle.y),
                           obstacle.width, obstacle.height,
                           color='#555555', alpha=0.7)
            self.ax.add_patch(rect)

    def _render_trails(self):
        """Render organism trails."""
        if self.config.enable_trails:
            self.trail_system.render(self.ax)

    def _render_food(self):
        """Render food particles."""
        for food in self.env.food_particles:
            if not food.consumed:
                circle = Circle((food.x, food.y), 2, color='#00ff00', alpha=0.6)
                self.ax.add_patch(circle)

    def _render_organisms(self):
        """Render organisms with effects."""
        for organism in self.env.organisms:
            if organism.alive:
                # Update trail
                if self.config.enable_trails:
                    org_id = id(organism)
                    color_rgb = self._hex_to_rgb(organism.morphology.color)
                    self.trail_system.register_organism(org_id, color_rgb)
                    self.trail_system.update(org_id, organism.x, organism.y)

                # Get color
                if hasattr(organism, 'morphology') and hasattr(organism.morphology, 'color'):
                    color = organism.morphology.color
                else:
                    energy_ratio = min(organism.energy / 150.0, 1.0)
                    color = plt.cm.plasma(energy_ratio)

                # Draw flagella (tail) if present
                if hasattr(organism, 'morphology') and organism.morphology.flagella_length > 0.1:
                    tail_length = organism.morphology.flagella_length * 15
                    if len(organism.trail) >= 2:
                        dx = organism.trail[-1][0] - organism.trail[-2][0]
                        dy = organism.trail[-1][1] - organism.trail[-2][1]
                        if dx != 0 or dy != 0:
                            angle = np.arctan2(dy, dx)
                            tail_x = organism.x - np.cos(angle) * tail_length
                            tail_y = organism.y - np.sin(angle) * tail_length
                            self.ax.plot([organism.x, tail_x], [organism.y, tail_y],
                                       color=color, linewidth=2, alpha=0.7)

                # Draw glow effect if enabled
                if self.config.enable_glow:
                    # Check if AI organism
                    if hasattr(organism, 'brain') and organism.brain:
                        # Draw glow for AI organisms
                        glow_circle = Circle((organism.x, organism.y),
                                            organism.size * 1.5,
                                            color='yellow',
                                            alpha=0.2,
                                            zorder=5)
                        self.ax.add_patch(glow_circle)

                # Draw organism body
                circle = Circle((organism.x, organism.y),
                              organism.size,
                              color=color,
                              alpha=0.8,
                              edgecolor='white',
                              linewidth=0.5,
                              zorder=10)
                self.ax.add_patch(circle)

                # Draw cilia if present
                if hasattr(organism, 'morphology') and organism.morphology.cilia_density > 0.3:
                    num_cilia = int(organism.morphology.cilia_density * 12)
                    for i in range(num_cilia):
                        angle = (2 * np.pi * i) / num_cilia
                        cilia_length = 5
                        x1 = organism.x + np.cos(angle) * organism.size
                        y1 = organism.y + np.sin(angle) * organism.size
                        x2 = organism.x + np.cos(angle) * (organism.size + cilia_length)
                        y2 = organism.y + np.sin(angle) * (organism.size + cilia_length)
                        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=1, alpha=0.6)

    def _render_particles(self):
        """Render particle effects."""
        if self.config.enable_particles:
            self.particle_system.render(self.ax)

    def _render_minimap(self):
        """Render mini-map."""
        if self.config.enable_minimap:
            viewport = (
                self.ax.get_xlim()[0],
                self.ax.get_xlim()[1],
                self.ax.get_ylim()[0],
                self.ax.get_ylim()[1]
            )
            self.minimap.render(self.env.organisms, self.env.food_particles, viewport)

    def _render_stats(self):
        """Render statistics overlay."""
        if self.config.show_fps:
            # FPS counter in top-left
            fps_text = f'FPS: {self.fps:.1f}'
            self.ax.text(0.02, 0.98, fps_text,
                        transform=self.ax.transAxes,
                        fontsize=10, color='yellow',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # Timestep
        timestep_text = f'Timestep: {self.env.timestep}'
        self.ax.text(0.02, 0.02, timestep_text,
                    transform=self.ax.transAxes,
                    fontsize=9, color='white',
                    verticalalignment='bottom')

    def _track_events(self):
        """Track events and emit particles."""
        if not self.config.enable_particles:
            return

        # Track food consumption
        for organism in self.env.organisms:
            if organism.alive and hasattr(organism, 'food_consumed_count'):
                event_key = (id(organism), 'food', organism.food_consumed_count)
                if event_key not in self.tracked_events:
                    self.particle_system.emit(ParticleType.FOOD_CONSUME, organism.x, organism.y)
                    self.tracked_events.add(event_key)

        # Track deaths
        for organism in self.env.organisms:
            if not organism.alive:
                event_key = (id(organism), 'death')
                if event_key not in self.tracked_events:
                    self.particle_system.emit(ParticleType.DEATH, organism.x, organism.y)
                    self.tracked_events.add(event_key)
                    # Remove from trail system
                    self.trail_system.remove_organism(id(organism))

        # Clean old events (prevent memory leak)
        if len(self.tracked_events) > 10000:
            self.tracked_events.clear()

    def _update_fps(self, start_time):
        """Update FPS counter."""
        render_time = time.time() - start_time
        self.render_times.append(render_time)

        # Keep last 30 frames
        if len(self.render_times) > 30:
            self.render_times.pop(0)

        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            avg_render_time = np.mean(self.render_times)
            self.fps = 1.0 / avg_render_time if avg_render_time > 0 else 0.0
            self.last_fps_time = current_time

    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple (0-255)."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def toggle_trails(self):
        """Toggle trail rendering."""
        self.config.enable_trails = not self.config.enable_trails
        self.trail_system.set_enabled(self.config.enable_trails)
        print(f"Trails: {'ON' if self.config.enable_trails else 'OFF'}")

    def toggle_particles(self):
        """Toggle particle effects."""
        self.config.enable_particles = not self.config.enable_particles
        self.particle_system.set_enabled(self.config.enable_particles)
        print(f"Particles: {'ON' if self.config.enable_particles else 'OFF'}")

    def toggle_heatmap(self):
        """Toggle heatmap."""
        self.config.enable_heatmap = not self.config.enable_heatmap
        self.heatmap.set_enabled(self.config.enable_heatmap)
        print(f"Heatmap: {'ON' if self.config.enable_heatmap else 'OFF'}")

    def toggle_minimap(self):
        """Toggle mini-map."""
        self.config.enable_minimap = not self.config.enable_minimap
        self.minimap.set_enabled(self.config.enable_minimap)
        print(f"MiniMap: {'ON' if self.config.enable_minimap else 'OFF'}")

    def get_performance_stats(self):
        """Get rendering performance statistics."""
        avg_render_time = np.mean(self.render_times) if self.render_times else 0.0
        return {
            'fps': self.fps,
            'avg_render_time_ms': avg_render_time * 1000,
            'frame_count': self.frame_count,
            'trail_count': self.trail_system.get_trail_count(),
            'particle_count': self.particle_system.get_particle_count()
        }
