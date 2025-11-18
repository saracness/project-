"""
Mini-Map
Provides an overview of the entire simulation
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class MiniMap:
    """
    Small overview map showing all organisms and food.
    """

    def __init__(self, env_width, env_height, size=100, position='top-right', enabled=True):
        """
        Args:
            env_width: Environment width
            env_height: Environment height
            size: Mini-map size in pixels
            position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
            enabled: Enable mini-map
        """
        self.env_width = env_width
        self.env_height = env_height
        self.size = size
        self.position = position
        self.enabled = enabled

        # Scale factors
        self.scale_x = size / env_width
        self.scale_y = size / env_height

        # Position on screen
        self.x_offset = 0
        self.y_offset = 0

        # Minimap axis
        self.minimap_ax = None

    def update_position(self, fig_width, fig_height):
        """Update mini-map position based on figure size."""
        padding = 10

        if self.position == 'top-right':
            self.x_offset = fig_width - self.size - padding
            self.y_offset = fig_height - self.size - padding
        elif self.position == 'top-left':
            self.x_offset = padding
            self.y_offset = fig_height - self.size - padding
        elif self.position == 'bottom-right':
            self.x_offset = fig_width - self.size - padding
            self.y_offset = padding
        elif self.position == 'bottom-left':
            self.x_offset = padding
            self.y_offset = padding

    def initialize(self, fig, ax_main):
        """
        Initialize mini-map on figure.

        Args:
            fig: Matplotlib figure
            ax_main: Main simulation axis
        """
        if not self.enabled:
            return

        # Get figure size in pixels
        bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_width_px = bbox.width * fig.dpi
        fig_height_px = bbox.height * fig.dpi

        # Update position
        self.update_position(fig_width_px, fig_height_px)

        # Convert to figure coordinates (0-1)
        fig_x = self.x_offset / fig_width_px
        fig_y = self.y_offset / fig_height_px
        fig_w = self.size / fig_width_px
        fig_h = self.size / fig_height_px

        # Create minimap axis
        self.minimap_ax = fig.add_axes([fig_x, fig_y, fig_w, fig_h], facecolor='#1a1a1a')
        self.minimap_ax.set_xlim(0, self.env_width)
        self.minimap_ax.set_ylim(0, self.env_height)
        self.minimap_ax.set_aspect('equal')
        self.minimap_ax.axis('off')

        # Add border
        border = patches.Rectangle(
            (0, 0), self.env_width, self.env_height,
            linewidth=2, edgecolor='white', facecolor='none'
        )
        self.minimap_ax.add_patch(border)

    def render(self, organisms, food_particles, viewport=None):
        """
        Render mini-map.

        Args:
            organisms: List of organisms
            food_particles: List of food particles
            viewport: Optional (x_min, x_max, y_min, y_max) to show current view
        """
        if not self.enabled or self.minimap_ax is None:
            return

        # Clear minimap
        self.minimap_ax.clear()
        self.minimap_ax.set_xlim(0, self.env_width)
        self.minimap_ax.set_ylim(0, self.env_height)
        self.minimap_ax.set_aspect('equal')
        self.minimap_ax.axis('off')
        self.minimap_ax.set_facecolor('#1a1a1a')

        # Draw border
        border = patches.Rectangle(
            (0, 0), self.env_width, self.env_height,
            linewidth=2, edgecolor='white', facecolor='none'
        )
        self.minimap_ax.add_patch(border)

        # Draw food (small dots)
        if food_particles:
            food_x = [f.x for f in food_particles]
            food_y = [f.y for f in food_particles]
            self.minimap_ax.scatter(food_x, food_y, c='#90EE90', s=1, alpha=0.5, zorder=1)

        # Draw organisms (colored dots)
        if organisms:
            # Separate by alive/dead
            alive_orgs = [o for o in organisms if o.alive]

            if alive_orgs:
                # Get positions and colors
                org_x = [o.x for o in alive_orgs]
                org_y = [o.y for o in alive_orgs]
                org_colors = [o.morphology.color for o in alive_orgs]

                self.minimap_ax.scatter(org_x, org_y, c=org_colors, s=4, alpha=0.8, zorder=2)

                # Highlight AI organisms
                ai_orgs = [o for o in alive_orgs if hasattr(o, 'brain') and o.brain]
                if ai_orgs:
                    ai_x = [o.x for o in ai_orgs]
                    ai_y = [o.y for o in ai_orgs]
                    # Draw rings around AI organisms
                    self.minimap_ax.scatter(ai_x, ai_y, s=8, facecolors='none',
                                           edgecolors='yellow', linewidths=0.5, zorder=3)

        # Draw viewport rectangle (if provided)
        if viewport:
            x_min, x_max, y_min, y_max = viewport
            width = x_max - x_min
            height = y_max - y_min
            viewport_rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1, edgecolor='cyan', facecolor='none',
                linestyle='--', zorder=4
            )
            self.minimap_ax.add_patch(viewport_rect)

    def set_enabled(self, enabled):
        """Enable or disable mini-map."""
        self.enabled = enabled
        if not enabled and self.minimap_ax is not None:
            self.minimap_ax.remove()
            self.minimap_ax = None

    def set_position(self, position):
        """
        Change mini-map position.

        Args:
            position: 'top-right', 'top-left', 'bottom-right', 'bottom-left'
        """
        self.position = position

    def world_to_minimap(self, x, y):
        """
        Convert world coordinates to mini-map coordinates.

        Args:
            x, y: World coordinates

        Returns:
            (mini_x, mini_y) in mini-map space
        """
        mini_x = x * self.scale_x
        mini_y = y * self.scale_y
        return mini_x, mini_y

    def minimap_to_world(self, mini_x, mini_y):
        """
        Convert mini-map coordinates to world coordinates.

        Args:
            mini_x, mini_y: Mini-map coordinates

        Returns:
            (x, y) in world space
        """
        x = mini_x / self.scale_x
        y = mini_y / self.scale_y
        return x, y

    def get_stats(self):
        """Get mini-map statistics."""
        return {
            'size': self.size,
            'position': self.position,
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'enabled': self.enabled
        }
