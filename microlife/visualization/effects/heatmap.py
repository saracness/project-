"""
Heatmap Generator
Visualizes population density as a heat map
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class HeatmapGenerator:
    """
    Generates population density heatmap.
    """

    def __init__(self, width, height, resolution=50, blur=True, enabled=False):
        """
        Args:
            width: Environment width
            height: Environment height
            resolution: Grid resolution (e.g., 50x50)
            blur: Apply Gaussian blur
            enabled: Enable heatmap
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.blur = blur
        self.enabled = enabled

        # Create grid
        self.grid = np.zeros((resolution, resolution))

        # Grid cell size
        self.cell_width = width / resolution
        self.cell_height = height / resolution

        # Create custom colormap (blue -> green -> yellow -> red)
        colors = ['#000033', '#0000FF', '#00FF00', '#FFFF00', '#FF0000']
        n_bins = 100
        self.cmap = LinearSegmentedColormap.from_list('density', colors, N=n_bins)

        # Heatmap image object (for updating)
        self.heatmap_image = None

    def update(self, organisms):
        """
        Update heatmap with current organism positions.

        Args:
            organisms: List of organisms
        """
        if not self.enabled:
            return

        # Reset grid
        self.grid.fill(0)

        # Count organisms in each cell
        for org in organisms:
            if org.alive:
                # Convert position to grid coordinates
                grid_x = int(org.x / self.cell_width)
                grid_y = int(org.y / self.cell_height)

                # Clamp to grid bounds
                grid_x = max(0, min(grid_x, self.resolution - 1))
                grid_y = max(0, min(grid_y, self.resolution - 1))

                # Increment count
                self.grid[grid_y, grid_x] += 1

        # Apply Gaussian blur for smooth heatmap
        if self.blur and np.any(self.grid > 0):
            sigma = self.resolution / 20  # Adaptive blur based on resolution
            self.grid = gaussian_filter(self.grid, sigma=sigma)

    def render(self, ax):
        """
        Render heatmap on matplotlib axis.

        Args:
            ax: Matplotlib axis
        """
        if not self.enabled:
            return

        # Normalize grid to 0-1
        if np.max(self.grid) > 0:
            normalized_grid = self.grid / np.max(self.grid)
        else:
            normalized_grid = self.grid

        # Render heatmap
        if self.heatmap_image is None:
            # First time - create image
            self.heatmap_image = ax.imshow(
                normalized_grid,
                extent=[0, self.width, 0, self.height],
                origin='lower',
                cmap=self.cmap,
                alpha=0.4,  # Semi-transparent
                interpolation='bilinear',
                zorder=0  # Draw behind everything
            )
        else:
            # Update existing image
            self.heatmap_image.set_data(normalized_grid)

    def clear(self):
        """Clear heatmap."""
        self.grid.fill(0)

    def set_enabled(self, enabled):
        """Enable or disable heatmap."""
        self.enabled = enabled
        if not enabled:
            self.clear()
            if self.heatmap_image is not None:
                self.heatmap_image.remove()
                self.heatmap_image = None

    def get_density_at(self, x, y):
        """
        Get density value at a specific position.

        Args:
            x, y: Position

        Returns:
            Density value
        """
        grid_x = int(x / self.cell_width)
        grid_y = int(y / self.cell_height)

        # Clamp to grid bounds
        grid_x = max(0, min(grid_x, self.resolution - 1))
        grid_y = max(0, min(grid_y, self.resolution - 1))

        return self.grid[grid_y, grid_x]

    def get_hotspots(self, threshold=0.5):
        """
        Get locations of high-density areas.

        Args:
            threshold: Density threshold (0-1)

        Returns:
            List of (x, y) coordinates
        """
        if np.max(self.grid) == 0:
            return []

        normalized_grid = self.grid / np.max(self.grid)
        hotspot_coords = []

        for y in range(self.resolution):
            for x in range(self.resolution):
                if normalized_grid[y, x] >= threshold:
                    # Convert grid coords back to world coords
                    world_x = (x + 0.5) * self.cell_width
                    world_y = (y + 0.5) * self.cell_height
                    hotspot_coords.append((world_x, world_y))

        return hotspot_coords

    def get_stats(self):
        """Get heatmap statistics."""
        total_density = np.sum(self.grid)
        max_density = np.max(self.grid)
        avg_density = np.mean(self.grid)

        return {
            'total_density': total_density,
            'max_density': max_density,
            'avg_density': avg_density,
            'resolution': self.resolution,
            'enabled': self.enabled,
            'blur': self.blur
        }
