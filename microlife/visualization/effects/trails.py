"""
Trail System
Renders movement trails for organisms
"""
import numpy as np
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class Trail:
    """
    Trail for a single organism.
    """

    def __init__(self, max_length=20, color=(0, 255, 0), fade=True):
        self.positions = deque(maxlen=max_length)
        self.max_length = max_length
        self.base_color = np.array(color) / 255.0  # Normalize to 0-1
        self.fade = fade

    def add_position(self, x, y):
        """Add a new position to the trail."""
        self.positions.append((x, y))

    def get_segments(self):
        """
        Get line segments for rendering.

        Returns:
            segments: List of line segments [(x1,y1), (x2,y2)]
            colors: List of RGBA colors for each segment
        """
        if len(self.positions) < 2:
            return [], []

        segments = []
        colors = []

        positions_list = list(self.positions)
        num_segments = len(positions_list) - 1

        for i in range(num_segments):
            p1 = positions_list[i]
            p2 = positions_list[i + 1]
            segments.append([p1, p2])

            # Calculate alpha based on position in trail
            if self.fade:
                # Older positions are more transparent
                alpha = (i + 1) / num_segments  # 0.0 to 1.0
            else:
                alpha = 1.0

            # Create RGBA color
            rgba = (*self.base_color, alpha)
            colors.append(rgba)

        return segments, colors

    def clear(self):
        """Clear all trail positions."""
        self.positions.clear()


class TrailSystem:
    """
    Manages trails for all organisms.
    """

    def __init__(self, max_length=20, fade=True, enabled=True):
        self.max_length = max_length
        self.fade = fade
        self.enabled = enabled
        self.trails = {}  # organism_id -> Trail

    def register_organism(self, organism_id, color=(0, 255, 0)):
        """Register an organism for trail tracking."""
        if organism_id not in self.trails:
            self.trails[organism_id] = Trail(
                max_length=self.max_length,
                color=color,
                fade=self.fade
            )

    def update(self, organism_id, x, y):
        """Update trail with new position."""
        if not self.enabled:
            return

        if organism_id not in self.trails:
            self.register_organism(organism_id)

        self.trails[organism_id].add_position(x, y)

    def remove_organism(self, organism_id):
        """Remove trail for dead organism."""
        if organism_id in self.trails:
            del self.trails[organism_id]

    def render(self, ax):
        """
        Render all trails on matplotlib axis.

        Args:
            ax: Matplotlib axis
        """
        if not self.enabled:
            return

        all_segments = []
        all_colors = []

        for trail in self.trails.values():
            segments, colors = trail.get_segments()
            all_segments.extend(segments)
            all_colors.extend(colors)

        if all_segments:
            # Use LineCollection for efficient batch rendering
            lc = LineCollection(all_segments, colors=all_colors, linewidths=2, alpha=0.6)
            ax.add_collection(lc)

    def clear(self):
        """Clear all trails."""
        for trail in self.trails.values():
            trail.clear()

    def clear_all(self):
        """Remove all trail objects."""
        self.trails.clear()

    def get_trail_count(self):
        """Get number of active trails."""
        return len(self.trails)

    def set_enabled(self, enabled):
        """Enable or disable trail rendering."""
        self.enabled = enabled
        if not enabled:
            self.clear()

    def set_max_length(self, max_length):
        """Change maximum trail length."""
        self.max_length = max_length
        # Update existing trails
        for trail in self.trails.values():
            # Create new deque with new maxlen
            new_positions = deque(trail.positions, maxlen=max_length)
            trail.positions = new_positions
            trail.max_length = max_length
