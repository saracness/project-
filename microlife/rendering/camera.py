"""
Camera System
Handles view/projection matrices, zoom, pan with smooth interpolation
"""
import numpy as np
import pyrr
from typing import Tuple


class Camera:
    """
    2D camera for simulation rendering.

    Features:
    - Smooth zoom/pan with interpolation
    - Bounds checking
    - Screen <-> World coordinate conversion
    - Optimized matrix calculations
    """

    def __init__(self, width: int, height: int, world_width: float, world_height: float):
        """
        Initialize camera.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
            world_width: World width in simulation units
            world_height: World height in simulation units
        """
        self.screen_width = width
        self.screen_height = height
        self.world_width = world_width
        self.world_height = world_height

        # Camera state
        self.position = np.array([world_width / 2, world_height / 2], dtype=np.float32)
        self.zoom = 1.0

        # Target state (for smooth interpolation)
        self.target_position = self.position.copy()
        self.target_zoom = self.zoom

        # Interpolation speed
        self.lerp_speed = 10.0  # Higher = faster

        # Zoom limits
        self.min_zoom = 0.1
        self.max_zoom = 10.0

        # Cached matrices
        self._view_matrix = None
        self._projection_matrix = None
        self._view_projection_matrix = None
        self._matrices_dirty = True

    def update(self, dt: float):
        """
        Update camera state with smooth interpolation.

        Args:
            dt: Delta time in seconds
        """
        # Smooth interpolation
        t = min(1.0, self.lerp_speed * dt)

        # Interpolate position
        self.position += (self.target_position - self.position) * t

        # Interpolate zoom
        self.zoom += (self.target_zoom - self.zoom) * t

        # Clamp zoom
        self.zoom = np.clip(self.zoom, self.min_zoom, self.max_zoom)

        # Mark matrices as dirty
        if t > 0:
            self._matrices_dirty = True

    def pan(self, dx: float, dy: float, immediate: bool = False):
        """
        Pan camera.

        Args:
            dx: Delta X in screen pixels
            dy: Delta Y in screen pixels
            immediate: If True, apply immediately (no interpolation)
        """
        # Convert screen delta to world delta
        world_dx = dx / self.screen_width * self.world_width / self.zoom
        world_dy = dy / self.screen_height * self.world_height / self.zoom

        if immediate:
            self.position[0] -= world_dx
            self.position[1] += world_dy  # Y is inverted in screen space
            self.target_position = self.position.copy()
        else:
            self.target_position[0] -= world_dx
            self.target_position[1] += world_dy

    def zoom_to(self, zoom: float, immediate: bool = False):
        """
        Set zoom level.

        Args:
            zoom: New zoom level (1.0 = default)
            immediate: If True, apply immediately
        """
        zoom = np.clip(zoom, self.min_zoom, self.max_zoom)

        if immediate:
            self.zoom = zoom
            self.target_zoom = zoom
        else:
            self.target_zoom = zoom

    def zoom_delta(self, delta: float, immediate: bool = False):
        """
        Adjust zoom by delta.

        Args:
            delta: Zoom delta (positive = zoom in, negative = zoom out)
            immediate: If True, apply immediately
        """
        self.zoom_to(self.target_zoom * (1.0 + delta), immediate)

    def reset(self):
        """Reset camera to default state."""
        self.target_position = np.array([self.world_width / 2, self.world_height / 2], dtype=np.float32)
        self.target_zoom = 1.0

    def get_view_matrix(self) -> np.ndarray:
        """
        Get view matrix.

        Returns:
            4x4 view matrix
        """
        if self._matrices_dirty or self._view_matrix is None:
            self._update_matrices()
        return self._view_matrix

    def get_projection_matrix(self) -> np.ndarray:
        """
        Get orthographic projection matrix.

        Returns:
            4x4 projection matrix
        """
        if self._matrices_dirty or self._projection_matrix is None:
            self._update_matrices()
        return self._projection_matrix

    def get_view_projection_matrix(self) -> np.ndarray:
        """
        Get combined view-projection matrix.

        Returns:
            4x4 view-projection matrix
        """
        if self._matrices_dirty or self._view_projection_matrix is None:
            self._update_matrices()
        return self._view_projection_matrix

    def _update_matrices(self):
        """Update cached matrices."""
        # Calculate visible area
        half_width = (self.world_width / self.zoom) / 2
        half_height = (self.world_height / self.zoom) / 2

        left = self.position[0] - half_width
        right = self.position[0] + half_width
        bottom = self.position[1] - half_height
        top = self.position[1] + half_height

        # Projection matrix (orthographic)
        self._projection_matrix = pyrr.matrix44.create_orthogonal_projection(
            left, right, bottom, top, -1.0, 1.0, dtype=np.float32
        )

        # View matrix (identity for 2D orthographic)
        self._view_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Combined
        self._view_projection_matrix = self._projection_matrix @ self._view_matrix

        self._matrices_dirty = False

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen X (pixels)
            screen_y: Screen Y (pixels)

        Returns:
            (world_x, world_y) tuple
        """
        # Normalize to [0, 1]
        norm_x = screen_x / self.screen_width
        norm_y = screen_y / self.screen_height

        # Calculate visible area
        half_width = (self.world_width / self.zoom) / 2
        half_height = (self.world_height / self.zoom) / 2

        # Convert to world coordinates
        world_x = self.position[0] - half_width + norm_x * (half_width * 2)
        world_y = self.position[1] + half_height - norm_y * (half_height * 2)  # Y inverted

        return world_x, world_y

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates to screen coordinates.

        Args:
            world_x: World X
            world_y: World Y

        Returns:
            (screen_x, screen_y) tuple
        """
        # Calculate visible area
        half_width = (self.world_width / self.zoom) / 2
        half_height = (self.world_height / self.zoom) / 2

        left = self.position[0] - half_width
        top = self.position[1] + half_height

        # Normalize
        norm_x = (world_x - left) / (half_width * 2)
        norm_y = (top - world_y) / (half_height * 2)  # Y inverted

        # Convert to screen
        screen_x = norm_x * self.screen_width
        screen_y = norm_y * self.screen_height

        return screen_x, screen_y

    def is_visible(self, world_x: float, world_y: float, radius: float = 0) -> bool:
        """
        Check if a world position is visible.

        Args:
            world_x: World X
            world_y: World Y
            radius: Object radius (for margin)

        Returns:
            True if visible
        """
        half_width = (self.world_width / self.zoom) / 2 + radius
        half_height = (self.world_height / self.zoom) / 2 + radius

        return (abs(world_x - self.position[0]) < half_width and
                abs(world_y - self.position[1]) < half_height)

    def resize(self, width: int, height: int):
        """
        Handle window resize.

        Args:
            width: New screen width
            height: New screen height
        """
        self.screen_width = width
        self.screen_height = height
        self._matrices_dirty = True
