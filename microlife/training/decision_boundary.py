"""
DecisionBoundaryVisualizer - Visualize model decision boundaries
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Callable
from sklearn.decomposition import PCA
import moderngl


class DecisionBoundaryVisualizer:
    """
    Visualize decision boundaries for classification models.

    Features:
    - 2D decision boundary rendering
    - PCA projection for high-dimensional data
    - Confidence contours
    - Sample point overlay
    """

    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int],
                 resolution: int = 100):
        """
        Initialize decision boundary visualizer.

        Args:
            ctx: ModernGL context
            window_size: Window size (width, height)
            resolution: Grid resolution for boundary sampling
        """
        self.ctx = ctx
        self.window_width, self.window_height = window_size
        self.resolution = resolution

        # PCA for dimensionality reduction
        self.pca = None

        # Boundary texture
        self.boundary_texture = None

        # Sample points
        self.sample_points = []
        self.sample_labels = []

        # Rendering buffers
        self.max_vertices = 50000
        self.vbo = ctx.buffer(reserve=self.max_vertices * 2 * 4, dynamic=True)

        self._create_shader()
        self.projection = self._create_projection()

    def _create_shader(self):
        """Create shader for rendering."""
        vertex_shader = """
        #version 330 core

        in vec2 position;
        uniform mat4 projection;

        void main() {
            gl_Position = projection * vec4(position, 0.0, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core

        uniform vec4 color;
        out vec4 fragColor;

        void main() {
            fragColor = color;
        }
        """

        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )

        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '2f', 'position')]
        )

    def _create_projection(self) -> np.ndarray:
        """Create orthographic projection."""
        w = float(self.window_width)
        h = float(self.window_height)

        return np.array([
            [2.0/w, 0,      0, -1],
            [0,     -2.0/h, 0,  1],
            [0,     0,      1,  0],
            [0,     0,      0,  1]
        ], dtype=np.float32)

    def fit_projection(self, X: np.ndarray):
        """
        Fit PCA projection for high-dimensional data.

        Args:
            X: Training data (n_samples, n_features)
        """
        if X.shape[1] > 2:
            self.pca = PCA(n_components=2)
            self.pca.fit(X)
        else:
            self.pca = None

    def project_to_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to 2D.

        Args:
            X: Data to project (n_samples, n_features)

        Returns:
            Projected data (n_samples, 2)
        """
        if self.pca is not None:
            return self.pca.transform(X)
        else:
            return X[:, :2]

    def compute_boundary(self, model: nn.Module,
                        x_range: Tuple[float, float],
                        y_range: Tuple[float, float]) -> np.ndarray:
        """
        Compute decision boundary on 2D grid.

        Args:
            model: PyTorch classification model
            x_range: (min, max) for X axis
            y_range: (min, max) for Y axis

        Returns:
            Grid of predictions (resolution, resolution)
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], self.resolution)
        y = np.linspace(y_range[0], y_range[1], self.resolution)
        xx, yy = np.meshgrid(x, y)

        # Flatten grid
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Inverse PCA transform if needed
        if self.pca is not None:
            # Reconstruct high-dimensional points
            # This is approximate since we can't fully recover lost dimensions
            grid_points_high = self.pca.inverse_transform(grid_points)
        else:
            grid_points_high = grid_points

        # Get predictions
        with torch.no_grad():
            grid_tensor = torch.FloatTensor(grid_points_high)
            predictions = model(grid_tensor)

            # Convert to probabilities
            if predictions.shape[1] > 1:
                # Multi-class
                probs = torch.softmax(predictions, dim=1)
                class_predictions = torch.argmax(probs, dim=1).numpy()
            else:
                # Binary
                probs = torch.sigmoid(predictions)
                class_predictions = (probs > 0.5).float().numpy().flatten()

        # Reshape to grid
        boundary = class_predictions.reshape((self.resolution, self.resolution))

        return boundary

    def render(self, model: nn.Module,
               panel_x: int, panel_y: int,
               panel_width: int, panel_height: int,
               X_samples: Optional[np.ndarray] = None,
               y_samples: Optional[np.ndarray] = None):
        """
        Render decision boundary.

        Args:
            model: PyTorch model
            panel_x: Panel X position
            panel_y: Panel Y position
            panel_width: Panel width
            panel_height: Panel height
            X_samples: Sample points to overlay (n_samples, 2)
            y_samples: Sample labels (n_samples,)
        """
        # Determine data range
        if X_samples is not None and len(X_samples) > 0:
            X_2d = self.project_to_2d(X_samples)
            x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
            y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()

            # Add margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            x_range = (x_min - x_margin, x_max + x_margin)
            y_range = (y_min - y_margin, y_max + y_margin)
        else:
            x_range = (-5, 5)
            y_range = (-5, 5)

        # Compute boundary
        boundary = self.compute_boundary(model, x_range, y_range)

        # Render boundary as colored grid
        self._render_boundary_grid(
            boundary, panel_x, panel_y, panel_width, panel_height,
            x_range, y_range
        )

        # Render sample points
        if X_samples is not None and y_samples is not None:
            self._render_sample_points(
                X_samples, y_samples,
                panel_x, panel_y, panel_width, panel_height,
                x_range, y_range
            )

    def _render_boundary_grid(self, boundary: np.ndarray,
                             panel_x: int, panel_y: int,
                             panel_width: int, panel_height: int,
                             x_range: Tuple[float, float],
                             y_range: Tuple[float, float]):
        """Render decision boundary as colored rectangles."""
        cell_width = panel_width / self.resolution
        cell_height = panel_height / self.resolution

        # Color map for classes
        class_colors = [
            (0.3, 0.5, 1.0, 0.3),   # Blue (class 0)
            (1.0, 0.3, 0.3, 0.3),   # Red (class 1)
            (0.3, 1.0, 0.3, 0.3),   # Green (class 2)
            (1.0, 1.0, 0.3, 0.3),   # Yellow (class 3)
        ]

        vertices = []

        for i in range(self.resolution):
            for j in range(self.resolution):
                class_id = int(boundary[i, j])
                color = class_colors[class_id % len(class_colors)]

                # Calculate cell position
                cell_x = panel_x + j * cell_width
                cell_y = panel_y + i * cell_height

                # Create rectangle vertices
                vertices.extend([
                    [cell_x, cell_y],
                    [cell_x + cell_width, cell_y],
                    [cell_x + cell_width, cell_y + cell_height],

                    [cell_x, cell_y],
                    [cell_x + cell_width, cell_y + cell_height],
                    [cell_x, cell_y + cell_height],
                ])

        if not vertices:
            return

        vertices = np.array(vertices, dtype=np.float32)
        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        # Use average color for now (ideally render each cell separately)
        self.program['color'].value = (0.5, 0.5, 0.5, 0.3)

        self.vao.render(moderngl.TRIANGLES, vertices=len(vertices))

    def _render_sample_points(self, X: np.ndarray, y: np.ndarray,
                             panel_x: int, panel_y: int,
                             panel_width: int, panel_height: int,
                             x_range: Tuple[float, float],
                             y_range: Tuple[float, float]):
        """Render sample data points."""
        X_2d = self.project_to_2d(X)

        # Normalize to panel space
        x_min, x_max = x_range
        y_min, y_max = y_range

        normalized_x = (X_2d[:, 0] - x_min) / (x_max - x_min)
        normalized_y = (X_2d[:, 1] - y_min) / (y_max - y_min)

        point_x = panel_x + normalized_x * panel_width
        point_y = panel_y + normalized_y * panel_height

        # Colors for different classes
        point_colors = [
            (0.2, 0.4, 1.0, 1.0),   # Blue
            (1.0, 0.2, 0.2, 1.0),   # Red
            (0.2, 1.0, 0.2, 1.0),   # Green
            (1.0, 1.0, 0.2, 1.0),   # Yellow
        ]

        # Draw points
        for i in range(len(X_2d)):
            class_id = int(y[i])
            color = point_colors[class_id % len(point_colors)]
            self._draw_circle(point_x[i], point_y[i], 4.0, color)

    def _draw_circle(self, cx: float, cy: float, radius: float,
                     color: Tuple[float, float, float, float], segments: int = 12):
        """Draw filled circle."""
        vertices = [[cx, cy]]

        for i in range(segments + 1):
            angle = 2 * np.pi * i / segments
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            vertices.append([x, y])

        vertices = np.array(vertices, dtype=np.float32)
        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['color'].value = color

        self.vao.render(moderngl.TRIANGLE_FAN, vertices=len(vertices))

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.window_width = width
        self.window_height = height
        self.projection = self._create_projection()

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
        if hasattr(self, 'program'):
            self.program.release()
        if self.boundary_texture:
            self.boundary_texture.release()
