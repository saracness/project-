"""
TrainingVisualizer - Real-time training graphs and charts
GPU-accelerated visualization using ModernGL
"""
import moderngl
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .metrics_tracker import MetricsTracker


class TrainingVisualizer:
    """
    Real-time training visualization with GPU rendering.

    Features:
    - Multi-panel graph layout
    - Line plots with auto-scaling
    - Moving average overlays
    - Color-coded metrics
    - Performance optimized
    """

    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int]):
        """
        Initialize training visualizer.

        Args:
            ctx: ModernGL context
            window_size: Window size (width, height)
        """
        self.ctx = ctx
        self.window_width, self.window_height = window_size

        # Panels configuration
        self.panels: Dict[str, Dict[str, Any]] = {}

        # Vertex buffer for line rendering
        self.max_vertices = 50000
        self.vbo = ctx.buffer(reserve=self.max_vertices * 2 * 4, dynamic=True)  # vec2

        # Create shader program
        self._create_shader()

        # Colors for different metrics
        self.metric_colors = {
            'loss': (1.0, 0.3, 0.3),          # Red
            'accuracy': (0.3, 1.0, 0.3),      # Green
            'reward': (0.3, 0.6, 1.0),        # Blue
            'learning_rate': (1.0, 0.8, 0.2), # Yellow
            'gradient_norm': (0.8, 0.4, 1.0), # Purple
        }

        # Orthographic projection
        self.projection = self._create_projection()

    def _create_shader(self):
        """Create simple line shader."""
        vertex_shader = """
        #version 330 core

        in vec2 position;
        uniform mat4 projection;
        uniform mat4 transform;

        void main() {
            gl_Position = projection * transform * vec4(position, 0.0, 1.0);
        }
        """

        fragment_shader = """
        #version 330 core

        uniform vec3 color;
        out vec4 fragColor;

        void main() {
            fragColor = vec4(color, 1.0);
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
        """Create orthographic projection for screen space."""
        w = float(self.window_width)
        h = float(self.window_height)

        return np.array([
            [2.0/w, 0,      0, -1],
            [0,     -2.0/h, 0,  1],
            [0,     0,      1,  0],
            [0,     0,      0,  1]
        ], dtype=np.float32)

    def add_panel(self, name: str, metric_name: str,
                  position: Tuple[int, int], size: Tuple[int, int],
                  title: Optional[str] = None,
                  y_range: Optional[Tuple[float, float]] = None,
                  show_smooth: bool = True):
        """
        Add visualization panel.

        Args:
            name: Panel identifier
            metric_name: Metric to display
            position: (x, y) position in pixels
            size: (width, height) in pixels
            title: Panel title (None = use metric name)
            y_range: Y-axis range (None = auto-scale)
            show_smooth: Show smoothed line overlay
        """
        self.panels[name] = {
            'metric_name': metric_name,
            'position': position,
            'size': size,
            'title': title or metric_name,
            'y_range': y_range,
            'show_smooth': show_smooth,
            'auto_scale': y_range is None,
        }

    def render(self, metrics_tracker: MetricsTracker):
        """
        Render all panels.

        Args:
            metrics_tracker: Metrics tracker with data
        """
        # Disable depth test for 2D rendering
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Enable line smoothing
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Render each panel
        for panel_name, panel_config in self.panels.items():
            self._render_panel(panel_config, metrics_tracker)

    def _render_panel(self, config: Dict[str, Any], tracker: MetricsTracker):
        """Render single panel."""
        metric_name = config['metric_name']
        x, y = config['position']
        width, height = config['size']

        # Get metric data
        values = tracker.get(metric_name)
        if not values or len(values) < 2:
            return

        # Draw panel background
        self._draw_rect(x, y, width, height, color=(0.1, 0.1, 0.1, 0.8))

        # Draw panel border
        self._draw_rect_outline(x, y, width, height, color=(0.3, 0.3, 0.3))

        # Determine Y range
        if config['auto_scale']:
            y_min = min(values)
            y_max = max(values)
            # Add 10% padding
            y_range_size = y_max - y_min
            if y_range_size == 0:
                y_range_size = 1.0
            y_min -= y_range_size * 0.1
            y_max += y_range_size * 0.1
        else:
            y_min, y_max = config['y_range']

        # Draw grid
        self._draw_grid(x, y, width, height, y_min, y_max)

        # Draw main line
        color = self.metric_colors.get(metric_name, (0.5, 0.5, 0.5))
        self._draw_line_graph(
            values, x, y, width, height, y_min, y_max, color
        )

        # Draw smooth line overlay
        if config['show_smooth']:
            smooth_values = tracker.get(metric_name, smoothed=True)
            if smooth_values and len(smooth_values) >= 2:
                smooth_color = tuple(c * 1.2 for c in color)  # Brighter
                self._draw_line_graph(
                    smooth_values, x, y, width, height, y_min, y_max,
                    smooth_color, line_width=3.0
                )

    def _draw_line_graph(self, values: List[float],
                        x: int, y: int, width: int, height: int,
                        y_min: float, y_max: float,
                        color: Tuple[float, float, float],
                        line_width: float = 2.0):
        """Draw line graph."""
        if len(values) < 2:
            return

        num_points = len(values)
        x_step = width / (num_points - 1)

        # Generate line vertices
        vertices = []
        for i, value in enumerate(values):
            # Normalize value to panel space
            normalized_y = (value - y_min) / (y_max - y_min)
            point_x = x + i * x_step
            point_y = y + height - (normalized_y * height)

            vertices.append([point_x, point_y])

        vertices = np.array(vertices, dtype=np.float32)

        # Update VBO
        self.vbo.write(vertices.tobytes())

        # Set uniforms
        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['transform'].write(np.eye(4, dtype='f4').tobytes())
        self.program['color'].value = color

        # Set line width
        self.ctx.line_width = line_width

        # Render
        self.vao.render(moderngl.LINE_STRIP, vertices=num_points)

    def _draw_rect(self, x: float, y: float, width: float, height: float,
                   color: Tuple[float, float, float, float]):
        """Draw filled rectangle."""
        vertices = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y],
            [x + width, y + height],
            [x, y + height],
        ], dtype=np.float32)

        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['transform'].write(np.eye(4, dtype='f4').tobytes())
        self.program['color'].value = color[:3]

        self.vao.render(moderngl.TRIANGLES, vertices=6)

    def _draw_rect_outline(self, x: float, y: float, width: float, height: float,
                          color: Tuple[float, float, float]):
        """Draw rectangle outline."""
        vertices = np.array([
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
            [x, y],  # Close the loop
        ], dtype=np.float32)

        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['transform'].write(np.eye(4, dtype='f4').tobytes())
        self.program['color'].value = color

        self.ctx.line_width = 1.0
        self.vao.render(moderngl.LINE_STRIP, vertices=5)

    def _draw_grid(self, x: int, y: int, width: int, height: int,
                   y_min: float, y_max: float, num_lines: int = 5):
        """Draw grid lines."""
        vertices = []

        # Horizontal grid lines
        for i in range(num_lines + 1):
            grid_y = y + (i / num_lines) * height
            vertices.append([x, grid_y])
            vertices.append([x + width, grid_y])

        # Vertical grid lines (fewer)
        num_vertical = 5
        for i in range(num_vertical + 1):
            grid_x = x + (i / num_vertical) * width
            vertices.append([grid_x, y])
            vertices.append([grid_x, y + height])

        vertices = np.array(vertices, dtype=np.float32)

        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['transform'].write(np.eye(4, dtype='f4').tobytes())
        self.program['color'].value = (0.2, 0.2, 0.2)

        self.ctx.line_width = 1.0
        total_lines = (num_lines + 1) * 2 + (num_vertical + 1) * 2
        self.vao.render(moderngl.LINES, vertices=total_lines)

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


class MatplotlibVisualizer:
    """
    Matplotlib-based training visualizer (fallback option).

    Slower but higher quality plots, good for publication.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize matplotlib visualizer.

        Args:
            figsize: Figure size (width, height) in inches
        """
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('TkAgg')  # Interactive backend

        self.plt = plt
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.tight_layout(pad=3.0)

        # Configure style
        plt.style.use('dark_background')

        # Panel configuration
        self.panels = {
            'loss': (0, 0),
            'accuracy': (0, 1),
            'reward': (1, 0),
            'learning_rate': (1, 1),
        }

    def update(self, tracker: MetricsTracker):
        """Update all plots."""
        for metric_name, (row, col) in self.panels.items():
            ax = self.axes[row, col]
            ax.clear()

            # Get data
            values = tracker.get(metric_name)
            smooth_values = tracker.get(metric_name, smoothed=True)

            if not values:
                continue

            episodes = list(range(len(values)))

            # Plot raw values
            ax.plot(episodes, values, alpha=0.3, label='Raw')

            # Plot smooth values
            if smooth_values:
                ax.plot(episodes, smooth_values, linewidth=2, label='Smooth')

            # Configure
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def show(self):
        """Show plot window."""
        self.plt.show(block=False)

    def close(self):
        """Close plot window."""
        self.plt.close(self.fig)

    def save(self, filepath: str):
        """Save figure to file."""
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✅ Saved figure to {filepath}")
