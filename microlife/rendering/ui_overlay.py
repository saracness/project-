"""
UI Overlay - Performance Monitoring & Statistics Display
Lightweight text and shape rendering for UI elements
"""
import moderngl
import numpy as np
from typing import Dict, Any, List, Optional
from .shaders.shader_manager import ShaderManager


class UIOverlay:
    """
    Renders UI overlay with performance stats.

    Features:
    - FPS counter
    - Frame time graph
    - Organism count
    - System stats
    - Keyboard shortcuts help
    """

    def __init__(self, ctx: moderngl.Context, shader_manager: ShaderManager,
                 window_size: tuple):
        """
        Initialize UI overlay.

        Args:
            ctx: ModernGL context
            shader_manager: Shader manager
            window_size: Window size (width, height)
        """
        self.ctx = ctx
        self.shader_manager = shader_manager
        self.window_width, self.window_height = window_size

        # Dynamic vertex buffer for UI elements
        self.max_vertices = 10000
        self.vbo = ctx.buffer(reserve=self.max_vertices * 8 * 4, dynamic=True)  # 8 floats per vertex

        # Create VAO
        self._create_vao()

        # Orthographic projection (screen space)
        self.projection = self._create_orthographic_projection()

        # UI settings
        self.show_help = False
        self.background_alpha = 0.7

    def _create_vao(self):
        """Create vertex array object for UI rendering."""
        shader = self.shader_manager.get_shader('ui')
        if not shader:
            raise RuntimeError("UI shader not loaded")

        self.vao = self.ctx.vertex_array(
            shader.program,
            [
                (self.vbo, '2f 2f 4f', 'position', 'texcoord', 'color')
            ]
        )

    def _create_orthographic_projection(self) -> np.ndarray:
        """Create orthographic projection matrix for screen space."""
        # Simple 2D orthographic projection
        width = float(self.window_width)
        height = float(self.window_height)

        projection = np.array([
            [2.0/width, 0,           0, -1],
            [0,         -2.0/height, 0,  1],
            [0,         0,           1,  0],
            [0,         0,           0,  1]
        ], dtype=np.float32)

        return projection

    def render(self, fps: float, simulation_data: Optional[Dict[str, Any]],
               frame_times: List[float]):
        """
        Render UI overlay.

        Args:
            fps: Current FPS
            simulation_data: Simulation data (organisms, etc.)
            frame_times: Recent frame times in milliseconds
        """
        # Collect UI elements
        vertices = []

        # FPS panel (top-left)
        vertices.extend(self._create_fps_panel(10, 10, fps, frame_times))

        # Statistics panel (top-left, below FPS)
        if simulation_data:
            vertices.extend(self._create_stats_panel(10, 80, simulation_data))

        # Controls help (bottom-left)
        if self.show_help:
            vertices.extend(self._create_help_panel(10, self.window_height - 300))

        if not vertices:
            return

        # Convert to numpy array
        vertices_array = np.array(vertices, dtype=np.float32)
        vertex_count = len(vertices_array)

        if vertex_count == 0:
            return

        # Update VBO
        self.vbo.write(vertices_array.tobytes())

        # Get shader
        shader = self.shader_manager.get_shader('ui')
        if not shader:
            return

        # Disable depth test for UI
        self.ctx.disable(moderngl.DEPTH_TEST)

        # Set uniforms
        shader.use()
        shader.set_uniforms(
            projection=self.projection.T.flatten().astype('f4'),
            use_texture=False  # We're not using font textures yet
        )

        # Render UI
        self.vao.render(moderngl.TRIANGLES, vertices=vertex_count)

    def _create_fps_panel(self, x: float, y: float, fps: float,
                          frame_times: List[float]) -> List[float]:
        """Create FPS panel with graph."""
        vertices = []
        panel_width = 200
        panel_height = 60

        # Background
        bg_color = (0.1, 0.1, 0.1, self.background_alpha)
        vertices.extend(self._create_rect(x, y, panel_width, panel_height, bg_color))

        # FPS text area (dark background)
        text_bg_color = (0.05, 0.05, 0.05, self.background_alpha)
        vertices.extend(self._create_rect(x + 5, y + 5, panel_width - 10, 20, text_bg_color))

        # FPS indicator color (green if >60, yellow if >30, red otherwise)
        if fps >= 60:
            indicator_color = (0.2, 0.8, 0.2, 1.0)
        elif fps >= 30:
            indicator_color = (0.8, 0.8, 0.2, 1.0)
        else:
            indicator_color = (0.8, 0.2, 0.2, 1.0)

        # FPS indicator bar
        bar_width = min(panel_width - 10, (fps / 120) * (panel_width - 10))
        vertices.extend(self._create_rect(x + 5, y + 5, bar_width, 20, indicator_color))

        # Frame time graph
        if frame_times:
            vertices.extend(self._create_graph(
                x + 5, y + 30, panel_width - 10, 25,
                frame_times, max_value=33.0,  # 33ms = 30 FPS
                color=(0.3, 0.6, 0.9, 0.8)
            ))

        return vertices

    def _create_stats_panel(self, x: float, y: float,
                           simulation_data: Dict[str, Any]) -> List[float]:
        """Create statistics panel."""
        vertices = []
        panel_width = 200
        panel_height = 100

        # Background
        bg_color = (0.1, 0.1, 0.1, self.background_alpha)
        vertices.extend(self._create_rect(x, y, panel_width, panel_height, bg_color))

        # Statistics bars
        organisms = simulation_data.get('organisms', [])
        organism_count = len(organisms)

        # Organism count bar
        count_ratio = min(1.0, organism_count / 1000.0)
        count_color = (0.3, 0.7, 0.4, 0.9)
        vertices.extend(self._create_rect(
            x + 5, y + 10,
            count_ratio * (panel_width - 10), 15,
            count_color
        ))

        # Energy average (if available)
        if organisms and 'energy' in organisms[0]:
            avg_energy = sum(org.get('energy', 0) for org in organisms) / len(organisms)
            energy_ratio = min(1.0, avg_energy / 100.0)
            energy_color = (0.9, 0.6, 0.2, 0.9)
            vertices.extend(self._create_rect(
                x + 5, y + 35,
                energy_ratio * (panel_width - 10), 15,
                energy_color
            ))

        return vertices

    def _create_help_panel(self, x: float, y: float) -> List[float]:
        """Create keyboard shortcuts help panel."""
        vertices = []
        panel_width = 250
        panel_height = 250

        # Background
        bg_color = (0.1, 0.1, 0.1, self.background_alpha)
        vertices.extend(self._create_rect(x, y, panel_width, panel_height, bg_color))

        # Header
        header_color = (0.2, 0.3, 0.5, 0.9)
        vertices.extend(self._create_rect(x + 5, y + 5, panel_width - 10, 25, header_color))

        # Key bindings (visual indicators only, text rendering would require font atlas)
        # In a production system, you'd render actual text here
        key_color = (0.3, 0.3, 0.3, 0.8)
        y_offset = 40
        for i in range(8):  # 8 key bindings
            vertices.extend(self._create_rect(
                x + 10, y + y_offset + i * 25,
                panel_width - 20, 20,
                key_color
            ))

        return vertices

    def _create_rect(self, x: float, y: float, width: float, height: float,
                    color: tuple) -> List[float]:
        """Create rectangle vertices."""
        # Two triangles for a rectangle
        # Vertex format: x, y, u, v, r, g, b, a
        r, g, b, a = color

        return [
            # Triangle 1
            x, y, 0, 0, r, g, b, a,
            x + width, y, 1, 0, r, g, b, a,
            x + width, y + height, 1, 1, r, g, b, a,
            # Triangle 2
            x, y, 0, 0, r, g, b, a,
            x + width, y + height, 1, 1, r, g, b, a,
            x, y + height, 0, 1, r, g, b, a,
        ]

    def _create_graph(self, x: float, y: float, width: float, height: float,
                     values: List[float], max_value: float,
                     color: tuple) -> List[float]:
        """Create line graph."""
        vertices = []

        if not values or len(values) < 2:
            return vertices

        num_points = len(values)
        x_step = width / (num_points - 1)

        # Draw as series of thin rectangles (lines)
        line_width = 2.0
        r, g, b, a = color

        for i in range(num_points):
            value = min(values[i], max_value)
            normalized = value / max_value
            point_x = x + i * x_step
            point_y = y + height - (normalized * height)

            # Small rectangle for each point
            vertices.extend([
                point_x, point_y, 0, 0, r, g, b, a,
                point_x + line_width, point_y, 1, 0, r, g, b, a,
                point_x + line_width, point_y + line_width, 1, 1, r, g, b, a,

                point_x, point_y, 0, 0, r, g, b, a,
                point_x + line_width, point_y + line_width, 1, 1, r, g, b, a,
                point_x, point_y + line_width, 0, 1, r, g, b, a,
            ])

        return vertices

    def toggle_help(self):
        """Toggle help panel visibility."""
        self.show_help = not self.show_help

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.window_width = width
        self.window_height = height
        self.projection = self._create_orthographic_projection()

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
