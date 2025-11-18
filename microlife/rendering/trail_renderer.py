"""
Trail Renderer - Dynamic Vertex Buffer Trails
Efficient trail rendering with fade effects
"""
import moderngl
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Any
from .shaders.shader_manager import ShaderManager


class Trail:
    """Single organism trail."""

    def __init__(self, max_length: int = 30, color: tuple = (0, 1, 0)):
        self.positions = deque(maxlen=max_length)
        self.max_length = max_length
        self.base_color = color

    def add_position(self, x: float, y: float):
        """Add position to trail."""
        self.positions.append((x, y))

    def get_vertices(self) -> np.ndarray:
        """Get trail vertices with fade."""
        if len(self.positions) < 2:
            return np.array([], dtype=np.float32)

        num_points = len(self.positions)
        vertices = np.zeros((num_points, 6), dtype=np.float32)  # x, y, r, g, b, a

        for i, (x, y) in enumerate(self.positions):
            # Position
            vertices[i, 0] = x
            vertices[i, 1] = y

            # Color with fade (older = more transparent)
            alpha = (i + 1) / num_points  # 0.0 (oldest) to 1.0 (newest)
            vertices[i, 2] = self.base_color[0]
            vertices[i, 3] = self.base_color[1]
            vertices[i, 4] = self.base_color[2]
            vertices[i, 5] = alpha

        return vertices


class TrailRenderer:
    """
    Renders organism movement trails.

    Features:
    - Per-organism trails
    - Fade-out effect
    - Dynamic vertex buffers
    - Batch rendering
    """

    def __init__(self, ctx: moderngl.Context, shader_manager: ShaderManager,
                 max_length: int = 30):
        """
        Initialize trail renderer.

        Args:
            ctx: ModernGL context
            shader_manager: Shader manager
            max_length: Maximum trail length (positions)
        """
        self.ctx = ctx
        self.shader_manager = shader_manager
        self.max_length = max_length

        # Trails per organism
        self.trails: Dict[int, Trail] = {}

        # Dynamic vertex buffer
        self.max_vertices = 10000  # Pre-allocate
        self.vbo = self.ctx.buffer(reserve=self.max_vertices * 6 * 4, dynamic=True)  # 6 floats per vertex

        # Create VAO
        self._create_vao()

        # Statistics
        self.vertex_count = 0

    def _create_vao(self):
        """Create vertex array object."""
        shader = self.shader_manager.get_shader('trail')
        if not shader:
            raise RuntimeError("Trail shader not loaded")

        self.vao = self.ctx.vertex_array(
            shader.program,
            [
                (self.vbo, '2f 4f', 'position', 'color')
            ]
        )

    def update(self, organisms: List[Dict[str, Any]]):
        """
        Update trails from organism data.

        Args:
            organisms: List of organism dictionaries
        """
        active_ids = set()

        for org in organisms:
            org_id = org.get('id', id(org))
            active_ids.add(org_id)

            # Get or create trail
            if org_id not in self.trails:
                color = self._hex_to_rgb(org.get('color', '#00FF00'))
                self.trails[org_id] = Trail(self.max_length, color)

            # Add current position
            self.trails[org_id].add_position(org['x'], org['y'])

        # Remove trails for dead organisms
        dead_ids = set(self.trails.keys()) - active_ids
        for org_id in dead_ids:
            del self.trails[org_id]

    def render(self, simulation_data: Dict[str, Any],
               projection: np.ndarray, view: np.ndarray):
        """
        Render all trails.

        Args:
            simulation_data: Simulation data (not used, trails already updated)
            projection: Projection matrix
            view: View matrix
        """
        if not self.trails:
            return

        # Collect all trail vertices
        all_vertices = []
        for trail in self.trails.values():
            vertices = trail.get_vertices()
            if len(vertices) > 0:
                all_vertices.append(vertices)

        if not all_vertices:
            return

        # Concatenate all vertices
        vertices = np.vstack(all_vertices)
        self.vertex_count = len(vertices)

        if self.vertex_count == 0:
            return

        # Update VBO
        self.vbo.write(vertices.tobytes())

        # Get shader
        shader = self.shader_manager.get_shader('trail')
        if not shader:
            return

        # Set uniforms
        shader.use()
        shader.set_uniforms(
            projection=projection.T.flatten().astype('f4'),
            view=view.T.flatten().astype('f4')
        )

        # Render as line strip
        # Note: This renders all trails as one continuous line strip
        # For better results, we could use multiple draw calls or index buffer
        self.vao.render(moderngl.LINE_STRIP, vertices=self.vertex_count)

    def _hex_to_rgb(self, hex_color: str) -> tuple:
        """Convert hex color to RGB (0-1)."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    def clear(self):
        """Clear all trails."""
        self.trails.clear()

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'vbo'):
            self.vbo.release()
