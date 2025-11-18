"""
Organism Renderer - Instanced Rendering
Renders 10,000+ organisms efficiently using GPU instancing
"""
import moderngl
import numpy as np
from typing import List, Dict, Any
from .shaders.shader_manager import ShaderManager


class OrganismRenderer:
    """
    High-performance organism renderer using instanced rendering.

    Features:
    - Single draw call for all organisms
    - GPU instancing (10k+ organisms)
    - Morphology-based visual representation
    - Glow effects for AI organisms
    - Energy-based coloring
    """

    def __init__(self, ctx: moderngl.Context, shader_manager: ShaderManager):
        """
        Initialize organism renderer.

        Args:
            ctx: ModernGL context
            shader_manager: Shader manager
        """
        self.ctx = ctx
        self.shader_manager = shader_manager

        # Circle geometry (unit circle)
        self.circle_vertices = self._create_circle_vertices(segments=32)

        # Create vertex buffer for circle geometry
        self.circle_vbo = self.ctx.buffer(self.circle_vertices.astype('f4').tobytes())

        # Instance data buffer (will be updated each frame)
        self.instance_buffer = None
        self.max_instances = 10000  # Pre-allocate for 10k organisms

        # Create VAO
        self._create_vao()

        # Statistics
        self.render_count = 0

    def _create_circle_vertices(self, segments: int = 32) -> np.ndarray:
        """
        Create circle vertices (unit circle).

        Args:
            segments: Number of segments

        Returns:
            Vertex array (x, y)
        """
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        vertices = np.column_stack([
            np.cos(angles),
            np.sin(angles)
        ])
        return vertices.astype(np.float32)

    def _create_vao(self):
        """Create vertex array object."""
        shader = self.shader_manager.get_shader('organism')
        if not shader:
            raise RuntimeError("Organism shader not loaded")

        # Pre-allocate instance buffer
        instance_dtype = np.dtype([
            ('position', np.float32, 2),    # xy position
            ('color', np.float32, 3),       # rgb color
            ('size', np.float32),           # radius
            ('glow', np.float32),           # glow intensity
            ('energy', np.float32),         # energy level (0-1)
        ])

        # Create empty buffer
        instance_data = np.zeros(self.max_instances, dtype=instance_dtype)
        self.instance_buffer = self.ctx.buffer(instance_data.tobytes(), dynamic=True)

        # Create VAO
        self.vao = self.ctx.vertex_array(
            shader.program,
            [
                # Circle vertices (per-vertex)
                (self.circle_vbo, '2f', 'vertex_position'),

                # Instance data (per-instance)
                (self.instance_buffer, '2f 3f 1f 1f 1f /i',
                 'instance_position', 'instance_color', 'instance_size',
                 'instance_glow', 'instance_energy'),
            ]
        )

    def render(self, simulation_data: Dict[str, Any], projection: np.ndarray,
               view: np.ndarray, time: float, enable_glow: bool = True):
        """
        Render all organisms using instanced rendering.

        Args:
            simulation_data: Simulation data dictionary
            projection: Projection matrix
            view: View matrix
            time: Current time (for animations)
            enable_glow: Enable glow effects
        """
        if not simulation_data or 'organisms' not in simulation_data:
            return

        organisms = simulation_data['organisms']
        if not organisms:
            return

        num_organisms = min(len(organisms), self.max_instances)

        # Prepare instance data
        instance_data = self._prepare_instance_data(organisms[:num_organisms])

        # Update instance buffer
        self.instance_buffer.write(instance_data.tobytes())

        # Get shader
        shader = self.shader_manager.get_shader('organism')
        if not shader:
            return

        # Set uniforms
        shader.use()
        shader.set_uniforms(
            projection=projection.T.flatten().astype('f4'),
            view=view.T.flatten().astype('f4'),
            time=time,
            enable_glow=enable_glow,
            enable_energy_overlay=True
        )

        # Render (single draw call!)
        self.vao.render(moderngl.TRIANGLE_FAN, instances=num_organisms)

        self.render_count = num_organisms

    def _prepare_instance_data(self, organisms: List[Dict]) -> np.ndarray:
        """
        Prepare instance data from organism list.

        Args:
            organisms: List of organism dictionaries

        Returns:
            Numpy structured array
        """
        num_organisms = len(organisms)

        instance_dtype = np.dtype([
            ('position', np.float32, 2),
            ('color', np.float32, 3),
            ('size', np.float32),
            ('glow', np.float32),
            ('energy', np.float32),
        ])

        instance_data = np.zeros(num_organisms, dtype=instance_dtype)

        for i, org in enumerate(organisms):
            # Position
            instance_data[i]['position'] = [org['x'], org['y']]

            # Color (from morphology or default)
            color = org.get('color', '#00FF00')  # Default green
            instance_data[i]['color'] = self._hex_to_rgb(color)

            # Size (from morphology or default)
            size = org.get('size', 5.0)
            instance_data[i]['size'] = size

            # Glow (1.0 if AI, 0.0 otherwise)
            has_ai = org.get('has_brain', False)
            instance_data[i]['glow'] = 1.0 if has_ai else 0.0

            # Energy (normalized 0-1)
            energy = org.get('energy', 100.0)
            max_energy = org.get('max_energy', 200.0)
            instance_data[i]['energy'] = np.clip(energy / max_energy, 0.0, 1.0)

        return instance_data

    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        """
        Convert hex color to RGB (0-1 range).

        Args:
            hex_color: Hex color string (e.g., '#FF0000')

        Returns:
            RGB array (0-1 range)
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return np.array([r, g, b], dtype=np.float32)

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'circle_vbo'):
            self.circle_vbo.release()
        if hasattr(self, 'instance_buffer'):
            self.instance_buffer.release()
