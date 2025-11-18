"""
Heatmap Renderer - GPU-Accelerated Density Visualization
Real-time population density with Gaussian blur
"""
import moderngl
import numpy as np
from typing import Dict, Any, Optional
from .shaders.shader_manager import ShaderManager


class HeatmapRenderer:
    """
    Renders population density heatmap.

    Features:
    - GPU compute shader for density calculation
    - Separable Gaussian blur (two-pass)
    - Color gradient (blue -> green -> yellow -> red)
    - Real-time updates
    - Ping-pong framebuffers for blur
    """

    def __init__(self, ctx: moderngl.Context, shader_manager: ShaderManager,
                 world_width: float, world_height: float,
                 resolution: tuple = (256, 256), sigma: float = 20.0):
        """
        Initialize heatmap renderer.

        Args:
            ctx: ModernGL context
            shader_manager: Shader manager
            world_width: World width in simulation units
            world_height: World height in simulation units
            resolution: Heatmap texture resolution (width, height)
            sigma: Gaussian blur sigma (larger = more blur)
        """
        self.ctx = ctx
        self.shader_manager = shader_manager
        self.world_width = world_width
        self.world_height = world_height
        self.resolution = resolution
        self.sigma = sigma

        # Heatmap textures (ping-pong for blur)
        self.density_texture = self._create_texture()
        self.blur_temp_texture = self._create_texture()
        self.blur_final_texture = self._create_texture()

        # Framebuffers for rendering to textures
        self.density_fbo = ctx.framebuffer(color_attachments=[self.density_texture])
        self.blur_temp_fbo = ctx.framebuffer(color_attachments=[self.blur_temp_texture])
        self.blur_final_fbo = ctx.framebuffer(color_attachments=[self.blur_final_texture])

        # Full-screen quad for rendering heatmap
        self._create_quad()

        # Organism position buffer (for compute shader)
        self.max_organisms = 10000
        self.position_buffer = ctx.buffer(reserve=self.max_organisms * 2 * 4)  # vec2 positions
        self.position_buffer.bind_to_storage_buffer(1)  # Binding point 1 (matches shader)

        # Compute shader work groups
        self.compute_groups_x = (resolution[0] + 7) // 8  # 8x8 threads per group
        self.compute_groups_y = (resolution[1] + 7) // 8

        # Statistics
        self.organism_count = 0
        self.alpha = 0.5  # Overall transparency

    def _create_texture(self) -> moderngl.Texture:
        """Create floating-point texture for density data."""
        texture = self.ctx.texture(
            self.resolution,
            components=4,
            dtype='f2'  # 16-bit float
        )
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = False
        texture.repeat_y = False
        return texture

    def _create_quad(self):
        """Create full-screen quad for rendering heatmap."""
        # Quad vertices (position + texcoord)
        vertices = np.array([
            # x, y, u, v
            0, 0, 0, 0,
            self.world_width, 0, 1, 0,
            self.world_width, self.world_height, 1, 1,
            0, 0, 0, 0,
            self.world_width, self.world_height, 1, 1,
            0, self.world_height, 0, 1,
        ], dtype=np.float32)

        self.quad_vbo = self.ctx.buffer(vertices.tobytes())

        # Create VAO
        shader = self.shader_manager.get_shader('heatmap')
        if not shader:
            raise RuntimeError("Heatmap shader not loaded")

        self.quad_vao = self.ctx.vertex_array(
            shader.program,
            [
                (self.quad_vbo, '2f 2f', 'position', 'texcoord')
            ]
        )

    def update(self, organisms: list):
        """
        Update heatmap with current organism positions.

        Args:
            organisms: List of organism dictionaries with 'x' and 'y' keys
        """
        if not organisms:
            self.organism_count = 0
            return

        # Extract positions
        positions = np.array([[org['x'], org['y']] for org in organisms], dtype=np.float32)
        self.organism_count = len(positions)

        # Update position buffer
        self.position_buffer.write(positions.tobytes())

    def _compute_density(self):
        """Run compute shader to calculate density map."""
        density_shader = self.shader_manager.get_shader('heatmap_density')
        if not density_shader:
            return

        # Bind density texture as image (writable)
        self.density_texture.bind_to_image(0, read=False, write=True)

        # Set uniforms
        density_shader.use()
        density_shader.set_uniforms(
            world_size=(self.world_width, self.world_height),
            map_size=self.resolution,
            organism_count=self.organism_count,
            sigma=self.sigma
        )

        # Run compute shader
        density_shader.program.run(
            group_x=self.compute_groups_x,
            group_y=self.compute_groups_y
        )

        # Memory barrier to ensure writes are visible
        self.ctx.memory_barrier()

    def _apply_blur(self):
        """Apply separable Gaussian blur (two passes using compute shader)."""
        blur_shader = self.shader_manager.get_shader('heatmap_blur')
        if not blur_shader:
            return

        blur_shader.use()

        # Pass 1: Horizontal blur (density -> temp)
        self.density_texture.bind_to_image(0, read=True, write=False)
        self.blur_temp_texture.bind_to_image(1, read=False, write=True)

        blur_shader.set_uniforms(
            horizontal=True,
            blur_radius=self.sigma
        )

        blur_shader.program.run(
            group_x=self.compute_groups_x,
            group_y=self.compute_groups_y
        )
        self.ctx.memory_barrier()

        # Pass 2: Vertical blur (temp -> final)
        self.blur_temp_texture.bind_to_image(0, read=True, write=False)
        self.blur_final_texture.bind_to_image(1, read=False, write=True)

        blur_shader.set_uniforms(
            horizontal=False,
            blur_radius=self.sigma
        )

        blur_shader.program.run(
            group_x=self.compute_groups_x,
            group_y=self.compute_groups_y
        )
        self.ctx.memory_barrier()

    def render(self, simulation_data: Dict[str, Any],
               projection: np.ndarray, view: np.ndarray):
        """
        Render heatmap overlay.

        Args:
            simulation_data: Simulation data (not used, already updated)
            projection: Projection matrix
            view: View matrix
        """
        if self.organism_count == 0:
            return

        # Step 1: Compute density map
        self._compute_density()

        # Step 2: Apply Gaussian blur
        self._apply_blur()

        # Step 3: Render blurred heatmap to screen
        # Restore default framebuffer
        self.ctx.screen.use()

        # Get heatmap shader
        shader = self.shader_manager.get_shader('heatmap')
        if not shader:
            return

        # Enable blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Bind blurred density texture
        self.blur_final_texture.use(0)

        # Set uniforms
        shader.use()
        shader.set_uniforms(
            projection=projection.T.flatten().astype('f4'),
            view=view.T.flatten().astype('f4'),
            density_texture=0,
            alpha=self.alpha
        )

        # Render heatmap quad
        self.quad_vao.render(moderngl.TRIANGLES)

    def set_alpha(self, alpha: float):
        """Set heatmap transparency (0.0 = invisible, 1.0 = opaque)."""
        self.alpha = np.clip(alpha, 0.0, 1.0)

    def set_sigma(self, sigma: float):
        """Set blur amount (larger = more blur)."""
        self.sigma = max(1.0, sigma)

    def clear(self):
        """Clear heatmap data."""
        self.organism_count = 0
        self.density_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.blur_temp_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.blur_final_fbo.clear(0.0, 0.0, 0.0, 0.0)

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'quad_vao'):
            self.quad_vao.release()
        if hasattr(self, 'quad_vbo'):
            self.quad_vbo.release()

        self.density_texture.release()
        self.blur_temp_texture.release()
        self.blur_final_texture.release()

        self.density_fbo.release()
        self.blur_temp_fbo.release()
        self.blur_final_fbo.release()

        self.position_buffer.release()
