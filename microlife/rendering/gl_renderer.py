"""
GLRenderer - Main OpenGL Rendering Engine
Production-grade renderer with ModernGL

Performance Target: 100+ FPS with 1000+ organisms
"""
import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import numpy as np
import time
from typing import Optional, Dict, Any

from .camera import Camera
from .shaders.shader_manager import ShaderManager
from .organism_renderer import OrganismRenderer
from .particle_renderer import ParticleRenderer
from .trail_renderer import TrailRenderer
from .heatmap_renderer import HeatmapRenderer
from .ui_overlay import UIOverlay


class GLRenderer(mglw.WindowConfig):
    """
    Main OpenGL renderer using ModernGL.

    Features:
    - 100+ FPS with 1000+ organisms
    - Instanced rendering
    - GPU compute shaders
    - Advanced visual effects
    - Real-time performance monitoring
    """

    gl_version = (3, 3)
    title = "Micro-Life Simulation - ModernGL [Production]"
    window_size = (1920, 1080)
    aspect_ratio = 16 / 9
    resizable = True
    vsync = False  # Disable VSync for max FPS

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # OpenGL context (provided by moderngl_window)
        self.ctx: moderngl.Context = self.ctx

        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Camera
        self.camera = Camera(
            self.window_size[0],
            self.window_size[1],
            world_width=800,
            world_height=600
        )

        # Shader manager
        self.shader_manager = ShaderManager(self.ctx)
        self.shader_manager.hot_reload_enabled = True  # Enable hot-reload

        # Load all shaders
        self._load_shaders()

        # Renderers
        self.organism_renderer = OrganismRenderer(self.ctx, self.shader_manager)
        self.particle_renderer = ParticleRenderer(self.ctx, self.shader_manager)
        self.trail_renderer = TrailRenderer(self.ctx, self.shader_manager)
        self.heatmap_renderer = HeatmapRenderer(
            self.ctx, self.shader_manager,
            world_width=800, world_height=600
        )
        self.ui_overlay = UIOverlay(self.ctx, self.shader_manager, self.window_size)

        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.frame_times = []
        self.last_fps_update = time.time()

        # Simulation state
        self.simulation_data = None
        self.paused = False

        # Settings
        self.enable_trails = True
        self.enable_particles = True
        self.enable_heatmap = False
        self.enable_glow = True
        self.enable_ui = True

        print("✅ GLRenderer initialized")
        print(f"   OpenGL: {self.ctx.version_code}")
        print(f"   Vendor: {self.ctx.info['GL_VENDOR']}")
        print(f"   Renderer: {self.ctx.info['GL_RENDERER']}")

    def _load_shaders(self):
        """Load all shader programs."""
        # Organism shader
        self.shader_manager.load_shader(
            'organism',
            vertex_file='organism.vert',
            fragment_file='organism.frag'
        )

        # Particle shader
        self.shader_manager.load_shader(
            'particle',
            vertex_file='particle.vert',
            fragment_file='particle.frag'
        )

        # Particle compute shader (physics update)
        self.shader_manager.load_shader(
            'particle_update',
            compute_file='particle_update.comp'
        )

        # Trail shader
        self.shader_manager.load_shader(
            'trail',
            vertex_file='trail.vert',
            fragment_file='trail.frag'
        )

        # Heatmap shaders
        self.shader_manager.load_shader(
            'heatmap_density',
            compute_file='heatmap_density.comp'
        )
        self.shader_manager.load_shader(
            'heatmap_blur',
            compute_file='heatmap_blur.comp'
        )
        self.shader_manager.load_shader(
            'heatmap',
            vertex_file='heatmap.vert',
            fragment_file='heatmap.frag'
        )

        # UI shader
        self.shader_manager.load_shader(
            'ui',
            vertex_file='ui.vert',
            fragment_file='ui.frag'
        )

        print("✅ Shaders loaded")

    def render(self, time_elapsed: float, frame_time: float):
        """
        Main render method (called by moderngl_window).

        Args:
            time_elapsed: Total time elapsed since start
            frame_time: Time since last frame
        """
        # Update camera
        self.camera.update(frame_time)

        # Check for shader hot-reload
        self.shader_manager.check_hot_reload()

        # Clear screen
        self.ctx.clear(0.05, 0.05, 0.05)  # Dark gray background

        # Get camera matrices
        projection = self.camera.get_projection_matrix()
        view = self.camera.get_view_matrix()

        # Render heatmap (if enabled and simulation data available)
        if self.enable_heatmap and self.simulation_data:
            self.heatmap_renderer.render(
                self.simulation_data,
                projection,
                view
            )

        # Render trails (if enabled)
        if self.enable_trails and self.simulation_data:
            self.trail_renderer.render(
                self.simulation_data,
                projection,
                view
            )

        # Render organisms
        if self.simulation_data:
            self.organism_renderer.render(
                self.simulation_data,
                projection,
                view,
                time_elapsed,
                enable_glow=self.enable_glow
            )

        # Render particles (if enabled)
        if self.enable_particles:
            self.particle_renderer.update(frame_time)
            self.particle_renderer.render(projection, view)

        # Render UI overlay (if enabled)
        if self.enable_ui:
            self.ui_overlay.render(
                self.fps,
                self.simulation_data,
                self.frame_times
            )

        # Update performance stats
        self._update_performance(frame_time)

    def _update_performance(self, frame_time: float):
        """Update performance statistics."""
        self.frame_count += 1
        self.frame_times.append(frame_time * 1000)  # Convert to ms

        # Keep last 60 frames
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        # Update FPS every 0.5 seconds
        current_time = time.time()
        if current_time - self.last_fps_update >= 0.5:
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            self.last_fps_update = current_time

    def update_simulation_data(self, simulation_data: Dict[str, Any]):
        """
        Update simulation data for rendering.

        Args:
            simulation_data: Dictionary containing:
                - 'organisms': List of organism data
                - 'particles': Particle spawn events
                - 'trails': Trail data
                - etc.
        """
        self.simulation_data = simulation_data

        # Update trail renderer
        if self.enable_trails and 'organisms' in simulation_data:
            self.trail_renderer.update(simulation_data['organisms'])

        # Update heatmap renderer
        if self.enable_heatmap and 'organisms' in simulation_data:
            self.heatmap_renderer.update(simulation_data['organisms'])

        # Spawn particles from events
        if self.enable_particles and 'particle_events' in simulation_data:
            for event in simulation_data['particle_events']:
                self.particle_renderer.emit(
                    event['type'],
                    event['position'],
                    event.get('color', (1, 1, 1, 1))
                )

    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action == self.wnd.keys.ACTION_PRESS:
            # Space - Pause/Resume
            if key == self.wnd.keys.SPACE:
                self.paused = not self.paused
                print(f"{'⏸️  Paused' if self.paused else '▶️  Resumed'}")

            # T - Toggle trails
            elif key == self.wnd.keys.T:
                self.enable_trails = not self.enable_trails
                print(f"Trails: {'ON' if self.enable_trails else 'OFF'}")

            # P - Toggle particles
            elif key == self.wnd.keys.P:
                self.enable_particles = not self.enable_particles
                print(f"Particles: {'ON' if self.enable_particles else 'OFF'}")

            # H - Toggle heatmap
            elif key == self.wnd.keys.H:
                self.enable_heatmap = not self.enable_heatmap
                print(f"Heatmap: {'ON' if self.enable_heatmap else 'OFF'}")

            # G - Toggle glow
            elif key == self.wnd.keys.G:
                self.enable_glow = not self.enable_glow
                print(f"Glow: {'ON' if self.enable_glow else 'OFF'}")

            # U - Toggle UI
            elif key == self.wnd.keys.U:
                self.enable_ui = not self.enable_ui
                print(f"UI: {'ON' if self.enable_ui else 'OFF'}")

            # R - Reset camera
            elif key == self.wnd.keys.R:
                self.camera.reset()
                print("🎥 Camera reset")

            # F - Print FPS
            elif key == self.wnd.keys.F:
                print(f"📊 FPS: {self.fps:.1f}")

            # ESC - Exit
            elif key == self.wnd.keys.ESCAPE:
                self.wnd.close()

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        """Handle mouse scroll (zoom)."""
        zoom_factor = 0.1
        self.camera.zoom_delta(y_offset * zoom_factor)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Handle mouse drag (pan)."""
        self.camera.pan(dx, dy)

    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.camera.resize(width, height)
        self.ui_overlay.resize(width, height)

    def close(self):
        """Cleanup on exit."""
        print("🧹 Cleaning up...")

        # Cleanup renderers
        if hasattr(self, 'organism_renderer'):
            self.organism_renderer.cleanup()
        if hasattr(self, 'particle_renderer'):
            self.particle_renderer.cleanup()
        if hasattr(self, 'trail_renderer'):
            self.trail_renderer.cleanup()
        if hasattr(self, 'heatmap_renderer'):
            self.heatmap_renderer.cleanup()
        if hasattr(self, 'ui_overlay'):
            self.ui_overlay.cleanup()

        # Cleanup shaders
        if hasattr(self, 'shader_manager'):
            self.shader_manager.cleanup()

        print("✅ Cleanup complete")


def run_standalone():
    """Run renderer standalone (for testing)."""
    # This function allows running the renderer standalone for testing
    # In production, this will be called from a main demo script
    mglw.run_window_config(GLRenderer)


if __name__ == '__main__':
    run_standalone()
