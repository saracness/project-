"""
Particle Renderer - GPU-Accelerated Particles
Uses compute shaders for physics simulation on GPU
"""
import moderngl
import numpy as np
from typing import Tuple
from enum import Enum
from .shaders.shader_manager import ShaderManager


class ParticleType(Enum):
    """Particle effect types."""
    FOOD_CONSUME = 0
    DEATH = 1
    REPRODUCTION = 2
    ENERGY_GAIN = 3
    ENERGY_LOSS = 4


class ParticleRenderer:
    """
    GPU-accelerated particle system.

    Features:
    - Compute shader physics (fully on GPU)
    - No CPU particle updates!
    - Point sprite rendering
    - Particle pooling
    - Multiple effect types
    """

    def __init__(self, ctx: moderngl.Context, shader_manager: ShaderManager,
                 max_particles: int = 10000):
        """
        Initialize particle renderer.

        Args:
            ctx: ModernGL context
            max_particles: Maximum number of active particles
        """
        self.ctx = ctx
        self.shader_manager = shader_manager
        self.max_particles = max_particles

        # Particle buffer (SSBO - Shader Storage Buffer Object)
        self._create_particle_buffer()

        # Particle spawn queue
        self.spawn_queue = []

        # Next available particle index
        self.next_particle = 0

        # Particle configurations
        self.configs = {
            ParticleType.FOOD_CONSUME: {
                'color': (0.0, 1.0, 0.0, 1.0),  # Green
                'size': 5.0,
                'lifetime': 0.5,
                'count': 8,
                'speed': 3.0
            },
            ParticleType.DEATH: {
                'color': (1.0, 0.0, 0.0, 1.0),  # Red
                'size': 8.0,
                'lifetime': 1.0,
                'count': 20,
                'speed': 5.0
            },
            ParticleType.REPRODUCTION: {
                'color': (0.0, 0.6, 1.0, 1.0),  # Blue
                'size': 6.0,
                'lifetime': 0.8,
                'count': 12,
                'speed': 4.0
            },
            ParticleType.ENERGY_GAIN: {
                'color': (1.0, 1.0, 0.0, 1.0),  # Yellow
                'size': 4.0,
                'lifetime': 0.4,
                'count': 5,
                'speed': 2.0
            },
            ParticleType.ENERGY_LOSS: {
                'color': (1.0, 0.5, 0.0, 1.0),  # Orange
                'size': 3.0,
                'lifetime': 0.3,
                'count': 3,
                'speed': 1.5
            }
        }

        # Create VAO for rendering
        self._create_vao()

        print(f"✅ ParticleRenderer initialized ({max_particles} particles)")

    def _create_particle_buffer(self):
        """Create particle SSBO."""
        # Particle structure (must match compute shader)
        particle_dtype = np.dtype([
            ('position', np.float32, 2),     # vec2
            ('velocity', np.float32, 2),     # vec2
            ('color', np.float32, 4),        # vec4
            ('size', np.float32),            # float
            ('lifetime', np.float32),        # float
            ('max_lifetime', np.float32),    # float
            ('padding', np.float32),         # float (alignment)
        ])

        # Initialize all particles as dead
        particles = np.zeros(self.max_particles, dtype=particle_dtype)
        particles['lifetime'] = -1.0  # Negative = dead

        # Create SSBO
        self.particle_buffer = self.ctx.buffer(particles.tobytes())

        # Bind to binding point 0 (matches compute shader)
        self.particle_buffer.bind_to_storage_buffer(0)

    def _create_vao(self):
        """Create VAO for particle rendering."""
        shader = self.shader_manager.get_shader('particle')
        if not shader:
            raise RuntimeError("Particle shader not loaded")

        # VAO using particle buffer
        self.vao = self.ctx.vertex_array(
            shader.program,
            [
                (self.particle_buffer,
                 '2f 2f 4f 1f 1f 1f 1f',  # position, velocity, color, size, lifetime, max_lifetime, padding
                 'particle_position', None, 'particle_color', 'particle_size', 'particle_lifetime', None, None)
            ]
        )

    def emit(self, particle_type: ParticleType, position: Tuple[float, float],
             color: Tuple[float, float, float, float] = None):
        """
        Emit particles at position.

        Args:
            particle_type: Type of particle effect
            position: (x, y) position
            color: Optional custom color (overrides default)
        """
        config = self.configs[particle_type]
        count = config['count']

        # Queue particles for spawning
        for _ in range(count):
            # Random velocity (explosion pattern)
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(0.5, 1.0) * config['speed']
            velocity = (np.cos(angle) * speed, np.sin(angle) * speed)

            # Random offset
            offset_x = np.random.uniform(-2, 2)
            offset_y = np.random.uniform(-2, 2)
            pos = (position[0] + offset_x, position[1] + offset_y)

            particle_data = {
                'position': pos,
                'velocity': velocity,
                'color': color if color else config['color'],
                'size': config['size'],
                'lifetime': config['lifetime'],
                'max_lifetime': config['lifetime']
            }

            self.spawn_queue.append(particle_data)

    def update(self, dt: float):
        """
        Update particles using compute shader.

        Args:
            dt: Delta time
        """
        # Spawn queued particles
        self._spawn_queued_particles()

        # Run compute shader for physics update
        compute_shader = self.shader_manager.get_shader('particle_update')
        if not compute_shader:
            return

        # Set uniforms
        compute_shader.use()
        compute_shader.set_uniforms(
            dt=dt,
            gravity=(0.0, -50.0),  # Downward gravity
            damping=0.95,  # Air resistance
            particle_count=self.max_particles
        )

        # Run compute shader
        # 256 threads per group, calculate number of groups
        num_groups = (self.max_particles + 255) // 256
        compute_shader.program.run(group_x=num_groups)

        # Memory barrier to ensure compute shader writes are visible
        self.ctx.memory_barrier()

    def _spawn_queued_particles(self):
        """Spawn particles from spawn queue."""
        if not self.spawn_queue:
            return

        # Read current particle buffer
        data = np.frombuffer(self.particle_buffer.read(), dtype=np.dtype([
            ('position', np.float32, 2),
            ('velocity', np.float32, 2),
            ('color', np.float32, 4),
            ('size', np.float32),
            ('lifetime', np.float32),
            ('max_lifetime', np.float32),
            ('padding', np.float32),
        ]))

        # Find dead particles and replace
        spawned = 0
        while self.spawn_queue and spawned < 100:  # Limit spawns per frame
            # Find next dead particle
            for i in range(self.next_particle, self.max_particles):
                if data[i]['lifetime'] <= 0:
                    # Spawn new particle
                    particle_data = self.spawn_queue.pop(0)

                    data[i]['position'] = particle_data['position']
                    data[i]['velocity'] = particle_data['velocity']
                    data[i]['color'] = particle_data['color']
                    data[i]['size'] = particle_data['size']
                    data[i]['lifetime'] = particle_data['lifetime']
                    data[i]['max_lifetime'] = particle_data['max_lifetime']

                    self.next_particle = (i + 1) % self.max_particles
                    spawned += 1
                    break
            else:
                # No dead particles found, wrap around
                self.next_particle = 0
                break

        # Write back to GPU
        if spawned > 0:
            self.particle_buffer.write(data.tobytes())

    def render(self, projection: np.ndarray, view: np.ndarray):
        """
        Render particles.

        Args:
            projection: Projection matrix
            view: View matrix
        """
        shader = self.shader_manager.get_shader('particle')
        if not shader:
            return

        # Enable point sprites
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)

        # Set uniforms
        shader.use()
        shader.set_uniforms(
            projection=projection.T.flatten().astype('f4'),
            view=view.T.flatten().astype('f4')
        )

        # Render all particles (GPU will skip dead ones via vertex shader)
        self.vao.render(moderngl.POINTS, vertices=self.max_particles)

        self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)

    def get_stats(self) -> dict:
        """Get particle statistics."""
        # Count active particles (expensive - use sparingly)
        data = np.frombuffer(self.particle_buffer.read(), dtype=np.dtype([
            ('position', np.float32, 2),
            ('velocity', np.float32, 2),
            ('color', np.float32, 4),
            ('size', np.float32),
            ('lifetime', np.float32),
            ('max_lifetime', np.float32),
            ('padding', np.float32),
        ]))

        active = np.sum(data['lifetime'] > 0)

        return {
            'active': int(active),
            'max': self.max_particles,
            'queued': len(self.spawn_queue)
        }

    def cleanup(self):
        """Release GPU resources."""
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'particle_buffer'):
            self.particle_buffer.release()
