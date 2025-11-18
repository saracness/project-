"""
Configuration System for Micro-Life Simulation
Handles GPU/CPU selection, performance settings, visual effects
"""
import torch


class SimulationConfig:
    """
    Central configuration for simulation performance and features.
    """

    def __init__(
        self,
        # GPU Settings
        use_gpu=None,  # Auto-detect if None
        gpu_device=None,  # Auto-select if None
        batch_size=32,

        # Simulation Settings
        max_organisms=1000,
        max_food=500,
        max_timesteps=None,  # Unlimited if None

        # Visual Effects
        enable_trails=True,
        enable_particles=True,
        enable_heatmap=False,
        enable_minimap=True,
        enable_glow=True,

        # Performance Settings
        target_fps=60,
        skip_render_frames=0,  # Render every Nth frame
        cull_offscreen=True,

        # Trail Settings
        trail_length=20,
        trail_fade=True,

        # Particle Settings
        max_particles=1000,
        particle_lifetime=1.0,

        # Heatmap Settings
        heatmap_resolution=50,
        heatmap_blur=True,

        # AI Settings
        enable_ai_metrics=True,
        metrics_update_interval=10,  # Timesteps

        # Debug
        debug_mode=False,
        show_fps=True,
        profile_performance=False
    ):
        # GPU Detection and Configuration
        if use_gpu is None:
            self.use_gpu = torch.cuda.is_available()
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()

        if self.use_gpu:
            if gpu_device is None:
                self.gpu_device = 'cuda:0'
            else:
                self.gpu_device = gpu_device
            self.device = torch.device(self.gpu_device)
        else:
            self.gpu_device = 'cpu'
            self.device = torch.device('cpu')

        # Simulation
        self.batch_size = batch_size
        self.max_organisms = max_organisms
        self.max_food = max_food
        self.max_timesteps = max_timesteps

        # Visual Effects
        self.enable_trails = enable_trails
        self.enable_particles = enable_particles
        self.enable_heatmap = enable_heatmap
        self.enable_minimap = enable_minimap
        self.enable_glow = enable_glow

        # Performance
        self.target_fps = target_fps
        self.skip_render_frames = skip_render_frames
        self.cull_offscreen = cull_offscreen

        # Trail Settings
        self.trail_length = trail_length
        self.trail_fade = trail_fade

        # Particle Settings
        self.max_particles = max_particles
        self.particle_lifetime = particle_lifetime

        # Heatmap Settings
        self.heatmap_resolution = heatmap_resolution
        self.heatmap_blur = heatmap_blur

        # AI Settings
        self.enable_ai_metrics = enable_ai_metrics
        self.metrics_update_interval = metrics_update_interval

        # Debug
        self.debug_mode = debug_mode
        self.show_fps = show_fps
        self.profile_performance = profile_performance

    def __repr__(self):
        return (
            f"SimulationConfig(\n"
            f"  GPU: {self.use_gpu} ({self.gpu_device})\n"
            f"  Max Organisms: {self.max_organisms}\n"
            f"  Trails: {self.enable_trails}\n"
            f"  Particles: {self.enable_particles}\n"
            f"  Heatmap: {self.enable_heatmap}\n"
            f"  MiniMap: {self.enable_minimap}\n"
            f"  Target FPS: {self.target_fps}\n"
            f")"
        )

    def get_info(self):
        """Get detailed configuration info."""
        info = []
        info.append("=" * 60)
        info.append("SIMULATION CONFIGURATION")
        info.append("=" * 60)

        # GPU Info
        info.append("\n🖥️  GPU/CPU:")
        info.append(f"  Use GPU: {self.use_gpu}")
        info.append(f"  Device: {self.gpu_device}")
        if self.use_gpu:
            if torch.cuda.is_available():
                info.append(f"  GPU Name: {torch.cuda.get_device_name(0)}")
                info.append(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                info.append("  ⚠️  CUDA not available")
        info.append(f"  Batch Size: {self.batch_size}")

        # Simulation Settings
        info.append("\n⚙️  Simulation:")
        info.append(f"  Max Organisms: {self.max_organisms}")
        info.append(f"  Max Food: {self.max_food}")
        info.append(f"  Max Timesteps: {self.max_timesteps or 'Unlimited'}")

        # Visual Effects
        info.append("\n✨ Visual Effects:")
        info.append(f"  Trails: {'✅' if self.enable_trails else '❌'} (length={self.trail_length})")
        info.append(f"  Particles: {'✅' if self.enable_particles else '❌'} (max={self.max_particles})")
        info.append(f"  Heatmap: {'✅' if self.enable_heatmap else '❌'} (resolution={self.heatmap_resolution})")
        info.append(f"  MiniMap: {'✅' if self.enable_minimap else '❌'}")
        info.append(f"  Glow: {'✅' if self.enable_glow else '❌'}")

        # Performance
        info.append("\n⚡ Performance:")
        info.append(f"  Target FPS: {self.target_fps}")
        info.append(f"  Skip Render Frames: {self.skip_render_frames}")
        info.append(f"  Cull Offscreen: {'✅' if self.cull_offscreen else '❌'}")

        # AI
        info.append("\n🧠 AI:")
        info.append(f"  Metrics Tracking: {'✅' if self.enable_ai_metrics else '❌'}")
        info.append(f"  Update Interval: {self.metrics_update_interval} timesteps")

        # Debug
        info.append("\n🐛 Debug:")
        info.append(f"  Debug Mode: {'✅' if self.debug_mode else '❌'}")
        info.append(f"  Show FPS: {'✅' if self.show_fps else '❌'}")
        info.append(f"  Profile Performance: {'✅' if self.profile_performance else '❌'}")

        info.append("\n" + "=" * 60)

        return "\n".join(info)


# Preset Configurations

def get_performance_config():
    """High performance config - minimal visuals, max speed."""
    return SimulationConfig(
        use_gpu=True,
        batch_size=128,
        max_organisms=2000,
        enable_trails=False,
        enable_particles=False,
        enable_heatmap=False,
        enable_minimap=False,
        enable_glow=False,
        skip_render_frames=2,
        cull_offscreen=True
    )


def get_quality_config():
    """High quality config - all visuals enabled."""
    return SimulationConfig(
        use_gpu=True,
        batch_size=64,
        max_organisms=500,
        enable_trails=True,
        enable_particles=True,
        enable_heatmap=True,
        enable_minimap=True,
        enable_glow=True,
        trail_length=30,
        max_particles=2000
    )


def get_balanced_config():
    """Balanced config - good visuals, good performance."""
    return SimulationConfig(
        use_gpu=True,
        batch_size=64,
        max_organisms=1000,
        enable_trails=True,
        enable_particles=True,
        enable_heatmap=False,
        enable_minimap=True,
        enable_glow=True,
        trail_length=20,
        max_particles=1000
    )


def get_cpu_config():
    """CPU-only config - optimized for CPU."""
    return SimulationConfig(
        use_gpu=False,
        batch_size=16,
        max_organisms=200,
        enable_trails=True,
        enable_particles=False,
        enable_heatmap=False,
        enable_minimap=True,
        enable_glow=False,
        trail_length=15,
        skip_render_frames=1
    )


def get_debug_config():
    """Debug config - all debug features enabled."""
    return SimulationConfig(
        use_gpu=True,
        batch_size=32,
        max_organisms=100,
        enable_trails=True,
        enable_particles=True,
        enable_heatmap=True,
        enable_minimap=True,
        debug_mode=True,
        show_fps=True,
        profile_performance=True
    )


# Auto-select best config
def get_auto_config():
    """Automatically select best config based on hardware."""
    if torch.cuda.is_available():
        # Check GPU memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_mem_gb >= 6:
            print("🚀 High-end GPU detected - Using Quality Config")
            return get_quality_config()
        elif gpu_mem_gb >= 4:
            print("⚡ Mid-range GPU detected - Using Balanced Config")
            return get_balanced_config()
        else:
            print("💻 Low-end GPU detected - Using Performance Config")
            return get_performance_config()
    else:
        print("🖥️  No GPU detected - Using CPU Config")
        return get_cpu_config()
