"""
Shader Manager - Production Grade
Handles shader compilation, linking, caching, hot-reload, and uniform management
"""
import moderngl
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ShaderProgram:
    """
    Wrapper for ModernGL shader program with enhanced features.

    Features:
    - Automatic uniform caching
    - Type-safe uniform setting
    - Hot-reload support
    - Error handling with detailed messages
    """

    def __init__(self, ctx: moderngl.Context, program: moderngl.Program,
                 name: str, vertex_path: str = None, fragment_path: str = None,
                 geometry_path: str = None, compute_path: str = None):
        """
        Initialize shader program wrapper.

        Args:
            ctx: ModernGL context
            program: Compiled shader program
            name: Shader name
            vertex_path: Path to vertex shader source
            fragment_path: Path to fragment shader source
            geometry_path: Path to geometry shader source (optional)
            compute_path: Path to compute shader source (optional)
        """
        self.ctx = ctx
        self.program = program
        self.name = name

        # Source paths for hot-reload
        self.vertex_path = vertex_path
        self.fragment_path = fragment_path
        self.geometry_path = geometry_path
        self.compute_path = compute_path

        # Uniform cache
        self._uniform_cache: Dict[str, Any] = {}

        # File modification times for hot-reload
        self._mtime: Dict[str, float] = {}
        self._update_mtimes()

    def _update_mtimes(self):
        """Update file modification times."""
        for path in [self.vertex_path, self.fragment_path,
                     self.geometry_path, self.compute_path]:
            if path and os.path.exists(path):
                self._mtime[path] = os.path.getmtime(path)

    def needs_reload(self) -> bool:
        """
        Check if shader files have been modified.

        Returns:
            True if any shader file has been modified
        """
        for path, old_mtime in self._mtime.items():
            if os.path.exists(path):
                new_mtime = os.path.getmtime(path)
                if new_mtime > old_mtime:
                    return True
        return False

    def use(self):
        """Bind this shader program."""
        if self.program:
            self.program.use()

    def release(self):
        """Release shader program resources."""
        if self.program:
            self.program.release()
            self.program = None

    def set_uniform(self, name: str, value: Any):
        """
        Set uniform value with caching.

        Args:
            name: Uniform name
            value: Uniform value (int, float, tuple, list, numpy array)
        """
        # Check cache to avoid redundant GPU calls
        if name in self._uniform_cache and self._uniform_cache[name] == value:
            return

        try:
            if name in self.program:
                self.program[name].value = value
                self._uniform_cache[name] = value
        except KeyError:
            logger.warning(f"Uniform '{name}' not found in shader '{self.name}'")
        except Exception as e:
            logger.error(f"Error setting uniform '{name}' in shader '{self.name}': {e}")

    def set_uniforms(self, **uniforms):
        """
        Set multiple uniforms at once.

        Args:
            **uniforms: Keyword arguments with uniform names and values
        """
        for name, value in uniforms.items():
            self.set_uniform(name, value)

    def get_uniform_location(self, name: str) -> int:
        """
        Get uniform location.

        Args:
            name: Uniform name

        Returns:
            Uniform location or -1 if not found
        """
        try:
            return self.program[name].location
        except KeyError:
            return -1

    def __getitem__(self, name: str):
        """Direct access to uniforms."""
        return self.program[name]

    def __contains__(self, name: str) -> bool:
        """Check if uniform exists."""
        return name in self.program


class ShaderManager:
    """
    Manages all shaders in the application.

    Features:
    - Automatic shader discovery
    - Hot-reload support (development mode)
    - Error recovery
    - Shader caching
    - Include directive support
    """

    def __init__(self, ctx: moderngl.Context, shader_dir: str = None):
        """
        Initialize shader manager.

        Args:
            ctx: ModernGL context
            shader_dir: Directory containing shader files
        """
        self.ctx = ctx

        # Default shader directory
        if shader_dir is None:
            shader_dir = Path(__file__).parent / 'shaders'
        self.shader_dir = Path(shader_dir)

        # Shader cache
        self._shaders: Dict[str, ShaderProgram] = {}

        # Hot-reload settings
        self.hot_reload_enabled = False
        self._last_reload_check = 0
        self.reload_check_interval = 1.0  # Check every second

        logger.info(f"ShaderManager initialized with shader dir: {self.shader_dir}")

    def load_shader(self, name: str, vertex_file: str = None, fragment_file: str = None,
                   geometry_file: str = None, compute_file: str = None) -> Optional[ShaderProgram]:
        """
        Load and compile shader program.

        Args:
            name: Shader program name
            vertex_file: Vertex shader filename
            fragment_file: Fragment shader filename
            geometry_file: Geometry shader filename (optional)
            compute_file: Compute shader filename (optional)

        Returns:
            ShaderProgram instance or None on error
        """
        try:
            # Read shader sources
            vertex_src = None
            fragment_src = None
            geometry_src = None
            compute_src = None

            vertex_path = None
            fragment_path = None
            geometry_path = None
            compute_path = None

            if vertex_file:
                vertex_path = self.shader_dir / vertex_file
                vertex_src = self._read_shader(vertex_path)

            if fragment_file:
                fragment_path = self.shader_dir / fragment_file
                fragment_src = self._read_shader(fragment_path)

            if geometry_file:
                geometry_path = self.shader_dir / geometry_file
                geometry_src = self._read_shader(geometry_path)

            if compute_file:
                compute_path = self.shader_dir / compute_file
                compute_src = self._read_shader(compute_path)

            # Compile program
            program = self.ctx.program(
                vertex_shader=vertex_src,
                fragment_shader=fragment_src,
                geometry_shader=geometry_src,
                # Compute shaders are separate programs in ModernGL
            )

            # If compute shader provided, create compute program instead
            if compute_src:
                program = self.ctx.compute_shader(compute_src)

            # Create wrapper
            shader_program = ShaderProgram(
                self.ctx, program, name,
                str(vertex_path) if vertex_path else None,
                str(fragment_path) if fragment_path else None,
                str(geometry_path) if geometry_path else None,
                str(compute_path) if compute_path else None
            )

            # Cache
            self._shaders[name] = shader_program

            logger.info(f"Loaded shader '{name}'")
            return shader_program

        except Exception as e:
            logger.error(f"Failed to load shader '{name}': {e}")
            return None

    def _read_shader(self, path: Path) -> str:
        """
        Read shader source with include directive support.

        Args:
            path: Path to shader file

        Returns:
            Shader source code
        """
        if not path.exists():
            raise FileNotFoundError(f"Shader file not found: {path}")

        source = path.read_text()

        # Process #include directives
        lines = []
        for line in source.split('\n'):
            if line.strip().startswith('#include'):
                # Extract include path
                include_file = line.split('"')[1]
                include_path = self.shader_dir / include_file

                # Read included file (recursive)
                included_src = self._read_shader(include_path)
                lines.append(included_src)
            else:
                lines.append(line)

        return '\n'.join(lines)

    def get_shader(self, name: str) -> Optional[ShaderProgram]:
        """
        Get shader by name.

        Args:
            name: Shader name

        Returns:
            ShaderProgram instance or None
        """
        return self._shaders.get(name)

    def reload_shader(self, name: str) -> bool:
        """
        Reload shader from disk.

        Args:
            name: Shader name

        Returns:
            True if successful
        """
        shader = self._shaders.get(name)
        if not shader:
            logger.warning(f"Cannot reload shader '{name}': not found")
            return False

        try:
            # Store old program
            old_program = shader.program

            # Reload sources and compile
            vertex_src = self._read_shader(shader.vertex_path) if shader.vertex_path else None
            fragment_src = self._read_shader(shader.fragment_path) if shader.fragment_path else None
            geometry_src = self._read_shader(shader.geometry_path) if shader.geometry_path else None
            compute_src = self._read_shader(shader.compute_path) if shader.compute_path else None

            if compute_src:
                new_program = self.ctx.compute_shader(compute_src)
            else:
                new_program = self.ctx.program(
                    vertex_shader=vertex_src,
                    fragment_shader=fragment_src,
                    geometry_shader=geometry_src
                )

            # Replace program
            shader.program = new_program
            shader._update_mtimes()
            shader._uniform_cache.clear()

            # Release old program
            old_program.release()

            logger.info(f"Reloaded shader '{name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to reload shader '{name}': {e}")
            return False

    def reload_all(self):
        """Reload all shaders from disk."""
        for name in list(self._shaders.keys()):
            self.reload_shader(name)

    def check_hot_reload(self):
        """Check and reload modified shaders (if hot-reload enabled)."""
        if not self.hot_reload_enabled:
            return

        current_time = time.time()
        if current_time - self._last_reload_check < self.reload_check_interval:
            return

        self._last_reload_check = current_time

        # Check each shader for modifications
        for name, shader in self._shaders.items():
            if shader.needs_reload():
                logger.info(f"Detected modification in shader '{name}', reloading...")
                self.reload_shader(name)

    def cleanup(self):
        """Release all shader resources."""
        for shader in self._shaders.values():
            shader.release()
        self._shaders.clear()
        logger.info("ShaderManager cleaned up")
