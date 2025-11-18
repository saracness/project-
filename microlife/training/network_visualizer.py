"""
NetworkVisualizer - Neural network structure and activation visualization
"""
import moderngl
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any


class NetworkVisualizer:
    """
    Visualize neural network structure and activations.

    Features:
    - Network topology diagram
    - Real-time neuron activations
    - Weight heatmaps
    - Gradient flow visualization
    """

    def __init__(self, ctx: moderngl.Context, window_size: Tuple[int, int]):
        """
        Initialize network visualizer.

        Args:
            ctx: ModernGL context
            window_size: Window size (width, height)
        """
        self.ctx = ctx
        self.window_width, self.window_height = window_size

        # Network structure
        self.layer_sizes: List[int] = []
        self.activations: List[np.ndarray] = []
        self.weights: List[np.ndarray] = []

        # Rendering
        self.max_vertices = 100000
        self.vbo = ctx.buffer(reserve=self.max_vertices * 2 * 4, dynamic=True)

        # Create shader
        self._create_shader()

        # Layout configuration
        self.neuron_radius = 8.0
        self.layer_spacing = 150.0
        self.neuron_spacing = 30.0

        # Orthographic projection
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

    def set_network(self, layer_sizes: List[int]):
        """
        Set network structure.

        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
        """
        self.layer_sizes = layer_sizes
        self.activations = [np.zeros(size) for size in layer_sizes]

    def update_from_pytorch(self, model: nn.Module, input_sample: torch.Tensor):
        """
        Update visualization from PyTorch model.

        Args:
            model: PyTorch neural network model
            input_sample: Sample input tensor
        """
        # Extract layer sizes
        layer_sizes = []
        activations = []

        # Forward pass with hooks to capture activations
        hooks = []

        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                activations.append(output.detach().cpu().numpy().flatten())

        # Register hooks
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        with torch.no_grad():
            output = model(input_sample)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Update structure
        if activations:
            self.layer_sizes = [len(act) for act in activations]
            self.activations = activations

    def render(self, panel_x: int, panel_y: int, panel_width: int, panel_height: int):
        """
        Render network visualization.

        Args:
            panel_x: Panel X position
            panel_y: Panel Y position
            panel_width: Panel width
            panel_height: Panel height
        """
        if not self.layer_sizes:
            return

        # Calculate layout
        positions = self._calculate_neuron_positions(
            panel_x, panel_y, panel_width, panel_height
        )

        # Render connections
        self._render_connections(positions)

        # Render neurons
        self._render_neurons(positions)

    def _calculate_neuron_positions(self, x: int, y: int, width: int, height: int) -> List[List[Tuple[float, float]]]:
        """Calculate neuron positions for rendering."""
        num_layers = len(self.layer_sizes)

        # Calculate layer X positions
        if num_layers > 1:
            layer_x_step = width / (num_layers - 1)
        else:
            layer_x_step = 0

        positions = []

        for layer_idx, layer_size in enumerate(self.layer_sizes):
            layer_positions = []

            # Calculate neuron Y positions
            if layer_size > 1:
                neuron_y_step = height / (layer_size - 1)
            else:
                neuron_y_step = 0

            layer_x = x + layer_idx * layer_x_step

            for neuron_idx in range(layer_size):
                if layer_size == 1:
                    neuron_y = y + height / 2
                else:
                    neuron_y = y + neuron_idx * neuron_y_step

                layer_positions.append((layer_x, neuron_y))

            positions.append(layer_positions)

        return positions

    def _render_connections(self, positions: List[List[Tuple[float, float]]]):
        """Render connections between neurons."""
        vertices = []

        # Draw connections between consecutive layers
        for layer_idx in range(len(positions) - 1):
            curr_layer = positions[layer_idx]
            next_layer = positions[layer_idx + 1]

            for src_x, src_y in curr_layer:
                for dst_x, dst_y in next_layer:
                    vertices.extend([[src_x, src_y], [dst_x, dst_y]])

        if not vertices:
            return

        vertices = np.array(vertices, dtype=np.float32)
        self.vbo.write(vertices.tobytes())

        self.program['projection'].write(self.projection.T.astype('f4').tobytes())
        self.program['color'].value = (0.3, 0.3, 0.3, 0.3)

        self.ctx.line_width = 1.0
        self.vao.render(moderngl.LINES, vertices=len(vertices))

    def _render_neurons(self, positions: List[List[Tuple[float, float]]]):
        """Render neurons as circles."""
        for layer_idx, layer_positions in enumerate(positions):
            for neuron_idx, (x, y) in enumerate(layer_positions):
                # Get activation value
                if layer_idx < len(self.activations):
                    activations = self.activations[layer_idx]
                    if neuron_idx < len(activations):
                        activation = float(activations[neuron_idx])
                    else:
                        activation = 0.0
                else:
                    activation = 0.0

                # Color based on activation
                color = self._activation_to_color(activation)

                # Draw circle
                self._draw_circle(x, y, self.neuron_radius, color)

    def _activation_to_color(self, activation: float) -> Tuple[float, float, float, float]:
        """Convert activation value to color."""
        # Normalize to 0-1 (assuming tanh/sigmoid activation)
        normalized = (np.tanh(activation) + 1.0) / 2.0

        # Blue (low) to Red (high)
        r = normalized
        g = 0.3
        b = 1.0 - normalized
        a = 0.9

        return (r, g, b, a)

    def _draw_circle(self, cx: float, cy: float, radius: float,
                     color: Tuple[float, float, float, float], segments: int = 16):
        """Draw filled circle."""
        vertices = []

        # Center vertex
        vertices.append([cx, cy])

        # Circle vertices
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


class WeightHeatmapVisualizer:
    """
    Visualize weight matrices as heatmaps.
    """

    def __init__(self, ctx: moderngl.Context):
        """Initialize weight heatmap visualizer."""
        self.ctx = ctx

    def render_weight_matrix(self, weights: np.ndarray,
                            x: int, y: int, width: int, height: int):
        """
        Render weight matrix as heatmap.

        Args:
            weights: Weight matrix (2D array)
            x: Panel X position
            y: Panel Y position
            width: Panel width
            height: Panel height
        """
        # Normalize weights to 0-1
        w_min = weights.min()
        w_max = weights.max()

        if w_max - w_min > 0:
            normalized = (weights - w_min) / (w_max - w_min)
        else:
            normalized = np.zeros_like(weights)

        # Create texture
        texture = self.ctx.texture(
            weights.shape[::-1],  # (width, height)
            components=3,
            data=self._colorize_weights(normalized).tobytes()
        )

        # TODO: Render texture to screen
        # This requires a textured quad shader

        texture.release()

    def _colorize_weights(self, normalized_weights: np.ndarray) -> np.ndarray:
        """Convert normalized weights to RGB colors."""
        # Blue (negative) to White (zero) to Red (positive)
        # Assuming weights are centered around 0

        h, w = normalized_weights.shape
        colors = np.zeros((h, w, 3), dtype=np.uint8)

        # Map 0-0.5 to blue, 0.5-1.0 to red
        for i in range(h):
            for j in range(w):
                val = normalized_weights[i, j]

                if val < 0.5:
                    # Blue to white
                    t = val * 2
                    colors[i, j] = [int(t * 255), int(t * 255), 255]
                else:
                    # White to red
                    t = (val - 0.5) * 2
                    colors[i, j] = [255, int((1 - t) * 255), int((1 - t) * 255)]

        return colors
