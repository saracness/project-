"""
CNN-Based Brain for Visual Perception
Processes environment as image/grid
"""
import numpy as np
from .brain_base import Brain


class CNNBrain(Brain):
    """
    Convolutional Neural Network brain.
    Processes environment as a 2D grid (visual perception).
    Inspired by visual cortex in biology.
    """

    def __init__(self, grid_size=20, hidden_size=64):
        super().__init__(brain_type="CNN")
        self.grid_size = grid_size
        self.hidden_size = hidden_size

        # CNN layers (simplified - 2D convolution)
        # Conv layer: 1 input channel, 8 filters, 3x3 kernel
        self.conv1_filters = np.random.randn(8, 3, 3) * 0.1
        self.conv1_bias = np.zeros(8)

        # Flatten and fully connected
        conv_output_size = 8 * (grid_size - 2) * (grid_size - 2)  # After 3x3 conv
        self.fc1_weights = np.random.randn(conv_output_size, hidden_size) * 0.1
        self.fc1_bias = np.zeros(hidden_size)

        # Output layer: 9 actions
        self.fc2_weights = np.random.randn(hidden_size, 9) * 0.1
        self.fc2_bias = np.zeros(9)

        self.epsilon = 0.3
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.998
        self.lr = 0.001

        # Actions
        self.actions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)
        ]

    def _create_visual_grid(self, state):
        """
        Create a 2D visual grid of the environment.
        This is like what the organism "sees".
        """
        grid = np.zeros((self.grid_size, self.grid_size))

        # Organism is always at center
        center = self.grid_size // 2

        # Add food locations (if available)
        if 'visible_food' in state:
            for food_pos in state['visible_food']:
                # Convert world coords to grid coords
                rel_x = int(food_pos[0] / 25)  # Scale down
                rel_y = int(food_pos[1] / 25)
                grid_x = center + rel_x
                grid_y = center + rel_y

                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    grid[grid_y, grid_x] = 1.0  # Food

        # Add temperature zones
        if state.get('in_temperature_zone', False):
            # Mark area around organism as temperature zone
            grid[center-2:center+3, center-2:center+3] = -0.5

        # Add nearest food direction (simplified)
        angle = state.get('nearest_food_angle', 0)
        if angle is not None:
            offset_x = int(3 * np.cos(angle))
            offset_y = int(3 * np.sin(angle))
            mark_x = center + offset_x
            mark_y = center + offset_y
            if 0 <= mark_x < self.grid_size and 0 <= mark_y < self.grid_size:
                grid[mark_y, mark_x] = 0.8

        return grid

    def _convolve2d(self, image, kernel):
        """Simple 2D convolution."""
        h, w = image.shape
        kh, kw = kernel.shape
        output_h = h - kh + 1
        output_w = w - kw + 1
        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                region = image[i:i+kh, j:j+kw]
                output[i, j] = np.sum(region * kernel)

        return output

    def _forward(self, grid):
        """Forward pass through CNN."""
        # Convolutional layer
        conv_outputs = []
        for filter_idx in range(self.conv1_filters.shape[0]):
            conv_out = self._convolve2d(grid, self.conv1_filters[filter_idx])
            # ReLU activation
            conv_out = np.maximum(0, conv_out + self.conv1_bias[filter_idx])
            conv_outputs.append(conv_out)

        # Flatten
        flattened = np.concatenate([c.flatten() for c in conv_outputs])

        # Ensure correct size
        expected_size = self.fc1_weights.shape[0]
        if len(flattened) != expected_size:
            # Resize if needed
            if len(flattened) < expected_size:
                flattened = np.pad(flattened, (0, expected_size - len(flattened)))
            else:
                flattened = flattened[:expected_size]

        # Fully connected layers
        h1 = np.maximum(0, np.dot(flattened, self.fc1_weights) + self.fc1_bias)
        q_values = np.dot(h1, self.fc2_weights) + self.fc2_bias

        return q_values, flattened

    def decide_action(self, state):
        """Decide action using CNN."""
        self.decision_count += 1

        # Create visual grid
        grid = self._create_visual_grid(state)

        # Forward pass
        q_values, features = self._forward(grid)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, len(self.actions))
        else:
            action_idx = np.argmax(q_values)

        direction = self.actions[action_idx]

        return {
            'move_direction': direction,
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0,
            '_action_idx': action_idx,
            '_grid': grid,
            '_features': features
        }

    def learn(self, state, action, reward, next_state, done):
        """Learn from visual experience."""
        if '_grid' not in action or '_features' not in action:
            return

        grid = action['_grid']
        features = action['_features']
        action_idx = action['_action_idx']

        # Forward pass
        q_values, _ = self._forward(grid)

        # Calculate target
        if done:
            target = reward
        else:
            next_grid = self._create_visual_grid(next_state)
            next_q, _ = self._forward(next_grid)
            target = reward + 0.95 * np.max(next_q)

        # Error
        q_values[action_idx] = target
        output_error = q_values - (np.dot(np.maximum(0, np.dot(features, self.fc1_weights) + self.fc1_bias), self.fc2_weights) + self.fc2_bias)

        # Gradient descent on FC layers (simplified)
        h1 = np.maximum(0, np.dot(features, self.fc1_weights) + self.fc1_bias)

        self.fc2_weights += self.lr * np.outer(h1, output_error)
        self.fc2_bias += self.lr * output_error

        # Backprop to FC1
        fc2_error = np.dot(output_error, self.fc2_weights.T) * (h1 > 0)
        self.fc1_weights += self.lr * np.outer(features, fc2_error)
        self.fc1_bias += self.lr * fc2_error

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_reward += reward


class ResidualCNNBrain(CNNBrain):
    """
    CNN with residual connections (ResNet-inspired).
    Used in modern computer vision and biological neural modeling.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brain_type = "ResNet-CNN"

        # Additional residual layer
        self.residual_weights = np.random.randn(self.hidden_size, self.hidden_size) * 0.1
        self.residual_bias = np.zeros(self.hidden_size)

    def _forward(self, grid):
        """Forward with residual connection."""
        q_values, flattened = super()._forward(grid)

        # Add residual connection in hidden layer
        h1 = np.maximum(0, np.dot(flattened, self.fc1_weights) + self.fc1_bias)

        # Residual block: h2 = h1 + F(h1)
        residual = np.maximum(0, np.dot(h1, self.residual_weights) + self.residual_bias)
        h2 = h1 + residual  # Skip connection!

        # Recalculate output with residual features
        q_values = np.dot(h2, self.fc2_weights) + self.fc2_bias

        return q_values, flattened
