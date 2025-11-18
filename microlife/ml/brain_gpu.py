"""
GPU-Accelerated Brains using PyTorch
High-performance neural networks for large-scale simulations
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .brain_base import Brain


class GPUBrain(Brain):
    """
    Base class for GPU-accelerated brains.
    """

    def __init__(self, brain_type, device=None):
        super().__init__(brain_type)

        # GPU device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Track if using GPU
        self.using_gpu = self.device.type == 'cuda'

        # Metrics for training visualization
        self.last_reward = 0.0
        self.last_loss = 0.0
        self.last_q_value = 0.0
        self.last_action_idx = 0

    def to_gpu(self):
        """Move brain to GPU."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            if hasattr(self, 'network'):
                self.network = self.network.to(self.device)
            if hasattr(self, 'target_network'):
                self.target_network = self.target_network.to(self.device)
            self.using_gpu = True
            print(f"✅ {self.brain_type} moved to GPU")

    def to_cpu(self):
        """Move brain to CPU."""
        self.device = torch.device('cpu')
        if hasattr(self, 'network'):
            self.network = self.network.to(self.device)
        if hasattr(self, 'target_network'):
            self.target_network = self.target_network.to(self.device)
        self.using_gpu = False
        print(f"✅ {self.brain_type} moved to CPU")

    def get_device_info(self):
        """Get device information."""
        if self.using_gpu:
            return {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,  # MB
                'gpu_memory_cached': torch.cuda.memory_reserved(0) / 1024**2  # MB
            }
        else:
            return {'device': 'cpu'}


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""

    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.network(x)


class GPUDQNBrain(GPUBrain):
    """
    GPU-accelerated Deep Q-Network.
    Uses PyTorch for fast neural network training.
    """

    def __init__(self,
                 state_size=7,
                 action_size=9,
                 hidden_size=128,
                 learning_rate=0.001,
                 discount_factor=0.95,
                 epsilon=0.3,
                 device=None,
                 batch_size=32,
                 memory_size=10000):
        super().__init__('GPU-DQN', device)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size

        # Create network
        self.network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=memory_size)

        # Action space
        self.actions = [
            (0, 1),    # North
            (1, 1),    # NE
            (1, 0),    # East
            (1, -1),   # SE
            (0, -1),   # South
            (-1, -1),  # SW
            (-1, 0),   # West
            (-1, 1),   # NW
            (0, 0)     # Stay
        ]

    def get_state_vector(self, state):
        """Convert state dict to vector."""
        return np.array([
            state.get('energy', 0) / 200.0,  # Normalize
            state.get('nearest_food_distance', 999) / 500.0,
            state.get('nearest_food_angle', 0) / np.pi,
            1.0 if state.get('in_temperature_zone', False) else 0.0,
            state.get('speed_multiplier', 1.0),
            state.get('maneuverability', 1.0),
            state.get('perception', 100.0) / 200.0
        ], dtype=np.float32)

    def decide_action(self, state):
        """Decide action using DQN."""
        self.decision_count += 1

        state_vector = self.get_state_vector(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
            q_value = 0.0
        else:
            # Use network to select action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                action_idx = q_values.argmax().item()
                q_value = q_values.max().item()

        direction = self.actions[action_idx]

        # Store for metrics
        self.last_action_idx = action_idx
        self.last_q_value = q_value

        return {
            'move_direction': direction,
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0,
            '_action_idx': action_idx,
            '_state_vector': state_vector,
            '_q_value': q_value
        }

    def learn(self, state, action, reward, next_state, done):
        """Learn from experience using DQN."""
        state_vector = action.get('_state_vector')
        next_vector = self.get_state_vector(next_state)
        action_idx = action.get('_action_idx', 0)

        # Store experience
        self.memory.append((state_vector, action_idx, reward, next_vector, done))

        # Learn from batch
        if len(self.memory) >= self.batch_size:
            self._replay()

        self.total_reward += reward
        self.last_reward = reward

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay(self):
        """Experience replay with batch training."""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)

        # Separate batch components
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Next Q-values
        with torch.no_grad():
            next_q = self.network(next_states_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss for metrics
        self.last_loss = loss.item()

    def batch_decide_action(self, states):
        """
        Batch process multiple states (for future optimization).

        Args:
            states: List of state dicts

        Returns:
            List of actions
        """
        state_vectors = np.array([self.get_state_vector(s) for s in states])
        states_tensor = torch.FloatTensor(state_vectors).to(self.device)

        with torch.no_grad():
            q_values = self.network(states_tensor)
            action_indices = q_values.argmax(dim=1).cpu().numpy()

        actions = []
        for i, action_idx in enumerate(action_indices):
            direction = self.actions[action_idx]
            actions.append({
                'move_direction': direction,
                'should_reproduce': states[i].get('energy', 0) > 150,
                'speed_multiplier': 1.0,
                '_action_idx': action_idx,
                '_state_vector': state_vectors[i]
            })

        return actions

    def save_weights(self, filepath):
        """Save network weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward
        }, filepath)
        print(f"✅ Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        print(f"✅ Weights loaded from {filepath}")


class GPUDoubleDQNBrain(GPUDQNBrain):
    """
    GPU-accelerated Double DQN.
    Reduces overestimation bias using target network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brain_type = 'GPU-Double-DQN'

        # Create target network
        self.target_network = DQNNetwork(
            self.state_size,
            self.action_size,
            self.hidden_size
        ).to(self.device)

        # Copy weights to target network
        self.target_network.load_state_dict(self.network.state_dict())

        # Update target network every N steps
        self.update_target_every = 100
        self.update_counter = 0

    def _replay(self):
        """Double DQN experience replay."""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)

        # Separate batch components
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.network(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Double DQN: Use main network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.network(next_states_tensor).argmax(1)
            next_q = self.target_network(next_states_tensor).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss for metrics
        self.last_loss = loss.item()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter >= self.update_target_every:
            self.target_network.load_state_dict(self.network.state_dict())
            self.update_counter = 0

    def save_weights(self, filepath):
        """Save network weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward,
            'update_counter': self.update_counter
        }, filepath)
        print(f"✅ Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        self.update_counter = checkpoint.get('update_counter', 0)
        print(f"✅ Weights loaded from {filepath}")


class CNNNetwork(nn.Module):
    """Convolutional Neural Network for spatial state processing."""

    def __init__(self, grid_size, action_size):
        super(CNNNetwork, self).__init__()

        self.grid_size = grid_size

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Calculate size after conv layers
        conv_output_size = (grid_size // 4) * (grid_size // 4) * 32

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        # x shape: (batch, 1, grid_size, grid_size)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class GPUCNNBrain(GPUBrain):
    """
    GPU-accelerated Convolutional Neural Network.
    Uses spatial awareness for decision making.
    """

    def __init__(self,
                 grid_size=20,
                 action_size=9,
                 learning_rate=0.001,
                 discount_factor=0.95,
                 epsilon=0.3,
                 device=None,
                 batch_size=32,
                 memory_size=10000,
                 perception_radius=100.0):
        super().__init__('GPU-CNN', device)

        self.grid_size = grid_size
        self.action_size = action_size
        self.perception_radius = perception_radius
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size

        # Create network
        self.network = CNNNetwork(grid_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=memory_size)

        # Action space
        self.actions = [
            (0, 1),    # North
            (1, 1),    # NE
            (1, 0),    # East
            (1, -1),   # SE
            (0, -1),   # South
            (-1, -1),  # SW
            (-1, 0),   # West
            (-1, 1),   # NW
            (0, 0)     # Stay
        ]

    def get_spatial_grid(self, state):
        """
        Create a spatial grid representation of the environment.

        Returns:
            Grid of shape (grid_size, grid_size) with food and organism positions
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Get organism position (center of grid)
        org_x = state.get('position', (0, 0))[0]
        org_y = state.get('position', (0, 0))[1]

        # Get food positions (if available)
        food_positions = state.get('food_positions', [])

        # Map food to grid
        for fx, fy in food_positions:
            # Calculate relative position
            rel_x = fx - org_x
            rel_y = fy - org_y

            # Check if within perception radius
            dist = np.sqrt(rel_x**2 + rel_y**2)
            if dist <= self.perception_radius:
                # Map to grid coordinates
                grid_x = int((rel_x / self.perception_radius + 1) * self.grid_size / 2)
                grid_y = int((rel_y / self.perception_radius + 1) * self.grid_size / 2)

                # Clamp to grid bounds
                grid_x = max(0, min(grid_x, self.grid_size - 1))
                grid_y = max(0, min(grid_y, self.grid_size - 1))

                # Mark food on grid (intensity based on distance)
                intensity = 1.0 - (dist / self.perception_radius)
                grid[grid_y, grid_x] = max(grid[grid_y, grid_x], intensity)

        return grid

    def decide_action(self, state):
        """Decide action using CNN."""
        self.decision_count += 1

        # Get spatial grid
        grid = self.get_spatial_grid(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
            q_value = 0.0
        else:
            # Use network to select action
            with torch.no_grad():
                grid_tensor = torch.FloatTensor(grid).unsqueeze(0).unsqueeze(0).to(self.device)
                q_values = self.network(grid_tensor)
                action_idx = q_values.argmax().item()
                q_value = q_values.max().item()

        direction = self.actions[action_idx]

        # Store for metrics
        self.last_action_idx = action_idx
        self.last_q_value = q_value

        return {
            'move_direction': direction,
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0,
            '_action_idx': action_idx,
            '_grid': grid,
            '_q_value': q_value
        }

    def learn(self, state, action, reward, next_state, done):
        """Learn from experience using CNN."""
        grid = action.get('_grid')
        next_grid = self.get_spatial_grid(next_state)
        action_idx = action.get('_action_idx', 0)

        # Store experience
        self.memory.append((grid, action_idx, reward, next_grid, done))

        # Learn from batch
        if len(self.memory) >= self.batch_size:
            self._replay()

        self.total_reward += reward
        self.last_reward = reward

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay(self):
        """Experience replay with batch training."""
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)

        # Separate batch components
        grids = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_grids = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Convert to tensors (add channel dimension)
        grids_tensor = torch.FloatTensor(grids).unsqueeze(1).to(self.device)
        next_grids_tensor = torch.FloatTensor(next_grids).unsqueeze(1).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.network(grids_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Next Q-values
        with torch.no_grad():
            next_q = self.network(next_grids_tensor).max(1)[0]
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q

        # Compute loss
        loss = self.criterion(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store loss for metrics
        self.last_loss = loss.item()

    def save_weights(self, filepath):
        """Save network weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_reward': self.total_reward
        }, filepath)
        print(f"✅ Weights saved to {filepath}")

    def load_weights(self, filepath):
        """Load network weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.total_reward = checkpoint.get('total_reward', 0.0)
        print(f"✅ Weights loaded from {filepath}")
