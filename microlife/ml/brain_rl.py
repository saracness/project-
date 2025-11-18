"""
Reinforcement Learning Brains:
1. Q-Learning (Table-based)
2. DQN (Deep Q-Network)
"""
import numpy as np
import random
from collections import deque
from .brain_base import Brain


class QLearningBrain(Brain):
    """
    Q-Learning brain using table-based RL.
    Good for discrete state spaces.
    """

    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.3):
        super().__init__(brain_type="Q-Learning")
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Q-table: state -> action -> Q-value
        self.q_table = {}

        # Action space: 8 directions + stay
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

    def _discretize_state(self, state):
        """Convert continuous state to discrete for Q-table."""
        # Discretize key features
        energy_bin = int(state.get('energy', 0) / 20)  # 0-10
        food_dist_bin = min(int(state.get('nearest_food_distance', 500) / 50), 10)
        angle_bin = int(state.get('nearest_food_angle', 0) / (np.pi / 4))  # 0-7
        in_temp = 1 if state.get('in_temperature_zone', False) else 0

        return (energy_bin, food_dist_bin, angle_bin, in_temp)

    def _get_q_value(self, state, action_idx):
        """Get Q-value for state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        return self.q_table[state][action_idx]

    def decide_action(self, state):
        """Decide action using epsilon-greedy Q-learning."""
        self.decision_count += 1
        discrete_state = self._discretize_state(state)

        # Epsilon-greedy: explore vs exploit
        if random.random() < self.epsilon:
            # Explore: random action
            action_idx = random.randint(0, len(self.actions) - 1)
        else:
            # Exploit: best Q-value
            if discrete_state not in self.q_table:
                self.q_table[discrete_state] = np.zeros(len(self.actions))
            action_idx = np.argmax(self.q_table[discrete_state])

        # Convert action index to action dict
        direction = self.actions[action_idx]

        return {
            'move_direction': direction,
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0,
            '_action_idx': action_idx  # Store for learning
        }

    def learn(self, state, action, reward, next_state, done):
        """Q-Learning update."""
        discrete_state = self._discretize_state(state)
        discrete_next = self._discretize_state(next_state)
        action_idx = action.get('_action_idx', 0)

        # Initialize Q-tables if needed
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next not in self.q_table:
            self.q_table[discrete_next] = np.zeros(len(self.actions))

        # Q-Learning formula:
        # Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[discrete_state][action_idx]

        if done:
            target_q = reward  # No future reward if dead
        else:
            max_next_q = np.max(self.q_table[discrete_next])
            target_q = reward + self.gamma * max_next_q

        # Update Q-value
        self.q_table[discrete_state][action_idx] = \
            current_q + self.lr * (target_q - current_q)

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.total_reward += reward


class DQNBrain(Brain):
    """
    Deep Q-Network brain using neural network.
    Better for continuous/complex state spaces.

    Note: Requires PyTorch/TensorFlow. For now, using simple numpy NN.
    """

    def __init__(self, state_size=7, hidden_size=32, learning_rate=0.001):
        super().__init__(brain_type="DQN")
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.action_size = 9  # 8 directions + stay
        self.lr = learning_rate

        # Neural network weights (simple feedforward)
        self.w1 = np.random.randn(state_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, self.action_size) * 0.1
        self.b2 = np.zeros(self.action_size)

        # Experience replay
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95

        # Action space
        self.actions = [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 0)
        ]

    def _forward(self, state_vector):
        """Forward pass through network."""
        # Hidden layer with ReLU
        h = np.maximum(0, np.dot(state_vector, self.w1) + self.b1)
        # Output layer
        q_values = np.dot(h, self.w2) + self.b2
        return q_values

    def decide_action(self, state):
        """Decide action using DQN."""
        self.decision_count += 1
        state_vector = self.get_state_vector(state)

        # Epsilon-greedy
        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_size - 1)
        else:
            q_values = self._forward(state_vector)
            action_idx = np.argmax(q_values)

        direction = self.actions[action_idx]

        return {
            'move_direction': direction,
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0,
            '_action_idx': action_idx,
            '_state_vector': state_vector
        }

    def learn(self, state, action, reward, next_state, done):
        """DQN learning with experience replay."""
        state_vector = action.get('_state_vector')
        next_vector = self.get_state_vector(next_state)
        action_idx = action.get('_action_idx', 0)

        # Store experience
        self.memory.append((state_vector, action_idx, reward, next_vector, done))

        # Learn from batch
        if len(self.memory) >= self.batch_size:
            self._replay()

        self.total_reward += reward

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _replay(self):
        """Experience replay learning."""
        # Sample random batch
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for state_vec, action_idx, reward, next_vec, done in batch:
            # Predict Q-values
            q_values = self._forward(state_vec)

            if done:
                target = reward
            else:
                next_q = self._forward(next_vec)
                target = reward + self.gamma * np.max(next_q)

            # Calculate error
            q_values[action_idx] = target

            # Backpropagation (simplified gradient descent)
            # Hidden layer
            h = np.maximum(0, np.dot(state_vec, self.w1) + self.b1)

            # Output error
            output_error = q_values - (np.dot(h, self.w2) + self.b2)

            # Update weights (simplified)
            self.w2 += self.lr * np.outer(h, output_error)
            self.b2 += self.lr * output_error

            # Hidden error
            hidden_error = np.dot(output_error, self.w2.T) * (h > 0)  # ReLU derivative

            # Update first layer
            self.w1 += self.lr * np.outer(state_vec, hidden_error)
            self.b1 += self.lr * hidden_error


class DoubleDQNBrain(DQNBrain):
    """
    Double DQN: Reduces overestimation bias.
    Used by modern RL researchers.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.brain_type = "Double-DQN"

        # Target network (for stable learning)
        self.target_w1 = self.w1.copy()
        self.target_b1 = self.b1.copy()
        self.target_w2 = self.w2.copy()
        self.target_b2 = self.b2.copy()

        self.update_target_every = 100
        self.update_counter = 0

    def _forward_target(self, state_vector):
        """Forward pass through target network."""
        h = np.maximum(0, np.dot(state_vector, self.target_w1) + self.target_b1)
        q_values = np.dot(h, self.target_w2) + self.target_b2
        return q_values

    def _replay(self):
        """Double DQN replay."""
        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for state_vec, action_idx, reward, next_vec, done in batch:
            q_values = self._forward(state_vec)

            if done:
                target = reward
            else:
                # Double DQN: use main network to select action
                next_q_main = self._forward(next_vec)
                best_action = np.argmax(next_q_main)

                # Use target network to evaluate
                next_q_target = self._forward_target(next_vec)
                target = reward + self.gamma * next_q_target[best_action]

            q_values[action_idx] = target

            # Backprop (same as DQN)
            h = np.maximum(0, np.dot(state_vec, self.w1) + self.b1)
            output_error = q_values - (np.dot(h, self.w2) + self.b2)

            self.w2 += self.lr * np.outer(h, output_error)
            self.b2 += self.lr * output_error

            hidden_error = np.dot(output_error, self.w2.T) * (h > 0)
            self.w1 += self.lr * np.outer(state_vec, hidden_error)
            self.b1 += self.lr * hidden_error

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter >= self.update_target_every:
            self.target_w1 = self.w1.copy()
            self.target_b1 = self.b1.copy()
            self.target_w2 = self.w2.copy()
            self.target_b2 = self.b2.copy()
            self.update_counter = 0
