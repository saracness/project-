"""
Neuron Spatial Dynamics and Learning Simulation

Implements:
1. Neuron migration and movement (chemotaxis, activity-based)
2. Dynamic synaptogenesis based on proximity and activity
3. Reward-based learning tasks
4. Network-level learning metrics

Scientific References:
- Hatten (2002): Neuronal migration mechanisms
- Lohmann & Wong (2005): Dendrite and axon guidance
- Turrigiano & Nelson (2004): Homeostatic plasticity
- Schultz et al. (1997): Reward prediction and dopamine
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
import math


@dataclass
class LearningTask:
    """
    A learning task for the neural network

    Tasks:
    - Pattern recognition: Learn to respond to specific input patterns
    - Temporal sequence: Learn temporal dependencies
    - Reward prediction: Learn to predict rewards
    """
    name: str
    input_patterns: List[np.ndarray]
    target_outputs: List[np.ndarray]
    rewards: List[float]

    # Performance tracking
    correct_responses: int = 0
    total_trials: int = 0
    recent_accuracy: List[float] = None

    def __post_init__(self):
        if self.recent_accuracy is None:
            self.recent_accuracy = []

    def get_accuracy(self) -> float:
        """Get current accuracy"""
        if self.total_trials == 0:
            return 0.0
        return self.correct_responses / self.total_trials

    def get_recent_accuracy(self, window: int = 100) -> float:
        """Get accuracy over recent trials"""
        if len(self.recent_accuracy) == 0:
            return 0.0
        return np.mean(self.recent_accuracy[-window:])


class NeuronWithDynamics:
    """
    Extended neuron with spatial dynamics for learning simulation

    Adds:
    - Migration velocity and direction
    - Chemotaxis (movement toward chemical signals)
    - Dynamic synapse formation based on proximity
    - Activity-dependent positioning
    """

    def __init__(self, base_neuron, environment):
        """
        Wrap existing neuron with dynamics

        Args:
            base_neuron: Neuron instance from neuron.py
            environment: NeuralLearningEnvironment instance
        """
        self.neuron = base_neuron
        self.env = environment

        # Movement properties
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        self.max_speed = 2.0  # μm per timestep

        # Chemotaxis sensitivity
        self.bdnf_sensitivity = 0.5  # Attraction to BDNF gradients
        self.activity_sensitivity = 0.3  # Attraction to active regions

        # Synaptogenesis properties
        self.synapse_formation_radius = 50.0  # μm
        self.synapse_formation_probability = 0.1  # per timestep
        self.axon_growth_cone_reach = 100.0  # Maximum connection distance

        # Learning properties
        self.input_value = 0.0  # Current input signal
        self.output_value = 0.0  # Current output
        self.reward_signal = 0.0  # Dopamine-like reward signal

        # Hebbian trace for learning
        self.pre_synaptic_trace = 0.0
        self.post_synaptic_trace = 0.0
        self.trace_decay = 0.9

    def update_position(self, dt: float):
        """
        Update neuron position based on migration dynamics

        Based on Hatten (2002): neuronal migration
        """
        if self.neuron.stage not in ["migration", "mature"]:
            return  # Only mobile during migration or if mature (activity-dependent)

        # Calculate movement forces
        chemotaxis_force = self._calculate_chemotaxis()
        activity_gradient_force = self._calculate_activity_gradient()
        random_motion = self._random_walk()

        # Sum forces
        total_force_x = (
            chemotaxis_force[0] * self.bdnf_sensitivity +
            activity_gradient_force[0] * self.activity_sensitivity +
            random_motion[0] * 0.1
        )
        total_force_y = (
            chemotaxis_force[1] * self.bdnf_sensitivity +
            activity_gradient_force[1] * self.activity_sensitivity +
            random_motion[1] * 0.1
        )
        total_force_z = (
            chemotaxis_force[2] * self.bdnf_sensitivity +
            activity_gradient_force[2] * self.activity_sensitivity +
            random_motion[2] * 0.1
        )

        # Update velocity (simple Euler integration)
        self.velocity_x += total_force_x * dt
        self.velocity_y += total_force_y * dt
        self.velocity_z += total_force_z * dt

        # Speed limiting
        speed = math.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
            self.velocity_z *= scale

        # Update position
        self.neuron.x += self.velocity_x * dt
        self.neuron.y += self.velocity_y * dt
        self.neuron.z += self.velocity_z * dt

        # Boundary conditions (bounce off walls)
        if self.neuron.x < 0 or self.neuron.x > self.env.width:
            self.velocity_x *= -0.5
            self.neuron.x = max(0, min(self.env.width, self.neuron.x))
        if self.neuron.y < 0 or self.neuron.y > self.env.height:
            self.velocity_y *= -0.5
            self.neuron.y = max(0, min(self.env.height, self.neuron.y))
        if self.neuron.z < 0 or self.neuron.z > self.env.depth:
            self.velocity_z *= -0.5
            self.neuron.z = max(0, min(self.env.depth, self.neuron.z))

    def _calculate_chemotaxis(self) -> Tuple[float, float, float]:
        """
        Calculate movement toward neurotrophic factor gradients
        Simple gradient ascent
        """
        gradient_samples = 5
        sample_radius = 5.0

        # Sample BDNF concentration around current position
        center_bdnf = self.env.get_bdnf_at(self.neuron.x, self.neuron.y, self.neuron.z)

        # Sample in each direction
        bdnf_x_plus = self.env.get_bdnf_at(self.neuron.x + sample_radius, self.neuron.y, self.neuron.z)
        bdnf_x_minus = self.env.get_bdnf_at(self.neuron.x - sample_radius, self.neuron.y, self.neuron.z)
        bdnf_y_plus = self.env.get_bdnf_at(self.neuron.x, self.neuron.y + sample_radius, self.neuron.z)
        bdnf_y_minus = self.env.get_bdnf_at(self.neuron.x, self.neuron.y - sample_radius, self.neuron.z)
        bdnf_z_plus = self.env.get_bdnf_at(self.neuron.x, self.neuron.y, self.neuron.z + sample_radius)
        bdnf_z_minus = self.env.get_bdnf_at(self.neuron.x, self.neuron.y, self.neuron.z - sample_radius)

        # Gradient
        grad_x = (bdnf_x_plus - bdnf_x_minus) / (2 * sample_radius)
        grad_y = (bdnf_y_plus - bdnf_y_minus) / (2 * sample_radius)
        grad_z = (bdnf_z_plus - bdnf_z_minus) / (2 * sample_radius)

        return (grad_x, grad_y, grad_z)

    def _calculate_activity_gradient(self) -> Tuple[float, float, float]:
        """
        Calculate movement toward regions of neural activity
        Activity-dependent migration
        """
        # Similar to chemotaxis but for neural activity
        sample_radius = 10.0

        center_activity = self.env.get_activity_at(self.neuron.x, self.neuron.y, self.neuron.z, radius=20.0)

        activity_x_plus = self.env.get_activity_at(self.neuron.x + sample_radius, self.neuron.y, self.neuron.z, radius=20.0)
        activity_x_minus = self.env.get_activity_at(self.neuron.x - sample_radius, self.neuron.y, self.neuron.z, radius=20.0)
        activity_y_plus = self.env.get_activity_at(self.neuron.x, self.neuron.y + sample_radius, self.neuron.z, radius=20.0)
        activity_y_minus = self.env.get_activity_at(self.neuron.x, self.neuron.y - sample_radius, self.neuron.z, radius=20.0)
        activity_z_plus = self.env.get_activity_at(self.neuron.x, self.neuron.y, self.neuron.z + sample_radius, radius=20.0)
        activity_z_minus = self.env.get_activity_at(self.neuron.x, self.neuron.y, self.neuron.z - sample_radius, radius=20.0)

        grad_x = (activity_x_plus - activity_x_minus) / (2 * sample_radius)
        grad_y = (activity_y_plus - activity_y_minus) / (2 * sample_radius)
        grad_z = (activity_z_plus - activity_z_minus) / (2 * sample_radius)

        return (grad_x, grad_y, grad_z)

    def _random_walk(self) -> Tuple[float, float, float]:
        """Random Brownian motion component"""
        return (
            random.gauss(0, 0.5),
            random.gauss(0, 0.5),
            random.gauss(0, 0.5)
        )

    def attempt_synapse_formation(self, other_neurons: List['NeuronWithDynamics'], dt: float):
        """
        Try to form new synapses with nearby neurons

        Based on:
        - Lohmann & Wong (2005): Activity-dependent synapse formation
        - Proximity-based connection probability
        - Axon growth cone dynamics
        """
        if self.neuron.stage != "mature":
            return

        # Probabilistic synapse formation
        if random.random() > self.synapse_formation_probability * dt:
            return

        # Find nearby neurons within axon reach
        candidates = []
        for other in other_neurons:
            if other.neuron.id == self.neuron.id:
                continue
            if other.neuron.stage != "mature":
                continue

            distance = math.sqrt(
                (self.neuron.x - other.neuron.x)**2 +
                (self.neuron.y - other.neuron.y)**2 +
                (self.neuron.z - other.neuron.z)**2
            )

            if distance < self.synapse_formation_radius:
                # Weight by activity correlation and distance
                activity_correlation = abs(self.neuron.firing_rate - other.neuron.firing_rate)
                connection_probability = (1.0 - distance / self.synapse_formation_radius)

                candidates.append((other, connection_probability))

        if not candidates:
            return

        # Select target based on probabilities
        neurons, probs = zip(*candidates)
        total_prob = sum(probs)
        if total_prob == 0:
            return

        normalized_probs = [p / total_prob for p in probs]
        target = random.choices(list(neurons), weights=normalized_probs)[0]

        # Check if connection already exists
        if target.neuron.id in self.neuron.synapses_out:
            return

        # Form synapse
        self.neuron.connect_to_neuron(target.neuron, initial_weight=0.3)

    def update_learning_traces(self, dt: float):
        """
        Update pre- and post-synaptic traces for learning

        Based on spike-timing-dependent plasticity (STDP)
        Markram et al. (1997)
        """
        # Decay traces
        self.pre_synaptic_trace *= self.trace_decay
        self.post_synaptic_trace *= self.trace_decay

        # Increment on spike
        if self.neuron.firing_rate > 0.1:  # Active
            self.post_synaptic_trace = 1.0

        # Pre-synaptic trace (would come from actual spikes)
        # Simplified: use current activity
        if self.input_value > 0.5:
            self.pre_synaptic_trace = 1.0

    def apply_reward_modulated_plasticity(self, reward: float, learning_rate: float = 0.01):
        """
        Reward-modulated Hebbian plasticity

        Based on Schultz et al. (1997): Dopamine and reward prediction

        Δw = learning_rate * reward * pre_trace * post_trace
        """
        self.reward_signal = reward

        # Update all incoming synapses
        for pre_id, synapse in self.neuron.synapses_in.items():
            # Weight change proportional to reward and activity correlation
            weight_change = (
                learning_rate *
                reward *
                self.pre_synaptic_trace *
                self.post_synaptic_trace
            )

            new_weight = synapse.weight + weight_change
            synapse.weight = max(0.0, min(1.0, new_weight))


class NeuralLearningEnvironment:
    """
    Extended neural environment with learning capabilities

    Adds:
    - Spatial neuron dynamics
    - Dynamic synaptogenesis
    - Learning tasks
    - Performance tracking
    """

    def __init__(self, width: float = 500.0, height: float = 500.0, depth: float = 100.0):
        self.width = width
        self.height = height
        self.depth = depth

        self.neurons: List[NeuronWithDynamics] = []
        self.time = 0.0
        self.dt = 0.1

        # Spatial fields
        self.bdnf_field = np.ones((50, 50, 10)) * 0.5  # Coarse 3D grid
        self.activity_field = np.zeros((50, 50, 10))

        # Learning
        self.tasks: List[LearningTask] = []
        self.current_task_idx = 0

        # Performance tracking
        self.learning_history = {
            'time': [],
            'accuracy': [],
            'reward': [],
            'num_synapses': [],
            'avg_firing_rate': []
        }

    def add_neuron_with_dynamics(self, base_neuron):
        """Add neuron with spatial dynamics"""
        neuron_dyn = NeuronWithDynamics(base_neuron, self)
        self.neurons.append(neuron_dyn)
        return neuron_dyn

    def get_bdnf_at(self, x: float, y: float, z: float) -> float:
        """Sample BDNF concentration at position"""
        # Map to grid coordinates
        grid_x = int((x / self.width) * (self.bdnf_field.shape[0] - 1))
        grid_y = int((y / self.height) * (self.bdnf_field.shape[1] - 1))
        grid_z = int((z / self.depth) * (self.bdnf_field.shape[2] - 1))

        grid_x = max(0, min(self.bdnf_field.shape[0] - 1, grid_x))
        grid_y = max(0, min(self.bdnf_field.shape[1] - 1, grid_y))
        grid_z = max(0, min(self.bdnf_field.shape[2] - 1, grid_z))

        return self.bdnf_field[grid_x, grid_y, grid_z]

    def get_activity_at(self, x: float, y: float, z: float, radius: float = 20.0) -> float:
        """Get average neural activity near position"""
        nearby_activity = []

        for neuron_dyn in self.neurons:
            neuron = neuron_dyn.neuron
            distance = math.sqrt(
                (neuron.x - x)**2 + (neuron.y - y)**2 + (neuron.z - z)**2
            )

            if distance < radius:
                nearby_activity.append(neuron.firing_rate)

        return np.mean(nearby_activity) if nearby_activity else 0.0

    def add_learning_task(self, task: LearningTask):
        """Add a learning task"""
        self.tasks.append(task)

    def update(self):
        """Main update loop with learning"""
        self.time += self.dt

        # Update all neurons
        for neuron_dyn in self.neurons:
            # Base neuron update
            neuron_dyn.neuron.update(self.dt, self.time)

            # Spatial dynamics
            neuron_dyn.update_position(self.dt)

            # Learning traces
            neuron_dyn.update_learning_traces(self.dt)

        # Dynamic synaptogenesis
        if int(self.time) % 5 == 0:  # Every 5 time units
            for neuron_dyn in self.neurons:
                neuron_dyn.attempt_synapse_formation(self.neurons, self.dt)

        # Run learning task
        if self.tasks:
            self._run_learning_trial()

        # Track performance
        self._update_learning_history()

    def _run_learning_trial(self):
        """Run one trial of the current learning task"""
        if not self.tasks:
            return

        task = self.tasks[self.current_task_idx]

        # Select random pattern
        pattern_idx = random.randint(0, len(task.input_patterns) - 1)
        input_pattern = task.input_patterns[pattern_idx]
        target_output = task.target_outputs[pattern_idx]
        reward = task.rewards[pattern_idx]

        # Present input to random subset of neurons (input layer)
        num_input_neurons = min(len(self.neurons), len(input_pattern))
        input_neurons = random.sample(self.neurons, num_input_neurons)

        for i, neuron_dyn in enumerate(input_neurons):
            if i < len(input_pattern):
                neuron_dyn.input_value = input_pattern[i]
                # Inject current to trigger firing
                if input_pattern[i] > 0.5:
                    neuron_dyn.neuron.membrane_potential += 20.0

        # Let network settle (simplified - instant propagation)
        # In real simulation would need multiple timesteps

        # Read output from random subset (output layer)
        output_neurons = self.neurons[-num_input_neurons:] if len(self.neurons) > num_input_neurons else self.neurons
        network_output = [n.neuron.firing_rate for n in output_neurons]

        # Calculate reward based on output
        # Simple: correct if firing rate matches target
        correct = True
        for i, target in enumerate(target_output):
            if i >= len(network_output):
                break
            actual = 1.0 if network_output[i] > 5.0 else 0.0
            if abs(actual - target) > 0.5:
                correct = False
                break

        if correct:
            actual_reward = reward
            task.correct_responses += 1
        else:
            actual_reward = -0.1  # Small punishment

        task.total_trials += 1
        task.recent_accuracy.append(1.0 if correct else 0.0)

        # Apply reward-modulated plasticity to all neurons
        for neuron_dyn in self.neurons:
            neuron_dyn.apply_reward_modulated_plasticity(actual_reward)

    def _update_learning_history(self):
        """Track learning metrics"""
        if int(self.time) % 10 == 0:  # Every 10 time units
            self.learning_history['time'].append(self.time)

            if self.tasks:
                task = self.tasks[self.current_task_idx]
                self.learning_history['accuracy'].append(task.get_recent_accuracy(50))
            else:
                self.learning_history['accuracy'].append(0.0)

            # Average reward
            avg_reward = np.mean([n.reward_signal for n in self.neurons])
            self.learning_history['reward'].append(avg_reward)

            # Total synapses
            total_synapses = sum([len(n.neuron.synapses_in) for n in self.neurons])
            self.learning_history['num_synapses'].append(total_synapses)

            # Average firing rate
            avg_firing = np.mean([n.neuron.firing_rate for n in self.neurons])
            self.learning_history['avg_firing_rate'].append(avg_firing)

    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        stats = {
            'time': self.time,
            'num_neurons': len(self.neurons),
            'total_synapses': sum([len(n.neuron.synapses_in) for n in self.neurons]),
            'avg_firing_rate': np.mean([n.neuron.firing_rate for n in self.neurons]) if self.neurons else 0.0
        }

        if self.tasks:
            task = self.tasks[self.current_task_idx]
            stats['task_name'] = task.name
            stats['accuracy'] = task.get_accuracy()
            stats['recent_accuracy'] = task.get_recent_accuracy(100)
            stats['total_trials'] = task.total_trials

        return stats

    def print_learning_stats(self):
        """Print learning statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print(f"Learning Environment (t={stats['time']:.1f})")
        print("="*60)
        print(f"Neurons: {stats['num_neurons']}")
        print(f"Total synapses: {stats['total_synapses']}")
        print(f"Avg firing rate: {stats['avg_firing_rate']:.2f} Hz")

        if 'task_name' in stats:
            print(f"\nTask: {stats['task_name']}")
            print(f"  Trials: {stats['total_trials']}")
            print(f"  Overall accuracy: {stats['accuracy']*100:.1f}%")
            print(f"  Recent accuracy (100 trials): {stats['recent_accuracy']*100:.1f}%")

        print("="*60)
