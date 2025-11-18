"""
Evolutionary Brains:
1. Genetic Algorithm (GA) - Gene evolution
2. NEAT (NeuroEvolution of Augmenting Topologies)
3. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

These are used extensively in biology research and artificial life!
"""
import numpy as np
import random
from .brain_base import Brain


class GeneticAlgorithmBrain(Brain):
    """
    Genetic Algorithm brain.
    Genome encodes behavior rules that evolve over generations.

    This mimics natural evolution - used by biologists!
    """

    def __init__(self, genome_size=20, mutation_rate=0.1):
        super().__init__(brain_type="Genetic-Algorithm")
        self.genome_size = genome_size
        self.mutation_rate = mutation_rate

        # Genome: array of behavioral genes
        # Each gene controls different behavior weights
        self.genome = np.random.randn(genome_size)

        # Fitness tracking
        self.fitness = 0.0
        self.generation = 0

    def decide_action(self, state):
        """Decide action based on genome."""
        self.decision_count += 1

        # Extract features
        energy = state.get('energy', 100) / 200.0
        food_dist = min(state.get('nearest_food_distance', 500) / 500.0, 1.0)
        food_angle = state.get('nearest_food_angle', 0)
        in_temp = 1.0 if state.get('in_temperature_zone', False) else 0.0

        # Genes control behavior rules
        # Gene 0-7: Direction preferences
        # Gene 8-11: Energy-based behavior
        # Gene 12-15: Food-seeking behavior
        # Gene 16-19: Risk-taking behavior

        # Calculate movement based on genes
        if food_dist < 0.5:  # Food nearby
            # Use food-seeking genes (12-15)
            dx = np.cos(food_angle) * abs(self.genome[12])
            dy = np.sin(food_angle) * abs(self.genome[13])
        elif energy < 0.3:  # Low energy - desperate
            # Use emergency genes (8-11)
            angle = food_angle if food_angle is not None else random.random() * 2 * np.pi
            dx = np.cos(angle) * abs(self.genome[8])
            dy = np.sin(angle) * abs(self.genome[9])
        else:  # Normal exploration
            # Use exploration genes (0-7)
            direction_gene = int(abs(self.genome[0]) * 8) % 8
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            angle = angles[direction_gene]
            dx = np.cos(angle) * abs(self.genome[direction_gene + 1])
            dy = np.sin(angle) * abs(self.genome[direction_gene + 1])

        # Risk-taking: avoid temperature zones?
        if in_temp > 0 and abs(self.genome[16]) > 0.5:
            # Risk-averse gene: flee temperature
            dx = -dx * 0.5
            dy = -dy * 0.5

        # Normalize
        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        # Reproduction decision based on gene
        should_reproduce = (energy > 0.75 and abs(self.genome[17]) > 0.3)

        return {
            'move_direction': (dx, dy),
            'should_reproduce': should_reproduce,
            'speed_multiplier': min(abs(self.genome[18]), 2.0)
        }

    def learn(self, state, action, reward, next_state, done):
        """Update fitness (evolution happens at population level)."""
        self.fitness += reward
        self.total_reward += reward

    def mutate(self):
        """Mutate genome."""
        for i in range(self.genome_size):
            if random.random() < self.mutation_rate:
                self.genome[i] += np.random.randn() * 0.5

    def crossover(self, other_brain):
        """Crossover with another brain to create offspring."""
        child = GeneticAlgorithmBrain(self.genome_size, self.mutation_rate)

        # Single-point crossover
        crossover_point = random.randint(0, self.genome_size)
        child.genome[:crossover_point] = self.genome[:crossover_point]
        child.genome[crossover_point:] = other_brain.genome[crossover_point:]

        # Small mutation
        child.mutate()
        child.generation = max(self.generation, other_brain.generation) + 1

        return child


class NEATBrain(Brain):
    """
    NEAT: NeuroEvolution of Augmenting Topologies
    Evolves both weights AND network structure!

    Famous for solving complex tasks. Used in AI research.
    Simplified version here.
    """

    def __init__(self, input_size=7, output_size=9):
        super().__init__(brain_type="NEAT")
        self.input_size = input_size
        self.output_size = output_size

        # Network structure
        self.nodes = {}  # node_id -> (type, value)
        self.connections = {}  # (from_id, to_id) -> weight

        # Innovation numbers for crossover
        self.innovation_number = 0

        # Initialize minimal network
        self._initialize_network()

        self.fitness = 0.0

    def _initialize_network(self):
        """Start with minimal topology: inputs -> outputs."""
        # Input nodes
        for i in range(self.input_size):
            self.nodes[i] = ('input', 0.0)

        # Output nodes
        for i in range(self.output_size):
            node_id = self.input_size + i
            self.nodes[node_id] = ('output', 0.0)

        # Full connections with random weights
        for inp in range(self.input_size):
            for out in range(self.output_size):
                out_id = self.input_size + out
                self.connections[(inp, out_id)] = np.random.randn() * 0.5
                self.innovation_number += 1

    def _forward(self, inputs):
        """Forward pass through evolved network."""
        # Set input values
        for i, val in enumerate(inputs):
            if i in self.nodes:
                self.nodes[i] = ('input', val)

        # Calculate output values
        outputs = []
        for out_idx in range(self.output_size):
            out_id = self.input_size + out_idx
            total = 0.0

            # Sum weighted inputs
            for inp_id in range(self.input_size):
                if (inp_id, out_id) in self.connections:
                    weight = self.connections[(inp_id, out_id)]
                    inp_val = self.nodes[inp_id][1]
                    total += weight * inp_val

            # Activation (tanh)
            activated = np.tanh(total)
            self.nodes[out_id] = ('output', activated)
            outputs.append(activated)

        return np.array(outputs)

    def decide_action(self, state):
        """Decide using evolved network."""
        self.decision_count += 1

        # Get state vector
        state_vec = self.get_state_vector(state)

        # Forward pass
        outputs = self._forward(state_vec)

        # Best output is chosen action
        action_idx = np.argmax(outputs)

        # Convert to direction
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4, 0]
        angle = angles[action_idx]
        dx = np.cos(angle)
        dy = np.sin(angle)

        return {
            'move_direction': (dx, dy) if action_idx < 8 else (0, 0),
            'should_reproduce': state.get('energy', 0) > 150,
            'speed_multiplier': 1.0
        }

    def learn(self, state, action, reward, next_state, done):
        """Update fitness."""
        self.fitness += reward
        self.total_reward += reward

    def mutate(self, add_node_prob=0.03, add_conn_prob=0.05, weight_mut_prob=0.8):
        """Mutate network structure."""
        # Mutate weights
        for conn in self.connections:
            if random.random() < weight_mut_prob:
                self.connections[conn] += np.random.randn() * 0.3

        # Add new connection
        if random.random() < add_conn_prob:
            # Pick random nodes
            from_nodes = [n for n in self.nodes if self.nodes[n][0] != 'output']
            to_nodes = [n for n in self.nodes if self.nodes[n][0] != 'input']

            if from_nodes and to_nodes:
                from_id = random.choice(from_nodes)
                to_id = random.choice(to_nodes)

                if (from_id, to_id) not in self.connections:
                    self.connections[(from_id, to_id)] = np.random.randn() * 0.5
                    self.innovation_number += 1

        # Add new node (split connection)
        if random.random() < add_node_prob and self.connections:
            # Pick random connection
            conn = random.choice(list(self.connections.keys()))
            old_weight = self.connections[conn]

            # Create new node
            new_id = len(self.nodes)
            self.nodes[new_id] = ('hidden', 0.0)

            # Split connection
            from_id, to_id = conn
            del self.connections[conn]

            self.connections[(from_id, new_id)] = 1.0
            self.connections[(new_id, to_id)] = old_weight
            self.innovation_number += 2


class CMAESBrain(Brain):
    """
    CMA-ES: Covariance Matrix Adaptation Evolution Strategy
    State-of-the-art evolution strategy used in biology/AI research.

    Adapts the search distribution based on successful mutations.
    """

    def __init__(self, param_size=20):
        super().__init__(brain_type="CMA-ES")
        self.param_size = param_size

        # Parameters (like genome but with adaptive distribution)
        self.params = np.random.randn(param_size)

        # CMA-ES specific
        self.mean = self.params.copy()
        self.sigma = 1.0  # Step size
        self.cov_matrix = np.eye(param_size)  # Covariance

        self.fitness = 0.0

    def decide_action(self, state):
        """Decide based on evolved parameters."""
        self.decision_count += 1

        # Similar to GA but uses parameters
        energy = state.get('energy', 100) / 200.0
        food_dist = min(state.get('nearest_food_distance', 500) / 500.0, 1.0)
        food_angle = state.get('nearest_food_angle', 0)

        # Use parameters to decide
        if food_dist < 0.5:
            dx = np.cos(food_angle) * abs(self.params[0])
            dy = np.sin(food_angle) * abs(self.params[1])
        else:
            direction_idx = int(abs(self.params[2]) * 8) % 8
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
            angle = angles[direction_idx]
            dx = np.cos(angle) * abs(self.params[3])
            dy = np.sin(angle) * abs(self.params[4])

        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        return {
            'move_direction': (dx, dy),
            'should_reproduce': energy > 0.75 and abs(self.params[5]) > 0.3,
            'speed_multiplier': min(abs(self.params[6]), 2.0)
        }

    def learn(self, state, action, reward, next_state, done):
        """Update fitness."""
        self.fitness += reward
        self.total_reward += reward

    def update_distribution(self, successful_params, fitness_values):
        """Update search distribution based on successful individuals."""
        # Sort by fitness
        sorted_indices = np.argsort(fitness_values)[::-1]
        top_half = sorted_indices[:len(sorted_indices)//2]

        # Update mean toward successful parameters
        self.mean = np.mean([successful_params[i] for i in top_half], axis=0)

        # Update covariance (simplified)
        deviations = [successful_params[i] - self.mean for i in top_half]
        self.cov_matrix = np.cov(np.array(deviations).T)

        # Update sigma (step size adaptation)
        self.sigma *= 0.95

        # Sample new parameters
        self.params = self.mean + self.sigma * np.random.multivariate_normal(
            np.zeros(self.param_size), self.cov_matrix
        )
