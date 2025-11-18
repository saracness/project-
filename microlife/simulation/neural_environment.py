"""
Neural Environment Simulation with Neurovascular Coupling
Based on peer-reviewed neuroscience literature

References:
-----------
1. Attwell, D., Buchan, A. M., Charpak, S., Lauritzen, M., MacVicar, B. A., & Newman, E. A. (2010).
   Glial and neuronal control of brain blood flow. Nature, 468(7321), 232-243.

2. Iadecola, C. (2017). The neurovascular unit coming of age: a journey through neurovascular
   coupling in health and disease. Neuron, 96(1), 17-42.

3. Hall, C. N., Reynell, C., Gesslein, B., Hamilton, N. B., Mishra, A., Sutherland, B. A., ... &
   Attwell, D. (2014). Capillary pericytes regulate cerebral blood flow in health and disease.
   Nature, 508(7494), 55-60.

4. Girouard, H., & Iadecola, C. (2006). Neurovascular coupling in the normal brain and in
   hypertension, stroke, and Alzheimer disease. Journal of Applied Physiology, 100(1), 328-335.

5. Zlokovic, B. V. (2011). Neurovascular pathways to neurodegeneration in Alzheimer's disease
   and other disorders. Nature Reviews Neuroscience, 12(12), 723-738.

6. Howarth, C. (2014). The contribution of astrocytes to the regulation of cerebral blood flow.
   Frontiers in Neuroscience, 8, 103.

7. Abbott, N. J., Rönnbäck, L., & Hansson, E. (2006). Astrocyte–endothelial interactions at the
   blood–brain barrier. Nature Reviews Neuroscience, 7(1), 41-53.

8. Laughlin, S. B., & Sejnowski, T. J. (2003). Communication in neuronal networks.
   Science, 301(5641), 1870-1874.
"""

import random
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from .neuron import Neuron
from .neuron_morphology import NeuronMorphology, NEURON_TYPE_TEMPLATES


@dataclass
class BloodVessel:
    """
    Capillary blood vessel supplying oxygen and glucose to neurons

    Based on Attwell et al. (2010) and Iadecola (2017):
    - Brain has ~400 miles of capillaries per cubic inch
    - Average distance neuron-to-capillary: 8-25 μm
    - Neurovascular coupling: neural activity → blood flow increase (1-2 seconds delay)
    """
    x: float
    y: float
    z: float
    diameter: float = 5.0           # Capillary diameter (μm), typically 3-7 μm
    flow_rate: float = 1.0          # Blood flow rate (relative)
    oxygen_supply: float = 1.0      # Oxygen delivery capacity
    glucose_supply: float = 1.0     # Glucose delivery capacity

    # Neurovascular coupling state
    dilation: float = 1.0           # Vessel dilation (1.0 = baseline, >1.0 = dilated)
    nearby_neural_activity: float = 0.0  # Local neural activity level

    def update(self, dt: float, neural_activity: float):
        """
        Update blood flow based on local neural activity

        Attwell et al. (2010): functional hyperemia
        - Neural activity → astrocyte signaling → vessel dilation → increased blood flow
        - Delay: 1-2 seconds
        - Magnitude: 10-30% increase in flow
        """
        self.nearby_neural_activity = neural_activity

        # Neurovascular coupling: activity drives dilation
        target_dilation = 1.0 + (neural_activity * 0.3)  # Up to 30% increase

        # Gradual dilation/constriction (hemodynamic response takes 1-2 seconds)
        dilation_rate = 0.1  # Rate of change
        if self.dilation < target_dilation:
            self.dilation += dilation_rate * dt
        else:
            self.dilation -= dilation_rate * dt

        self.dilation = max(0.7, min(1.3, self.dilation))  # Physiological limits

        # Flow rate increases with dilation (Poiseuille's law: flow ∝ r^4)
        self.flow_rate = self.dilation ** 4

        # Supply capacity scales with flow
        self.oxygen_supply = self.flow_rate
        self.glucose_supply = self.flow_rate


@dataclass
class Astrocyte:
    """
    Astrocyte (glial cell) supporting neurons

    Based on Howarth (2014) and Abbott et al. (2006):
    - Astrocytes ensheath blood vessels and synapses
    - Regulate neurovascular coupling
    - Provide metabolic support (astrocyte-neuron lactate shuttle)
    - Maintain extracellular environment
    """
    x: float
    y: float
    z: float
    domain_radius: float = 50.0     # Astrocyte territory (30-60 μm typical)

    # Metabolic support
    glycogen_stores: float = 100.0   # Energy reserves
    lactate_production: float = 0.0  # Lactate for neurons

    # Neurotransmitter uptake (glutamate recycling)
    glutamate_uptake_rate: float = 1.0

    # Vasomotor signaling
    ca2_level: float = 0.5           # Calcium signaling
    vasodilator_release: float = 0.0  # PGE2, EETs, etc.

    def update(self, dt: float, local_neural_activity: float):
        """
        Update astrocyte state based on local neural activity

        Howarth (2014): astrocytes sense synaptic glutamate → Ca²⁺ increase
        → vasodilator release → blood vessel dilation
        """
        # Neural activity → glutamate release → astrocyte Ca²⁺
        if local_neural_activity > 0.5:
            self.ca2_level = min(1.0, self.ca2_level + 0.1 * dt)
        else:
            self.ca2_level = max(0.3, self.ca2_level - 0.05 * dt)

        # Ca²⁺ → vasodilator release
        self.vasodilator_release = self.ca2_level * 0.5

        # Glycogenolysis: break down glycogen to lactate under high activity
        if local_neural_activity > 0.7:
            if self.glycogen_stores > 0:
                lactate_production = min(5.0, self.glycogen_stores * 0.1 * dt)
                self.lactate_production = lactate_production
                self.glycogen_stores -= lactate_production
        else:
            # Replenish glycogen stores during low activity
            self.glycogen_stores = min(100.0, self.glycogen_stores + 2.0 * dt)
            self.lactate_production = 0.0


class NeuralEnvironment:
    """
    Complete neural tissue environment simulation

    Components:
    1. Neurons with life cycles
    2. Blood vessels with neurovascular coupling
    3. Astrocytes providing metabolic support
    4. Extracellular space with diffusion

    Based on the "Neurovascular Unit" concept (Iadecola, 2017):
    - Neurons, astrocytes, vessels form integrated functional unit
    - Neural activity → metabolic demand → blood flow increase
    - Coupling ensures adequate oxygen/glucose supply
    """

    def __init__(self, width: float = 500.0, height: float = 500.0, depth: float = 100.0):
        """
        Initialize neural tissue environment

        Args:
            width, height, depth: Tissue dimensions (μm)
        """
        self.width = width
        self.height = height
        self.depth = depth

        self.neurons: List[Neuron] = []
        self.blood_vessels: List[BloodVessel] = []
        self.astrocytes: List[Astrocyte] = []

        self.time = 0.0
        self.dt = 0.1  # Timestep (seconds)

        # Environmental parameters
        self.ambient_oxygen = 1.0
        self.ambient_glucose = 1.0
        self.temperature = 37.0  # °C (mammalian body temperature)

        # Neurotrophic factors (produced by glia and blood)
        self.bdnf_concentration = 0.5
        self.ngf_concentration = 0.5

        # Statistics
        self.total_neurons_born = 0
        self.total_neurons_died = 0
        self.total_synapses_formed = 0
        self.total_synapses_pruned = 0

    def initialize_vasculature(self, vessel_density: float = 0.01):
        """
        Create capillary network

        Attwell et al. (2010): brain has dense vascular network
        - 1 capillary per ~50 μm spacing
        - Ensures no neuron is >25 μm from oxygen supply

        Args:
            vessel_density: Vessels per cubic μm
        """
        volume = self.width * self.height * self.depth
        num_vessels = int(volume * vessel_density)

        print(f"Initializing vasculature: {num_vessels} capillaries")

        for _ in range(num_vessels):
            vessel = BloodVessel(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                z=random.uniform(0, self.depth),
                diameter=random.uniform(4.0, 7.0)
            )
            self.blood_vessels.append(vessel)

    def initialize_astrocytes(self, astrocyte_density: float = 0.0001):
        """
        Create astrocyte network

        Howarth (2014): astrocytes tile the brain
        - Minimal overlap of domains
        - 1 astrocyte per ~50 μm radius sphere

        Args:
            astrocyte_density: Astrocytes per cubic μm
        """
        volume = self.width * self.height * self.depth
        num_astrocytes = int(volume * astrocyte_density)

        print(f"Initializing astrocytes: {num_astrocytes} cells")

        for _ in range(num_astrocytes):
            astrocyte = Astrocyte(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                z=random.uniform(0, self.depth)
            )
            self.astrocytes.append(astrocyte)

    def add_neuron(self, neuron: Neuron):
        """Add neuron to environment"""
        self.neurons.append(neuron)
        self.total_neurons_born += 1

        # Calculate distance to nearest blood vessel
        neuron.blood_vessel_distance = self._distance_to_nearest_vessel(
            neuron.x, neuron.y, neuron.z
        )

    def create_neuron_at(self, x: float, y: float, z: float = 0.0,
                        neuron_type: Optional[str] = None) -> Neuron:
        """
        Neurogenesis: create new neuron

        Args:
            x, y, z: Position
            neuron_type: Type template (pyramidal, granule, etc.)

        Returns:
            New neuron
        """
        neuron = Neuron(x, y, z, neuron_type=neuron_type)
        self.add_neuron(neuron)
        return neuron

    def create_random_population(self, num_neurons: int,
                                neuron_type_distribution: Optional[Dict[str, float]] = None):
        """
        Create initial neuronal population

        Args:
            num_neurons: Number of neurons to create
            neuron_type_distribution: Dict of {type: probability}
        """
        if neuron_type_distribution is None:
            # Default: 80% excitatory (pyramidal), 20% inhibitory
            neuron_type_distribution = {
                "pyramidal": 0.7,
                "granule": 0.1,
                "interneuron": 0.2
            }

        print(f"Creating population of {num_neurons} neurons")

        types = list(neuron_type_distribution.keys())
        probs = list(neuron_type_distribution.values())

        for _ in range(num_neurons):
            neuron_type = random.choices(types, weights=probs)[0]

            neuron = self.create_neuron_at(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                z=random.uniform(0, self.depth),
                neuron_type=neuron_type
            )

            # Mature neurons start in mature stage
            neuron.stage = "mature"

    def form_random_synapses(self, connection_probability: float = 0.05):
        """
        Create synaptic connectivity

        Args:
            connection_probability: Probability of connection between nearby neurons
        """
        print(f"Forming synapses with p={connection_probability}")

        for i, pre_neuron in enumerate(self.neurons):
            if pre_neuron.stage != "mature":
                continue

            for j, post_neuron in enumerate(self.neurons):
                if i == j or post_neuron.stage != "mature":
                    continue

                # Distance-dependent connectivity
                distance = self._distance_3d(
                    pre_neuron.x, pre_neuron.y, pre_neuron.z,
                    post_neuron.x, post_neuron.y, post_neuron.z
                )

                # Probability decreases with distance
                # Typical axon reach: 100-1000 μm for local connections
                max_distance = 200.0
                if distance < max_distance:
                    prob = connection_probability * (1.0 - distance / max_distance)
                    if random.random() < prob:
                        pre_neuron.connect_to_neuron(post_neuron)
                        self.total_synapses_formed += 1

        avg_synapses = np.mean([len(n.synapses_in) for n in self.neurons])
        print(f"Average synapses per neuron: {avg_synapses:.1f}")

    def update(self):
        """
        Update all components of neural environment for one timestep
        """
        self.time += self.dt

        # 1. Update blood vessels with neurovascular coupling
        self._update_vasculature()

        # 2. Update astrocytes
        self._update_astrocytes()

        # 3. Supply metabolites to neurons
        self._supply_metabolites_to_neurons()

        # 4. Update all neurons
        self._update_neurons()

        # 5. Remove dead neurons
        self._remove_dead_neurons()

    def _update_vasculature(self):
        """
        Update blood vessels based on local neural activity
        Attwell et al. (2010): neurovascular coupling
        """
        for vessel in self.blood_vessels:
            # Calculate local neural activity
            local_activity = self._calculate_local_neural_activity(
                vessel.x, vessel.y, vessel.z, radius=50.0
            )

            vessel.update(self.dt, local_activity)

    def _update_astrocytes(self):
        """Update astrocytes based on local neural activity"""
        for astrocyte in self.astrocytes:
            local_activity = self._calculate_local_neural_activity(
                astrocyte.x, astrocyte.y, astrocyte.z, radius=astrocyte.domain_radius
            )

            astrocyte.update(self.dt, local_activity)

    def _supply_metabolites_to_neurons(self):
        """
        Supply oxygen and glucose to neurons from blood vessels
        Based on distance and blood flow
        """
        for neuron in self.neurons:
            if not neuron.alive:
                continue

            # Find nearest blood vessel
            nearest_vessel, distance = self._find_nearest_vessel(neuron.x, neuron.y, neuron.z)

            if nearest_vessel and distance < 50.0:  # Within supply range
                # Supply decreases with distance (diffusion)
                diffusion_factor = 1.0 - (distance / 50.0)

                oxygen_supply = nearest_vessel.oxygen_supply * diffusion_factor * 0.2
                glucose_supply = nearest_vessel.glucose_supply * diffusion_factor * 0.2

                neuron.receive_metabolites(glucose=glucose_supply, oxygen=oxygen_supply)

            # Astrocyte lactate shuttle
            nearest_astrocyte = self._find_nearest_astrocyte(neuron.x, neuron.y, neuron.z)
            if nearest_astrocyte:
                # Lactate is alternative fuel for neurons
                lactate_uptake = nearest_astrocyte.lactate_production * 0.1
                neuron.receive_metabolites(glucose=lactate_uptake * 0.5)

            # Neurotrophic factors
            neuron.receive_neurotrophic_factors(
                bdnf=self.bdnf_concentration * 0.01,
                ngf=self.ngf_concentration * 0.01
            )

    def _update_neurons(self):
        """Update all neurons"""
        for neuron in self.neurons:
            neuron.update(self.dt, self.time)

    def _remove_dead_neurons(self):
        """Remove dead neurons from simulation"""
        initial_count = len(self.neurons)
        self.neurons = [n for n in self.neurons if n.alive]
        removed = initial_count - len(self.neurons)

        if removed > 0:
            self.total_neurons_died += removed

    def _calculate_local_neural_activity(self, x: float, y: float, z: float,
                                        radius: float = 50.0) -> float:
        """
        Calculate average neural activity within radius

        Args:
            x, y, z: Center position
            radius: Search radius

        Returns:
            Average firing rate of nearby neurons
        """
        nearby_activity = []

        for neuron in self.neurons:
            if not neuron.alive or neuron.stage != "mature":
                continue

            distance = self._distance_3d(x, y, z, neuron.x, neuron.y, neuron.z)

            if distance < radius:
                nearby_activity.append(neuron.firing_rate)

        return np.mean(nearby_activity) if nearby_activity else 0.0

    def _find_nearest_vessel(self, x: float, y: float, z: float) -> Tuple[Optional[BloodVessel], float]:
        """Find nearest blood vessel to position"""
        if not self.blood_vessels:
            return None, float('inf')

        min_distance = float('inf')
        nearest = None

        for vessel in self.blood_vessels:
            distance = self._distance_3d(x, y, z, vessel.x, vessel.y, vessel.z)
            if distance < min_distance:
                min_distance = distance
                nearest = vessel

        return nearest, min_distance

    def _find_nearest_astrocyte(self, x: float, y: float, z: float) -> Optional[Astrocyte]:
        """Find nearest astrocyte to position"""
        if not self.astrocytes:
            return None

        min_distance = float('inf')
        nearest = None

        for astrocyte in self.astrocytes:
            distance = self._distance_3d(x, y, z, astrocyte.x, astrocyte.y, astrocyte.z)
            if distance < min_distance:
                min_distance = distance
                nearest = astrocyte

        return nearest

    def _distance_to_nearest_vessel(self, x: float, y: float, z: float) -> float:
        """Calculate distance to nearest blood vessel"""
        _, distance = self._find_nearest_vessel(x, y, z)
        return distance

    @staticmethod
    def _distance_3d(x1: float, y1: float, z1: float,
                    x2: float, y2: float, z2: float) -> float:
        """Calculate 3D Euclidean distance"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    def get_statistics(self) -> Dict:
        """Get simulation statistics"""
        alive_neurons = [n for n in self.neurons if n.alive]

        if not alive_neurons:
            return {
                "time": self.time,
                "total_neurons": 0,
                "alive_neurons": 0,
                "dead_neurons": self.total_neurons_died
            }

        stages = {}
        for neuron in alive_neurons:
            stages[neuron.stage] = stages.get(neuron.stage, 0) + 1

        avg_energy = np.mean([n.energy for n in alive_neurons])
        avg_firing_rate = np.mean([n.firing_rate for n in alive_neurons if n.stage == "mature"])
        avg_synapses = np.mean([len(n.synapses_in) for n in alive_neurons])

        return {
            "time": self.time,
            "total_neurons_born": self.total_neurons_born,
            "alive_neurons": len(alive_neurons),
            "dead_neurons": self.total_neurons_died,
            "stages": stages,
            "avg_energy": avg_energy,
            "avg_firing_rate": avg_firing_rate,
            "avg_synapses": avg_synapses,
            "blood_vessels": len(self.blood_vessels),
            "astrocytes": len(self.astrocytes),
            "total_synapses_formed": self.total_synapses_formed
        }

    def print_statistics(self):
        """Print current statistics"""
        stats = self.get_statistics()

        print("\n" + "="*60)
        print(f"Neural Environment Statistics (t={stats['time']:.1f})")
        print("="*60)
        print(f"Neurons: {stats['alive_neurons']} alive, {stats['dead_neurons']} died")
        print(f"Life stages: {stats.get('stages', {})}")
        print(f"Average energy: {stats.get('avg_energy', 0):.1f}")
        print(f"Average firing rate: {stats.get('avg_firing_rate', 0):.2f} Hz")
        print(f"Average synapses/neuron: {stats.get('avg_synapses', 0):.1f}")
        print(f"Blood vessels: {stats['blood_vessels']}")
        print(f"Astrocytes: {stats['astrocytes']}")
        print(f"Total synapses formed: {stats['total_synapses_formed']}")
        print("="*60 + "\n")
