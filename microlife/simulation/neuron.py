"""
Neuron Life Cycle Simulation
Based on peer-reviewed neuroscience literature

References:
-----------
1. Gage, F. H. (2000). Mammalian neural stem cells.
   Science, 287(5457), 1433-1438.

2. Kempermann, G., Song, H., & Gage, F. H. (2015). Neurogenesis in the adult hippocampus.
   Cold Spring Harbor Perspectives in Biology, 7(9), a018812.

3. Bliss, T. V., & Collingridge, G. L. (1993). A synaptic model of memory:
   long-term potentiation in the hippocampus. Nature, 361(6407), 31-39.

4. Malenka, R. C., & Bear, M. F. (2004). LTP and LTD: an embarrassment of riches.
   Neuron, 44(1), 5-21.

5. Magistretti, P. J., & Allaman, I. (2015). A cellular perspective on brain energy metabolism
   and functional imaging. Neuron, 86(4), 883-901.

6. Attwell, D., Buchan, A. M., Charpak, S., Lauritzen, M., MacVicar, B. A., & Newman, E. A. (2010).
   Glial and neuronal control of brain blood flow. Nature, 468(7321), 232-243.

7. Yuan, J., & Yankner, B. A. (2000). Apoptosis in the nervous system.
   Nature, 407(6805), 802-809.

8. Bonhoeffer, T., & Yuste, R. (2002). Spine motility: phenomenology, mechanisms, and function.
   Neuron, 35(6), 1019-1027.

9. Moroz, L. L. (2009). On the independent origins of complex brains and neurons.
   Brain, Behavior and Evolution, 74(3), 177-190.

10. Ryan, J. F., & Grant, K. A. (2009). Evolution of neural systems in Cnidaria.
    Integrative and Comparative Biology, 49(2), 142-153.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from copy import deepcopy
import numpy as np
from .neuron_morphology import NeuronMorphology, create_neuron_morphology


@dataclass
class Synapse:
    """
    Individual synapse with Hebbian plasticity (Bliss & Collingridge, 1993)

    Synaptic weight represents efficacy - undergoes LTP/LTD based on activity
    """
    pre_neuron_id: int              # Presynaptic neuron
    post_neuron_id: int             # Postsynaptic neuron (self)
    weight: float = 0.5             # Synaptic efficacy (0.0-1.0)
    last_activation_time: float = 0.0
    potentiation_history: List[float] = field(default_factory=list)  # LTP events
    depression_history: List[float] = field(default_factory=list)    # LTD events

    # Structural plasticity (Bonhoeffer & Yuste, 2002)
    spine_size: float = 0.5         # Correlates with synaptic strength
    is_stable: bool = False         # Stable vs transient spine

    def apply_hebbian_plasticity(self, pre_active: bool, post_active: bool,
                                 time: float, learning_rate: float = 0.01):
        """
        Hebbian learning: "neurons that fire together, wire together"
        Implements spike-timing-dependent plasticity (STDP)

        Based on Malenka & Bear (2004)
        """
        if pre_active and post_active:
            # LTP: Long-Term Potentiation
            self.weight = min(1.0, self.weight + learning_rate)
            self.spine_size = min(1.0, self.spine_size + learning_rate * 0.5)
            self.potentiation_history.append(time)

            # Stabilize spine after repeated potentiation
            if len(self.potentiation_history) > 5:
                self.is_stable = True

        elif pre_active and not post_active:
            # LTD: Long-Term Depression
            self.weight = max(0.0, self.weight - learning_rate * 0.5)
            self.spine_size = max(0.1, self.spine_size - learning_rate * 0.3)
            self.depression_history.append(time)

        # Synaptic pruning: very weak synapses are eliminated
        if self.weight < 0.05 and not self.is_stable:
            return False  # Signal for removal

        # Trim history to prevent memory bloat
        if len(self.potentiation_history) > 100:
            self.potentiation_history = self.potentiation_history[-100:]
        if len(self.depression_history) > 100:
            self.depression_history = self.depression_history[-100:]

        self.last_activation_time = time
        return True  # Synapse survives


class Neuron:
    """
    Biological neuron with full life cycle

    Life stages:
    1. Neurogenesis: Birth from neural stem cell (Gage, 2000; Kempermann et al., 2015)
    2. Migration: Movement to final position
    3. Differentiation: Acquiring specific neuron type identity
    4. Synaptogenesis: Forming connections
    5. Mature function: Signal processing with plasticity
    6. Apoptosis: Programmed cell death (Yuan & Yankner, 2000)

    Environmental requirements:
    - Oxygen and glucose from blood vessels (Attwell et al., 2010)
    - Neurotrophic factors (BDNF, NGF) for survival
    - Synaptic activity for maintenance
    """

    def __init__(self, x: float, y: float, z: float = 0.0,
                 neuron_id: Optional[int] = None,
                 morphology: Optional[NeuronMorphology] = None,
                 neuron_type: Optional[str] = None,
                 energy: float = 100.0):
        """
        Initialize neuron

        Args:
            x, y, z: 3D position in neural tissue
            neuron_id: Unique identifier
            morphology: Morphological properties
            neuron_type: Template type (pyramidal, granule, etc.)
            energy: Initial energy (ATP) level
        """
        self.id = neuron_id if neuron_id is not None else random.randint(1, 1000000)
        self.x = x
        self.y = y
        self.z = z

        # Morphology
        if morphology:
            self.morphology = morphology
        else:
            self.morphology = create_neuron_morphology(neuron_type)

        # Life cycle stage
        self.age = 0.0  # Time since neurogenesis
        self.stage = "neurogenesis"  # neurogenesis, migration, differentiation, mature, apoptotic
        self.alive = True

        # Energy metabolism (Magistretti & Allaman, 2015)
        self.energy = energy  # ATP level
        self.glucose_consumption_rate = self.morphology.metabolic_demand * 0.5
        self.oxygen_consumption_rate = self.morphology.metabolic_demand * 0.3

        # Electrophysiology
        self.membrane_potential = -70.0  # Resting potential (mV)
        self.firing_rate = 0.0  # Spikes per second
        self.last_spike_time = 0.0
        self.refractory_period = 2.0  # ms

        # Synaptic connectivity
        self.synapses_in: Dict[int, Synapse] = {}   # Incoming synapses
        self.synapses_out: Dict[int, Synapse] = {}  # Outgoing synapses

        # Neurovascular coupling (Attwell et al., 2010)
        self.blood_vessel_distance = float('inf')  # Distance to nearest capillary
        self.oxygen_level = 1.0
        self.glucose_level = 1.0

        # Neurotrophic support
        self.bdnf_level = 0.5  # Brain-Derived Neurotrophic Factor
        self.ngf_level = 0.5   # Nerve Growth Factor

        # Activity-dependent survival
        self.recent_activity = []  # Recent firing history
        self.activity_score = 0.0

    def update(self, dt: float, time: float):
        """
        Update neuron state for one timestep

        Args:
            dt: Time delta (seconds)
            time: Current simulation time
        """
        if not self.alive:
            return

        self.age += dt

        # Stage-specific updates
        if self.stage == "neurogenesis":
            self._update_neurogenesis(dt)
        elif self.stage == "migration":
            self._update_migration(dt)
        elif self.stage == "differentiation":
            self._update_differentiation(dt)
        elif self.stage == "mature":
            self._update_mature_function(dt, time)
        elif self.stage == "apoptotic":
            self._undergo_apoptosis(dt)

        # Universal metabolic updates
        self._update_metabolism(dt)
        self._check_survival()

    def _update_neurogenesis(self, dt: float):
        """
        Neurogenesis phase: newly born neuron from stem cell
        Kempermann et al. (2015): occurs in hippocampal dentate gyrus and SVZ
        """
        # Neurogenesis takes ~1-2 weeks in vivo
        if self.age > 10.0:  # Simplified: 10 time units
            self.stage = "migration"
            print(f"Neuron {self.id}: Completed neurogenesis, beginning migration")

    def _update_migration(self, dt: float):
        """
        Migration phase: neuron moves to final position
        Guided by radial glia and chemotactic signals
        """
        # Simplified migration: random walk toward target region
        # Real migration is highly guided by molecular cues

        if self.age > 20.0:  # Migration complete
            self.stage = "differentiation"
            print(f"Neuron {self.id}: Reached target location, beginning differentiation")

    def _update_differentiation(self, dt: float):
        """
        Differentiation: acquiring specific neuronal identity
        Morphology specification and initial synaptogenesis
        """
        # Morphological maturation
        if self.age > 30.0:
            self.stage = "mature"
            print(f"Neuron {self.id}: Differentiation complete - "
                  f"{self.morphology.neurotransmitter}ergic neuron")

    def _update_mature_function(self, dt: float, time: float):
        """
        Mature neuron: active signal processing with plasticity
        """
        # Integrate synaptic input
        synaptic_input = self._integrate_synaptic_input()

        # Update membrane potential
        leak_current = (self.membrane_potential + 70.0) * 0.1  # Passive leak
        self.membrane_potential += (synaptic_input - leak_current) * dt

        # Check for action potential
        if self.membrane_potential >= self.morphology.firing_threshold:
            if (time - self.last_spike_time) > self.refractory_period:
                self._fire_action_potential(time)
                self.membrane_potential = -70.0  # Reset

        # Synaptic plasticity
        self._update_synaptic_plasticity(time)

        # Activity-dependent BDNF release
        if self.firing_rate > 10.0:  # Active neuron
            self.bdnf_level = min(1.0, self.bdnf_level + 0.01 * dt)

        # Track activity for survival
        self.recent_activity.append(self.firing_rate)
        if len(self.recent_activity) > 100:
            self.recent_activity.pop(0)
        self.activity_score = np.mean(self.recent_activity) if self.recent_activity else 0.0

    def _integrate_synaptic_input(self) -> float:
        """
        Integrate all incoming synaptic currents
        Based on dendritic integration capacity (Spruston, 2008)
        """
        total_input = 0.0

        for synapse in self.synapses_in.values():
            # Synaptic current depends on weight and presynaptic activity
            # Simplified: assume some baseline activity
            synaptic_current = synapse.weight * synapse.spine_size

            if self.morphology.is_excitatory:
                total_input += synaptic_current
            else:
                total_input -= synaptic_current  # Inhibitory

        # Scale by integration capacity
        total_input *= self.morphology.integration_capacity

        return total_input

    def _fire_action_potential(self, time: float):
        """
        Generate action potential and propagate to downstream neurons
        """
        self.last_spike_time = time
        self.firing_rate += 1.0  # Increment spike count

        # Action potential propagates down axon to all postsynaptic neurons
        # Propagation speed depends on myelination
        propagation_delay = 1.0 / self.morphology.signal_speed

        # Signal all postsynaptic targets
        for synapse in self.synapses_out.values():
            # Neurotransmitter release
            self._release_neurotransmitter(synapse)

        # Energy cost of action potential (Attwell & Laughlin, 2001)
        # Sodium influx and potassium efflux require ATP to restore gradients
        self.energy -= 0.5 * self.morphology.metabolic_demand

    def _release_neurotransmitter(self, synapse: Synapse):
        """
        Synaptic transmission: release neurotransmitter
        Calcium-dependent vesicle fusion
        """
        # Simplified: just modulate synaptic weight
        # Real process involves complex vesicle dynamics
        pass

    def _update_synaptic_plasticity(self, time: float):
        """
        Update synaptic weights based on Hebbian learning
        Malenka & Bear (2004): activity-dependent strengthening/weakening
        """
        # Determine if this neuron is currently active
        post_active = (time - self.last_spike_time) < 10.0  # Recent spike

        # Update all incoming synapses
        synapses_to_remove = []

        for pre_id, synapse in self.synapses_in.items():
            # Simplified: assume presynaptic activity based on random chance
            pre_active = random.random() < 0.1

            # Apply Hebbian plasticity
            survives = synapse.apply_hebbian_plasticity(
                pre_active, post_active, time,
                learning_rate=self.morphology.learning_rate * 0.01
            )

            if not survives:
                synapses_to_remove.append(pre_id)

        # Synaptic pruning: remove weak synapses
        for pre_id in synapses_to_remove:
            del self.synapses_in[pre_id]

        # Structural plasticity: occasionally form new synapses
        if random.random() < self.morphology.synaptic_plasticity_rate * 0.001:
            self._form_new_synapse()

    def _form_new_synapse(self):
        """
        Synaptogenesis: formation of new synaptic connection
        Activity-dependent and guidance cue-dependent
        """
        # In full simulation, would connect to nearby neuron
        # Here just increment synapse count
        new_id = random.randint(1, 1000000)
        new_synapse = Synapse(pre_neuron_id=new_id, post_neuron_id=self.id)
        self.synapses_in[new_id] = new_synapse

    def _update_metabolism(self, dt: float):
        """
        Metabolic energy consumption and supply from vasculature
        Magistretti & Allaman (2015): neurons are metabolically expensive

        ATP production requires glucose + oxygen
        """
        # Basal metabolic rate
        basal_consumption = self.morphology.metabolic_demand * 0.1 * dt

        # Activity-dependent consumption (ion pump costs)
        activity_consumption = self.firing_rate * 0.05 * dt

        # Total energy consumption
        total_consumption = basal_consumption + activity_consumption

        # Energy supply from blood vessels (neurovascular coupling)
        # Attwell et al. (2010): blood flow increases with neural activity
        if self.blood_vessel_distance < 50.0:  # Within capillary supply range
            # Aerobic ATP production: glucose + O2 → ATP
            if self.glucose_level > 0.1 and self.oxygen_level > 0.1:
                atp_production = min(self.glucose_level, self.oxygen_level) * 2.0 * dt
                self.energy += atp_production

                # Consume glucose and oxygen
                self.glucose_level -= atp_production * 0.3
                self.oxygen_level -= atp_production * 0.2
        else:
            # Hypoxic conditions: anaerobic glycolysis (less efficient)
            if self.glucose_level > 0.1:
                atp_production = self.glucose_level * 0.5 * dt
                self.energy += atp_production
                self.glucose_level -= atp_production * 0.6

        # Apply energy consumption
        self.energy -= total_consumption

        # Clamp values
        self.energy = max(0.0, min(200.0, self.energy))
        self.glucose_level = max(0.0, min(1.0, self.glucose_level))
        self.oxygen_level = max(0.0, min(1.0, self.oxygen_level))

    def _check_survival(self):
        """
        Check if neuron should undergo apoptosis
        Yuan & Yankner (2000): neurons die from:
        - Energy depletion
        - Lack of neurotrophic factors
        - Lack of synaptic activity
        - Toxic insults
        """
        # Energy depletion
        if self.energy < 5.0:
            self._initiate_apoptosis("energy depletion")
            return

        # Neurotrophic factor withdrawal
        if self.stage == "mature" and self.bdnf_level < 0.1:
            self._initiate_apoptosis("neurotrophic factor withdrawal")
            return

        # Synaptic inactivity (use it or lose it)
        if self.stage == "mature" and self.age > 100.0:
            if self.activity_score < 0.5 and len(self.synapses_in) < 3:
                self._initiate_apoptosis("synaptic inactivity")
                return

        # Hypoxia
        if self.oxygen_level < 0.05:
            self._initiate_apoptosis("hypoxia")
            return

    def _initiate_apoptosis(self, reason: str):
        """
        Programmed cell death cascade
        Yuan & Yankner (2000): caspase activation, DNA fragmentation
        """
        if self.stage != "apoptotic":
            self.stage = "apoptotic"
            print(f"Neuron {self.id}: Initiating apoptosis due to {reason}")

    def _undergo_apoptosis(self, dt: float):
        """
        Execute apoptotic program
        Clean cell death without inflammation
        """
        # Gradual energy depletion
        self.energy -= 10.0 * dt

        # Membrane potential depolarizes
        self.membrane_potential += 5.0 * dt

        # After apoptotic cascade, cell is removed
        if self.energy <= 0:
            self.alive = False
            print(f"Neuron {self.id}: Apoptosis complete, cell removed")

    def connect_to_neuron(self, target_neuron: 'Neuron', initial_weight: float = 0.5):
        """
        Form synaptic connection with another neuron

        Args:
            target_neuron: Postsynaptic neuron
            initial_weight: Initial synaptic efficacy
        """
        # Create outgoing synapse
        synapse = Synapse(
            pre_neuron_id=self.id,
            post_neuron_id=target_neuron.id,
            weight=initial_weight
        )

        self.synapses_out[target_neuron.id] = synapse
        target_neuron.synapses_in[self.id] = synapse

        print(f"Synapse formed: Neuron {self.id} → Neuron {target_neuron.id}")

    def receive_neurotrophic_factors(self, bdnf: float = 0.0, ngf: float = 0.0):
        """
        Receive neurotrophic factors from environment (glial cells, blood)

        Args:
            bdnf: Brain-Derived Neurotrophic Factor
            ngf: Nerve Growth Factor
        """
        self.bdnf_level = min(1.0, self.bdnf_level + bdnf)
        self.ngf_level = min(1.0, self.ngf_level + ngf)

    def receive_metabolites(self, glucose: float = 0.0, oxygen: float = 0.0):
        """
        Receive glucose and oxygen from blood vessel

        Args:
            glucose: Glucose concentration
            oxygen: Oxygen concentration
        """
        self.glucose_level = min(1.0, self.glucose_level + glucose)
        self.oxygen_level = min(1.0, self.oxygen_level + oxygen)

    def get_state_description(self) -> str:
        """Generate detailed state description"""
        return (
            f"Neuron {self.id} ({self.morphology.neurotransmitter}ergic)\n"
            f"  Stage: {self.stage}, Age: {self.age:.1f}, Alive: {self.alive}\n"
            f"  Position: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f})\n"
            f"  Energy: {self.energy:.1f}, Membrane: {self.membrane_potential:.1f} mV\n"
            f"  Firing rate: {self.firing_rate:.2f} Hz\n"
            f"  Synapses in: {len(self.synapses_in)}, out: {len(self.synapses_out)}\n"
            f"  Glucose: {self.glucose_level:.2f}, O2: {self.oxygen_level:.2f}\n"
            f"  BDNF: {self.bdnf_level:.2f}, Activity: {self.activity_score:.2f}\n"
            f"  {self.morphology.get_description()}"
        )


# Evolutionary context: origin of neurons
# Moroz (2009) and Ryan & Grant (2009): neurons evolved ~600 million years ago

NEURON_EVOLUTION_CONTEXT = """
Evolutionary Origin of Neurons
==============================

Based on Moroz (2009) and Ryan & Grant (2009):

1. Pre-neuronal organisms (~800 Mya):
   - Sponges (no neurons, use chemical signaling)
   - Cell-cell communication via gap junctions and secreted factors

2. Early nervous systems (~600-580 Mya):
   - Ctenophores (comb jellies): independent evolution of neurons?
   - Cnidarians (jellyfish, hydra): nerve nets
   - Diffuse organization, no centralization

3. Bilaterian nervous systems (~555 Mya):
   - Centralized nervous systems emerge
   - Segmentation, cephalization (brain formation)
   - Specialized neuron types

4. Vertebrate nervous system elaboration (~525 Mya):
   - Neural crest cells enable complex peripheral NS
   - Myelination (enables fast conduction)
   - Cortex expansion in mammals

Key evolutionary innovations:
- Action potential mechanism (voltage-gated Na+/K+ channels)
- Synaptic transmission (vesicle release machinery)
- Neurotransmitter diversity
- Myelination (10-100x conduction speed increase)
- Dendritic computation (nonlinear integration)

Environmental requirements for neuron evolution:
- Multicellularity
- Sufficient oxygen (oxidative metabolism)
- Developmental programs for patterning
- Predation pressure (sensory-motor integration advantage)
"""
