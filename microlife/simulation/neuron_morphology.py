"""
Neuron Morphology System
Based on peer-reviewed neuroscience literature

References:
-----------
1. Spruston, N. (2008). Pyramidal neurons: dendritic structure and synaptic integration.
   Nature Reviews Neuroscience, 9(3), 206-221.

2. Ascoli, G. A., Donohue, D. E., & Halavi, M. (2007). NeuroMorpho.Org:
   a central resource for neuronal morphologies. Journal of Neuroscience, 27(35), 9247-9251.

3. Jan, Y. N., & Jan, L. Y. (2010). Branching out: mechanisms of dendritic arborization.
   Nature Reviews Neuroscience, 11(5), 316-328.

4. Tavosanis, G. (2012). Dendritic structural plasticity.
   Developmental Neurobiology, 72(1), 73-86.

5. Luebke, J. I. (2017). Pyramidal neurons are not generalizable building blocks of cortical networks.
   Frontiers in Neuroanatomy, 11, 11.
"""

import random
from dataclasses import dataclass
from copy import deepcopy
from typing import Optional, Dict
import numpy as np


@dataclass
class NeuronMorphology:
    """
    Neuronal morphological properties based on real neuron anatomy

    Based on Spruston (2008) and Ascoli et al. (2007) classifications
    All values normalized to 0.0-1.0 range for simulation purposes
    """

    # Dendritic properties (Spruston, 2008)
    dendritic_arbor_complexity: float = 0.5  # Branching complexity (Sholl analysis)
    dendritic_spine_density: float = 0.5     # Spines per μm (0-10 spines/μm typical)
    dendritic_total_length: float = 0.5      # Total dendritic length (μm)

    # Axonal properties (Jan & Jan, 2010)
    axon_length: float = 0.5                 # Axon length (local vs projection)
    axon_branching: float = 0.5              # Axonal collaterals count
    myelination: float = 0.3                 # Degree of myelination (0=unmyelinated, 1=heavily myelinated)

    # Soma properties
    soma_size: float = 0.5                   # Cell body diameter (5-100 μm)

    # Synaptic properties (Tavosanis, 2012)
    synapse_count: float = 0.5               # Total synapse count (100s to 10,000s)
    synaptic_plasticity_rate: float = 0.5    # Rate of structural plasticity

    # Ion channel density (affects excitability)
    sodium_channel_density: float = 0.5      # Nav channels (action potential generation)
    potassium_channel_density: float = 0.5   # Kv channels (repolarization)
    calcium_channel_density: float = 0.5     # Cav channels (synaptic transmission)

    # Neurotransmitter type
    neurotransmitter: str = "glutamate"      # Primary NT (glutamate, GABA, dopamine, etc.)
    is_excitatory: bool = True               # Excitatory vs inhibitory

    # Derived computational advantages
    signal_speed: float = 1.0                # Action potential propagation speed
    integration_capacity: float = 1.0        # Dendritic integration capability
    metabolic_demand: float = 1.0            # ATP/glucose consumption rate
    learning_rate: float = 1.0               # Synaptic plasticity efficiency
    firing_threshold: float = -55.0          # Membrane potential threshold (mV)

    def __post_init__(self):
        """Calculate derived properties based on morphology"""
        self._clamp_values()
        self._update_computational_properties()

    def _clamp_values(self):
        """Ensure all morphological values stay within valid ranges"""
        morphological_traits = [
            'dendritic_arbor_complexity', 'dendritic_spine_density', 'dendritic_total_length',
            'axon_length', 'axon_branching', 'myelination', 'soma_size',
            'synapse_count', 'synaptic_plasticity_rate',
            'sodium_channel_density', 'potassium_channel_density', 'calcium_channel_density'
        ]

        for trait in morphological_traits:
            value = getattr(self, trait)
            setattr(self, trait, max(0.0, min(1.0, value)))

    def _update_computational_properties(self):
        """
        Calculate functional properties from morphology
        Based on biophysical principles from Spruston (2008) and Luebke (2017)
        """

        # Signal speed: myelination and axon diameter
        # Heavily myelinated axons: up to 120 m/s
        # Unmyelinated: 0.5-2 m/s
        self.signal_speed = 0.5 + (self.myelination * 1.5) + (self.soma_size * 0.3)

        # Integration capacity: dendritic complexity and spine density
        # More spines = more synaptic inputs to integrate
        self.integration_capacity = (
            0.3 +
            (self.dendritic_arbor_complexity * 0.4) +
            (self.dendritic_spine_density * 0.3) +
            (self.dendritic_total_length * 0.2)
        )

        # Metabolic demand: size, synapse count, firing rate
        # Larger neurons with more synapses consume more ATP
        # Magistretti & Allaman (2015): synaptic transmission is metabolically expensive
        self.metabolic_demand = (
            0.5 +
            (self.soma_size * 0.2) +
            (self.synapse_count * 0.4) +
            (self.dendritic_total_length * 0.2) +
            (self.calcium_channel_density * 0.1)
        )

        # Learning rate: plasticity rate and spine dynamics
        # Tavosanis (2012): spine turnover correlates with learning
        self.learning_rate = (
            0.3 +
            (self.synaptic_plasticity_rate * 0.5) +
            (self.dendritic_spine_density * 0.2) +
            (self.calcium_channel_density * 0.2)  # Ca²⁺ drives plasticity
        )

        # Firing threshold: influenced by ion channel densities
        # More Nav channels = lower threshold (more excitable)
        # More Kv channels = higher threshold (less excitable)
        threshold_shift = (
            (self.sodium_channel_density * -5.0) +  # Lower threshold
            (self.potassium_channel_density * 5.0)   # Higher threshold
        )
        self.firing_threshold = -55.0 + threshold_shift

        # Set excitatory/inhibitory based on neurotransmitter
        excitatory_nts = {"glutamate", "acetylcholine", "dopamine", "serotonin"}
        self.is_excitatory = self.neurotransmitter in excitatory_nts

    def mutate(self, mutation_rate: float = 0.15, mutation_strength: float = 0.1):
        """
        Mutate morphological properties (genetic variation + developmental noise)

        Based on Jan & Jan (2010): dendritic development has genetic and stochastic components

        Args:
            mutation_rate: Probability of each trait mutating
            mutation_strength: Magnitude of mutations
        """
        morphological_traits = [
            'dendritic_arbor_complexity', 'dendritic_spine_density', 'dendritic_total_length',
            'axon_length', 'axon_branching', 'myelination', 'soma_size',
            'synapse_count', 'synaptic_plasticity_rate',
            'sodium_channel_density', 'potassium_channel_density', 'calcium_channel_density'
        ]

        for trait in morphological_traits:
            if random.random() < mutation_rate:
                current_value = getattr(self, trait)
                mutation = random.gauss(0, mutation_strength)
                new_value = current_value + mutation
                setattr(self, trait, new_value)

        # Occasionally mutate neurotransmitter type (rare evolutionary event)
        if random.random() < mutation_rate * 0.1:
            neurotransmitters = ["glutamate", "GABA", "dopamine", "serotonin", "acetylcholine"]
            self.neurotransmitter = random.choice(neurotransmitters)

        self._clamp_values()
        self._update_computational_properties()

    def get_description(self) -> str:
        """Generate human-readable description of this neuron's morphology"""
        nt_type = "excitatory" if self.is_excitatory else "inhibitory"

        # Classify neuron type based on morphology
        if self.dendritic_arbor_complexity > 0.7 and self.soma_size > 0.6:
            neuron_type = "Pyramidal-like"
        elif self.dendritic_arbor_complexity < 0.3 and self.soma_size < 0.4:
            neuron_type = "Granule-like"
        elif not self.is_excitatory:
            neuron_type = "Interneuron-like"
        else:
            neuron_type = "Stellate-like"

        return (
            f"{neuron_type} {nt_type} neuron "
            f"({self.neurotransmitter}ergic)\n"
            f"  Dendritic complexity: {self.dendritic_arbor_complexity:.2f}, "
            f"Spine density: {self.dendritic_spine_density:.2f}\n"
            f"  Synapses: {self.synapse_count:.2f}, "
            f"Myelination: {self.myelination:.2f}\n"
            f"  Signal speed: {self.signal_speed:.2f}, "
            f"Learning rate: {self.learning_rate:.2f}, "
            f"Metabolic demand: {self.metabolic_demand:.2f}"
        )


# Real neuron type templates based on NeuroMorpho.Org database (Ascoli et al., 2007)
NEURON_TYPE_TEMPLATES: Dict[str, NeuronMorphology] = {
    # Pyramidal neuron (cortex, hippocampus)
    # Spruston (2008): extensive dendritic arbor, high spine density
    "pyramidal": NeuronMorphology(
        dendritic_arbor_complexity=0.85,
        dendritic_spine_density=0.80,
        dendritic_total_length=0.90,
        axon_length=0.85,  # Long-range projections
        axon_branching=0.70,
        myelination=0.60,
        soma_size=0.75,
        synapse_count=0.85,
        synaptic_plasticity_rate=0.75,
        sodium_channel_density=0.70,
        potassium_channel_density=0.60,
        calcium_channel_density=0.75,
        neurotransmitter="glutamate",
        is_excitatory=True
    ),

    # Granule cell (dentate gyrus, cerebellum)
    # Small soma, sparse dendrites
    "granule": NeuronMorphology(
        dendritic_arbor_complexity=0.30,
        dendritic_spine_density=0.40,
        dendritic_total_length=0.35,
        axon_length=0.60,
        axon_branching=0.30,
        myelination=0.20,
        soma_size=0.25,
        synapse_count=0.40,
        synaptic_plasticity_rate=0.85,  # High plasticity in DG
        sodium_channel_density=0.60,
        potassium_channel_density=0.70,
        calcium_channel_density=0.50,
        neurotransmitter="glutamate",
        is_excitatory=True
    ),

    # GABAergic interneuron (inhibitory)
    # Diverse morphologies, local connectivity
    "interneuron": NeuronMorphology(
        dendritic_arbor_complexity=0.60,
        dendritic_spine_density=0.30,  # Fewer spines on inhibitory neurons
        dendritic_total_length=0.55,
        axon_length=0.40,  # Local connections
        axon_branching=0.80,  # Highly branched
        myelination=0.15,
        soma_size=0.50,
        synapse_count=0.70,
        synaptic_plasticity_rate=0.55,
        sodium_channel_density=0.85,  # Fast-spiking
        potassium_channel_density=0.80,
        calcium_channel_density=0.60,
        neurotransmitter="GABA",
        is_excitatory=False
    ),

    # Purkinje cell (cerebellum)
    # Most extensive dendritic tree in nervous system
    "purkinje": NeuronMorphology(
        dendritic_arbor_complexity=0.98,  # Extremely complex
        dendritic_spine_density=0.95,     # ~200,000 spines per cell
        dendritic_total_length=0.95,
        axon_length=0.70,
        axon_branching=0.50,
        myelination=0.50,
        soma_size=0.85,
        synapse_count=0.98,  # Receives ~200,000 synapses
        synaptic_plasticity_rate=0.80,
        sodium_channel_density=0.70,
        potassium_channel_density=0.75,
        calcium_channel_density=0.90,  # Rich Ca²⁺ signaling
        neurotransmitter="GABA",
        is_excitatory=False
    ),

    # Motor neuron (spinal cord)
    # Large soma, long myelinated axon
    "motor": NeuronMorphology(
        dendritic_arbor_complexity=0.65,
        dendritic_spine_density=0.50,
        dendritic_total_length=0.60,
        axon_length=0.98,  # Can be >1 meter in humans
        axon_branching=0.40,
        myelination=0.95,  # Heavily myelinated for fast conduction
        soma_size=0.90,    # Large cell bodies
        synapse_count=0.65,
        synaptic_plasticity_rate=0.45,
        sodium_channel_density=0.80,
        potassium_channel_density=0.70,
        calcium_channel_density=0.65,
        neurotransmitter="acetylcholine",
        is_excitatory=True
    ),

    # Dopaminergic neuron (substantia nigra, VTA)
    # Moderately complex, neuromodulatory
    "dopaminergic": NeuronMorphology(
        dendritic_arbor_complexity=0.55,
        dendritic_spine_density=0.45,
        dendritic_total_length=0.60,
        axon_length=0.80,  # Widespread projections
        axon_branching=0.85,  # Extensive arborization
        myelination=0.40,
        soma_size=0.60,
        synapse_count=0.60,
        synaptic_plasticity_rate=0.70,
        sodium_channel_density=0.50,
        potassium_channel_density=0.60,
        calcium_channel_density=0.70,
        neurotransmitter="dopamine",
        is_excitatory=True
    )
}


def create_neuron_morphology(neuron_type: Optional[str] = None) -> NeuronMorphology:
    """
    Create a neuron with specified or random morphology

    Args:
        neuron_type: One of NEURON_TYPE_TEMPLATES keys, or None for random

    Returns:
        NeuronMorphology instance
    """
    if neuron_type and neuron_type in NEURON_TYPE_TEMPLATES:
        return deepcopy(NEURON_TYPE_TEMPLATES[neuron_type])
    elif neuron_type is None:
        # Random neuron
        return NeuronMorphology()
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}. "
                        f"Available: {list(NEURON_TYPE_TEMPLATES.keys())}")
