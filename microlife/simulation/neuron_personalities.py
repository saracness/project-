"""
Neuron Personalities - Specialized Neuron Types with Unique Behaviors

Based on neuroscience literature, implements different neuron "personalities"
with distinct firing patterns, connectivity, and functional roles.

Scientific References:
----------------------
1. Grace, A. A., & Bunney, B. S. (1984). The control of firing pattern in nigral
   dopamine neurons: single spike firing. Journal of Neuroscience, 4(11), 2866-2876.

2. Jacobs, B. L., & Azmitia, E. C. (1992). Structure and function of the brain
   serotonin system. Physiological Reviews, 72(1), 165-229.

3. O'Keefe, J., & Dostrovsky, J. (1971). The hippocampus as a spatial map.
   Brain Research, 34(1), 171-175.

4. Hafting, T., Fyhn, M., Molden, S., Moser, M. B., & Moser, E. I. (2005).
   Microstructure of a spatial map in the entorhinal cortex. Nature, 436(7052), 801-806.

5. Rizzolatti, G., & Craighero, L. (2004). The mirror-neuron system.
   Annual Review of Neuroscience, 27, 169-192.

6. Allman, J. M., Watson, K. K., Tetreault, N. A., & Hakeem, A. Y. (2005).
   Intuition and autism: a possible role for Von Economo neurons.
   Trends in Cognitive Sciences, 9(8), 367-373.

7. Markram, H., Toledo-Rodriguez, M., Wang, Y., Gupta, A., Silberberg, G., & Wu, C. (2004).
   Interneurons of the neocortical inhibitory system. Nature Reviews Neuroscience, 5(10), 793-807.

8. McCormick, D. A., Connors, B. W., Lighthall, J. W., & Prince, D. A. (1985).
   Comparative electrophysiology of pyramidal and sparsely spiny stellate neurons.
   Journal of Neurophysiology, 54(4), 782-806.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
import numpy as np
import random
from enum import Enum


class FiringPattern(Enum):
    """Different neuronal firing patterns"""
    REGULAR = "regular"              # Regular spiking
    BURST = "burst"                  # Burst firing
    FAST_SPIKING = "fast_spiking"   # Fast spiking (interneurons)
    IRREGULAR = "irregular"          # Irregular firing
    ADAPTIVE = "adaptive"            # Spike frequency adaptation
    PACEMAKER = "pacemaker"          # Spontaneous rhythmic firing
    CHATTERING = "chattering"        # High-frequency bursts


class NeuronRole(Enum):
    """Functional roles of neurons"""
    REWARD_CODING = "reward_coding"          # Dopaminergic - reward prediction
    MOOD_REGULATION = "mood_regulation"      # Serotonergic - mood, sleep
    ATTENTION = "attention"                  # Cholinergic - attention, arousal
    SPATIAL_NAVIGATION = "spatial_nav"       # Place cells - location encoding
    SPATIAL_GRID = "spatial_grid"            # Grid cells - metric encoding
    ACTION_UNDERSTANDING = "action_mirror"   # Mirror neurons - action observation
    SOCIAL_COGNITION = "social_cognition"    # VEN - social awareness
    TIMING = "timing"                        # Timing and rhythm
    PATTERN_RECOGNITION = "pattern_recog"    # Pattern detection
    GENERAL_COMPUTATION = "general"          # Standard pyramidal


@dataclass
class PersonalityTraits:
    """
    Defines personality traits for a neuron type

    These traits determine how the neuron behaves, connects, and responds
    """
    # Identity
    name: str
    role: NeuronRole

    # Firing characteristics
    firing_pattern: FiringPattern
    baseline_firing_rate: float  # Hz (spontaneous activity)
    max_firing_rate: float       # Hz
    burst_probability: float     # 0-1
    adaptation_rate: float       # How quickly firing adapts

    # Connectivity preferences
    preferred_targets: List[str]  # Which neuron types to connect to
    connection_radius: float      # μm
    synapse_formation_rate: float # Probability per timestep

    # Response characteristics
    response_latency: float       # ms (delay to respond)
    response_gain: float          # Input amplification
    noise_tolerance: float        # Resistance to noise

    # Neuromodulation
    dopamine_sensitivity: float   # How much dopamine affects it
    serotonin_sensitivity: float
    acetylcholine_sensitivity: float

    # Special properties
    has_pacemaker: bool          # Spontaneous rhythmic activity
    is_plastic: bool             # Can change connections easily
    is_homeostatic: bool         # Self-regulates activity

    # Behavioral functions
    special_functions: Dict[str, any] = None


# Predefined neuron personalities based on literature

DOPAMINERGIC_VTA = PersonalityTraits(
    name="Dopaminergic VTA Neuron",
    role=NeuronRole.REWARD_CODING,

    # Grace & Bunney (1984): Dopamine neurons fire in two modes
    firing_pattern=FiringPattern.BURST,
    baseline_firing_rate=4.0,  # 1-10 Hz baseline
    max_firing_rate=20.0,      # Can burst up to 20 Hz
    burst_probability=0.3,     # 30% chance of burst mode
    adaptation_rate=0.1,

    # Connectivity: Broadcasts widely to striatum, PFC
    preferred_targets=["pyramidal", "medium_spiny", "interneuron"],
    connection_radius=500.0,   # Long-range projections
    synapse_formation_rate=0.05,

    # Response: Sensitive to reward prediction errors
    response_latency=50.0,     # ~50 ms
    response_gain=2.0,         # Strong amplification
    noise_tolerance=0.7,

    # Modulation: Self-regulates via autoreceptors
    dopamine_sensitivity=0.5,  # Auto-inhibition
    serotonin_sensitivity=0.3,
    acetylcholine_sensitivity=0.4,

    # Special: Encodes reward prediction errors
    has_pacemaker=True,
    is_plastic=True,
    is_homeostatic=True,

    special_functions={
        "reward_prediction_error": True,
        "motivational_salience": True,
        "learning_signal": True
    }
)


SEROTONERGIC_RAPHE = PersonalityTraits(
    name="Serotonergic Raphe Neuron",
    role=NeuronRole.MOOD_REGULATION,

    # Jacobs & Azmitia (1992): Regular, slow firing
    firing_pattern=FiringPattern.REGULAR,
    baseline_firing_rate=1.0,  # Very slow (0.5-2 Hz)
    max_firing_rate=5.0,
    burst_probability=0.05,    # Rarely bursts
    adaptation_rate=0.05,

    # Connectivity: Diffuse, widespread projections
    preferred_targets=["pyramidal", "interneuron", "dopaminergic"],
    connection_radius=800.0,   # Very long-range
    synapse_formation_rate=0.03,

    # Response: Slow, sustained
    response_latency=100.0,    # Slow response
    response_gain=0.5,         # Weak but sustained
    noise_tolerance=0.9,       # Very stable

    # Modulation: Sensitive to stress, circadian
    dopamine_sensitivity=0.3,
    serotonin_sensitivity=0.8,  # Auto-regulation
    acetylcholine_sensitivity=0.2,

    # Special: Regulates mood, sleep-wake
    has_pacemaker=True,
    is_plastic=False,          # Stable connections
    is_homeostatic=True,

    special_functions={
        "mood_regulation": True,
        "sleep_wake_cycle": True,
        "anxiety_control": True,
        "impulse_control": True
    }
)


CHOLINERGIC_BASAL_FOREBRAIN = PersonalityTraits(
    name="Cholinergic Basal Forebrain Neuron",
    role=NeuronRole.ATTENTION,

    # Irregular firing, state-dependent
    firing_pattern=FiringPattern.IRREGULAR,
    baseline_firing_rate=5.0,
    max_firing_rate=30.0,      # Can fire rapidly during attention
    burst_probability=0.4,
    adaptation_rate=0.2,

    # Connectivity: Cortex and hippocampus
    preferred_targets=["pyramidal", "interneuron"],
    connection_radius=600.0,
    synapse_formation_rate=0.08,

    # Response: Fast, enhances cortical processing
    response_latency=30.0,
    response_gain=1.5,
    noise_tolerance=0.5,       # Sensitive to arousal state

    # Modulation: Responds to arousal
    dopamine_sensitivity=0.4,
    serotonin_sensitivity=0.3,
    acetylcholine_sensitivity=0.6,  # Self-modulation

    # Special: Attention and arousal
    has_pacemaker=False,
    is_plastic=True,
    is_homeostatic=False,

    special_functions={
        "attention_enhancement": True,
        "cortical_activation": True,
        "memory_encoding": True,
        "sensory_gating": True
    }
)


HIPPOCAMPAL_PLACE_CELL = PersonalityTraits(
    name="Hippocampal Place Cell",
    role=NeuronRole.SPATIAL_NAVIGATION,

    # O'Keefe & Dostrovsky (1971): Fires at specific locations
    firing_pattern=FiringPattern.ADAPTIVE,
    baseline_firing_rate=0.5,  # Silent until in place field
    max_firing_rate=40.0,      # High rate in place field
    burst_probability=0.6,     # Often bursts
    adaptation_rate=0.3,

    # Connectivity: Local hippocampal circuit
    preferred_targets=["place_cell", "interneuron", "grid_cell"],
    connection_radius=200.0,   # Local connections
    synapse_formation_rate=0.15,  # High plasticity

    # Response: Location-specific
    response_latency=20.0,
    response_gain=3.0,         # Strong when in field
    noise_tolerance=0.6,

    # Modulation: NMDA-dependent plasticity
    dopamine_sensitivity=0.4,
    serotonin_sensitivity=0.2,
    acetylcholine_sensitivity=0.8,  # Important for encoding

    # Special: Spatial encoding
    has_pacemaker=False,
    is_plastic=True,           # Place fields can remap
    is_homeostatic=False,

    special_functions={
        "place_field_encoding": True,
        "spatial_memory": True,
        "path_integration": True,
        "context_encoding": True
    }
)


ENTORHINAL_GRID_CELL = PersonalityTraits(
    name="Entorhinal Grid Cell",
    role=NeuronRole.SPATIAL_GRID,

    # Hafting et al. (2005): Regular grid pattern
    firing_pattern=FiringPattern.REGULAR,
    baseline_firing_rate=2.0,
    max_firing_rate=30.0,
    burst_probability=0.4,
    adaptation_rate=0.1,

    # Connectivity: Projects to hippocampus
    preferred_targets=["place_cell", "grid_cell"],
    connection_radius=300.0,
    synapse_formation_rate=0.1,

    # Response: Periodic spatial firing
    response_latency=25.0,
    response_gain=2.0,
    noise_tolerance=0.8,       # Very stable grid

    # Modulation
    dopamine_sensitivity=0.2,
    serotonin_sensitivity=0.2,
    acetylcholine_sensitivity=0.5,

    # Special: Metric encoding
    has_pacemaker=True,        # Oscillatory input
    is_plastic=False,          # Stable grids
    is_homeostatic=True,

    special_functions={
        "grid_pattern": True,
        "metric_encoding": True,
        "path_integration": True,
        "speed_coding": True
    }
)


MIRROR_NEURON = PersonalityTraits(
    name="Mirror Neuron",
    role=NeuronRole.ACTION_UNDERSTANDING,

    # Rizzolatti & Craighero (2004): Fires during action and observation
    firing_pattern=FiringPattern.BURST,
    baseline_firing_rate=3.0,
    max_firing_rate=50.0,      # Strong response
    burst_probability=0.7,     # High bursting
    adaptation_rate=0.2,

    # Connectivity: Motor and sensory areas
    preferred_targets=["pyramidal", "mirror_neuron"],
    connection_radius=250.0,
    synapse_formation_rate=0.12,

    # Response: Action-specific
    response_latency=40.0,
    response_gain=2.5,
    noise_tolerance=0.5,

    # Modulation
    dopamine_sensitivity=0.5,  # Learning new actions
    serotonin_sensitivity=0.3,
    acetylcholine_sensitivity=0.6,

    # Special: Action understanding
    has_pacemaker=False,
    is_plastic=True,           # Learn new actions
    is_homeostatic=False,

    special_functions={
        "action_observation_matching": True,
        "imitation_learning": True,
        "empathy": True,
        "intention_understanding": True
    }
)


VON_ECONOMO_NEURON = PersonalityTraits(
    name="Von Economo Neuron",
    role=NeuronRole.SOCIAL_COGNITION,

    # Allman et al. (2005): Large, fast-conducting, social awareness
    firing_pattern=FiringPattern.FAST_SPIKING,
    baseline_firing_rate=8.0,
    max_firing_rate=80.0,      # Very fast
    burst_probability=0.2,
    adaptation_rate=0.15,

    # Connectivity: Long-range, frontoinsular
    preferred_targets=["pyramidal", "von_economo"],
    connection_radius=700.0,   # Long projections
    synapse_formation_rate=0.06,

    # Response: Fast social information
    response_latency=15.0,     # Very fast
    response_gain=1.8,
    noise_tolerance=0.6,

    # Modulation: Social context
    dopamine_sensitivity=0.6,
    serotonin_sensitivity=0.7,  # Social stress
    acetylcholine_sensitivity=0.4,

    # Special: Social cognition
    has_pacemaker=False,
    is_plastic=True,
    is_homeostatic=True,

    special_functions={
        "social_awareness": True,
        "self_consciousness": True,
        "empathy": True,
        "intuition": True,
        "emotional_regulation": True
    }
)


FAST_SPIKING_INTERNEURON = PersonalityTraits(
    name="Fast-Spiking Parvalbumin Interneuron",
    role=NeuronRole.GENERAL_COMPUTATION,

    # Markram et al. (2004): Fast, non-adapting
    firing_pattern=FiringPattern.FAST_SPIKING,
    baseline_firing_rate=10.0,
    max_firing_rate=200.0,     # Extremely fast
    burst_probability=0.1,
    adaptation_rate=0.0,       # No adaptation

    # Connectivity: Local inhibition
    preferred_targets=["pyramidal"],
    connection_radius=100.0,   # Very local
    synapse_formation_rate=0.2,  # High connectivity

    # Response: Extremely fast inhibition
    response_latency=5.0,      # <5 ms
    response_gain=1.0,
    noise_tolerance=0.8,

    # Modulation: Less sensitive
    dopamine_sensitivity=0.2,
    serotonin_sensitivity=0.2,
    acetylcholine_sensitivity=0.3,

    # Special: Rhythm generation
    has_pacemaker=False,
    is_plastic=False,          # Stable inhibition
    is_homeostatic=True,

    special_functions={
        "gamma_oscillations": True,
        "precise_timing": True,
        "feedforward_inhibition": True,
        "gain_control": True
    }
)


CHATTERING_NEURON = PersonalityTraits(
    name="Chattering Neuron",
    role=NeuronRole.PATTERN_RECOGNITION,

    # McCormick et al. (1985): High-frequency bursts
    firing_pattern=FiringPattern.CHATTERING,
    baseline_firing_rate=2.0,
    max_firing_rate=100.0,     # High frequency bursts
    burst_probability=0.9,     # Almost always bursts
    adaptation_rate=0.25,

    # Connectivity: Layer 2/3 pyramidal
    preferred_targets=["pyramidal", "interneuron"],
    connection_radius=150.0,
    synapse_formation_rate=0.1,

    # Response: Burst pattern detection
    response_latency=35.0,
    response_gain=2.2,
    noise_tolerance=0.5,

    # Modulation
    dopamine_sensitivity=0.3,
    serotonin_sensitivity=0.3,
    acetylcholine_sensitivity=0.5,

    # Special: Pattern detection
    has_pacemaker=False,
    is_plastic=True,
    is_homeostatic=False,

    special_functions={
        "pattern_detection": True,
        "feature_binding": True,
        "attention": True,
        "beta_oscillations": True
    }
)


# Collection of all personalities
NEURON_PERSONALITIES: Dict[str, PersonalityTraits] = {
    "dopaminergic": DOPAMINERGIC_VTA,
    "serotonergic": SEROTONERGIC_RAPHE,
    "cholinergic": CHOLINERGIC_BASAL_FOREBRAIN,
    "place_cell": HIPPOCAMPAL_PLACE_CELL,
    "grid_cell": ENTORHINAL_GRID_CELL,
    "mirror_neuron": MIRROR_NEURON,
    "von_economo": VON_ECONOMO_NEURON,
    "fast_spiking": FAST_SPIKING_INTERNEURON,
    "chattering": CHATTERING_NEURON,
}


class PersonalizedNeuron:
    """
    Wrapper that adds personality to a base neuron

    This modifies neuron behavior based on personality traits without
    changing the base neuron class.
    """

    def __init__(self, base_neuron, personality: PersonalityTraits):
        """
        Wrap a base neuron with personality

        Args:
            base_neuron: Instance from neuron.py
            personality: PersonalityTraits instance
        """
        self.neuron = base_neuron
        self.personality = personality

        # Apply personality to base neuron
        self._apply_personality()

        # Personality-specific state
        self.burst_timer = 0.0
        self.pacemaker_phase = 0.0
        self.adaptation_level = 0.0
        self.place_field_center = None  # For place cells
        self.grid_phase = (0.0, 0.0)    # For grid cells
        self.observed_action = None      # For mirror neurons

    def _apply_personality(self):
        """Apply personality traits to base neuron"""
        # Modify firing characteristics
        self.neuron.morphology.learning_rate = (
            self.neuron.morphology.learning_rate *
            (1.0 if self.personality.is_plastic else 0.5)
        )

        # Set neurotransmitter based on personality
        if self.personality.role == NeuronRole.REWARD_CODING:
            self.neuron.morphology.neurotransmitter = "dopamine"
        elif self.personality.role == NeuronRole.MOOD_REGULATION:
            self.neuron.morphology.neurotransmitter = "serotonin"
        elif self.personality.role == NeuronRole.ATTENTION:
            self.neuron.morphology.neurotransmitter = "acetylcholine"

    def update(self, dt: float, time: float, context: Dict = None):
        """
        Update neuron with personality-specific behavior

        Args:
            dt: Time step
            time: Current time
            context: Environmental context (position, actions, etc.)
        """
        # Update base neuron
        self.neuron.update(dt, time)

        # Apply personality-specific updates
        self._update_firing_pattern(dt, time)

        if self.personality.has_pacemaker:
            self._update_pacemaker(dt)

        if self.personality.role == NeuronRole.SPATIAL_NAVIGATION:
            self._update_place_cell(context)
        elif self.personality.role == NeuronRole.SPATIAL_GRID:
            self._update_grid_cell(context)
        elif self.personality.role == NeuronRole.ACTION_UNDERSTANDING:
            self._update_mirror_neuron(context)

        if self.personality.is_homeostatic:
            self._homeostatic_regulation(dt)

    def _update_firing_pattern(self, dt: float, time: float):
        """Implement personality-specific firing pattern"""
        pattern = self.personality.firing_pattern

        if pattern == FiringPattern.BURST:
            if random.random() < self.personality.burst_probability * dt:
                # Enter burst mode
                self.burst_timer = 50.0  # 50 ms burst

            if self.burst_timer > 0:
                self.burst_timer -= dt * 1000  # Convert to ms
                self.neuron.firing_rate = self.personality.max_firing_rate

        elif pattern == FiringPattern.PACEMAKER:
            # Regular rhythmic firing
            base_rate = self.personality.baseline_firing_rate
            self.neuron.firing_rate = base_rate + np.sin(self.pacemaker_phase) * 2.0

        elif pattern == FiringPattern.ADAPTIVE:
            # Spike frequency adaptation
            if self.neuron.firing_rate > 10.0:
                self.adaptation_level += dt * self.personality.adaptation_rate
                self.neuron.firing_rate *= (1.0 - self.adaptation_level)
            else:
                self.adaptation_level *= 0.95  # Recover

        elif pattern == FiringPattern.CHATTERING:
            # High-frequency bursts
            if random.random() < 0.05 * dt:  # Occasional bursts
                for _ in range(5):  # 5-spike burst
                    self.neuron.membrane_potential += 30.0

    def _update_pacemaker(self, dt: float):
        """Update pacemaker oscillation"""
        freq = self.personality.baseline_firing_rate  # Hz
        self.pacemaker_phase += 2 * np.pi * freq * dt

        # Spontaneous depolarization
        if np.sin(self.pacemaker_phase) > 0.9:
            self.neuron.membrane_potential += 10.0

    def _update_place_cell(self, context: Dict):
        """Update place cell behavior"""
        if context and 'position' in context:
            pos = context['position']

            # Initialize place field on first update
            if self.place_field_center is None:
                self.place_field_center = pos

            # Calculate distance from place field center
            dist = np.linalg.norm(np.array(pos) - np.array(self.place_field_center))

            # Gaussian place field (sigma ~ 30 cm in rats, scale appropriately)
            field_size = 50.0  # μm in simulation
            activity = np.exp(-(dist ** 2) / (2 * field_size ** 2))

            # Modulate firing rate
            self.neuron.firing_rate = activity * self.personality.max_firing_rate

    def _update_grid_cell(self, context: Dict):
        """Update grid cell behavior"""
        if context and 'position' in context:
            pos = np.array(context['position'])

            # Grid cell: hexagonal pattern (simplified)
            # Real grid cells use path integration, this is simplified
            grid_spacing = 100.0  # μm

            # Calculate grid activity (simplified hexagonal)
            x, y = pos[0] / grid_spacing, pos[1] / grid_spacing

            # Three sinusoidal gratings at 60° angles
            g1 = np.cos(2 * np.pi * x)
            g2 = np.cos(2 * np.pi * (0.5 * x + 0.866 * y))
            g3 = np.cos(2 * np.pi * (-0.5 * x + 0.866 * y))

            grid_activity = (g1 + g2 + g3) / 3.0
            grid_activity = max(0, grid_activity)  # Rectify

            self.neuron.firing_rate = grid_activity * self.personality.max_firing_rate

    def _update_mirror_neuron(self, context: Dict):
        """Update mirror neuron behavior"""
        if context and 'observed_action' in context:
            action = context['observed_action']

            # Fire if observing relevant action
            if action == self.observed_action or context.get('self_action') == self.observed_action:
                self.neuron.firing_rate = self.personality.max_firing_rate
            else:
                self.neuron.firing_rate *= 0.9  # Decay

    def _homeostatic_regulation(self, dt: float):
        """Maintain firing rate homeostasis"""
        target_rate = self.personality.baseline_firing_rate
        current_rate = self.neuron.firing_rate

        # Slowly adjust excitability to maintain target
        if current_rate > target_rate * 1.5:
            # Too active, reduce excitability
            self.neuron.morphology.firing_threshold += 0.5 * dt
        elif current_rate < target_rate * 0.5:
            # Too quiet, increase excitability
            self.neuron.morphology.firing_threshold -= 0.5 * dt

    def get_description(self) -> str:
        """Get personality description"""
        return (
            f"{self.personality.name}\n"
            f"  Role: {self.personality.role.value}\n"
            f"  Firing: {self.personality.firing_pattern.value}\n"
            f"  Baseline rate: {self.personality.baseline_firing_rate:.1f} Hz\n"
            f"  Special functions: {list(self.personality.special_functions.keys()) if self.personality.special_functions else 'None'}\n"
            f"  Current firing rate: {self.neuron.firing_rate:.1f} Hz"
        )


def create_personalized_neuron(base_neuron, personality_type: str):
    """
    Create a neuron with specified personality

    Args:
        base_neuron: Neuron instance from neuron.py
        personality_type: Key from NEURON_PERSONALITIES

    Returns:
        PersonalizedNeuron instance
    """
    if personality_type not in NEURON_PERSONALITIES:
        raise ValueError(f"Unknown personality: {personality_type}. "
                        f"Available: {list(NEURON_PERSONALITIES.keys())}")

    personality = NEURON_PERSONALITIES[personality_type]
    return PersonalizedNeuron(base_neuron, personality)
