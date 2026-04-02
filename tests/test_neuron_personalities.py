"""
Tests for neuron personality system.

Validates that:
- All 9 personality types can be instantiated correctly
- PersonalityTraits fields are within biologically plausible ranges
- PersonalizedNeuron wraps a base Neuron without errors
- Key functional behaviours (burst, pacemaker, plasticity flags) are set correctly
"""

import pytest

from microlife.simulation.neuron_personalities import (
    NEURON_PERSONALITIES,
    PersonalityTraits,
    PersonalizedNeuron,
    FiringPattern,
    NeuronRole,
    create_personalized_neuron,
    DOPAMINERGIC_VTA,
    SEROTONERGIC_RAPHE,
    CHOLINERGIC_BASAL_FOREBRAIN,
    HIPPOCAMPAL_PLACE_CELL,
    ENTORHINAL_GRID_CELL,
    MIRROR_NEURON,
    VON_ECONOMO_NEURON,
    FAST_SPIKING_INTERNEURON,
    CHATTERING_NEURON,
)
from microlife.simulation.neuron import Neuron


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_neuron():
    return Neuron(x=0.0, y=0.0, z=0.0)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

class TestRegistry:
    EXPECTED_KEYS = {
        "dopaminergic", "serotonergic", "cholinergic",
        "place_cell", "grid_cell", "mirror_neuron",
        "von_economo", "fast_spiking", "chattering",
    }

    def test_all_nine_personalities_present(self):
        assert set(NEURON_PERSONALITIES.keys()) == self.EXPECTED_KEYS

    def test_all_values_are_personality_traits(self):
        for key, val in NEURON_PERSONALITIES.items():
            assert isinstance(val, PersonalityTraits), \
                f"{key} is not a PersonalityTraits instance"

    def test_all_have_valid_firing_pattern(self):
        valid_patterns = set(FiringPattern)
        for key, p in NEURON_PERSONALITIES.items():
            assert p.firing_pattern in valid_patterns, \
                f"{key} has invalid firing_pattern"

    def test_all_have_valid_role(self):
        valid_roles = set(NeuronRole)
        for key, p in NEURON_PERSONALITIES.items():
            assert p.role in valid_roles, \
                f"{key} has invalid role"


# ---------------------------------------------------------------------------
# Biological plausibility — firing rates
# ---------------------------------------------------------------------------

class TestFiringRates:
    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_baseline_firing_rate_positive(self, key):
        p = NEURON_PERSONALITIES[key]
        assert p.baseline_firing_rate > 0, \
            f"{key}: baseline_firing_rate must be > 0"

    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_max_greater_than_baseline(self, key):
        p = NEURON_PERSONALITIES[key]
        assert p.max_firing_rate >= p.baseline_firing_rate, \
            f"{key}: max_firing_rate must be >= baseline_firing_rate"

    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_burst_probability_in_range(self, key):
        p = NEURON_PERSONALITIES[key]
        assert 0.0 <= p.burst_probability <= 1.0, \
            f"{key}: burst_probability out of [0,1]"

    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_neuromodulator_sensitivities_in_range(self, key):
        p = NEURON_PERSONALITIES[key]
        for attr in ("dopamine_sensitivity", "serotonin_sensitivity",
                     "acetylcholine_sensitivity"):
            val = getattr(p, attr)
            assert 0.0 <= val <= 1.0, \
                f"{key}: {attr}={val} out of [0,1]"

    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_noise_tolerance_in_range(self, key):
        p = NEURON_PERSONALITIES[key]
        assert 0.0 <= p.noise_tolerance <= 1.0, \
            f"{key}: noise_tolerance out of [0,1]"


# ---------------------------------------------------------------------------
# Specific known properties from literature
# ---------------------------------------------------------------------------

class TestLiteratureProperties:
    def test_dopaminergic_is_plastic(self):
        """Dopamine neurons show plasticity (Schultz et al. reward coding)."""
        assert DOPAMINERGIC_VTA.is_plastic is True

    def test_dopaminergic_has_pacemaker(self):
        """DA neurons show autonomous pacemaker activity (Grace & Bunney 1984)."""
        assert DOPAMINERGIC_VTA.has_pacemaker is True

    def test_dopaminergic_reward_coding_role(self):
        assert DOPAMINERGIC_VTA.role == NeuronRole.REWARD_CODING

    def test_serotonergic_is_stable(self):
        """5-HT neurons are characteristically non-plastic (Jacobs & Azmitia 1992)."""
        assert SEROTONERGIC_RAPHE.is_plastic is False

    def test_serotonergic_slow_firing(self):
        """Raphe 5-HT neurons fire very slowly (0.5–2 Hz)."""
        assert SEROTONERGIC_RAPHE.baseline_firing_rate <= 2.0

    def test_place_cell_spatial_role(self):
        """Place cells encode spatial location (O'Keefe & Dostrovsky 1971)."""
        assert HIPPOCAMPAL_PLACE_CELL.role == NeuronRole.SPATIAL_NAVIGATION

    def test_grid_cell_spatial_role(self):
        """Grid cells encode metric space (Hafting et al. 2005)."""
        assert ENTORHINAL_GRID_CELL.role == NeuronRole.SPATIAL_GRID

    def test_fast_spiking_interneuron_high_max_rate(self):
        """Fast-spiking interneurons can fire >100 Hz (Markram et al. 2004)."""
        assert FAST_SPIKING_INTERNEURON.max_firing_rate >= 100.0

    def test_mirror_neuron_action_understanding_role(self):
        """Mirror neurons respond to observed actions (Rizzolatti 2004)."""
        assert MIRROR_NEURON.role == NeuronRole.ACTION_UNDERSTANDING


# ---------------------------------------------------------------------------
# PersonalizedNeuron wrapping
# ---------------------------------------------------------------------------

class TestPersonalizedNeuron:
    @pytest.mark.parametrize("key", list(NEURON_PERSONALITIES.keys()))
    def test_create_personalized_neuron(self, key):
        """Every personality type can wrap a base Neuron without error."""
        base = make_neuron()
        pn = create_personalized_neuron(base, key)
        assert isinstance(pn, PersonalizedNeuron)

    def test_personality_assigned(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "dopaminergic")
        assert pn.personality.name == "Dopaminergic VTA Neuron"

    def test_unknown_personality_raises(self):
        base = make_neuron()
        with pytest.raises(ValueError, match="Unknown personality"):
            create_personalized_neuron(base, "nonexistent_type")

    def test_dopaminergic_sets_neurotransmitter(self):
        """Dopaminergic personality should set dopamine as neurotransmitter."""
        base = make_neuron()
        pn = create_personalized_neuron(base, "dopaminergic")
        assert pn.neuron.morphology.neurotransmitter == "dopamine"

    def test_serotonergic_sets_neurotransmitter(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "serotonergic")
        assert pn.neuron.morphology.neurotransmitter == "serotonin"

    def test_cholinergic_sets_neurotransmitter(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "cholinergic")
        assert pn.neuron.morphology.neurotransmitter == "acetylcholine"

    def test_non_plastic_halves_learning_rate(self):
        """Non-plastic neurons should have reduced learning rate."""
        base_plastic = make_neuron()
        base_stable = make_neuron()
        initial_lr = base_plastic.morphology.learning_rate
        pn_stable = create_personalized_neuron(base_stable, "serotonergic")
        # is_plastic=False → lr * 0.5
        assert pn_stable.neuron.morphology.learning_rate == pytest.approx(
            initial_lr * 0.5, rel=1e-5
        )

    def test_place_cell_has_place_field_attr(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "place_cell")
        assert hasattr(pn, "place_field_center")

    def test_grid_cell_has_grid_phase_attr(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "grid_cell")
        assert hasattr(pn, "grid_phase")

    def test_mirror_neuron_has_observed_action_attr(self):
        base = make_neuron()
        pn = create_personalized_neuron(base, "mirror_neuron")
        assert hasattr(pn, "observed_action")
