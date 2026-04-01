"""
Tests for NeuroBrain — verifies that each personality type produces
measurably different RL behaviour compared to plain DQN.

Each test has a scientific comment explaining WHY the behaviour should differ.
"""

import numpy as np
import pytest

from microlife.ml.brain_neuro import NeuroBrain, SUPPORTED_PERSONALITIES
from microlife.ml.brain_rl import DQNBrain
from microlife.gym_env import MicrolifeEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _state(energy=80.0, food_dist=50.0, food_angle=0.5,
           in_temp=False, near_obs=False, age=10, food_count=15.0):
    return dict(energy=energy, nearest_food_distance=food_dist,
                nearest_food_angle=food_angle, in_temperature_zone=in_temp,
                near_obstacle=near_obs, age=age, speed=1.0,
                food_count=food_count)


def _run_episodes(brain, n_ep=20, max_steps=100, seed=42):
    """Run brain in env, return list of episode rewards."""
    env = MicrolifeEnv(width=200, height=200, n_food=15,
                       n_temp_zones=2, max_steps=max_steps, seed=seed)
    rewards = []
    for _ in range(n_ep):
        obs = env.reset()
        ep_r = 0.0
        while True:
            state = env.obs_to_brain_state(obs)
            ad    = brain.decide_action(state)
            obs2, r, done, _ = env.step(ad["_action_idx"])
            brain.learn(state, ad, r, env.obs_to_brain_state(obs2), done)
            ep_r += r
            obs   = obs2
            if done:
                break
        rewards.append(ep_r)
    return rewards


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:
    @pytest.mark.parametrize("p", sorted(SUPPORTED_PERSONALITIES))
    def test_all_personalities_instantiate(self, p):
        brain = NeuroBrain(personality_type=p, state_size=8, hidden_size=32)
        assert brain.personality_type == p

    def test_brain_type_label_includes_personality(self):
        brain = NeuroBrain("dopaminergic", state_size=8)
        assert "dopaminergic" in brain.brain_type

    def test_unsupported_personality_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            NeuroBrain("grid_cell", state_size=8)   # grid_cell has no RL modulation yet

    def test_is_dqn_subclass(self):
        brain = NeuroBrain("serotonergic", state_size=8)
        assert isinstance(brain, DQNBrain)


# ---------------------------------------------------------------------------
# 2. decide_action — interface contract
# ---------------------------------------------------------------------------

class TestDecideAction:
    @pytest.mark.parametrize("p", sorted(SUPPORTED_PERSONALITIES))
    def test_returns_valid_action_dict(self, p):
        brain = NeuroBrain(p, state_size=8)
        s = _state()
        ad = brain.decide_action(s)
        assert "_action_idx" in ad
        assert 0 <= ad["_action_idx"] <= 8
        dx, dy = ad["move_direction"]
        assert dx in (-1, 0, 1) and dy in (-1, 0, 1)

    def test_decision_count_increments(self):
        brain = NeuroBrain("fast_spiking", state_size=8)
        for _ in range(5):
            brain.decide_action(_state())
        assert brain.decision_count == 5


# ---------------------------------------------------------------------------
# 3. Personality-specific hyperparameter reconfiguration
# ---------------------------------------------------------------------------

class TestHyperparams:
    def test_serotonergic_higher_gamma_than_default(self):
        """
        5-HT neurons promote patience and long-term planning (Jacobs & Azmitia 1992).
        Higher γ means the agent values future rewards more.
        """
        sero  = NeuroBrain("serotonergic", state_size=8)
        plain = DQNBrain(state_size=8)
        assert sero.gamma > plain.gamma

    def test_serotonergic_slower_epsilon_decay(self):
        """
        Serotonin stabilises behaviour; slower ε decay keeps exploration alive longer.
        """
        sero  = NeuroBrain("serotonergic", state_size=8)
        plain = DQNBrain(state_size=8)
        assert sero.epsilon_decay > plain.epsilon_decay

    def test_fast_spiking_faster_epsilon_decay(self):
        """
        GABAergic inhibition = decisive action selection, rapid exploitation.
        ε decays faster → agent commits to best action sooner.
        """
        fast  = NeuroBrain("fast_spiking", state_size=8)
        plain = DQNBrain(state_size=8)
        assert fast.epsilon_decay < plain.epsilon_decay


# ---------------------------------------------------------------------------
# 4. Reward shaping — verifies that shaped != raw
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_serotonergic_adds_survival_bonus(self):
        """
        Serotonergic personality adds a per-step survival bonus.
        Shaped reward must be greater than raw reward for any positive step.
        """
        brain = NeuroBrain("serotonergic", state_size=8)
        s  = _state(energy=80.0)
        ns = _state(energy=79.0)   # slight energy loss (normal movement)
        brain.epsilon = 0.0

        ad = brain.decide_action(s)
        # Capture shaped reward by running learn and comparing total_reward
        before = brain.total_reward
        raw_reward = -0.1
        brain.learn(s, ad, raw_reward, ns, False)
        shaped_reward = brain.total_reward - before
        assert shaped_reward > raw_reward, \
            "Serotonergic should add survival bonus on top of raw reward"

    def test_cholinergic_amplifies_food_reward(self):
        """
        Cholinergic neurons boost attention to food (Hasselmo 2006).
        A large reward (eating food) should be amplified further.
        """
        brain = NeuroBrain("cholinergic", state_size=8)
        s  = _state(energy=80.0)
        ns = _state(energy=100.0)  # ate food
        brain.epsilon = 0.0

        ad = brain.decide_action(s)
        before = brain.total_reward
        raw_food_reward = 5.0   # typical food reward
        brain.learn(s, ad, raw_food_reward, ns, False)
        shaped = brain.total_reward - before
        assert shaped > raw_food_reward, \
            "Cholinergic should amplify food reward"

    def test_fast_spiking_penalises_stay_repetition(self):
        """
        Fast-spiking interneurons prevent motor perseveration.
        Repeating 'Stay' (action 8) should incur a penalty.
        """
        brain = NeuroBrain("fast_spiking", state_size=8)
        s  = _state()
        ns = _state()
        # Force Stay action twice in a row
        stay_action = {"_action_idx": 8, "move_direction": (0, 0),
                       "should_reproduce": False, "speed_multiplier": 1.0,
                       "_state_vector": brain.get_state_vector(s)}
        brain.learn(s, stay_action, 0.0, ns, False)   # first Stay
        before = brain.total_reward
        brain.learn(s, stay_action, 0.0, ns, False)   # second Stay
        shaped = brain.total_reward - before
        assert shaped < 0.0, "Second consecutive Stay should yield negative reward"

    def test_place_cell_gives_novelty_bonus(self):
        """
        Place cells reward exploration of novel spatial contexts (O'Keefe 1971).
        First visit to a new region should yield a positive bonus.
        """
        brain = NeuroBrain("place_cell", state_size=8)
        # Clear visited cells so next step is definitely novel
        brain._visited_cells.clear()
        s  = _state(food_dist=50.0,  food_angle=0.5)
        ns = _state(food_dist=100.0, food_angle=1.0)  # different region

        ad = brain.decide_action(s)
        before = brain.total_reward
        brain.learn(s, ad, 0.0, ns, False)
        shaped = brain.total_reward - before
        # Place cell adds novelty bonus → shaped > 0 even for raw=0 step
        # (Survival in parent also adds 0.05; combined should be positive)
        assert shaped > 0.0, \
            "Place cell should add novelty bonus for first visit to region"


# ---------------------------------------------------------------------------
# 5. Dopaminergic — RPE-scaled learning rate
# ---------------------------------------------------------------------------

class TestDopaminergic:
    def test_large_rpe_boosts_effective_lr(self):
        """
        Schultz (1997): unexpected reward → dopamine burst → stronger LTP.
        Large reward magnitude should produce a higher effective learning rate.
        """
        brain = NeuroBrain("dopaminergic", state_size=8)
        s  = _state()
        ns = _state()

        lr_small = brain._effective_lr(0.01, s, ns)   # small reward
        lr_large = brain._effective_lr(5.0,  s, ns)   # large reward (unexpected)
        assert lr_large > lr_small, \
            "Dopaminergic: large RPE should boost effective learning rate"

    def test_zero_reward_uses_base_lr(self):
        brain = NeuroBrain("dopaminergic", state_size=8)
        s  = _state()
        ns = _state()
        assert brain._effective_lr(0.0, s, ns) == pytest.approx(brain.lr)


# ---------------------------------------------------------------------------
# 6. Cholinergic — adaptive learning rate
# ---------------------------------------------------------------------------

class TestCholinergic:
    def test_food_nearby_boosts_lr(self):
        """
        ACh increases cortical gain when attention is focused on a stimulus.
        Food proximity is the attention signal here.
        """
        brain = NeuroBrain("cholinergic", state_size=8)
        s_near = _state(food_dist=10.0)   # food very close
        s_far  = _state(food_dist=290.0)  # food far away
        ns = _state()
        lr_near = brain._effective_lr(0.0, s_near, ns)
        lr_far  = brain._effective_lr(0.0, s_far,  ns)
        assert lr_near > lr_far, \
            "Cholinergic: food proximity should boost learning rate"


# ---------------------------------------------------------------------------
# 7. set_personality — live switching
# ---------------------------------------------------------------------------

class TestSetPersonality:
    @pytest.mark.parametrize("target", sorted(SUPPORTED_PERSONALITIES))
    def test_can_switch_to_any_personality(self, target):
        brain = NeuroBrain("dopaminergic", state_size=8)
        brain.set_personality(target)
        assert brain.personality_type == target

    def test_switch_clears_place_cell_memory(self):
        brain = NeuroBrain("place_cell", state_size=8)
        brain._visited_cells.add((1, 2))
        brain.set_personality("cholinergic")
        brain.set_personality("place_cell")
        assert len(brain._visited_cells) == 0

    def test_switch_updates_hyperparams(self):
        brain = NeuroBrain("fast_spiking", state_size=8)
        fast_decay = brain.epsilon_decay
        brain.set_personality("serotonergic")
        assert brain.epsilon_decay > fast_decay

    def test_invalid_personality_raises(self):
        brain = NeuroBrain("dopaminergic", state_size=8)
        with pytest.raises(ValueError):
            brain.set_personality("nonexistent")


# ---------------------------------------------------------------------------
# 8. End-to-end: does NeuroBrain actually run without error?
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.mark.parametrize("p", sorted(SUPPORTED_PERSONALITIES))
    def test_runs_episodes_without_crash(self, p):
        brain = NeuroBrain(p, state_size=8, hidden_size=32)
        rewards = _run_episodes(brain, n_ep=5, max_steps=50)
        assert len(rewards) == 5
        assert all(isinstance(r, float) for r in rewards)

    def test_serotonergic_vs_fast_spiking_different_variance(self):
        """
        Fast-spiking interneurons commit to exploitation quickly (low ε early).
        This creates high-variance rewards: great episodes when the committed
        policy is good, poor when not.  Serotonergic stays exploratory longer
        → more consistent (lower std) reward early in training.

        We compare std of episode rewards as the distinguishing metric.
        """
        sero = NeuroBrain("serotonergic", state_size=8, hidden_size=32)
        fast = NeuroBrain("fast_spiking",  state_size=8, hidden_size=32)
        r_sero = _run_episodes(sero, n_ep=30, max_steps=150, seed=7)
        r_fast = _run_episodes(fast, n_ep=30, max_steps=150, seed=7)
        std_sero = np.std(r_sero)
        std_fast = np.std(r_fast)
        # Fast-spiking commits early → more variable outcome; difference > 0.5
        assert abs(std_fast - std_sero) > 0.5, \
            (f"Expected reward std to differ between personalities. "
             f"std_sero={std_sero:.2f}, std_fast={std_fast:.2f}")

    def test_get_neuro_info_returns_dict(self):
        brain = NeuroBrain("place_cell", state_size=8)
        _run_episodes(brain, n_ep=3, max_steps=30)
        info = brain.get_neuro_info()
        assert "personality" in info
        assert "novel_cells" in info
        assert info["novel_cells"] >= 0
