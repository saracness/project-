"""
MicrolifeEnv — OpenAI Gym-compatible RL environment

Wraps the existing Python simulation (Environment + Organism) into a standard
Gym interface so that any brain from microlife/ml/ can be trained with it.

State vector (8 dims, all normalised to [0, 1]):
    0  energy              / 200
    1  nearest_food_dist   / env_diagonal
    2  sin(food_angle)     ∈ [-1, 1] → mapped to [0, 1]
    3  cos(food_angle)     ∈ [-1, 1] → mapped to [0, 1]
    4  in_temperature_zone  (0 / 1)
    5  near_obstacle        (0 / 1)
    6  age                 / 2000   (capped)
    7  food_count          / 50     (capped)

Actions (discrete, 9):
    0  North   (0,  1)
    1  NE      (1,  1)
    2  East    (1,  0)
    3  SE      (1, -1)
    4  South   (0, -1)
    5  SW      (-1,-1)
    6  West    (-1, 0)
    7  NW      (-1, 1)
    8  Stay    (0,  0)
"""

import math
import random
import numpy as np
from .simulation.environment import Environment
from .simulation.organism import Organism, Food


# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------
ACTIONS = [
    (0,  1),   # 0 North
    (1,  1),   # 1 NE
    (1,  0),   # 2 East
    (1, -1),   # 3 SE
    (0, -1),   # 4 South
    (-1, -1),  # 5 SW
    (-1,  0),  # 6 West
    (-1,  1),  # 7 NW
    (0,  0),   # 8 Stay
]
N_ACTIONS = len(ACTIONS)
OBS_DIM = 8


class MicrolifeEnv:
    """
    Gym-style single-organism RL environment.

    Parameters
    ----------
    width, height : int
        World dimensions.
    n_food : int
        Initial food count (food respawns to keep count ≈ n_food).
    n_temp_zones : int
        Number of temperature hazard zones.
    max_steps : int
        Episode length cap (organism may die earlier).
    seed : int | None
        RNG seed for reproducibility.
    """

    # Gym-style spaces (lightweight dicts, no gym dependency required)
    observation_space = {"shape": (OBS_DIM,), "low": 0.0, "high": 1.0}
    action_space      = {"n": N_ACTIONS}

    def __init__(
        self,
        width: int = 300,
        height: int = 300,
        n_food: int = 20,
        n_temp_zones: int = 3,
        max_steps: int = 1000,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.n_food = n_food
        self.n_temp_zones = n_temp_zones
        self.max_steps = max_steps
        self._diagonal = math.sqrt(width ** 2 + height ** 2)

        self._rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)

        # Will be initialised in reset()
        self._env: Environment | None = None
        self._organism: Organism | None = None
        self._step_count = 0
        self._episode_reward = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset episode and return initial observation."""
        self._env = Environment(
            width=self.width,
            height=self.height,
            use_intelligent_movement=False,  # we drive the agent ourselves
        )

        # Spawn food
        for _ in range(self.n_food):
            self._env.add_food()

        # Add temperature hazard zones
        for _ in range(self.n_temp_zones):
            temp = self._rng.choice([-1, 1])
            self._env.add_temperature_zone(radius=40, temperature=temp)

        # Spawn the controlled organism at centre with full energy
        cx, cy = self.width / 2, self.height / 2
        self._organism = Organism(cx, cy, energy=100.0)
        self._env.organisms.append(self._organism)

        self._step_count = 0
        self._episode_reward = 0.0

        return self._get_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """
        Apply action, advance one simulation tick.

        Returns
        -------
        obs : np.ndarray  shape (OBS_DIM,)
        reward : float
        done : bool
        info : dict
        """
        assert self._organism is not None, "Call reset() before step()"

        old_energy = self._organism.energy
        old_food_dist = self._nearest_food_dist()

        # Apply movement
        dx, dy = ACTIONS[action]
        speed = self._organism.speed
        self._organism.x = max(0.0, min(self.width,  self._organism.x + dx * speed))
        self._organism.y = max(0.0, min(self.height, self._organism.y + dy * speed))
        self._organism._update_after_movement()

        # Environment effects (temperature, food consumption)
        self._env._apply_temperature_effects(self._organism)
        self._env._check_food_consumption(self._organism)

        # Respawn food to keep density roughly constant
        available = sum(1 for f in self._env.food_particles if not f.consumed)
        if available < self.n_food:
            self._env.add_food()

        self._step_count += 1

        # Reward shaping
        reward = self._compute_reward(old_energy, old_food_dist)
        self._episode_reward += reward

        done = (not self._organism.alive) or (self._step_count >= self.max_steps)

        info = {
            "step": self._step_count,
            "energy": self._organism.energy,
            "age": self._organism.age,
            "episode_reward": self._episode_reward,
        }

        return self._get_obs(), reward, done, info

    def render(self) -> dict:
        """Return current stats (no graphical rendering in headless mode)."""
        return self._env.get_statistics() if self._env else {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_food(self):
        """Return (food_object, distance) for nearest unconsumed food."""
        best, best_dist = None, float("inf")
        for food in self._env.food_particles:
            if food.consumed:
                continue
            dx = food.x - self._organism.x
            dy = food.y - self._organism.y
            d = math.sqrt(dx * dx + dy * dy)
            if d < best_dist:
                best, best_dist = food, d
        return best, best_dist

    def _nearest_food_dist(self) -> float:
        _, d = self._nearest_food()
        return d

    def _get_obs(self) -> np.ndarray:
        """Build normalised observation vector."""
        o = self._organism
        food, food_dist = self._nearest_food()

        if food is not None:
            angle = math.atan2(food.y - o.y, food.x - o.x)
            sin_a = math.sin(angle)
            cos_a = math.cos(angle)
        else:
            sin_a, cos_a = 0.0, 0.0
            food_dist = self._diagonal  # max possible

        in_temp = float(
            any(z.affects(o) for z in self._env.temperature_zones)
        )
        near_obs = float(
            any(ob.collides_with(o) for ob in self._env.obstacles)
        )
        food_count = sum(1 for f in self._env.food_particles if not f.consumed)

        obs = np.array([
            o.energy / 200.0,
            min(food_dist / self._diagonal, 1.0),
            (sin_a + 1.0) / 2.0,      # remap [-1,1] → [0,1]
            (cos_a + 1.0) / 2.0,
            in_temp,
            near_obs,
            min(o.age / 2000.0, 1.0),
            min(food_count / 50.0, 1.0),
        ], dtype=np.float32)

        return obs

    def _compute_reward(self, old_energy: float, old_food_dist: float) -> float:
        o = self._organism

        reward = 0.0

        # Survival bonus every step
        reward += 0.05

        # Energy change (eating gives large positive, starvation negative)
        delta_e = o.energy - old_energy
        reward += delta_e * 0.3

        # Shaping: getting closer to food
        new_food_dist = self._nearest_food_dist()
        if new_food_dist < old_food_dist:
            reward += 0.1

        # Penalty for critical energy (< 20 % of max)
        if o.energy < 40.0:
            reward -= 0.2

        # Death penalty
        if not o.alive:
            reward -= 5.0

        return float(reward)

    # ------------------------------------------------------------------
    # Convenience helpers for training scripts
    # ------------------------------------------------------------------

    def obs_to_brain_state(self, obs: np.ndarray) -> dict:
        """
        Convert numpy observation to the dict format expected by
        Brain.decide_action() / Brain.learn().

        All 8 observation dimensions are preserved so that Brain.get_state_vector()
        can reconstruct the same 8-dim vector without information loss.
        """
        return {
            "energy":                  float(obs[0]) * 200.0,
            "nearest_food_distance":   float(obs[1]) * self._diagonal,
            "nearest_food_angle":      math.atan2(
                float(obs[2]) * 2 - 1,   # sin
                float(obs[3]) * 2 - 1,   # cos
            ),
            "in_temperature_zone":     bool(obs[4] > 0.5),
            "near_obstacle":           bool(obs[5] > 0.5),
            "age":                     int(float(obs[6]) * 2000),
            "speed":                   getattr(self._organism, "speed", 1.0),
            "food_count":              float(obs[7]) * 50.0,  # dim 8 — matches brain_base
        }
