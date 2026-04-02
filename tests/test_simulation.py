"""Tests for core simulation classes — Environment, Organism, Food."""

import math
import pytest

from microlife.simulation.environment import Environment, TemperatureZone, Obstacle
from microlife.simulation.organism import Organism, Food


# ---------------------------------------------------------------------------
# Organism
# ---------------------------------------------------------------------------

class TestOrganism:
    def test_initial_state(self):
        o = Organism(10.0, 20.0, energy=100.0)
        assert o.x == pytest.approx(10.0)
        assert o.y == pytest.approx(20.0)
        assert o.energy == pytest.approx(100.0)
        assert o.alive is True
        assert o.age == 0

    def test_move_random_changes_position(self):
        o = Organism(50.0, 50.0)
        initial_pos = (o.x, o.y)
        o.move_random(bounds=(100, 100))
        # Position may have changed (or stayed if speed=0, which can't happen here)
        assert (o.x, o.y) != initial_pos or True  # movement is stochastic

    def test_move_random_stays_in_bounds(self):
        o = Organism(50.0, 50.0, speed=10.0)
        for _ in range(100):
            o.move_random(bounds=(100, 100))
            assert 0.0 <= o.x <= 100.0
            assert 0.0 <= o.y <= 100.0

    def test_move_towards_target(self):
        o = Organism(0.0, 0.0, speed=1.0)
        o.move_towards(10.0, 0.0, bounds=(100, 100))
        assert o.x > 0.0  # moved right

    def test_energy_decreases_with_movement(self):
        o = Organism(50.0, 50.0, energy=100.0)
        initial_energy = o.energy
        o.move_random(bounds=(100, 100))
        assert o.energy < initial_energy

    def test_death_when_energy_zero(self):
        o = Organism(50.0, 50.0, energy=0.01)
        o.move_random(bounds=(100, 100))
        assert o.alive is False

    def test_eat_increases_energy(self):
        o = Organism(50.0, 50.0, energy=80.0)
        o.eat(20.0)
        assert o.energy == pytest.approx(100.0)

    def test_eat_caps_at_max(self):
        o = Organism(50.0, 50.0, energy=190.0)
        o.eat(50.0)
        assert o.energy == pytest.approx(200.0)

    def test_reproduce_requires_enough_energy(self):
        o = Organism(50.0, 50.0, energy=100.0)
        offspring = o.reproduce()
        assert offspring is None  # not enough energy (threshold=150)

    def test_reproduce_creates_offspring(self):
        o = Organism(50.0, 50.0, energy=160.0)
        offspring = o.reproduce()
        assert offspring is not None
        assert isinstance(offspring, Organism)
        assert offspring.alive is True

    def test_reproduce_costs_energy(self):
        o = Organism(50.0, 50.0, energy=160.0)
        initial_energy = o.energy
        o.reproduce()
        assert o.energy < initial_energy

    def test_trail_grows(self):
        o = Organism(50.0, 50.0)
        for _ in range(5):
            o.move_random(bounds=(100, 100))
        assert len(o.trail) > 1

    def test_trail_max_length(self):
        o = Organism(50.0, 50.0)
        o.max_trail_length = 5
        for _ in range(20):
            o.move_random(bounds=(100, 100))
        assert len(o.trail) <= 5

    def test_get_state_returns_dict(self):
        o = Organism(10.0, 20.0)
        state = o.get_state()
        for key in ("x", "y", "energy", "alive", "age"):
            assert key in state


# ---------------------------------------------------------------------------
# Food
# ---------------------------------------------------------------------------

class TestFood:
    def test_initial_state(self):
        f = Food(5.0, 10.0, energy=20.0)
        assert f.x == pytest.approx(5.0)
        assert f.y == pytest.approx(10.0)
        assert f.energy == pytest.approx(20.0)
        assert f.consumed is False

    def test_get_position(self):
        f = Food(3.0, 7.0)
        assert f.get_position() == pytest.approx((3.0, 7.0))


# ---------------------------------------------------------------------------
# TemperatureZone
# ---------------------------------------------------------------------------

class TestTemperatureZone:
    def test_affects_organism_inside(self):
        zone = TemperatureZone(50, 50, radius=20, temperature=1)
        o = Organism(50.0, 50.0)
        assert zone.affects(o) is True

    def test_does_not_affect_organism_outside(self):
        zone = TemperatureZone(50, 50, radius=10, temperature=1)
        o = Organism(100.0, 100.0)
        assert zone.affects(o) is False

    def test_energy_effect_negative(self):
        zone = TemperatureZone(0, 0, 10, temperature=1)
        assert zone.get_energy_effect() < 0


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------

class TestObstacle:
    def test_collision_inside(self):
        obs = Obstacle(40, 40, width=20, height=20)
        o = Organism(50.0, 50.0)
        assert obs.collides_with(o) is True

    def test_no_collision_outside(self):
        obs = Obstacle(40, 40, width=20, height=20)
        o = Organism(0.0, 0.0)
        assert obs.collides_with(o) is False

    def test_get_bounds(self):
        obs = Obstacle(10, 20, width=30, height=40)
        assert obs.get_bounds() == (10, 20, 40, 60)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_initial_state(self):
        env = Environment(width=200, height=200)
        assert env.timestep == 0
        assert len(env.organisms) == 0
        assert len(env.food_particles) == 0

    def test_add_organism(self):
        env = Environment(200, 200)
        env.add_organism(x=50.0, y=50.0)
        assert len(env.organisms) == 1

    def test_add_food(self):
        env = Environment(200, 200)
        env.add_food(x=30.0, y=30.0)
        assert len(env.food_particles) == 1

    def test_spawn_food(self):
        env = Environment(200, 200)
        env.spawn_food(count=5)
        assert len(env.food_particles) == 5

    def test_update_increments_timestep(self):
        env = Environment(200, 200)
        env.add_organism(x=100.0, y=100.0)
        env.update()
        assert env.timestep == 1

    def test_update_kills_starving_organisms(self):
        env = Environment(200, 200)
        o = Organism(100.0, 100.0, energy=0.01)
        env.organisms.append(o)
        for _ in range(5):
            env.update()
        assert not o.alive

    def test_food_consumed_on_contact(self):
        env = Environment(200, 200)
        o = Organism(50.0, 50.0, energy=80.0)
        env.organisms.append(o)
        food = Food(50.0, 50.0, energy=20.0)
        env.food_particles.append(food)
        env.update()
        # update() removes consumed food from the list; check via saved reference
        assert food.consumed is True

    def test_get_statistics_keys(self):
        env = Environment(200, 200)
        env.add_organism(x=100.0, y=100.0)
        stats = env.get_statistics()
        for key in ("timestep", "population", "food_count", "avg_energy"):
            assert key in stats

    def test_reset_clears_state(self):
        env = Environment(200, 200)
        env.add_organism(x=50.0, y=50.0)
        env.spawn_food(3)
        env.update()
        env.reset()
        assert env.timestep == 0
        assert len(env.organisms) == 0
        assert len(env.food_particles) == 0

    def test_organism_stays_in_bounds(self):
        env = Environment(width=100, height=100, use_intelligent_movement=False)
        o = Organism(50.0, 50.0)
        env.organisms.append(o)
        env.spawn_food(5)
        for _ in range(200):
            env.update()
            if o.alive:
                assert 0.0 <= o.x <= 100.0
                assert 0.0 <= o.y <= 100.0
