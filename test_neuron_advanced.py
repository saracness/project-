#!/usr/bin/env python3
"""
Advanced Neuron Simulation Tests
Tests edge cases, apoptosis, metabolic stress, etc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microlife'))

from simulation.neural_environment import NeuralEnvironment
from simulation.neuron import Neuron
from simulation.neuron_morphology import create_neuron_morphology

def test_apoptosis_energy_depletion():
    """Test that neurons die when energy is depleted"""
    print("\n" + "="*70)
    print("TEST 1: Apoptosis from Energy Depletion")
    print("="*70)

    neuron = Neuron(x=0, y=0, neuron_type="pyramidal")
    neuron.stage = "mature"
    neuron.energy = 10.0  # Low energy
    neuron.glucose_level = 0.0  # No glucose - prevent ATP production
    neuron.oxygen_level = 0.0   # No oxygen

    print(f"Initial: Energy={neuron.energy:.1f}, Glucose={neuron.glucose_level:.1f}, Alive={neuron.alive}, Stage={neuron.stage}")

    apoptosis_entered = False

    # Don't provide metabolites - force starvation
    for t in range(200):  # Increased iterations
        neuron.update(dt=0.1, time=t*0.1)

        if neuron.stage == "apoptotic" and not apoptosis_entered:
            print(f"t={t*0.1:.1f}: Entered apoptosis, Energy={neuron.energy:.1f}")
            apoptosis_entered = True

        # Debug output every 10 steps during apoptosis
        if neuron.stage == "apoptotic" and t % 10 == 0:
            print(f"t={t*0.1:.1f}: Apoptotic, Energy={neuron.energy:.1f}, Alive={neuron.alive}")

        if not neuron.alive:
            print(f"t={t*0.1:.1f}: Neuron died, Energy={neuron.energy:.1f}")
            break

    assert not neuron.alive, f"Neuron should have died from energy depletion. Final: Energy={neuron.energy:.1f}, Alive={neuron.alive}, Stage={neuron.stage}"
    print("✓ Test passed: Neuron correctly died from energy depletion")

def test_apoptosis_hypoxia():
    """Test that neurons die from oxygen deprivation"""
    print("\n" + "="*70)
    print("TEST 2: Apoptosis from Hypoxia")
    print("="*70)

    neuron = Neuron(x=0, y=0, neuron_type="pyramidal")
    neuron.stage = "mature"
    neuron.oxygen_level = 0.02  # Severe hypoxia

    print(f"Initial: O2={neuron.oxygen_level:.3f}, Alive={neuron.alive}")

    # Provide glucose but no oxygen
    for t in range(100):
        neuron.receive_metabolites(glucose=0.5, oxygen=0.0)  # No oxygen
        neuron.update(dt=0.1, time=t*0.1)

        if neuron.stage == "apoptotic" and t == 0:
            print(f"t={t*0.1:.1f}: Hypoxia triggered apoptosis")
            break

    assert neuron.stage == "apoptotic", "Neuron should enter apoptosis from hypoxia"
    print("✓ Test passed: Hypoxia correctly triggers apoptosis")

def test_synaptic_plasticity_ltp():
    """Test that LTP increases synaptic weight"""
    print("\n" + "="*70)
    print("TEST 3: Long-Term Potentiation (LTP)")
    print("="*70)

    pre = Neuron(x=0, y=0, neuron_type="pyramidal", energy=200)
    post = Neuron(x=10, y=0, neuron_type="pyramidal", energy=200)
    pre.stage = "mature"
    post.stage = "mature"

    pre.connect_to_neuron(post, initial_weight=0.5)
    synapse = post.synapses_in[pre.id]

    initial_weight = synapse.weight
    print(f"Initial synaptic weight: {initial_weight:.3f}")

    # Correlated firing (Hebbian: fire together, wire together)
    for t in range(200):
        if t % 5 == 0:
            pre._fire_action_potential(t*0.1)
            post._fire_action_potential(t*0.1 + 0.001)  # Post fires just after pre

        pre.receive_metabolites(glucose=0.5, oxygen=0.5)
        post.receive_metabolites(glucose=0.5, oxygen=0.5)

        pre.update(dt=0.1, time=t*0.1)
        post.update(dt=0.1, time=t*0.1)

    final_weight = synapse.weight
    print(f"Final synaptic weight: {final_weight:.3f}")
    print(f"Weight change: {(final_weight - initial_weight):.3f} ({(final_weight/initial_weight - 1)*100:.1f}%)")

    assert final_weight > initial_weight, "LTP should increase synaptic weight"
    print("✓ Test passed: LTP correctly strengthened synapse")

def test_synaptic_pruning():
    """Test that weak synapses are pruned"""
    print("\n" + "="*70)
    print("TEST 4: Synaptic Pruning")
    print("="*70)

    post = Neuron(x=0, y=0, neuron_type="pyramidal", energy=200)
    post.stage = "mature"

    # Create multiple weak synapses
    from simulation.neuron import Synapse
    for i in range(10):
        weak_synapse = Synapse(pre_neuron_id=i*1000, post_neuron_id=post.id, weight=0.03)
        post.synapses_in[i*1000] = weak_synapse

    initial_count = len(post.synapses_in)
    print(f"Initial synapse count: {initial_count}")

    # Run simulation - weak synapses should be pruned
    for t in range(200):
        post.receive_metabolites(glucose=0.5, oxygen=0.5)
        post.update(dt=0.1, time=t*0.1)

    final_count = len(post.synapses_in)
    print(f"Final synapse count: {final_count}")
    print(f"Pruned: {initial_count - final_count} weak synapses")

    assert final_count < initial_count, "Weak synapses should be pruned"
    print("✓ Test passed: Weak synapses correctly pruned")

def test_neurovascular_coupling():
    """Test that neural activity increases blood flow"""
    print("\n" + "="*70)
    print("TEST 5: Neurovascular Coupling")
    print("="*70)

    from simulation.neural_environment import BloodVessel

    vessel = BloodVessel(x=0, y=0, z=0, diameter=5.0)
    initial_dilation = vessel.dilation
    initial_flow = vessel.flow_rate

    print(f"Initial: Dilation={initial_dilation:.3f}, Flow={initial_flow:.3f}")

    # Simulate high neural activity nearby
    high_activity = 0.8
    for t in range(50):
        vessel.update(dt=0.1, neural_activity=high_activity)

    final_dilation = vessel.dilation
    final_flow = vessel.flow_rate

    print(f"After high activity: Dilation={final_dilation:.3f}, Flow={final_flow:.3f}")
    print(f"Flow increase: {(final_flow/initial_flow - 1)*100:.1f}%")

    assert final_flow > initial_flow, "High neural activity should increase blood flow"
    print("✓ Test passed: Neurovascular coupling works correctly")

def test_astrocyte_lactate_shuttle():
    """Test astrocyte lactate production during high activity"""
    print("\n" + "="*70)
    print("TEST 6: Astrocyte-Neuron Lactate Shuttle")
    print("="*70)

    from simulation.neural_environment import Astrocyte

    astrocyte = Astrocyte(x=0, y=0, z=0)
    initial_glycogen = astrocyte.glycogen_stores

    print(f"Initial glycogen: {initial_glycogen:.1f}")

    # High neural activity triggers glycogenolysis
    high_activity = 0.9
    for t in range(100):
        astrocyte.update(dt=0.1, local_neural_activity=high_activity)

    final_glycogen = astrocyte.glycogen_stores
    lactate = astrocyte.lactate_production

    print(f"Final glycogen: {final_glycogen:.1f}")
    print(f"Lactate production: {lactate:.3f}")
    print(f"Glycogen consumed: {initial_glycogen - final_glycogen:.1f}")

    assert final_glycogen < initial_glycogen, "High activity should consume glycogen"
    print("✓ Test passed: Astrocyte lactate shuttle works correctly")

def test_morphology_mutation():
    """Test that morphology mutation works"""
    print("\n" + "="*70)
    print("TEST 7: Morphology Mutation")
    print("="*70)

    original = create_neuron_morphology("pyramidal")
    mutated = create_neuron_morphology("pyramidal")

    print("Original morphology:")
    print(f"  Dendritic complexity: {original.dendritic_arbor_complexity:.3f}")
    print(f"  Spine density: {original.dendritic_spine_density:.3f}")
    print(f"  Myelination: {original.myelination:.3f}")

    # Apply mutations
    mutated.mutate(mutation_rate=1.0, mutation_strength=0.2)  # High mutation for testing

    print("\nMutated morphology:")
    print(f"  Dendritic complexity: {mutated.dendritic_arbor_complexity:.3f}")
    print(f"  Spine density: {mutated.dendritic_spine_density:.3f}")
    print(f"  Myelination: {mutated.myelination:.3f}")

    # At least one property should have changed
    changed = (
        abs(original.dendritic_arbor_complexity - mutated.dendritic_arbor_complexity) > 0.01 or
        abs(original.dendritic_spine_density - mutated.dendritic_spine_density) > 0.01 or
        abs(original.myelination - mutated.myelination) > 0.01
    )

    assert changed, "Mutation should change at least one property"
    print("✓ Test passed: Morphology mutation works correctly")

def test_life_cycle_stages():
    """Test that neurons progress through life stages"""
    print("\n" + "="*70)
    print("TEST 8: Life Cycle Stage Progression")
    print("="*70)

    neuron = Neuron(x=0, y=0, neuron_type="pyramidal")

    stages_seen = set()
    stages_seen.add(neuron.stage)

    print(f"Initial stage: {neuron.stage}")

    # Simulate until mature
    for t in range(100):
        neuron.receive_metabolites(glucose=0.5, oxygen=0.5)
        neuron.receive_neurotrophic_factors(bdnf=0.1, ngf=0.1)
        neuron.update(dt=1.0, time=t)

        if neuron.stage not in stages_seen:
            stages_seen.add(neuron.stage)
            print(f"t={t}: Progressed to {neuron.stage}")

        if neuron.stage == "mature":
            break

    expected_stages = {"neurogenesis", "migration", "differentiation", "mature"}
    assert expected_stages.issubset(stages_seen), f"Should see stages: {expected_stages}"
    print(f"✓ Test passed: Neuron progressed through stages: {stages_seen}")

def test_environment_population():
    """Test neural environment with longer simulation"""
    print("\n" + "="*70)
    print("TEST 9: Neural Environment Long Simulation")
    print("="*70)

    env = NeuralEnvironment(width=100, height=100, depth=50)
    env.initialize_vasculature(vessel_density=0.02)
    env.initialize_astrocytes(astrocyte_density=0.0005)

    env.create_random_population(num_neurons=10, neuron_type_distribution={
        "pyramidal": 0.8,
        "interneuron": 0.2
    })

    env.form_random_synapses(connection_probability=0.15)

    print(f"Initial: {len(env.neurons)} neurons, {len(env.blood_vessels)} vessels")

    # Run longer simulation
    for step in range(100):
        env.update()

    stats = env.get_statistics()

    print(f"\nFinal statistics (t={stats['time']:.1f}):")
    print(f"  Alive neurons: {stats['alive_neurons']}")
    print(f"  Average energy: {stats['avg_energy']:.1f}")
    print(f"  Average synapses: {stats['avg_synapses']:.1f}")

    assert stats['alive_neurons'] > 0, "Some neurons should survive"
    print("✓ Test passed: Environment simulation runs correctly")

def main():
    """Run all advanced tests"""
    print("\n" + "#"*70)
    print("# Advanced Neuron Simulation Tests")
    print("#"*70)

    tests = [
        test_apoptosis_energy_depletion,
        test_apoptosis_hypoxia,
        test_synaptic_plasticity_ltp,
        test_synaptic_pruning,
        test_neurovascular_coupling,
        test_astrocyte_lactate_shuttle,
        test_morphology_mutation,
        test_life_cycle_stages,
        test_environment_population
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ Test error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "#"*70)
    print(f"# Test Results: {passed} passed, {failed} failed")
    print("#"*70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
