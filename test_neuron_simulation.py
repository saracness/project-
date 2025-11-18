#!/usr/bin/env python3
"""
Neuron Life Cycle Simulation Demo

This script demonstrates the neuron simulation system with:
1. Individual neuron life cycle
2. Neural tissue with blood vessels and astrocytes
3. Synaptic plasticity
4. Neurovascular coupling
"""

import sys
import os

# Add microlife to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'microlife'))

from simulation.neural_environment import NeuralEnvironment
from simulation.neuron import Neuron, NEURON_EVOLUTION_CONTEXT
from simulation.neuron_morphology import (
    create_neuron_morphology,
    NEURON_TYPE_TEMPLATES,
    NeuronMorphology
)

def demo_neuron_types():
    """Demonstrate different neuron types"""
    print("\n" + "="*70)
    print("DEMO 1: Neuron Type Morphologies")
    print("="*70)

    for neuron_type in NEURON_TYPE_TEMPLATES.keys():
        morphology = create_neuron_morphology(neuron_type)
        print(f"\n{neuron_type.upper()}:")
        print(morphology.get_description())

def demo_individual_neuron():
    """Demonstrate individual neuron life cycle"""
    print("\n" + "="*70)
    print("DEMO 2: Individual Neuron Life Cycle")
    print("="*70)

    # Create pyramidal neuron in neurogenesis stage
    neuron = Neuron(x=100, y=100, z=50, neuron_type="pyramidal")

    print(f"\nInitial state:")
    print(neuron.get_state_description())

    # Simulate development
    print("\n--- Simulating neuron development (50 timesteps) ---")
    for t in range(50):
        neuron.update(dt=1.0, time=t)

        # Provide metabolic support
        neuron.receive_metabolites(glucose=0.5, oxygen=0.5)
        neuron.receive_neurotrophic_factors(bdnf=0.1, ngf=0.1)

        # Print stage transitions
        if t % 10 == 0:
            print(f"\nTimestep {t}: Stage = {neuron.stage}, Energy = {neuron.energy:.1f}")

    print(f"\nFinal state:")
    print(neuron.get_state_description())

def demo_synaptic_plasticity():
    """Demonstrate Hebbian learning"""
    print("\n" + "="*70)
    print("DEMO 3: Synaptic Plasticity (Hebbian Learning)")
    print("="*70)

    # Create two mature neurons
    pre_neuron = Neuron(x=0, y=0, neuron_type="pyramidal", energy=100)
    pre_neuron.stage = "mature"

    post_neuron = Neuron(x=50, y=0, neuron_type="pyramidal", energy=100)
    post_neuron.stage = "mature"

    # Form synapse
    pre_neuron.connect_to_neuron(post_neuron, initial_weight=0.5)

    synapse = post_neuron.synapses_in[pre_neuron.id]
    print(f"\nInitial synaptic weight: {synapse.weight:.3f}")
    print(f"Initial spine size: {synapse.spine_size:.3f}")

    # Simulate correlated activity (LTP expected)
    print("\n--- Simulating correlated activity (LTP) ---")
    for t in range(100):
        # Fire both neurons (correlated)
        if t % 5 == 0:  # Fire periodically
            pre_neuron._fire_action_potential(t)
            post_neuron._fire_action_potential(t + 0.01)  # Post slightly after pre

        pre_neuron.update(dt=0.1, time=t * 0.1)
        post_neuron.update(dt=0.1, time=t * 0.1)

        # Metabolic support
        pre_neuron.receive_metabolites(glucose=0.5, oxygen=0.5)
        post_neuron.receive_metabolites(glucose=0.5, oxygen=0.5)

    print(f"\nAfter correlated activity:")
    print(f"Final synaptic weight: {synapse.weight:.3f} (LTP expected)")
    print(f"Final spine size: {synapse.spine_size:.3f}")
    print(f"LTP events: {len(synapse.potentiation_history)}")
    print(f"LTD events: {len(synapse.depression_history)}")
    print(f"Synapse is {'stable' if synapse.is_stable else 'transient'}")

def demo_neural_environment():
    """Demonstrate full neural tissue simulation"""
    print("\n" + "="*70)
    print("DEMO 4: Neural Tissue Environment")
    print("="*70)

    # Create environment (smaller for demo)
    env = NeuralEnvironment(width=200, height=200, depth=50)

    # Initialize vasculature
    print("\n--- Initializing tissue components ---")
    env.initialize_vasculature(vessel_density=0.02)  # Increased density for demo
    env.initialize_astrocytes(astrocyte_density=0.0002)

    # Create neuronal population
    env.create_random_population(
        num_neurons=20,  # Small population for demo
        neuron_type_distribution={
            "pyramidal": 0.6,
            "granule": 0.2,
            "interneuron": 0.2
        }
    )

    # Form synapses
    env.form_random_synapses(connection_probability=0.1)

    # Run simulation
    print("\n--- Running simulation ---")
    for step in range(50):
        env.update()

        # Print statistics every 10 steps
        if step % 10 == 0:
            env.print_statistics()

    # Final statistics
    print("\n--- Final State ---")
    env.print_statistics()

    # Show some individual neurons
    print("\n--- Sample Neurons ---")
    for i, neuron in enumerate(env.neurons[:3]):  # Show first 3
        print(f"\nNeuron {i+1}:")
        print(neuron.get_state_description())

def demo_evolution_context():
    """Show evolutionary context"""
    print("\n" + "="*70)
    print("DEMO 5: Evolutionary Origin of Neurons")
    print("="*70)
    print(NEURON_EVOLUTION_CONTEXT)

def main():
    """Run all demos"""
    print("\n" + "#"*70)
    print("# Neuron Life Cycle Simulation - Comprehensive Demo")
    print("# Based on peer-reviewed neuroscience literature")
    print("#"*70)

    try:
        demo_neuron_types()
        demo_individual_neuron()
        demo_synaptic_plasticity()
        demo_neural_environment()
        demo_evolution_context()

        print("\n" + "#"*70)
        print("# All demos completed successfully!")
        print("#"*70)
        print("\nFor more information, see NEURON_BIOLOGY.md")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
