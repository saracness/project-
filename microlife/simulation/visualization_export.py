"""
Export simulation data for C++ visualization

Creates JSON files that can be read by the C++ visualizer
"""

import json
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


class VisualizationExporter:
    """Export neural simulation data for visualization"""

    def __init__(self, output_dir: str = "visualization_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.frame_data = []

    def capture_frame(self, env, frame_number: int):
        """
        Capture one frame of simulation state

        Args:
            env: NeuralLearningEnvironment instance
            frame_number: Current frame number
        """
        frame = {
            "frame": frame_number,
            "time": float(env.time),
            "neurons": [],
            "synapses": [],
            "statistics": {}
        }

        # Capture neurons
        for neuron_dyn in env.neurons:
            neuron = neuron_dyn.neuron
            neuron_data = {
                "id": neuron.id,
                "position": {
                    "x": float(neuron.x),
                    "y": float(neuron.y),
                    "z": float(neuron.z)
                },
                "firing_rate": float(neuron.firing_rate),
                "energy": float(neuron.energy),
                "type": neuron.morphology.neurotransmitter + "ergic",
                "is_excitatory": bool(neuron.morphology.is_excitatory),
                "stage": neuron.stage
            }
            frame["neurons"].append(neuron_data)

        # Capture synapses
        synapse_set = set()  # Avoid duplicates
        for neuron_dyn in env.neurons:
            neuron = neuron_dyn.neuron
            for post_id, synapse in neuron.synapses_out.items():
                synapse_id = (synapse.pre_neuron_id, synapse.post_neuron_id)
                if synapse_id not in synapse_set:
                    synapse_set.add(synapse_id)
                    synapse_data = {
                        "pre_neuron_id": synapse.pre_neuron_id,
                        "post_neuron_id": synapse.post_neuron_id,
                        "weight": float(synapse.weight)
                    }
                    frame["synapses"].append(synapse_data)

        # Capture statistics
        stats = env.get_statistics()
        frame["statistics"] = {
            "accuracy": float(stats.get('recent_accuracy', 0.0)),
            "reward": float(np.mean([n.reward_signal for n in env.neurons]) if env.neurons else 0.0),
            "num_synapses": int(stats['total_synapses']),
            "avg_firing_rate": float(stats['avg_firing_rate']),
            "num_neurons": len(env.neurons)
        }

        self.frame_data.append(frame)

    def save_animation(self, filename: str = "neuron_animation.json"):
        """Save all captured frames to JSON file"""
        output_path = self.output_dir / filename

        animation_data = {
            "metadata": {
                "total_frames": len(self.frame_data),
                "format_version": "1.0"
            },
            "frames": self.frame_data
        }

        with open(output_path, 'w') as f:
            json.dump(animation_data, f, indent=2)

        print(f"✓ Saved {len(self.frame_data)} frames to {output_path}")
        return str(output_path)

    def save_current_state(self, env, filename: str = "neuron_state.json"):
        """Save just the current state"""
        output_path = self.output_dir / filename

        state = {
            "environment": {
                "width": float(env.width),
                "height": float(env.height),
                "depth": float(env.depth),
                "time": float(env.time)
            },
            "neurons": [],
            "synapses": [],
            "learning_history": {
                "time": [float(t) for t in env.learning_history['time']],
                "accuracy": [float(a) for a in env.learning_history['accuracy']],
                "reward": [float(r) for r in env.learning_history['reward']],
                "num_synapses": [int(s) for s in env.learning_history['num_synapses']],
                "avg_firing_rate": [float(f) for f in env.learning_history['avg_firing_rate']]
            }
        }

        # Save neurons
        for neuron_dyn in env.neurons:
            neuron = neuron_dyn.neuron
            neuron_data = {
                "id": neuron.id,
                "position": {
                    "x": float(neuron.x),
                    "y": float(neuron.y),
                    "z": float(neuron.z)
                },
                "velocity": {
                    "x": float(neuron_dyn.velocity_x),
                    "y": float(neuron_dyn.velocity_y),
                    "z": float(neuron_dyn.velocity_z)
                },
                "firing_rate": float(neuron.firing_rate),
                "energy": float(neuron.energy),
                "membrane_potential": float(neuron.membrane_potential),
                "type": neuron.morphology.neurotransmitter + "ergic",
                "is_excitatory": bool(neuron.morphology.is_excitatory),
                "stage": neuron.stage,
                "morphology": {
                    "dendritic_complexity": float(neuron.morphology.dendritic_arbor_complexity),
                    "spine_density": float(neuron.morphology.dendritic_spine_density),
                    "axon_length": float(neuron.morphology.axon_length),
                    "signal_speed": float(neuron.morphology.signal_speed)
                }
            }
            state["neurons"].append(neuron_data)

        # Save synapses
        synapse_set = set()
        for neuron_dyn in env.neurons:
            neuron = neuron_dyn.neuron
            for post_id, synapse in neuron.synapses_out.items():
                synapse_id = (synapse.pre_neuron_id, synapse.post_neuron_id)
                if synapse_id not in synapse_set:
                    synapse_set.add(synapse_id)
                    synapse_data = {
                        "pre_neuron_id": synapse.pre_neuron_id,
                        "post_neuron_id": synapse.post_neuron_id,
                        "weight": float(synapse.weight),
                        "spine_size": float(synapse.spine_size),
                        "is_stable": bool(synapse.is_stable)
                    }
                    state["synapses"].append(synapse_data)

        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"✓ Saved simulation state to {output_path}")
        return str(output_path)

    def export_learning_curves(self, env, filename: str = "learning_curves.csv"):
        """Export learning history as CSV"""
        output_path = self.output_dir / filename

        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Accuracy', 'Reward', 'NumSynapses', 'AvgFiringRate'])

            history = env.learning_history
            for i in range(len(history['time'])):
                writer.writerow([
                    history['time'][i],
                    history['accuracy'][i],
                    history['reward'][i],
                    history['num_synapses'][i],
                    history['avg_firing_rate'][i]
                ])

        print(f"✓ Exported learning curves to {output_path}")
        return str(output_path)
