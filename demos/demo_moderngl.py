#!/usr/bin/env python3
"""
ModernGL Renderer Demo
Production-grade high-performance rendering demo

Target: 100+ FPS with 1000+ organisms
"""
import sys
import os
import numpy as np
import moderngl_window as mglw

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from microlife.core.environment import Environment
from microlife.core.organism import Organism
from microlife.rendering.gl_renderer import GLRenderer


class MicroLifeGLDemo(GLRenderer):
    """
    Demo application combining simulation with ModernGL renderer.
    """

    def __init__(self, **kwargs):
        # Initialize renderer first
        super().__init__(**kwargs)

        # Create simulation environment
        self.environment = Environment(
            width=800,
            height=600,
            food_count=100
        )

        # Spawn initial organisms
        self._spawn_organisms(50)

        # Simulation timing
        self.simulation_dt = 0.016  # 60 updates per second
        self.simulation_accumulator = 0.0

        print("✅ MicroLife ModernGL Demo initialized")
        print("   Controls:")
        print("   - SPACE: Pause/Resume")
        print("   - T: Toggle trails")
        print("   - P: Toggle particles")
        print("   - H: Toggle heatmap")
        print("   - G: Toggle glow effects")
        print("   - U: Toggle UI")
        print("   - R: Reset camera")
        print("   - F: Print FPS")
        print("   - Mouse Wheel: Zoom")
        print("   - Mouse Drag: Pan")
        print("   - ESC: Exit")

    def _spawn_organisms(self, count: int):
        """Spawn random organisms."""
        for i in range(count):
            x = np.random.uniform(50, self.environment.width - 50)
            y = np.random.uniform(50, self.environment.height - 50)

            # Random brain type (simple or AI)
            brain_type = 'simple' if i % 3 == 0 else 'ai'

            organism = Organism(
                x=x,
                y=y,
                brain_type=brain_type,
                environment=self.environment
            )
            self.environment.add_organism(organism)

    def render(self, time_elapsed: float, frame_time: float):
        """Override render to include simulation updates."""
        # Update simulation (fixed timestep)
        if not self.paused:
            self.simulation_accumulator += frame_time

            # Run simulation at fixed 60 Hz
            while self.simulation_accumulator >= self.simulation_dt:
                self._update_simulation()
                self.simulation_accumulator -= self.simulation_dt

        # Prepare rendering data
        simulation_data = self._prepare_render_data()
        self.update_simulation_data(simulation_data)

        # Call parent render
        super().render(time_elapsed, frame_time)

    def _update_simulation(self):
        """Update simulation state."""
        # Update environment (organisms, food, etc.)
        self.environment.update(self.simulation_dt)

        # Spawn new organisms if population is low
        if len(self.environment.organisms) < 20:
            self._spawn_organisms(5)

        # Spawn new food
        if len(self.environment.food_sources) < 50:
            for _ in range(5):
                food_x = np.random.uniform(0, self.environment.width)
                food_y = np.random.uniform(0, self.environment.height)
                self.environment.add_food(food_x, food_y, value=20)

    def _prepare_render_data(self) -> dict:
        """Prepare data for renderer."""
        organisms_data = []
        particle_events = []

        for organism in self.environment.organisms:
            # Check if organism has AI brain
            has_ai_brain = hasattr(organism.brain, 'neural_network')

            org_data = {
                'id': id(organism),
                'x': organism.x,
                'y': organism.y,
                'size': organism.size,
                'color': self._organism_color(organism),
                'glow': 1.0 if has_ai_brain else 0.0,
                'energy': organism.energy / 100.0,  # Normalized
            }
            organisms_data.append(org_data)

            # Generate particle events (eating, reproducing, dying)
            if hasattr(organism, '_just_ate') and organism._just_ate:
                particle_events.append({
                    'type': 'eat',
                    'position': (organism.x, organism.y),
                    'color': (0.2, 0.8, 0.2, 1.0)
                })
                organism._just_ate = False

            if hasattr(organism, '_just_reproduced') and organism._just_reproduced:
                particle_events.append({
                    'type': 'reproduce',
                    'position': (organism.x, organism.y),
                    'color': (0.8, 0.2, 0.8, 1.0)
                })
                organism._just_reproduced = False

        return {
            'organisms': organisms_data,
            'particle_events': particle_events,
            'food_sources': [
                {'x': food.x, 'y': food.y, 'value': food.value}
                for food in self.environment.food_sources
            ]
        }

    def _organism_color(self, organism) -> str:
        """Get organism color based on type."""
        # Check if organism has AI brain
        if hasattr(organism.brain, 'neural_network'):
            return '#00FFFF'  # Cyan for AI
        else:
            return '#00FF00'  # Green for simple

    def key_event(self, key, action, modifiers):
        """Handle additional keyboard input."""
        if action == self.wnd.keys.ACTION_PRESS:
            # O - Spawn more organisms
            if key == self.wnd.keys.O:
                self._spawn_organisms(10)
                print("➕ Spawned 10 organisms")

            # C - Clear all organisms
            elif key == self.wnd.keys.C:
                self.environment.organisms.clear()
                print("🗑️  Cleared all organisms")

            # B - Spawn organism burst
            elif key == self.wnd.keys.B:
                self._spawn_organisms(100)
                print("💥 Spawned 100 organisms!")

        # Call parent handler for other keys
        super().key_event(key, action, modifiers)


def main():
    """Run the demo."""
    print("=" * 60)
    print("MicroLife - ModernGL Renderer Demo")
    print("Production-grade high-performance rendering")
    print("=" * 60)
    print()

    # Run the demo
    mglw.run_window_config(MicroLifeGLDemo)


if __name__ == '__main__':
    main()
