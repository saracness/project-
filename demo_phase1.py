"""
Phase 1 Demo: Basic Organism Simulation
Run this to see random-moving organisms in a 2D environment!
"""
from microlife.simulation.environment import Environment
from microlife.visualization.simple_renderer import SimpleRenderer


def main():
    """Run a basic Phase 1 simulation."""
    print("=" * 60)
    print("🦠 MICRO-LIFE ML PROJECT - PHASE 1 DEMO")
    print("=" * 60)
    print("\nInitializing environment...")

    # Create environment
    env = Environment(width=500, height=500)

    # Add initial organisms
    print("Adding organisms...")
    for _ in range(20):
        env.add_organism()

    # Add initial food
    print("Adding food particles...")
    env.spawn_food(count=30)

    # Create renderer
    print("Setting up visualization...")
    renderer = SimpleRenderer(env)

    print("\n" + "=" * 60)
    print("SIMULATION STARTING")
    print("=" * 60)
    print("\nWatch organisms (colored dots) move randomly!")
    print("- Color indicates energy level (red=low, yellow=high)")
    print("- Green dots are food particles")
    print("- Organisms consume energy as they move")
    print("- They eat food when nearby")
    print("- They reproduce when energy is high")
    print("\nClose the window to end simulation.\n")

    # Run animation
    try:
        renderer.animate(frames=1000, interval=50)
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    # Final statistics
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    stats = env.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n✅ Phase 1 Demo Complete!")
    print("Next: Check MICROLIFE_ML_GUIDE.md for Phase 2\n")


if __name__ == "__main__":
    main()
