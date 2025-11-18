"""
Phase 2 Demo: Intelligent Behaviors & Data Collection
Shows organisms with food-seeking, environmental challenges, and ML data logging!
"""
from microlife.simulation.environment import Environment
from microlife.visualization.simple_renderer import SimpleRenderer
from microlife.data.logger import DataLogger


def main():
    """Run an enhanced Phase 2 simulation."""
    print("=" * 70)
    print("🦠 MICRO-LIFE ML PROJECT - PHASE 2 DEMO")
    print("=" * 70)
    print("\n✨ NEW FEATURES:")
    print("  • Intelligent food-seeking behavior")
    print("  • Temperature zones (red=hot, blue=cold)")
    print("  • Obstacles (gray blocks)")
    print("  • Data logging for ML analysis")
    print("\n" + "=" * 70)

    # Create environment with Phase 2 features enabled
    print("\n[1/6] Creating environment...")
    env = Environment(width=500, height=500, use_intelligent_movement=True)

    # Add initial organisms
    print("[2/6] Adding organisms with intelligent behavior...")
    for _ in range(15):
        env.add_organism()

    # Add environmental features
    print("[3/6] Creating environmental challenges...")

    # Add obstacles
    env.add_obstacle(x=200, y=200, width=100, height=20)  # Horizontal wall
    env.add_obstacle(x=350, y=100, width=20, height=150)  # Vertical wall
    env.add_obstacle(x=50, y=350, width=80, height=80)    # Block

    # Add temperature zones
    env.add_temperature_zone(x=100, y=100, radius=60, temperature=1)   # Hot zone
    env.add_temperature_zone(x=400, y=400, radius=70, temperature=-1)  # Cold zone

    # Add initial food
    env.spawn_food(count=40)

    # Set up data logging
    print("[4/6] Initializing data logger...")
    logger = DataLogger()

    # Create renderer
    print("[5/6] Setting up visualization...")
    renderer = SimpleRenderer(env)

    print("\n" + "=" * 70)
    print("SIMULATION STARTING")
    print("=" * 70)
    print("\n🎯 WATCH FOR:")
    print("  • Organisms actively SEEKING food (when hungry)")
    print("  • WANDERING behavior (when energy is high)")
    print("  • Organisms avoiding OBSTACLES (gray blocks)")
    print("  • Energy drain in TEMPERATURE ZONES (red/blue areas)")
    print("  • Reproduction when energy is sufficient")
    print("\n📊 DATA COLLECTION:")
    print("  • Logging every timestep for ML analysis")
    print("  • Tracking positions, behaviors, and decisions")
    print("  • Recording survival outcomes")
    print("\nClose window to end simulation and save data.\n")

    # Run animation with data logging
    print("[6/6] Running simulation...\n")
    frames_to_run = 800

    try:
        # Animation with logging callback
        import matplotlib.animation as animation

        def update_with_logging(frame):
            env.update()
            # Log every 5 timesteps to reduce data size
            if env.timestep % 5 == 0:
                logger.log_timestep(env)
            renderer.render_frame()
            return renderer.ax.patches

        anim = animation.FuncAnimation(
            renderer.fig,
            update_with_logging,
            frames=frames_to_run,
            interval=50,
            blit=False
        )
        import matplotlib.pyplot as plt
        plt.show()

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")

    # Save logged data
    print("\n" + "=" * 70)
    print("💾 SAVING DATA FOR ML ANALYSIS")
    print("=" * 70)

    logger.save_to_csv()

    # Save metadata
    metadata = {
        'simulation_type': 'Phase 2 - Intelligent Behaviors',
        'total_timesteps': env.timestep,
        'environment_size': {'width': env.width, 'height': env.height},
        'initial_organisms': 15,
        'intelligent_movement': env.use_intelligent_movement,
        'temperature_zones': len(env.temperature_zones),
        'obstacles': len(env.obstacles),
        'features': [
            'food_seeking',
            'temperature_zones',
            'obstacles',
            'reproduction',
            'energy_system'
        ]
    }
    logger.save_metadata(metadata)

    # Display summary
    print("\n" + "=" * 70)
    print("📈 FINAL STATISTICS")
    print("=" * 70)
    stats = env.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("📁 DATA LOGGING SUMMARY")
    print("=" * 70)
    summary = logger.get_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\n✅ Phase 2 Demo Complete!")
    print("\n🎓 NEXT STEPS:")
    print("  1. Analyze the logged CSV files in microlife/data/logs/")
    print("  2. Check MICROLIFE_ML_GUIDE.md for Phase 3 (ML Pattern Recognition)")
    print("  3. Use collected data for K-Means clustering and decision trees")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
