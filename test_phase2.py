"""
Quick test script for Phase 2 functionality (no visualization)
"""
from microlife.simulation.environment import Environment
from microlife.data.logger import DataLogger


def test_phase2_features():
    """Test Phase 2 features without visualization."""
    print("Testing Phase 2 Implementation...")
    print("-" * 50)

    # Create environment
    print("\n1. Creating environment with intelligent movement...")
    env = Environment(width=500, height=500, use_intelligent_movement=True)
    assert env.use_intelligent_movement == True
    print("   ✓ Environment created")

    # Add organisms
    print("\n2. Adding organisms...")
    for i in range(10):
        env.add_organism()
    assert len(env.organisms) == 10
    print(f"   ✓ Added {len(env.organisms)} organisms")

    # Test organism has Phase 2 attributes
    org = env.organisms[0]
    assert hasattr(org, 'behavior_mode')
    assert hasattr(org, 'perception_radius')
    assert hasattr(org, 'hunger_threshold')
    print("   ✓ Organisms have Phase 2 behavior attributes")

    # Add environmental features
    print("\n3. Adding environmental features...")
    env.add_obstacle(x=100, y=100, width=50, height=50)
    env.add_temperature_zone(x=200, y=200, radius=60, temperature=1)
    env.spawn_food(count=30)

    assert len(env.obstacles) == 1
    assert len(env.temperature_zones) == 1
    assert len(env.food_particles) == 30
    print(f"   ✓ Added {len(env.obstacles)} obstacle(s)")
    print(f"   ✓ Added {len(env.temperature_zones)} temperature zone(s)")
    print(f"   ✓ Added {len(env.food_particles)} food particles")

    # Test data logger
    print("\n4. Testing data logger...")
    logger = DataLogger()
    logger.log_timestep(env)
    assert len(logger.timestep_data) == 1
    assert len(logger.organism_data) == 10  # 10 organisms logged
    print("   ✓ Data logger recording correctly")

    # Run simulation for a few timesteps
    print("\n5. Running simulation for 50 timesteps...")
    for i in range(50):
        env.update()
        if i % 10 == 0:
            logger.log_timestep(env)

    print(f"   ✓ Simulation ran to timestep {env.timestep}")

    # Check statistics
    stats = env.get_statistics()
    print("\n6. Checking statistics...")
    print(f"   Population: {stats['population']}")
    print(f"   Avg Energy: {stats['avg_energy']:.1f}")
    print(f"   Seeking: {stats['seeking_count']}")
    print(f"   Wandering: {stats['wandering_count']}")
    assert 'seeking_count' in stats
    assert 'wandering_count' in stats
    print("   ✓ Phase 2 statistics tracking correctly")

    # Test intelligent movement
    print("\n7. Testing intelligent movement...")
    initial_behavior_mode = org.behavior_mode
    org.move_intelligent(env.food_particles, bounds=(env.width, env.height))
    assert org.behavior_mode in ['seeking', 'wandering']
    print(f"   ✓ Organism behavior mode: {org.behavior_mode}")

    # Test data export
    print("\n8. Testing data export...")
    logger.save_to_csv()
    summary = logger.get_summary()
    print(f"   ✓ Logged {summary['total_organism_records']} organism records")
    print(f"   ✓ Logged {summary['total_timesteps']} timestep records")
    print(f"   ✓ Session ID: {summary['session_id']}")

    print("\n" + "=" * 50)
    print("✅ ALL PHASE 2 TESTS PASSED!")
    print("=" * 50)
    print("\nPhase 2 features are working correctly:")
    print("  ✓ Intelligent food-seeking behavior")
    print("  ✓ Temperature zones")
    print("  ✓ Obstacles")
    print("  ✓ Data logging system")
    print("\nYou can now run: python demo_phase2.py")
    print("=" * 50)


if __name__ == "__main__":
    test_phase2_features()
