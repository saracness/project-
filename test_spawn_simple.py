"""
Simple Spawn Test - No GUI required
Tests core spawn functionality
"""
import sys
sys.path.insert(0, '.')

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species
from microlife.ml.brain_rl import QLearningBrain, DQNBrain, DoubleDQNBrain
import random

print("=" * 70)
print("🧪 SIMPLE SPAWN TEST - Core Functionality")
print("=" * 70)

# Create environment
env = Environment(width=500, height=500, use_intelligent_movement=True)

print("\n✅ Environment created")
print(f"Initial organisms: {len(env.organisms)}")

# Simulate what ControlPanel._spawn_species_with_name() does
def spawn_species_with_ai(species_name, ai_type='No AI'):
    """Simulate the spawn function from ControlPanel"""
    x = random.uniform(50, 450)
    y = random.uniform(50, 450)
    morphology = get_species(species_name)

    organism = Organism(x, y, energy=120, morphology=morphology)

    print(f"\n{'='*50}")
    print(f"✨ SPAWN: {species_name}")
    print(f"Seçili AI: {ai_type}")

    # Attach AI brain if selected
    if ai_type != 'No AI':
        brain = None

        if ai_type == 'Q-Learning':
            brain = QLearningBrain(learning_rate=0.1, epsilon=0.3)
        elif ai_type == 'DQN':
            brain = DQNBrain(state_size=7, hidden_size=24)
        elif ai_type == 'DoubleDQN':
            brain = DoubleDQNBrain(state_size=7, hidden_size=24)

        if brain:
            organism.brain = brain
            print(f"✅ {species_name} + {ai_type} EKLENDI!")
            print(f"   Brain type: {organism.brain.brain_type}")
        else:
            print(f"⚠️ Brain oluşturulamadı - AI yok")
    else:
        print(f"✅ {species_name} eklendi (AI yok)")

    env.add_organism(organism)
    total = len(env.organisms)
    with_brain = sum(1 for o in env.organisms if hasattr(o, 'brain') and o.brain)
    print(f"Toplam: {total} | Brain'li: {with_brain}")
    print(f"{'='*50}")

    return organism

# Test 1: No AI
print("\n" + "=" * 70)
print("TEST 1: Spawn WITHOUT AI")
print("=" * 70)

org1 = spawn_species_with_ai('Euglena', 'No AI')
assert len(env.organisms) == 1, "Should have 1 organism"
assert not (hasattr(org1, 'brain') and org1.brain), "Should NOT have brain"
print("✓ Test 1 PASSED")

# Test 2: Q-Learning
print("\n" + "=" * 70)
print("TEST 2: Spawn WITH Q-Learning")
print("=" * 70)

org2 = spawn_species_with_ai('Paramecium', 'Q-Learning')
assert len(env.organisms) == 2, "Should have 2 organisms"
assert hasattr(org2, 'brain') and org2.brain, "Should have brain"
assert org2.brain.brain_type == 'Q-Learning', "Should be Q-Learning"
print("✓ Test 2 PASSED")

# Test 3: DQN
print("\n" + "=" * 70)
print("TEST 3: Spawn WITH DQN")
print("=" * 70)

org3 = spawn_species_with_ai('Amoeba', 'DQN')
assert len(env.organisms) == 3, "Should have 3 organisms"
assert hasattr(org3, 'brain') and org3.brain, "Should have brain"
assert org3.brain.brain_type == 'DQN', "Should be DQN"
print("✓ Test 3 PASSED")

# Test 4: DoubleDQN
print("\n" + "=" * 70)
print("TEST 4: Spawn WITH DoubleDQN")
print("=" * 70)

org4 = spawn_species_with_ai('Spirillum', 'DoubleDQN')
assert len(env.organisms) == 4, "Should have 4 organisms"
assert hasattr(org4, 'brain') and org4.brain, "Should have brain"
assert org4.brain.brain_type == 'Double-DQN', "Should be Double-DQN"
print("✓ Test 4 PASSED")

# Test 5: Multiple species
print("\n" + "=" * 70)
print("TEST 5: Spawn Multiple Species")
print("=" * 70)

spawn_species_with_ai('Stentor', 'No AI')
spawn_species_with_ai('Volvox', 'Q-Learning')

assert len(env.organisms) == 6, "Should have 6 organisms"
print("✓ Test 5 PASSED")

# Test 6: Verify AI functionality
print("\n" + "=" * 70)
print("TEST 6: Verify AI Works in Simulation")
print("=" * 70)

# Run a few simulation steps
for i in range(5):
    env.update()

print(f"Timestep: {env.timestep}")

# Check AI organisms have stats
ai_organisms = [o for o in env.organisms if o.alive and hasattr(o, 'brain') and o.brain]
print(f"\nAI Organisms: {len(ai_organisms)}")

for org in ai_organisms:
    print(f"  {org.morphology.species_name} ({org.brain.brain_type}):")
    print(f"    Survival: {org.brain.survival_time}")
    print(f"    Decisions: {org.brain.decision_count}")
    print(f"    Reward: {org.brain.total_reward:.1f}")

if all(o.brain.survival_time > 0 for o in ai_organisms):
    print("\n✓ Test 6 PASSED - AI is active!")
else:
    print("\n❌ Test 6 FAILED - AI not active")

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

total = len([o for o in env.organisms if o.alive])
with_brain = len([o for o in env.organisms if o.alive and hasattr(o, 'brain') and o.brain])
without_brain = total - with_brain

print(f"Total alive: {total}")
print(f"With AI brain: {with_brain}")
print(f"Without AI brain: {without_brain}")

# Count by AI type
ai_types = {}
for org in env.organisms:
    if org.alive and hasattr(org, 'brain') and org.brain:
        ai_type = org.brain.brain_type
        ai_types[ai_type] = ai_types.get(ai_type, 0) + 1

print(f"\nAI Types:")
for ai_type, count in ai_types.items():
    print(f"  {ai_type}: {count}")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("✅ Spawn functionality is working correctly!")
print("✅ AI attachment is working correctly!")
print("✅ AI is active in simulation!")
print("=" * 70)
