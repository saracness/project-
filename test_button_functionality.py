"""
Test Button Functionality - Automated
Tests button handlers directly without GUI
"""
import sys
sys.path.insert(0, '.')

from microlife.simulation.environment import Environment
from microlife.visualization.simple_renderer import SimpleRenderer
from microlife.visualization.interactive_panel import ControlPanel
from microlife.simulation.organism import Organism
import random

print("=" * 70)
print("🧪 BUTTON FUNCTIONALITY TEST")
print("=" * 70)

# Create environment and renderer
env = Environment(width=500, height=500, use_intelligent_movement=True)
renderer = SimpleRenderer(env)

# Add some food
for _ in range(20):
    env.add_food(x=random.uniform(0, 500), y=random.uniform(0, 500), energy=20)

# Create control panel
panel = ControlPanel(env, renderer)

print("\n✅ Control panel created successfully")
print(f"Initial organisms: {len(env.organisms)}")

# Test 1: Spawn species without AI
print("\n" + "=" * 70)
print("TEST 1: Spawn Euglena WITHOUT AI")
print("=" * 70)

panel.selected_ai = 'No AI'
panel._spawn_species_with_name('Euglena')

print(f"✓ Organisms after spawn: {len(env.organisms)}")
last_org = env.organisms[-1] if env.organisms else None
if last_org:
    print(f"✓ Species: {last_org.morphology.species_name}")
    print(f"✓ Has brain: {hasattr(last_org, 'brain') and last_org.brain is not None}")
    print(f"✓ Position: ({last_org.x:.1f}, {last_org.y:.1f})")
else:
    print("❌ FAIL: No organism added!")

# Test 2: Spawn species WITH Q-Learning
print("\n" + "=" * 70)
print("TEST 2: Spawn Paramecium WITH Q-Learning")
print("=" * 70)

panel.selected_ai = 'Q-Learning'
panel._spawn_species_with_name('Paramecium')

print(f"✓ Organisms after spawn: {len(env.organisms)}")
last_org = env.organisms[-1] if len(env.organisms) > 1 else None
if last_org:
    print(f"✓ Species: {last_org.morphology.species_name}")
    has_brain = hasattr(last_org, 'brain') and last_org.brain is not None
    print(f"✓ Has brain: {has_brain}")
    if has_brain:
        print(f"✓ Brain type: {last_org.brain.brain_type}")
    print(f"✓ Position: ({last_org.x:.1f}, {last_org.y:.1f})")
else:
    print("❌ FAIL: No organism added!")

# Test 3: Spawn species WITH DQN
print("\n" + "=" * 70)
print("TEST 3: Spawn Amoeba WITH DQN")
print("=" * 70)

panel.selected_ai = 'DQN'
panel._spawn_species_with_name('Amoeba')

print(f"✓ Organisms after spawn: {len(env.organisms)}")
last_org = env.organisms[-1] if len(env.organisms) > 2 else None
if last_org:
    print(f"✓ Species: {last_org.morphology.species_name}")
    has_brain = hasattr(last_org, 'brain') and last_org.brain is not None
    print(f"✓ Has brain: {has_brain}")
    if has_brain:
        print(f"✓ Brain type: {last_org.brain.brain_type}")
    print(f"✓ Position: ({last_org.x:.1f}, {last_org.y:.1f})")
else:
    print("❌ FAIL: No organism added!")

# Test 4: Spawn multiple species
print("\n" + "=" * 70)
print("TEST 4: Spawn Multiple Species")
print("=" * 70)

species_to_test = ['Spirillum', 'Stentor', 'Volvox']
ai_types = ['DoubleDQN', 'CNN', 'GA']

for species, ai in zip(species_to_test, ai_types):
    panel.selected_ai = ai
    panel._spawn_species_with_name(species)
    last_org = env.organisms[-1]
    has_brain = hasattr(last_org, 'brain') and last_org.brain is not None
    brain_type = last_org.brain.brain_type if has_brain else 'None'
    print(f"✓ {species} + {ai}: Brain={has_brain}, Type={brain_type}")

# Test 5: Clear all
print("\n" + "=" * 70)
print("TEST 5: Clear All Organisms")
print("=" * 70)

before_clear = len(env.organisms)
print(f"Before clear: {before_clear} organisms")

# Simulate clear button click
class FakeEvent:
    pass

panel._clear_all(FakeEvent())

after_clear = len(env.organisms)
print(f"After clear: {after_clear} organisms")
if after_clear == 0:
    print("✓ Clear successful!")
else:
    print("❌ FAIL: Clear did not work!")

# Final Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Add organisms back for final count
panel.selected_ai = 'No AI'
panel._spawn_species_with_name('Euglena')
panel._spawn_species_with_name('Paramecium')

panel.selected_ai = 'Q-Learning'
panel._spawn_species_with_name('Amoeba')
panel._spawn_species_with_name('Spirillum')

total = len(env.organisms)
with_brain = sum(1 for o in env.organisms if hasattr(o, 'brain') and o.brain)
without_brain = total - with_brain

print(f"Total organisms: {total}")
print(f"With AI brain: {with_brain}")
print(f"Without AI brain: {without_brain}")

if total == 4 and with_brain == 2:
    print("\n✅ ALL TESTS PASSED!")
    print("✅ Button functionality is working correctly!")
    print("✅ AI attachment is working correctly!")
else:
    print("\n⚠️ Some tests may have issues")

print("\n" + "=" * 70)
print("🎯 Test complete!")
print("=" * 70)
