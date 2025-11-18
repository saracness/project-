"""
Quick AI Test - Minimal Demo
Basit AI testi
"""
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species
from microlife.ml.brain_rl import QLearningBrain, DQNBrain

# Create environment
env = Environment(width=500, height=500, use_intelligent_movement=True)

# Add initial food
for _ in range(40):
    import random
    env.add_food(x=random.uniform(0, 500), y=random.uniform(0, 500), energy=20)

print("=" * 60)
print("🧪 AI TEST - Basit Karşılaştırma")
print("=" * 60)

# Add organisms WITHOUT AI
print("\n1. AI YOK - 3 Euglena ekle:")
for i in range(3):
    morph = get_species('euglena')
    org = Organism(x=100 + i*30, y=100, energy=120, morphology=morph)
    env.add_organism(org)
    print(f"   ✓ Euglena #{i+1} (AI YOK) - ID: {id(org)}")

# Add organisms WITH Q-Learning
print("\n2. Q-LEARNING - 3 Euglena ekle:")
for i in range(3):
    morph = get_species('euglena')
    org = Organism(x=100 + i*30, y=250, energy=120, morphology=morph)

    # ATTACH BRAIN
    brain = QLearningBrain(learning_rate=0.1, epsilon=0.3)
    org.brain = brain

    env.add_organism(org)
    print(f"   ✓ Euglena #{i+1} (Q-LEARNING) - ID: {id(org)}")
    print(f"      Brain attached: {hasattr(org, 'brain')}")
    print(f"      Brain type: {org.brain.brain_type if hasattr(org, 'brain') else 'NONE'}")

# Add organisms WITH DQN
print("\n3. DQN - 3 Euglena ekle:")
for i in range(3):
    morph = get_species('euglena')
    org = Organism(x=100 + i*30, y=400, energy=120, morphology=morph)

    # ATTACH BRAIN
    brain = DQNBrain(state_size=7, hidden_size=24)
    org.brain = brain

    env.add_organism(org)
    print(f"   ✓ Euglena #{i+1} (DQN) - ID: {id(org)}")
    print(f"      Brain attached: {hasattr(org, 'brain')}")
    print(f"      Brain type: {org.brain.brain_type if hasattr(org, 'brain') else 'NONE'}")

print("\n" + "=" * 60)
print(f"Toplam organizma: {len(env.organisms)}")
print("=" * 60)

# Verify brains
ai_count = sum(1 for o in env.organisms if hasattr(o, 'brain') and o.brain)
print(f"\nBrain'li organizma sayısı: {ai_count}/9")
print(f"Brain'siz organizma sayısı: {len(env.organisms) - ai_count}/9")

if ai_count != 6:
    print("\n❌ HATA: Brain attach edilmedi!")
else:
    print("\n✅ Tüm AI'lar doğru attach edildi!")

# Simple visualization
fig, ax = plt.subplots(figsize=(8, 8))

def update(frame):
    env.update()
    ax.clear()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_facecolor('#0a0a0a')
    ax.set_title(f'Timestep: {env.timestep}', color='white')

    # Draw food
    for food in env.food_particles:
        if not food.consumed:
            ax.plot(food.x, food.y, 'go', markersize=4)

    # Draw organisms
    for org in env.organisms:
        if org.alive:
            if hasattr(org, 'brain') and org.brain:
                if org.brain.brain_type == 'QLearning':
                    color = 'red'
                    marker = 's'  # square
                elif org.brain.brain_type == 'DQN':
                    color = 'cyan'
                    marker = '^'  # triangle
                else:
                    color = 'yellow'
                    marker = 'o'
            else:
                color = 'white'
                marker = 'o'

            ax.plot(org.x, org.y, marker=marker, color=color, markersize=10)

    # Stats
    alive = len([o for o in env.organisms if o.alive])
    ai_alive = len([o for o in env.organisms if o.alive and hasattr(o, 'brain') and o.brain])

    stats = f"Alive: {alive}\n"
    stats += f"AI: {ai_alive}\n"

    # AI performance
    for org in env.organisms:
        if org.alive and hasattr(org, 'brain') and org.brain:
            brain_type = org.brain.brain_type
            reward = org.brain.total_reward
            survival = org.brain.survival_time
            stats += f"\n{brain_type[:5]}:\n"
            stats += f"  R={reward:.0f}\n"
            stats += f"  S={survival}\n"

    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            verticalalignment='top', fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Legend
    legend_text = "○ Beyaz = AI Yok\n"
    legend_text += "■ Kırmızı = Q-Learning\n"
    legend_text += "▲ Cyan = DQN"
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=8, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    if frame % 50 == 0:
        print(f"T={env.timestep}: {alive} alive, {ai_alive} with AI")

print("\n🚀 Simülasyon başlıyor...")
print("   Beyaz = AI Yok")
print("   Kırmızı Kare = Q-Learning")
print("   Cyan Üçgen = DQN")
print("\nR = Total Reward, S = Survival Time")

anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()

print("\n🏁 Test tamamlandı!")
