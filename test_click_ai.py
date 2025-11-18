"""
Click to Add AI Test
Tıklayarak AI ekle
"""
import sys
sys.path.insert(0, '.')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species
from microlife.ml.brain_rl import QLearningBrain

# Setup
env = Environment(width=500, height=500, use_intelligent_movement=True)

# Add food
import random
for _ in range(40):
    env.add_food(x=random.uniform(0, 500), y=random.uniform(0, 500), energy=20)

# Track what we're adding
current_mode = 'AI YOK'

print("=" * 60)
print("🎮 CLICK TEST - Butona tıkla, organizma ekle")
print("=" * 60)
print("\nButonlar:")
print("  1. AI YOK")
print("  2. Q-Learning")
print("=" * 60)

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_position([0.1, 0.2, 0.8, 0.7])

# Buttons
btn1_ax = plt.axes([0.1, 0.05, 0.15, 0.06])
btn1 = Button(btn1_ax, 'AI YOK', color='lightgray')

btn2_ax = plt.axes([0.3, 0.05, 0.15, 0.06])
btn2 = Button(btn2_ax, 'Q-Learning', color='#FF6B6B')

btn_add_ax = plt.axes([0.55, 0.05, 0.15, 0.06])
btn_add = Button(btn_add_ax, '+ EKLE', color='#4ECDC4')

btn_clear_ax = plt.axes([0.75, 0.05, 0.15, 0.06])
btn_clear = Button(btn_clear_ax, 'SİL', color='#C0392B')

def select_no_ai(event):
    global current_mode
    current_mode = 'AI YOK'
    btn1.color = '#4ECDC4'
    btn2.color = 'lightgray'
    print(f"\n✓ Mod: {current_mode}")

def select_qlearning(event):
    global current_mode
    current_mode = 'Q-Learning'
    btn1.color = 'lightgray'
    btn2.color = '#4ECDC4'
    print(f"\n✓ Mod: {current_mode}")

def add_organism(event):
    global current_mode
    morph = get_species('euglena')
    x = random.uniform(100, 400)
    y = random.uniform(100, 400)
    org = Organism(x=x, y=y, energy=120, morphology=morph)

    if current_mode == 'Q-Learning':
        brain = QLearningBrain(learning_rate=0.1, epsilon=0.3)
        org.brain = brain
        print(f"✨ Euglena + Q-Learning eklendi!")
        print(f"   Pozisyon: ({x:.0f}, {y:.0f})")
        print(f"   Brain var mı? {hasattr(org, 'brain')}")
        print(f"   Brain tipi: {org.brain.brain_type if hasattr(org, 'brain') else 'YOK'}")
    else:
        print(f"✨ Euglena (AI YOK) eklendi!")
        print(f"   Pozisyon: ({x:.0f}, {y:.0f})")
        print(f"   Brain var mı? {hasattr(org, 'brain')}")

    env.add_organism(org)
    print(f"   Toplam: {len(env.organisms)}")

def clear_all(event):
    env.organisms = []
    print("\n🗑️ Tüm organizmalar silindi!")

btn1.on_clicked(select_no_ai)
btn2.on_clicked(select_qlearning)
btn_add.on_clicked(add_organism)
btn_clear.on_clicked(clear_all)

# Animation
paused = False

def update(frame):
    if not paused:
        env.update()

    ax.clear()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.set_facecolor('#0a0a0a')
    ax.set_title(f'Timestep: {env.timestep} | Mod: {current_mode}', color='white', fontsize=14)

    # Food
    for food in env.food_particles:
        if not food.consumed:
            ax.plot(food.x, food.y, 'go', markersize=4, alpha=0.6)

    # Organisms
    for org in env.organisms:
        if org.alive:
            if hasattr(org, 'brain') and org.brain:
                color = '#FF6B6B'  # Red for Q-Learning
                marker = 's'
                size = 12
            else:
                color = 'white'
                marker = 'o'
                size = 8

            ax.plot(org.x, org.y, marker=marker, color=color, markersize=size)

    # Stats
    total = len([o for o in env.organisms if o.alive])
    with_ai = len([o for o in env.organisms if o.alive and hasattr(o, 'brain') and o.brain])
    without_ai = total - with_ai

    stats = f"Toplam: {total}\n"
    stats += f"AI Yok: {without_ai}\n"
    stats += f"Q-Learn: {with_ai}\n\n"

    # AI stats
    ai_organisms = [o for o in env.organisms if o.alive and hasattr(o, 'brain') and o.brain]
    if ai_organisms:
        avg_reward = sum(o.brain.total_reward for o in ai_organisms) / len(ai_organisms)
        max_survival = max(o.brain.survival_time for o in ai_organisms)
        stats += f"AI Stats:\n"
        stats += f"  Avg R: {avg_reward:.0f}\n"
        stats += f"  Max S: {max_survival}\n"

    ax.text(0.02, 0.98, stats, transform=ax.transAxes,
            verticalalignment='top', fontsize=10, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

    # Instructions
    instructions = "1. Mod seç (AI YOK / Q-Learning)\n"
    instructions += "2. + EKLE butonuna tıkla\n"
    instructions += "3. İzle!\n\n"
    instructions += "○ Beyaz = AI Yok\n"
    instructions += "■ Kırmızı = Q-Learning\n\n"
    instructions += "R = Reward, S = Survival"

    ax.text(0.98, 0.98, instructions, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            fontsize=9, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

print("\n✅ Hazır! Butonlara tıkla:")
print("   1. Mod seç (AI YOK veya Q-Learning)")
print("   2. + EKLE'ye tıkla")
print("   3. Organizmaların davranışını izle!")
print("\nQ-Learning'li olanlar KIRMIZI KARE olarak görünür")

anim = FuncAnimation(fig, update, interval=50, cache_frame_data=False)
plt.show()

print("\n🏁 Test bitti!")
