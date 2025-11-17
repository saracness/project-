"""
Interactive Micro-Life Simulation with Control Panel
İnteraktif Mikro-Yaşam Simülasyonu

Features / Özellikler:
- 🎮 Real-time control panel (Gerçek zamanlı kontrol paneli)
- 🦠 Spawn different species (Farklı türler ekle)
- 🧠 Select AI models (AI model seç)
- 🌡️ Control environment (Çevre kontrolü)
- 🎨 Beautiful morphology visualization (Gelişmiş görsellik)
- ⏸️ Pause/resume (Duraklat/devam)
- 🏃 Speed control (Hız kontrolü)
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

sys.path.insert(0, '.')

from microlife.simulation.environment import Environment
from microlife.simulation.organism import Organism
from microlife.simulation.morphology import get_species, SPECIES_TEMPLATES
from microlife.visualization.simple_renderer import SimpleRenderer
from microlife.visualization.interactive_panel import ControlPanel
import random


def create_starting_organisms(environment, count=10):
    """Create initial diverse population."""
    organisms = []
    species_names = list(SPECIES_TEMPLATES.keys())

    print("🌱 Creating starting population...")

    for i in range(count):
        x = random.uniform(50, environment.width - 50)
        y = random.uniform(50, environment.height - 50)
        species_name = random.choice(species_names)
        morphology = get_species(species_name)
        organism = Organism(x, y, energy=120, morphology=morphology)
        organisms.append(organism)
        environment.add_organism(organism)

        print(f"  + {species_name}: Speed {morphology.speed_multiplier:.2f}x, "
              f"Energy Eff {morphology.energy_efficiency:.2f}x")

    return organisms


def main():
    """Run interactive simulation."""
    print("=" * 70)
    print("🎮 İNTERAKTİF MİKRO-YAŞAM SİMÜLASYONU")
    print("=" * 70)
    print()
    print("Hangi ortamda simülasyon yapılsın?")
    print()
    print("1. 🌊 Göl (Lake)")
    print("2. 🦠 Bağışıklık Sistemi (Immune System)")
    print("3. 🐠 Okyanus Resifi (Ocean Reef)")
    print("4. 🌲 Orman Tabanı (Forest Floor)")
    print("5. 🌋 Volkanik Kaynak (Volcanic Vent)")
    print("6. ❄️ Kuzey Kutbu (Arctic Ice)")
    print("7. ⚪ Basit Ortam (Basic)")
    print()
    choice = input("Seçim (1-7) [Enter=Basit]: ").strip()

    if choice in ['1', '2', '3', '4', '5', '6']:
        from microlife.simulation.environment_presets import create_environment
        env_types = {'1': 'lake', '2': 'immune', '3': 'reef', '4': 'forest', '5': 'volcanic', '6': 'arctic'}
        env = create_environment(env_types[choice])
    else:
        env = Environment(width=500, height=500, use_intelligent_movement=True)
        print("⚪ Basit ortam oluşturuldu")
        for _ in range(30):
            env.add_food(x=random.uniform(0, env.width), y=random.uniform(0, env.height), energy=20)

    # Create starting organisms
    print()
    create_starting_organisms(env, count=12)

    # Create visualization
    print()
    print("🎨 Creating visualization with control panel...")
    renderer = SimpleRenderer(env)

    # Create control panel
    control_panel = ControlPanel(env, renderer)

    print()
    print("✅ Simülasyon hazır!")
    print()
    print("KONTROLLER:")
    print("  ALT: Duraklat, Hız, Yemek, Sıcaklık")
    print("  SOL: Tür butonları (Euglena, Paramecium, vb.)")
    print("  SAĞ: AI seç → Tür ekle")
    print()
    print("💡 Kuyruk=Hız, Tüyler=Manevra")
    print("=" * 70)

    # Animation update function
    def update(frame):
        if control_panel.is_paused():
            return

        speed = control_panel.get_speed()
        steps = max(1, int(speed))

        for _ in range(steps):
            env.update()
            control_panel.spawn_food_if_needed()

        renderer.render_frame()
        control_panel.update_stats()

        if env.timestep % 100 == 0:
            alive = len([o for o in env.organisms if o.alive])
            print(f"Timestep {env.timestep}: {alive} organisms alive")

    # Create animation
    anim = FuncAnimation(
        renderer.fig,
        update,
        interval=50,
        blit=False,
        cache_frame_data=False
    )

    # Show plot
    plt.show()

    print()
    print("🏁 Simulation ended!")
    print(f"   Final timestep: {env.timestep}")
    print(f"   Final population: {len([o for o in env.organisms if o.alive])}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
