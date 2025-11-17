"""
🌍 ENVIRONMENT EXPLORER
Farklı ekosistemler demo - Hangisini keşfetmek istersin?
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from microlife.simulation.environment_presets import create_environment
from microlife.simulation.organism import Organism
from microlife.visualization.simple_renderer import SimpleRenderer
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def show_environment_menu():
    """Show environment selection menu."""
    print("=" * 70)
    print("🌍 MICRO-LIFE ENVIRONMENT EXPLORER")
    print("=" * 70)
    print("\nHangi ekosistemi keşfetmek istersin?\n")

    environments = [
        ("1", "🌊 Lake Ecosystem (Göl)", "lake",
         "Su katmanları, akıntılar, oksijen bölgeleri"),
        ("2", "🦠 Immune System (Bağışıklık Sistemi)", "immune",
         "Patojenler, kan akışı, organlar"),
        ("3", "🐠 Ocean Reef (Okyanus Resifi)", "reef",
         "Mercanlar, gelgit, ışık katmanları"),
        ("4", "🌲 Forest Floor (Orman Tabanı)", "forest",
         "Çürüyen yapraklar, nem bölgeleri, kökler"),
        ("5", "🌋 Volcanic Vent (Volkanik Kaynak)", "volcanic",
         "Aşırı sıcaklık, zehirli gazlar, mineral kaynakları"),
        ("6", "❄️  Arctic Ice (Kuzey Kutbu)", "arctic",
         "Aşırı soğuk, fırtınalar, sınırlı kaynak"),
    ]

    for num, emoji_name, code, desc in environments:
        print(f"{num}. {emoji_name}")
        print(f"   └─ {desc}\n")

    print("=" * 70)

    while True:
        choice = input("\nSeçiminiz (1-6) [veya 'q' çıkış]: ").strip()

        if choice.lower() == 'q':
            print("Çıkılıyor...")
            sys.exit(0)

        try:
            choice_num = int(choice)
            if 1 <= choice_num <= 6:
                selected = environments[choice_num - 1]
                return selected[2], selected[1]  # code, name
        except:
            pass

        print("❌ Geçersiz seçim! 1-6 arası bir sayı girin.")


def run_environment_demo(env_type, env_name):
    """Run simulation in selected environment."""
    print(f"\n{'=' * 70}")
    print(f"🚀 {env_name} Başlatılıyor...")
    print("=" * 70 + "\n")

    # Create environment
    env = create_environment(env_type)

    # Add organisms
    num_organisms = 20 if env_type != "arctic" else 10  # Fewer in arctic
    print(f"[1/3] Adding {num_organisms} organisms...")

    for _ in range(num_organisms):
        x = env.width * (0.2 + 0.6 * __import__('random').random())
        y = env.height * (0.2 + 0.6 * __import__('random').random())
        org = Organism(x, y, energy=100, speed=1.0)
        env.add_organism(org)

    # Create visualization
    print("[2/3] Setting up visualization...")
    renderer = SimpleRenderer(env)

    # Customize title
    renderer.ax.set_title(f'{env_name} Simulation',
                         color='white', fontsize=14, pad=20)

    print("[3/3] Starting simulation...")
    print("\n" + "=" * 70)
    print("🎮 SİMÜLASYON BAŞLADI!")
    print("=" * 70)
    print(f"\n📍 Environment: {env_name}")
    print(f"🦠 Organisms: {num_organisms}")
    print(f"🎯 Goal: Survive as long as possible!")

    # Environment-specific tips
    tips = {
        "lake": "💡 İpucu: Akıntılar seni iter! Düşük oksijen bölgelerinden kaçın!",
        "immune": "💡 İpucu: Kırmızı patojenlerden uzak dur! Organ bölgelerinde güvendesin!",
        "reef": "💡 İpucu: Gelgit seni iter! Yüzeye yakın bol yemek var!",
        "forest": "💡 İpucu: Nemli bölgeler daha güvenli! Çürüyen yaprakları bul!",
        "volcanic": "💡 İpucu: AŞIRI ZEHAH! Sıcak bölgelerden uzak dur ama yüksek enerjili yemek var!",
        "arctic": "💡 İpucu: ÇOK ZOR! Fırtınalara dikkat! Yemek çok az!"
    }

    if env_type in tips:
        print(f"\n{tips[env_type]}")

    print("\nClose window to end simulation.\n")

    # Run animation
    try:
        def update_frame(frame):
            env.update()

            # Special updates for immune system
            if env_type == "immune" and hasattr(env, 'pathogens'):
                # Show pathogen count
                pathogen_count = len([p for p in env.pathogens if p.alive])
                if frame % 100 == 0:
                    print(f"  🦠 Pathogens: {pathogen_count}")

            renderer.render_frame()

            # Add environment-specific info
            if env_type == "immune" and hasattr(env, 'pathogens'):
                pathogen_count = len([p for p in env.pathogens if p.alive])
                stats_text = f"Pathogens: {pathogen_count}"
                renderer.ax.text(0.02, 0.02, stats_text,
                               transform=renderer.ax.transAxes,
                               fontsize=10,
                               color='red',
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            return renderer.ax.patches

        anim = animation.FuncAnimation(
            renderer.fig,
            update_frame,
            frames=1500,
            interval=50,
            blit=False
        )

        plt.show()

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted!")

    # Final stats
    print("\n" + "=" * 70)
    print("📊 FINAL STATISTICS")
    print("=" * 70)

    stats = env.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    survivors = stats.get('population', 0)
    total = stats.get('total_organisms', 0)
    survival_rate = (survivors / total * 100) if total > 0 else 0

    print(f"\n🏆 Survival Rate: {survival_rate:.1f}%")

    if survival_rate > 50:
        print("✅ EXCELLENT! Most organisms survived!")
    elif survival_rate > 25:
        print("👍 GOOD! Some organisms adapted well!")
    elif survival_rate > 10:
        print("😅 CHALLENGING! Only the strongest survived!")
    else:
        print("💀 BRUTAL! This environment is deadly!")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main environment explorer."""
    while True:
        # Show menu
        env_type, env_name = show_environment_menu()

        # Run simulation
        run_environment_demo(env_type, env_name)

        # Ask if user wants to try another
        again = input("\nBaşka bir ortam denemek ister misin? (y/n): ").strip().lower()
        if again != 'y':
            print("\n👋 Görüşmek üzere!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program sonlandırıldı!")
