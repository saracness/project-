"""
Interactive Control Panel for Micro-Life Simulation
İnteraktif Kontrol Paneli

Features:
- Real-time environment controls (temperature, food spawn)
- Spawn different species during simulation
- Select AI brain for organisms (Q-Learning, CNN, GA, etc.)
- Pause/resume
- Speed control
- Statistics display
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
import random


class ControlPanel:
    """
    Interactive control panel for simulation.
    Simülasyon için interaktif kontrol paneli.
    """

    def __init__(self, environment, renderer):
        """
        Initialize control panel.

        Args:
            environment: The simulation environment
            renderer: The visualization renderer
        """
        self.environment = environment
        self.renderer = renderer
        self.paused = False
        self.simulation_speed = 1.0

        # Control values
        self.food_spawn_rate = 5  # Every N timesteps
        self.temperature_modifier = 0.0  # -1 to +1
        self.hazard_intensity = 1.0  # 0 to 2

        # AI selection
        self.selected_ai = 'No AI'

        # Create figure with control panel
        self._create_panel()

    def _create_panel(self):
        """Create the control panel UI."""
        fig = self.renderer.fig
        fig.subplots_adjust(left=0.15, right=0.85, top=0.88, bottom=0.12)

        # Pause/Resume button
        self.pause_ax = plt.axes([0.02, 0.02, 0.08, 0.04])
        self.pause_button = Button(self.pause_ax, 'Duraklat', color='lightgray')
        self.pause_button.on_clicked(self._toggle_pause)

        # Speed slider
        self.speed_ax = plt.axes([0.12, 0.03, 0.12, 0.02])
        self.speed_slider = Slider(
            self.speed_ax, 'Hız', 0.1, 3.0,
            valinit=self.simulation_speed,
            valstep=0.1,
            color='skyblue'
        )
        self.speed_slider.on_changed(self._update_speed)

        # Food spawn rate slider
        self.food_ax = plt.axes([0.26, 0.03, 0.12, 0.02])
        self.food_slider = Slider(
            self.food_ax, 'Yemek', 1, 20,
            valinit=self.food_spawn_rate,
            valstep=1,
            color='green'
        )
        self.food_slider.on_changed(self._update_food_rate)

        # Temperature modifier slider
        self.temp_ax = plt.axes([0.40, 0.03, 0.12, 0.02])
        self.temp_slider = Slider(
            self.temp_ax, 'Sıcaklık', -1.0, 1.0,
            valinit=self.temperature_modifier,
            valstep=0.1,
            color='red'
        )
        self.temp_slider.on_changed(self._update_temperature)

        # AI Brain selector - BOTTOM RIGHT
        self.ai_selector_ax = plt.axes([0.88, 0.12, 0.11, 0.30], facecolor='#f0f0f0')
        self.ai_selector_ax.set_title('🧠 AI Seç:', fontsize=8, loc='left', pad=2, color='black')

        ai_options = [
            'AI Yok',
            'Q-Learn',
            'DQN',
            'DblDQN',
            'CNN',
            'GA',
            'NEAT',
            'CMA-ES'
        ]

        self.ai_radio = RadioButtons(
            self.ai_selector_ax,
            ai_options,
            active=0,
            activecolor='#3498DB'
        )
        for label in self.ai_radio.labels:
            label.set_fontsize(7)
            label.set_color('black')
        self.ai_radio.on_clicked(self._select_ai)

        # Species spawn buttons - BOTTOM LEFT
        button_y = 0.40
        button_height = 0.035
        button_spacing = 0.04

        species_list = [
            ('Euglena', '#2ECC71'),
            ('Paramecium', '#3498DB'),
            ('Amoeba', '#E74C3C'),
            ('Spirillum', '#9B59B6'),
            ('Stentor', '#1ABC9C'),
            ('Volvox', '#16A085'),
        ]

        self.species_buttons = []
        for i, (species, color) in enumerate(species_list):
            y_pos = button_y - (i * button_spacing)
            ax = plt.axes([0.02, y_pos, 0.10, button_height])
            btn = Button(ax, f'+ {species}', color=color, hovercolor='lightgray')
            btn.species_name = species
            btn.on_clicked(self._spawn_species)
            self.species_buttons.append(btn)

        # Random species button
        y_pos = button_y - (len(species_list) * button_spacing)
        self.random_ax = plt.axes([0.02, y_pos, 0.10, button_height])
        self.random_button = Button(
            self.random_ax, '+ Rastgele',
            color='#95A5A6',
            hovercolor='lightgray'
        )
        self.random_button.on_clicked(self._spawn_random)

        # Clear all button
        y_pos -= button_spacing
        self.clear_ax = plt.axes([0.02, y_pos, 0.10, button_height])
        self.clear_button = Button(
            self.clear_ax, 'Hepsini Sil',
            color='#C0392B',
            hovercolor='#E74C3C'
        )
        self.clear_button.on_clicked(self._clear_all)

        # Statistics text - TOP RIGHT
        self.stats_ax = plt.axes([0.88, 0.45, 0.11, 0.43], facecolor='#f0f0f0')
        self.stats_ax.axis('off')
        self.stats_text = self.stats_ax.text(
            0.05, 0.98, '',
            transform=self.stats_ax.transAxes,
            verticalalignment='top',
            fontsize=7,
            family='monospace',
            color='black'
        )

    def _toggle_pause(self, event):
        """Toggle pause/resume."""
        self.paused = not self.paused
        if self.paused:
            self.pause_button.label.set_text('Devam Et')
            self.pause_button.color = '#E74C3C'
        else:
            self.pause_button.label.set_text('Duraklat')
            self.pause_button.color = 'lightgray'
        plt.draw()

    def _update_speed(self, val):
        """Update simulation speed."""
        self.simulation_speed = val

    def _update_food_rate(self, val):
        """Update food spawn rate."""
        self.food_spawn_rate = int(val)

    def _update_temperature(self, val):
        """Update temperature modifier."""
        self.temperature_modifier = val

    def _select_ai(self, label):
        """Select AI brain type."""
        ai_map = {
            'AI Yok': 'No AI',
            'Q-Learn': 'Q-Learning',
            'DblDQN': 'DoubleDQN'
        }
        self.selected_ai = ai_map.get(label, label)
        print(f"🧠 AI seçildi: {self.selected_ai}")

    def _create_brain(self, ai_type):
        """Create AI brain instance based on selection."""
        if ai_type == 'No AI':
            return None

        try:
            if ai_type == 'Q-Learning':
                from ..ml.brain_rl import QLearningBrain
                return QLearningBrain(learning_rate=0.1, epsilon=0.3)

            elif ai_type == 'DQN':
                from ..ml.brain_rl import DQNBrain
                return DQNBrain(state_size=7, hidden_size=24)

            elif ai_type == 'DoubleDQN':
                from ..ml.brain_rl import DoubleDQNBrain
                return DoubleDQNBrain(state_size=7, hidden_size=24)

            elif ai_type == 'CNN':
                from ..ml.brain_cnn import CNNBrain
                return CNNBrain(grid_size=20)

            elif ai_type == 'GA':
                from ..ml.brain_evolutionary import GeneticAlgorithmBrain
                return GeneticAlgorithmBrain(genome_size=20)

            elif ai_type == 'NEAT':
                from ..ml.brain_evolutionary import NEATBrain
                return NEATBrain(input_size=7, output_size=4)

            elif ai_type == 'CMA-ES':
                from ..ml.brain_evolutionary import CMAESBrain
                return CMAESBrain(genome_size=20)

        except Exception as e:
            print(f"⚠️ AI brain oluşturulamadı: {e}")
            return None

        return None

    def _spawn_species(self, event):
        """Spawn a specific species with selected AI."""
        from ..simulation.morphology import get_species
        from ..simulation.organism import Organism

        # Get species from button
        species_name = event.inaxes._button.species_name

        # Create organism with this species morphology
        x = random.uniform(50, self.environment.width - 50)
        y = random.uniform(50, self.environment.height - 50)
        morphology = get_species(species_name)

        organism = Organism(x, y, energy=120, morphology=morphology)

        print(f"\n{'='*50}")
        print(f"SPAWN: {species_name}")
        print(f"Seçili AI: {self.selected_ai}")

        # Attach AI brain if selected
        if self.selected_ai != 'No AI':
            brain = self._create_brain(self.selected_ai)
            print(f"Brain oluşturuldu: {brain is not None}")
            if brain:
                organism.brain = brain
                print(f"Brain attach edildi: {hasattr(organism, 'brain')}")
                print(f"Brain tipi: {organism.brain.brain_type}")
                print(f"✨ {species_name} + {self.selected_ai} BAŞARILI!")
            else:
                print(f"⚠️ Brain oluşturulamadı!")
        else:
            print(f"AI seçili değil - normal organizma")

        self.environment.add_organism(organism)
        total = len(self.environment.organisms)
        with_brain = sum(1 for o in self.environment.organisms if hasattr(o, 'brain') and o.brain)
        print(f"Toplam organizma: {total}")
        print(f"Brain'li: {with_brain}")
        print(f"{'='*50}\n")

    def _spawn_random(self, event):
        """Spawn random organism with selected AI."""
        from ..simulation.morphology import create_random_morphology
        from ..simulation.organism import Organism

        x = random.uniform(50, self.environment.width - 50)
        y = random.uniform(50, self.environment.height - 50)
        morphology = create_random_morphology()

        organism = Organism(x, y, energy=120, morphology=morphology)

        # Attach AI brain if selected
        if self.selected_ai != 'No AI':
            brain = self._create_brain(self.selected_ai)
            if brain:
                organism.brain = brain
                print(f"✨ Random + {self.selected_ai} eklendi!")
            else:
                print(f"✨ Random organizma eklendi (AI yok)")
        else:
            print(f"✨ Random organizma eklendi")

        self.environment.add_organism(organism)

    def _clear_all(self, event):
        """Clear all organisms."""
        self.environment.organisms = []
        print("🗑️ Tüm organizmalar silindi!")

    def update_stats(self):
        """Update statistics display."""
        alive = len([o for o in self.environment.organisms if o.alive])

        # Count by species
        species_count = {}
        ai_count = {}
        ai_stats = {}

        for org in self.environment.organisms:
            if org.alive:
                species = org.morphology.species_name
                species_count[species] = species_count.get(species, 0) + 1

                # Count AI types and gather stats
                if hasattr(org, 'brain') and org.brain:
                    ai_type = org.brain.brain_type
                    ai_count[ai_type] = ai_count.get(ai_type, 0) + 1

                    if ai_type not in ai_stats:
                        ai_stats[ai_type] = {
                            'avg_reward': 0,
                            'max_survival': 0,
                            'count': 0
                        }

                    ai_stats[ai_type]['avg_reward'] += org.brain.total_reward
                    ai_stats[ai_type]['max_survival'] = max(ai_stats[ai_type]['max_survival'],
                                                            org.brain.survival_time)
                    ai_stats[ai_type]['count'] += 1

        # Average energy
        avg_energy = 0
        if alive > 0:
            avg_energy = sum(o.energy for o in self.environment.organisms if o.alive) / alive

        # Create stats text
        stats = f"T: {self.environment.timestep}\n"
        stats += f"🦠 {alive}\n"
        stats += f"⚡ {avg_energy:.0f}\n"
        stats += f"🏃 {self.simulation_speed:.1f}x\n"

        if species_count:
            stats += "\n📊 Türler:\n"
            for species, count in sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:2]:
                name = species[:8]
                stats += f"{name}: {count}\n"

        if ai_stats:
            stats += "\n🧠 AI:\n"
            for ai_type, data in list(ai_stats.items())[:2]:
                avg_r = data['avg_reward'] / data['count'] if data['count'] > 0 else 0
                stats += f"{ai_type[:7]}\n"
                stats += f"  R:{avg_r:.0f}\n"
                stats += f"  S:{data['max_survival']}\n"

        self.stats_text.set_text(stats)

    def spawn_food_if_needed(self):
        """Spawn food based on spawn rate."""
        if self.environment.timestep % self.food_spawn_rate == 0:
            self.environment.add_food(
                x=random.uniform(0, self.environment.width),
                y=random.uniform(0, self.environment.height),
                energy=20
            )

    def is_paused(self):
        """Check if simulation is paused."""
        return self.paused

    def get_speed(self):
        """Get current simulation speed."""
        return self.simulation_speed
