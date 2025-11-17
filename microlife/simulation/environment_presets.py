"""
Environment Presets - Real-World Ecosystems
Gerçek dünya ekosistemlerini simüle eden environment'lar

1. Lake Ecosystem (Göl)
2. Animal Body / Immune System (Vücut İçi)
3. Ocean Reef (Okyanus Resifi)
4. Forest Floor (Orman Tabanı)
5. Extreme Environments (Volkan, Kuzey Kutbu, Çöl)
"""
import random
import math
from .environment import Environment, TemperatureZone, Obstacle, Food


class DynamicElement:
    """Base class for dynamic environmental elements."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update(self, timestep):
        """Update element state."""
        pass


class Current(DynamicElement):
    """Water current that pushes organisms."""

    def __init__(self, x, y, strength, direction):
        super().__init__(x, y)
        self.strength = strength  # 0.0 - 2.0
        self.direction = direction  # radians
        self.radius = 80

    def affects(self, organism):
        """Check if organism is in current."""
        dx = organism.x - self.x
        dy = organism.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        return dist <= self.radius

    def get_push(self):
        """Get push vector."""
        dx = math.cos(self.direction) * self.strength
        dy = math.sin(self.direction) * self.strength
        return (dx, dy)


class Toxin(DynamicElement):
    """Toxic area that damages organisms."""

    def __init__(self, x, y, radius, damage_rate):
        super().__init__(x, y)
        self.radius = radius
        self.damage_rate = damage_rate  # Energy loss per timestep

    def affects(self, organism):
        """Check if organism is in toxic zone."""
        dx = organism.x - self.x
        dy = organism.y - self.y
        dist = math.sqrt(dx**2 + dy**2)
        return dist <= self.radius


class Pathogen(DynamicElement):
    """Pathogen (virus/bacteria) for immune system simulation."""

    def __init__(self, x, y, pathogen_type="virus"):
        super().__init__(x, y)
        self.pathogen_type = pathogen_type
        self.health = 100
        self.alive = True
        self.replication_timer = 0

    def replicate(self):
        """Pathogen replicates."""
        if self.health > 50 and self.alive:
            offset = 10
            return Pathogen(
                self.x + random.uniform(-offset, offset),
                self.y + random.uniform(-offset, offset),
                self.pathogen_type
            )
        return None


class LakeEcosystem(Environment):
    """
    🌊 GÖL EKOSİSTEMİ

    Özellikler:
    - Su katmanları (yüzey, orta, dip)
    - Phytoplankton (producer)
    - Zooplankton (herbivore)
    - Balıklar (carnivore)
    - Oksijen seviyeleri
    - Su akıntıları
    - Sıcaklık gradyanları
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Lake Ecosystem"
        self.currents = []
        self.oxygen_zones = []
        self._setup_lake()

    def _setup_lake(self):
        """Setup lake ecosystem."""
        print("🌊 Setting up Lake Ecosystem...")

        # Water layers (temperature stratification)
        # Epilimnion (warm surface)
        self.add_temperature_zone(
            x=self.width/2, y=100,
            radius=200, temperature=0  # Warm
        )

        # Hypolimnion (cold bottom)
        self.add_temperature_zone(
            x=self.width/2, y=400,
            radius=180, temperature=-1  # Cold
        )

        # Currents
        self.currents.append(
            Current(x=100, y=250, strength=1.0, direction=0)  # Right
        )
        self.currents.append(
            Current(x=400, y=250, strength=1.0, direction=math.pi)  # Left
        )

        # Obstacles (rocks, logs)
        self.add_obstacle(x=200, y=200, width=30, height=80)  # Vertical rock
        self.add_obstacle(x=350, y=300, width=80, height=20)  # Horizontal log

        # Low oxygen zones (hypoxic zones)
        self.oxygen_zones.append(
            Toxin(x=250, y=450, radius=60, damage_rate=0.15)
        )

        # Rich phytoplankton zones (lots of food)
        for _ in range(30):
            self.add_food(
                x=random.uniform(50, 450),
                y=random.uniform(50, 200),  # Near surface
                energy=15
            )

        print("  ✓ Water layers created")
        print("  ✓ Currents added")
        print("  ✓ Hypoxic zones added")
        print("  ✓ Phytoplankton distributed")

    def update(self):
        """Update lake with currents."""
        super().update()

        # Apply currents to organisms
        for org in self.organisms:
            if org.alive:
                for current in self.currents:
                    if current.affects(org):
                        dx, dy = current.get_push()
                        org.x += dx
                        org.y += dy
                        # Keep in bounds
                        org.x = max(0, min(self.width, org.x))
                        org.y = max(0, min(self.height, org.y))

        # Apply low oxygen damage
        for org in self.organisms:
            if org.alive:
                for zone in self.oxygen_zones:
                    if zone.affects(org):
                        org.energy -= zone.damage_rate

        # Spawn phytoplankton near surface
        if self.timestep % 30 == 0:
            for _ in range(3):
                self.add_food(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, 150),
                    energy=15
                )


class ImmuneSystemEnvironment(Environment):
    """
    🦠 BAĞIŞIKLIK SİSTEMİ / VÜCUT İÇİ

    Özellikler:
    - Pathogens (virüsler, bakteriler)
    - White blood cells (beyaz kan hücreleri)
    - Antibodies
    - Organs (kalp, akciğer, karaciğer zones)
    - Blood flow (kan akışı)
    - Inflammation zones
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Immune System"
        self.pathogens = []
        self.blood_flow = []
        self.inflammation_zones = []
        self._setup_body()

    def _setup_body(self):
        """Setup immune system environment."""
        print("🦠 Setting up Immune System...")

        # Organs as zones
        # Heart (center, warm)
        self.add_temperature_zone(
            x=250, y=250,
            radius=50, temperature=1  # Hot (metabolically active)
        )

        # Lungs (upper, oxygen rich)
        self.add_obstacle(x=150, y=100, width=80, height=60)  # Left lung
        self.add_obstacle(x=270, y=100, width=80, height=60)  # Right lung

        # Liver (lower right, detox zone)
        self.add_temperature_zone(
            x=350, y=350,
            radius=70, temperature=1  # Hot (metabolic)
        )

        # Blood vessels (corridors)
        self.add_obstacle(x=240, y=0, width=20, height=150)   # Aorta
        self.add_obstacle(x=240, y=350, width=20, height=150) # Vena cava

        # Blood flow currents
        self.blood_flow.append(
            Current(x=250, y=150, strength=1.5, direction=math.pi/2)  # Down
        )
        self.blood_flow.append(
            Current(x=250, y=350, strength=1.5, direction=-math.pi/2)  # Up
        )

        # Initial pathogens (infection!)
        for _ in range(15):
            pathogen = Pathogen(
                x=random.uniform(100, 400),
                y=random.uniform(100, 400),
                pathogen_type=random.choice(["virus", "bacteria"])
            )
            self.pathogens.append(pathogen)

        # Nutrients (ATP, glucose) - energy sources
        for _ in range(40):
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                energy=10
            )

        print("  ✓ Organs created")
        print("  ✓ Blood flow established")
        print("  ✓ Pathogens introduced")
        print("  ✓ Nutrients distributed")

    def update(self):
        """Update immune system."""
        super().update()

        # Blood flow
        for org in self.organisms:
            if org.alive:
                for flow in self.blood_flow:
                    if flow.affects(org):
                        dx, dy = flow.get_push()
                        org.x += dx * 0.5  # Gentler than lake currents
                        org.y += dy * 0.5
                        org.x = max(0, min(self.width, org.x))
                        org.y = max(0, min(self.height, org.y))

        # Pathogen replication
        new_pathogens = []
        for pathogen in self.pathogens:
            if pathogen.alive:
                pathogen.replication_timer += 1
                if pathogen.replication_timer > 50:  # Replicate every 50 steps
                    offspring = pathogen.replicate()
                    if offspring:
                        new_pathogens.append(offspring)
                    pathogen.replication_timer = 0

        self.pathogens.extend(new_pathogens)

        # Pathogens as "negative food" (damage organisms that touch them)
        for org in self.organisms:
            if org.alive:
                for pathogen in self.pathogens:
                    if pathogen.alive:
                        dx = org.x - pathogen.x
                        dy = org.y - pathogen.y
                        dist = math.sqrt(dx**2 + dy**2)
                        if dist < 10:  # Contact
                            org.energy -= 0.5  # Infection damage
                            pathogen.health -= 10  # Immune response
                            if pathogen.health <= 0:
                                pathogen.alive = False

        # Remove dead pathogens
        self.pathogens = [p for p in self.pathogens if p.alive and p.health > 0]

        # Spawn nutrients near organs
        if self.timestep % 25 == 0:
            self.add_food(
                x=random.uniform(200, 300),
                y=random.uniform(200, 300),
                energy=10
            )


class OceanReef(Environment):
    """
    🐠 OKYANUS RESİFİ

    Özellikler:
    - Coral (mercan)
    - Algae (yosun)
    - Light zones (ışık katmanları)
    - Tide effects (gelgit)
    - Predator zones
    - Safe zones (hiding spots)
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Ocean Reef"
        self.tide_phase = 0
        self.light_gradient = []
        self._setup_reef()

    def _setup_reef(self):
        """Setup ocean reef."""
        print("🐠 Setting up Ocean Reef...")

        # Light zones (shallow to deep)
        # Photic zone (bright, near surface)
        self.add_temperature_zone(
            x=250, y=100,
            radius=200, temperature=0  # Warm and bright
        )

        # Aphotic zone (dark, deep)
        self.add_temperature_zone(
            x=250, y=450,
            radius=150, temperature=-1  # Cold and dark
        )

        # Coral structures (obstacles)
        self.add_obstacle(x=100, y=300, width=40, height=100)
        self.add_obstacle(x=200, y=350, width=60, height=80)
        self.add_obstacle(x=350, y=320, width=50, height=90)
        self.add_obstacle(x=420, y=380, width=30, height=70)

        # Predator zone (dangerous area)
        self.add_temperature_zone(
            x=50, y=50,
            radius=60, temperature=1  # Predator area
        )

        # Algae and plankton (food)
        for _ in range(50):
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(50, 250),  # Near surface
                energy=12
            )

        print("  ✓ Light zones created")
        print("  ✓ Coral structures placed")
        print("  ✓ Predator zone marked")
        print("  ✓ Algae distributed")

    def update(self):
        """Update reef with tide effects."""
        super().update()

        # Tide effect (periodic push)
        self.tide_phase += 0.05
        tide_strength = math.sin(self.tide_phase) * 0.3

        for org in self.organisms:
            if org.alive:
                # Tide pushes horizontally
                org.x += tide_strength
                org.x = max(0, min(self.width, org.x))

        # Spawn food in photic zone
        if self.timestep % 20 == 0:
            for _ in range(2):
                self.add_food(
                    x=random.uniform(0, self.width),
                    y=random.uniform(0, 200),
                    energy=12
                )


class ForestFloor(Environment):
    """
    🌲 ORMAN TABANI

    Özellikler:
    - Decomposing matter (çürüyen madde)
    - Fungi networks (mantar ağları)
    - Moisture zones (nemli alanlar)
    - Tree roots (obstacles)
    - Leaf litter (food source)
    - Nutrient hot spots
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Forest Floor"
        self.fungi_networks = []
        self._setup_forest()

    def _setup_forest(self):
        """Setup forest floor."""
        print("🌲 Setting up Forest Floor...")

        # Moisture zones
        # Wet areas (near water)
        self.add_temperature_zone(
            x=100, y=400,
            radius=80, temperature=-1  # Cool, moist
        )
        self.add_temperature_zone(
            x=400, y=100,
            radius=70, temperature=-1  # Cool, moist
        )

        # Dry areas (exposed)
        self.add_temperature_zone(
            x=400, y=400,
            radius=60, temperature=1  # Warm, dry
        )

        # Tree roots (obstacles)
        self.add_obstacle(x=150, y=150, width=80, height=15)
        self.add_obstacle(x=320, y=250, width=15, height=100)
        self.add_obstacle(x=200, y=380, width=120, height=20)

        # Decomposing leaves (food)
        for _ in range(60):
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                energy=8  # Lower energy than plankton
            )

        print("  ✓ Moisture zones created")
        print("  ✓ Tree roots placed")
        print("  ✓ Leaf litter distributed")

    def update(self):
        """Update forest floor."""
        super().update()

        # Decomposition: food slowly appears
        if self.timestep % 15 == 0:
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                energy=8
            )


class VolcanicVent(Environment):
    """
    🌋 VOLKANİK KAYNAK (Extreme)

    Özellikler:
    - Extreme heat zones
    - Toxic chemicals
    - Mineral-rich vents
    - Thermophiles (heat-loving organisms)
    - Unstable terrain
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Volcanic Vent"
        self.toxic_zones = []
        self._setup_volcanic()

    def _setup_volcanic(self):
        """Setup volcanic environment."""
        print("🌋 Setting up Volcanic Vent...")

        # Extreme heat zones
        self.add_temperature_zone(
            x=250, y=450,
            radius=100, temperature=2  # EXTREME heat
        )

        # Toxic zones
        self.toxic_zones.append(
            Toxin(x=250, y=450, radius=120, damage_rate=0.3)
        )

        # Mineral vents (high energy but dangerous)
        for _ in range(20):
            self.add_food(
                x=random.uniform(200, 300),
                y=random.uniform(400, 500),
                energy=30  # High energy!
            )

        # Stable zones (safe areas)
        self.add_temperature_zone(
            x=100, y=100,
            radius=80, temperature=0  # Normal
        )

        print("  ✓ Extreme heat zones created")
        print("  ✓ Toxic zones added")
        print("  ✓ Mineral vents placed")

    def update(self):
        """Update volcanic environment."""
        super().update()

        # Apply toxin damage
        for org in self.organisms:
            if org.alive:
                for toxin in self.toxic_zones:
                    if toxin.affects(org):
                        org.energy -= toxin.damage_rate

        # Spawn high-energy food near vents
        if self.timestep % 40 == 0:
            self.add_food(
                x=random.uniform(200, 300),
                y=random.uniform(400, 500),
                energy=30
            )


class ArcticIce(Environment):
    """
    ❄️ KUZEY KUTBU (Extreme)

    Özellikler:
    - Extreme cold
    - Limited food
    - Ice obstacles
    - Blizzards (periodic challenges)
    - Survival difficulty: HARD
    """

    def __init__(self, width=500, height=500):
        super().__init__(width, height, use_intelligent_movement=True)
        self.environment_type = "Arctic Ice"
        self.blizzard_active = False
        self.blizzard_timer = 0
        self._setup_arctic()

    def _setup_arctic(self):
        """Setup arctic environment."""
        print("❄️ Setting up Arctic Ice...")

        # Extreme cold everywhere
        self.add_temperature_zone(
            x=250, y=250,
            radius=300, temperature=-2  # EXTREME cold
        )

        # Ice obstacles
        self.add_obstacle(x=100, y=150, width=150, height=30)
        self.add_obstacle(x=300, y=300, width=100, height=50)
        self.add_obstacle(x=50, y=350, width=80, height=40)

        # Very limited food
        for _ in range(15):
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                energy=5  # Low energy
            )

        print("  ✓ Extreme cold zones created")
        print("  ✓ Ice obstacles placed")
        print("  ✓ Limited food distributed")

    def update(self):
        """Update arctic with blizzards."""
        super().update()

        # Blizzard mechanic
        if not self.blizzard_active:
            if random.random() < 0.01:  # 1% chance per timestep
                self.blizzard_active = True
                self.blizzard_timer = 50
                print("  ⚠️ BLIZZARD!")
        else:
            self.blizzard_timer -= 1
            # During blizzard: extra energy loss
            for org in self.organisms:
                if org.alive:
                    org.energy -= 0.2

            if self.blizzard_timer <= 0:
                self.blizzard_active = False
                print("  ✓ Blizzard ended")

        # Rarely spawn food
        if self.timestep % 50 == 0:
            self.add_food(
                x=random.uniform(0, self.width),
                y=random.uniform(0, self.height),
                energy=5
            )


# Factory function
def create_environment(environment_type="lake"):
    """
    Create environment by type.

    Args:
        environment_type (str): Type of environment
            - "lake": Lake ecosystem
            - "immune": Immune system
            - "reef": Ocean reef
            - "forest": Forest floor
            - "volcanic": Volcanic vent
            - "arctic": Arctic ice

    Returns:
        Environment: Configured environment
    """
    environments = {
        "lake": LakeEcosystem,
        "immune": ImmuneSystemEnvironment,
        "reef": OceanReef,
        "forest": ForestFloor,
        "volcanic": VolcanicVent,
        "arctic": ArcticIce
    }

    env_class = environments.get(environment_type.lower(), LakeEcosystem)
    return env_class()
