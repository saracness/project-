"""
Organism Morphology System
Organizma Morfoloji Sistemi

Different body features that affect behavior:
- Flagella (tail) → Speed
- Cilia (short hairs) → Maneuverability
- Size → Energy consumption, perception
- Shape → Drag resistance
"""
import random
import math


class Morphology:
    """
    Defines physical characteristics of an organism.
    Organizmanın fiziksel özelliklerini tanımlar.
    """

    def __init__(self, species_name="Generic", **kwargs):
        self.species_name = species_name

        # Physical features (0.0 - 1.0 normalized)
        self.flagella_length = kwargs.get('flagella_length', 0.5)  # 0 = none, 1 = very long
        self.cilia_density = kwargs.get('cilia_density', 0.3)      # 0 = none, 1 = full coverage
        self.body_size = kwargs.get('body_size', 0.5)              # 0 = tiny, 1 = large
        self.body_shape = kwargs.get('body_shape', 'round')        # round, oval, rod

        # Color for visualization
        self.color = kwargs.get('color', self._generate_color())

        # Calculated advantages
        self._update_advantages()

    def _generate_color(self):
        """Generate random color for species."""
        colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Cyan
            '#45B7D1',  # Blue
            '#FFA07A',  # Light salmon
            '#98D8C8',  # Mint
            '#F7DC6F',  # Yellow
            '#BB8FCE',  # Purple
            '#85C1E2',  # Sky blue
        ]
        return random.choice(colors)

    def _update_advantages(self):
        """Calculate advantages based on morphology."""
        # Speed: flagella increases speed, size decreases it
        self.speed_multiplier = 1.0 + (self.flagella_length * 0.8) - (self.body_size * 0.3)
        self.speed_multiplier = max(0.5, min(2.0, self.speed_multiplier))

        # Maneuverability: cilia increases turning ability
        self.maneuverability = 1.0 + (self.cilia_density * 0.6)
        self.maneuverability = max(0.7, min(1.8, self.maneuverability))

        # Energy efficiency: larger organisms consume more
        self.energy_efficiency = 1.0 - (self.body_size * 0.4)
        self.energy_efficiency = max(0.6, min(1.2, self.energy_efficiency))

        # Perception radius: larger organisms see further
        self.perception_multiplier = 1.0 + (self.body_size * 0.5)
        self.perception_multiplier = max(0.8, min(1.5, self.perception_multiplier))

        # Visual size for rendering
        self.visual_size = 3 + (self.body_size * 7)  # 3-10 pixels

    def get_advantages_summary(self):
        """Get human-readable summary of advantages."""
        return {
            'speed': f"{self.speed_multiplier:.2f}x",
            'maneuverability': f"{self.maneuverability:.2f}x",
            'energy_efficiency': f"{self.energy_efficiency:.2f}x",
            'perception': f"{self.perception_multiplier:.2f}x"
        }

    def mutate(self, mutation_rate=0.1):
        """Slightly mutate morphology (for evolution)."""
        if random.random() < mutation_rate:
            self.flagella_length += random.uniform(-0.1, 0.1)
            self.flagella_length = max(0.0, min(1.0, self.flagella_length))

        if random.random() < mutation_rate:
            self.cilia_density += random.uniform(-0.1, 0.1)
            self.cilia_density = max(0.0, min(1.0, self.cilia_density))

        if random.random() < mutation_rate:
            self.body_size += random.uniform(-0.05, 0.05)
            self.body_size = max(0.2, min(1.0, self.body_size))

        self._update_advantages()

    def to_dict(self):
        """Export morphology data."""
        return {
            'species': self.species_name,
            'flagella_length': self.flagella_length,
            'cilia_density': self.cilia_density,
            'body_size': self.body_size,
            'body_shape': self.body_shape,
            'color': self.color,
            'speed_multiplier': self.speed_multiplier,
            'maneuverability': self.maneuverability,
            'energy_efficiency': self.energy_efficiency,
            'perception_multiplier': self.perception_multiplier
        }


# Predefined species templates
SPECIES_TEMPLATES = {
    'euglena': Morphology(
        species_name='Euglena',
        flagella_length=0.9,  # Long tail for fast swimming
        cilia_density=0.1,
        body_size=0.5,
        body_shape='oval',
        color='#2ECC71'  # Green (has chloroplasts)
    ),

    'paramecium': Morphology(
        species_name='Paramecium',
        flagella_length=0.0,  # No flagella
        cilia_density=0.95,   # Covered in cilia
        body_size=0.7,        # Relatively large
        body_shape='oval',
        color='#3498DB'  # Blue
    ),

    'amoeba': Morphology(
        species_name='Amoeba',
        flagella_length=0.0,
        cilia_density=0.0,    # No appendages
        body_size=0.6,
        body_shape='round',
        color='#E74C3C'  # Red
    ),

    'spirillum': Morphology(
        species_name='Spirillum',
        flagella_length=0.7,  # Flagella at both ends
        cilia_density=0.0,
        body_size=0.3,        # Small bacteria
        body_shape='rod',
        color='#9B59B6'  # Purple
    ),

    'vorticella': Morphology(
        species_name='Vorticella',
        flagella_length=0.2,  # Stalk (not for swimming)
        cilia_density=0.8,    # Cilia around mouth
        body_size=0.4,
        body_shape='round',
        color='#F39C12'  # Orange
    ),

    'stentor': Morphology(
        species_name='Stentor',
        flagella_length=0.0,
        cilia_density=0.9,
        body_size=0.9,        # Very large
        body_shape='oval',
        color='#1ABC9C'  # Teal
    ),

    'chlamydomonas': Morphology(
        species_name='Chlamydomonas',
        flagella_length=0.85,  # Two flagella
        cilia_density=0.0,
        body_size=0.35,        # Small
        body_shape='round',
        color='#27AE60'  # Dark green
    ),

    'volvox': Morphology(
        species_name='Volvox',
        flagella_length=0.6,   # Each cell has flagella
        cilia_density=0.0,
        body_size=0.8,         # Colonial, appears large
        body_shape='round',
        color='#16A085'  # Green-blue
    ),
}


def create_random_morphology(species_name=None):
    """Create a random morphology."""
    if species_name is None:
        species_name = f"Species_{random.randint(100, 999)}"

    return Morphology(
        species_name=species_name,
        flagella_length=random.uniform(0.0, 1.0),
        cilia_density=random.uniform(0.0, 1.0),
        body_size=random.uniform(0.3, 0.9),
        body_shape=random.choice(['round', 'oval', 'rod']),
    )


def get_species_list():
    """Get list of available predefined species."""
    return list(SPECIES_TEMPLATES.keys())


def get_species(species_name):
    """Get a species template by name."""
    return SPECIES_TEMPLATES.get(species_name.lower(), create_random_morphology(species_name))
