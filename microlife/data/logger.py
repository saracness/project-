"""
Data Logger for Phase 2
Collects and saves organism behavioral data for ML analysis
"""
import csv
import json
import os
from datetime import datetime


class DataLogger:
    """
    Logs simulation data for machine learning analysis.

    Tracks:
    - Organism positions and movements
    - Energy levels over time
    - Behavioral decisions (seeking, wandering)
    - Environmental conditions
    - Survival outcomes
    """

    def __init__(self, log_dir="microlife/data/logs"):
        """
        Initialize the data logger.

        Args:
            log_dir (str): Directory to save log files
        """
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.organism_data = []
        self.timestep_data = []
        self.survival_data = []

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

    def log_timestep(self, environment):
        """
        Log data for a single timestep.

        Args:
            environment (Environment): The environment to log
        """
        timestep = environment.timestep
        stats = environment.get_statistics()

        # Log overall environment state
        timestep_record = {
            'timestep': timestep,
            'population': stats['population'],
            'food_count': stats['food_count'],
            'avg_energy': stats['avg_energy'],
            'avg_age': stats['avg_age'],
            'seeking_count': stats.get('seeking_count', 0),
            'wandering_count': stats.get('wandering_count', 0),
            'temperature_zones': stats.get('temperature_zones', 0),
            'obstacles': stats.get('obstacles', 0)
        }
        self.timestep_data.append(timestep_record)

        # Log individual organism states
        for i, organism in enumerate(environment.organisms):
            if organism.alive:
                org_state = organism.get_state()
                org_state['timestep'] = timestep
                org_state['organism_id'] = i

                # Calculate additional features for ML
                nearest_food_dist = self._calculate_nearest_food_distance(
                    organism, environment.food_particles
                )
                org_state['nearest_food_distance'] = nearest_food_dist

                # Check if in temperature zone
                in_temp_zone = any(zone.affects(organism)
                                  for zone in environment.temperature_zones)
                org_state['in_temperature_zone'] = in_temp_zone

                self.organism_data.append(org_state)

    def log_organism_death(self, organism, timestep, cause="energy_depletion"):
        """
        Log when an organism dies for survival analysis.

        Args:
            organism (Organism): The deceased organism
            timestep (int): Timestep of death
            cause (str): Cause of death
        """
        death_record = {
            'organism_id': id(organism),
            'death_timestep': timestep,
            'final_age': organism.age,
            'final_energy': organism.energy,
            'final_speed': organism.speed,
            'cause_of_death': cause
        }
        self.survival_data.append(death_record)

    def _calculate_nearest_food_distance(self, organism, food_list):
        """
        Calculate distance to nearest food.

        Args:
            organism (Organism): The organism
            food_list (list): List of food particles

        Returns:
            float: Distance to nearest food or -1 if no food
        """
        import math

        if not food_list:
            return -1

        min_distance = float('inf')
        for food in food_list:
            if food.consumed:
                continue
            dx = organism.x - food.x
            dy = organism.y - food.y
            distance = math.sqrt(dx**2 + dy**2)
            min_distance = min(min_distance, distance)

        return min_distance if min_distance != float('inf') else -1

    def save_to_csv(self):
        """
        Save logged data to CSV files.

        Creates three files:
        - organism_logs_[session].csv: Individual organism states
        - timestep_logs_[session].csv: Environment states per timestep
        - survival_logs_[session].csv: Death records
        """
        # Save organism data
        if self.organism_data:
            organism_file = os.path.join(
                self.log_dir, f"organism_logs_{self.session_id}.csv"
            )
            keys = self.organism_data[0].keys()
            with open(organism_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.organism_data)
            print(f"✓ Saved organism data: {organism_file}")

        # Save timestep data
        if self.timestep_data:
            timestep_file = os.path.join(
                self.log_dir, f"timestep_logs_{self.session_id}.csv"
            )
            keys = self.timestep_data[0].keys()
            with open(timestep_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.timestep_data)
            print(f"✓ Saved timestep data: {timestep_file}")

        # Save survival data
        if self.survival_data:
            survival_file = os.path.join(
                self.log_dir, f"survival_logs_{self.session_id}.csv"
            )
            keys = self.survival_data[0].keys()
            with open(survival_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.survival_data)
            print(f"✓ Saved survival data: {survival_file}")

    def save_metadata(self, metadata):
        """
        Save simulation metadata (configuration, parameters).

        Args:
            metadata (dict): Metadata to save
        """
        metadata_file = os.path.join(
            self.log_dir, f"metadata_{self.session_id}.json"
        )
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")

    def get_summary(self):
        """
        Get a summary of logged data.

        Returns:
            dict: Summary statistics
        """
        return {
            'session_id': self.session_id,
            'total_organism_records': len(self.organism_data),
            'total_timesteps': len(self.timestep_data),
            'total_deaths': len(self.survival_data),
            'log_directory': self.log_dir
        }

    def clear(self):
        """Clear all logged data (for new simulation run)."""
        self.organism_data = []
        self.timestep_data = []
        self.survival_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
