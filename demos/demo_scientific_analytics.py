#!/usr/bin/env python3
"""
Scientific Analytics Demo
Automatic data collection, graphing, and reporting
"""
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from microlife.core.environment import Environment
from microlife.core.organism import Organism
from microlife.analytics.data_logger import DataLogger
from microlife.analytics.scientific_plotter import ScientificPlotter
from microlife.analytics.statistical_analyzer import StatisticalAnalyzer
from microlife.analytics.report_generator import ReportGenerator


class ScientificExperiment:
    """
    Scientific experiment with automatic data logging and analysis.
    """

    def __init__(self,
                 experiment_name: str = "experiment_001",
                 max_episodes: int = 1000,
                 auto_save_interval: int = 100):
        """
        Initialize scientific experiment.

        Args:
            experiment_name: Experiment identifier
            max_episodes: Maximum episodes to run
            auto_save_interval: Save graphs every N episodes
        """
        self.experiment_name = experiment_name
        self.max_episodes = max_episodes
        self.auto_save_interval = auto_save_interval

        print("=" * 70)
        print(f"🔬 Scientific Experiment: {experiment_name}")
        print("=" * 70)
        print()

        # Create environment
        self.environment = Environment(
            width=800,
            height=600,
            food_count=150
        )

        # Spawn initial organisms
        self._spawn_organisms(100)

        # Initialize data logger (background mode)
        self.data_logger = DataLogger(
            db_path=f'outputs/data/{experiment_name}.db',
            sampling_rate=1,  # Log every step
            buffer_size=1000,
            auto_save_graphs=True,
            graph_interval=auto_save_interval
        )

        # Start background logging
        self.data_logger.start()

        # Initialize report generator
        self.report_generator = ReportGenerator(
            db_path=f'outputs/data/{experiment_name}.db',
            output_dir='outputs/reports',
            auto_generate=True,
            interval=200  # Generate report every 200 episodes
        )

        # Statistics
        self.episode = 0
        self.start_time = time.time()

        print("✅ Experiment initialized")
        print(f"   Database: outputs/data/{experiment_name}.db")
        print(f"   Graphs:   outputs/graphs/")
        print(f"   Reports:  outputs/reports/")
        print()
        print("📊 Background data collection ACTIVE")
        print(f"   - Graphs auto-save every {auto_save_interval} episodes")
        print(f"   - Reports auto-generate every 200 episodes")
        print()

    def _spawn_organisms(self, count: int):
        """Spawn random organisms."""
        for i in range(count):
            x = np.random.uniform(50, self.environment.width - 50)
            y = np.random.uniform(50, self.environment.height - 50)

            # Mix of simple and AI brains
            brain_type = 'simple' if i % 2 == 0 else 'ai'

            organism = Organism(
                x=x,
                y=y,
                brain_type=brain_type,
                environment=self.environment
            )
            self.environment.add_organism(organism)

    def run_episode(self):
        """Run single episode."""
        # Update environment
        self.environment.update(0.016)  # ~60 Hz

        # Replenish population if needed
        if len(self.environment.organisms) < 20:
            self._spawn_organisms(10)

        # Replenish food
        if len(self.environment.food_sources) < 50:
            for _ in range(10):
                food_x = np.random.uniform(0, self.environment.width)
                food_y = np.random.uniform(0, self.environment.height)
                self.environment.add_food(food_x, food_y, value=20)

        # Log data (background thread handles saving)
        organisms = self.environment.organisms

        self.data_logger.log_step(
            episode=self.episode,
            organisms=organisms,
            environment=self.environment
        )

        # Log custom metrics
        if organisms:
            avg_energy = sum(getattr(org, 'energy', 0) for org in organisms) / len(organisms)
            ai_organisms = sum(1 for org in organisms if hasattr(org.brain, 'neural_network'))

            self.data_logger.log_metric(self.episode, 'avg_energy', avg_energy)
            self.data_logger.log_metric(self.episode, 'ai_count', ai_organisms)
            self.data_logger.log_metric(self.episode, 'simple_count', len(organisms) - ai_organisms)

        # Check for auto-report generation
        self.report_generator.check_auto_generate(self.episode)

        self.episode += 1

    def run(self):
        """Run full experiment."""
        print("🚀 Starting experiment...")
        print(f"   Episodes: {self.max_episodes}")
        print(f"   Press Ctrl+C to stop early")
        print()

        try:
            for ep in range(self.max_episodes):
                self.run_episode()

                # Print progress
                if ep % 50 == 0 and ep > 0:
                    elapsed = time.time() - self.start_time
                    eps_per_sec = ep / elapsed
                    remaining = (self.max_episodes - ep) / eps_per_sec if eps_per_sec > 0 else 0

                    print(f"Episode {ep}/{self.max_episodes} "
                          f"| Organisms: {len(self.environment.organisms):3d} "
                          f"| Speed: {eps_per_sec:.1f} ep/s "
                          f"| ETA: {remaining:.0f}s")

        except KeyboardInterrupt:
            print("\n⏸️  Experiment stopped by user")

        # Final flush
        print("\n📊 Flushing final data...")
        self.data_logger.stop()

        # Generate final analysis
        self._generate_final_analysis()

    def _generate_final_analysis(self):
        """Generate comprehensive final analysis."""
        print("\n" + "=" * 70)
        print("📊 GENERATING FINAL ANALYSIS")
        print("=" * 70)
        print()

        # Database summary
        summary = self.data_logger.get_summary()
        print("Database Summary:")
        print(f"  - Total records: {summary['total_records']}")
        print(f"  - Episodes logged: {summary['max_episode']}")
        print(f"  - Unique metrics: {summary['unique_metrics']}")
        print()

        # Generate all plots
        print("📈 Generating publication-quality graphs...")
        plotter = ScientificPlotter(summary['database_path'])
        plotter.generate_all_plots('outputs/graphs')
        print()

        # Statistical analysis
        print("📊 Running statistical analysis...")
        analyzer = StatisticalAnalyzer(summary['database_path'])

        print("\n" + analyzer.generate_report())
        print()

        # Export summary
        analyzer.export_summary_csv('outputs/reports/summary.csv')
        print()

        # Generate final report
        print("📄 Generating final HTML report...")
        self.report_generator.generate_html_report(f"{self.experiment_name}_final")
        self.report_generator.generate_text_report(f"{self.experiment_name}_final")
        print()

        # Export raw data
        print("💾 Exporting raw data to CSV...")
        self.data_logger.export_csv(f'outputs/data/{self.experiment_name}_export.csv')
        print()

        print("=" * 70)
        print("✅ EXPERIMENT COMPLETE")
        print("=" * 70)
        print()
        print("Output files:")
        print(f"  - Database:  outputs/data/{self.experiment_name}.db")
        print(f"  - Graphs:    outputs/graphs/")
        print(f"  - Reports:   outputs/reports/")
        print(f"  - CSV Data:  outputs/data/{self.experiment_name}_export.csv")
        print()


def main():
    """Run scientific experiment demo."""
    import argparse

    parser = argparse.ArgumentParser(description='Scientific Analytics Demo')
    parser.add_argument('--name', type=str, default='experiment_001',
                       help='Experiment name')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Graph save interval')

    args = parser.parse_args()

    # Create and run experiment
    experiment = ScientificExperiment(
        experiment_name=args.name,
        max_episodes=args.episodes,
        auto_save_interval=args.save_interval
    )

    experiment.run()


if __name__ == '__main__':
    main()
