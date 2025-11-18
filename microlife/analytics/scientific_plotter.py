"""
ScientificPlotter - Publication-quality graph generation
"""
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for background saving
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any


class ScientificPlotter:
    """
    Generate publication-quality scientific plots.

    Features:
    - High-DPI output
    - Professional styling
    - Multiple plot types
    - Automatic formatting
    """

    def __init__(self, db_path: str, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize scientific plotter.

        Args:
            db_path: Path to SQLite database
            style: Matplotlib style
        """
        self.db_path = Path(db_path)

        # Set professional style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')  # Fallback

        sns.set_palette("husl")

        # Default figure settings
        self.dpi = 300
        self.figsize = (12, 8)
        self.font_size = 12

        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['axes.labelsize'] = self.font_size + 2
        plt.rcParams['axes.titlesize'] = self.font_size + 4
        plt.rcParams['xtick.labelsize'] = self.font_size
        plt.rcParams['ytick.labelsize'] = self.font_size
        plt.rcParams['legend.fontsize'] = self.font_size
        plt.rcParams['figure.titlesize'] = self.font_size + 6

    def _get_population_data(self) -> Dict[str, List]:
        """Retrieve population data from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT episode, organism_count, total_energy, avg_age, food_count
            FROM population_snapshots
            ORDER BY episode
        ''')

        episodes = []
        organism_counts = []
        total_energies = []
        avg_ages = []
        food_counts = []

        for row in cursor.fetchall():
            episodes.append(row[0])
            organism_counts.append(row[1])
            total_energies.append(row[2])
            avg_ages.append(row[3])
            food_counts.append(row[4])

        conn.close()

        return {
            'episodes': episodes,
            'organism_count': organism_counts,
            'total_energy': total_energies,
            'avg_age': avg_ages,
            'food_count': food_counts
        }

    def _get_metric_data(self, metric_name: str) -> Tuple[List, List]:
        """Retrieve metric data from database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT episode, metric_value
            FROM metrics
            WHERE metric_name = ?
            ORDER BY episode
        ''', (metric_name,))

        episodes = []
        values = []

        for row in cursor.fetchall():
            episodes.append(row[0])
            values.append(row[1])

        conn.close()

        return episodes, values

    def plot_population_dynamics(self, save_path: Optional[str] = None,
                                 show: bool = False):
        """
        Plot population dynamics over time.

        Args:
            save_path: Output file path
            show: Show plot interactively
        """
        data = self._get_population_data()

        if not data['episodes']:
            print("⚠️  No data available for population dynamics")
            return

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Population Dynamics Over Time', fontsize=16, fontweight='bold')

        # Plot 1: Organism count
        ax1 = axes[0, 0]
        ax1.plot(data['episodes'], data['organism_count'], linewidth=2, color='#3498db')
        ax1.fill_between(data['episodes'], data['organism_count'], alpha=0.3, color='#3498db')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Organism Count')
        ax1.set_title('Population Size')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Total energy
        ax2 = axes[0, 1]
        ax2.plot(data['episodes'], data['total_energy'], linewidth=2, color='#e74c3c')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Energy')
        ax2.set_title('System Energy')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Average age
        ax3 = axes[1, 0]
        ax3.plot(data['episodes'], data['avg_age'], linewidth=2, color='#2ecc71')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Age')
        ax3.set_title('Population Age')
        ax3.grid(True, alpha=0.3)

        # Plot 4: Food count
        ax4 = axes[1, 1]
        ax4.plot(data['episodes'], data['food_count'], linewidth=2, color='#f39c12')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Food Count')
        ax4.set_title('Available Resources')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved population dynamics to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_metric_timeseries(self, metric_name: str,
                               save_path: Optional[str] = None,
                               show: bool = False,
                               smooth_window: int = 20):
        """
        Plot metric time-series with smoothing.

        Args:
            metric_name: Name of metric to plot
            save_path: Output file path
            show: Show plot interactively
            smooth_window: Smoothing window size
        """
        episodes, values = self._get_metric_data(metric_name)

        if not episodes:
            print(f"⚠️  No data available for metric: {metric_name}")
            return

        # Calculate smoothed values
        values_array = np.array(values)
        smoothed = np.convolve(values_array, np.ones(smooth_window)/smooth_window, mode='valid')
        episodes_smoothed = episodes[smooth_window-1:]

        # Create plot
        plt.figure(figsize=self.figsize)

        # Raw values
        plt.plot(episodes, values, alpha=0.3, linewidth=1, label='Raw', color='#95a5a6')

        # Smoothed values
        plt.plot(episodes_smoothed, smoothed, linewidth=2, label=f'Smoothed (window={smooth_window})', color='#3498db')

        plt.xlabel('Episode')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'{metric_name.replace("_", " ").title()} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved {metric_name} timeseries to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_distribution(self, metric_name: str,
                         save_path: Optional[str] = None,
                         show: bool = False):
        """
        Plot metric distribution histogram with KDE.

        Args:
            metric_name: Metric to plot
            save_path: Output file path
            show: Show plot interactively
        """
        _, values = self._get_metric_data(metric_name)

        if not values:
            print(f"⚠️  No data available for metric: {metric_name}")
            return

        plt.figure(figsize=(10, 6))

        # Histogram with KDE
        sns.histplot(values, kde=True, bins=50, color='#3498db', alpha=0.6)

        plt.xlabel(metric_name.replace('_', ' ').title())
        plt.ylabel('Frequency')
        plt.title(f'{metric_name.replace("_", " ").title()} Distribution')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved {metric_name} distribution to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_correlation_matrix(self, metrics: List[str],
                                save_path: Optional[str] = None,
                                show: bool = False):
        """
        Plot correlation matrix heatmap.

        Args:
            metrics: List of metric names
            save_path: Output file path
            show: Show plot interactively
        """
        import pandas as pd

        # Collect data for all metrics
        data_dict = {}
        for metric in metrics:
            episodes, values = self._get_metric_data(metric)
            if values:
                data_dict[metric] = values

        if not data_dict:
            print("⚠️  No data available for correlation matrix")
            return

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Calculate correlation
        corr = df.corr()

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})

        plt.title('Metric Correlation Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved correlation matrix to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_comparative_boxplot(self, metrics: List[str],
                                 save_path: Optional[str] = None,
                                 show: bool = False):
        """
        Plot comparative box plots for multiple metrics.

        Args:
            metrics: List of metric names
            save_path: Output file path
            show: Show plot interactively
        """
        import pandas as pd

        # Collect data
        data_list = []
        for metric in metrics:
            _, values = self._get_metric_data(metric)
            if values:
                for v in values:
                    data_list.append({'Metric': metric.replace('_', ' ').title(), 'Value': v})

        if not data_list:
            print("⚠️  No data available for box plots")
            return

        df = pd.DataFrame(data_list)

        plt.figure(figsize=self.figsize)
        sns.boxplot(data=df, x='Metric', y='Value', palette='Set2')
        sns.swarmplot(data=df, x='Metric', y='Value', color='black', alpha=0.3, size=2)

        plt.xlabel('')
        plt.ylabel('Value')
        plt.title('Metric Distributions')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved box plots to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_multi_metric_comparison(self, metrics: List[str],
                                    save_path: Optional[str] = None,
                                    show: bool = False):
        """
        Plot multiple metrics on same graph (normalized).

        Args:
            metrics: List of metric names
            save_path: Output file path
            show: Show plot interactively
        """
        plt.figure(figsize=self.figsize)

        colors = sns.color_palette("husl", len(metrics))

        for idx, metric in enumerate(metrics):
            episodes, values = self._get_metric_data(metric)

            if not values:
                continue

            # Normalize to 0-1
            values_array = np.array(values)
            values_min = values_array.min()
            values_max = values_array.max()

            if values_max - values_min > 0:
                normalized = (values_array - values_min) / (values_max - values_min)
            else:
                normalized = np.zeros_like(values_array)

            plt.plot(episodes, normalized, linewidth=2,
                    label=metric.replace('_', ' ').title(),
                    color=colors[idx])

        plt.xlabel('Episode')
        plt.ylabel('Normalized Value (0-1)')
        plt.title('Multi-Metric Comparison (Normalized)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✅ Saved multi-metric comparison to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_all_plots(self, output_dir: str = 'outputs/graphs'):
        """
        Generate all available plots.

        Args:
            output_dir: Output directory for graphs
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("📊 Generating all scientific plots...")

        # Population dynamics
        self.plot_population_dynamics(
            save_path=output_path / 'population_dynamics.png'
        )

        # Get all available metrics
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT metric_name FROM metrics')
        available_metrics = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Individual metric plots
        for metric in available_metrics[:10]:  # Limit to first 10
            self.plot_metric_timeseries(
                metric,
                save_path=output_path / f'{metric}_timeseries.png'
            )

        # Multi-metric comparison
        if len(available_metrics) >= 2:
            self.plot_multi_metric_comparison(
                available_metrics[:5],  # First 5 metrics
                save_path=output_path / 'multi_metric_comparison.png'
            )

        print(f"✅ All plots generated in {output_dir}")
