#!/usr/bin/env python3
"""
MICROLIFE Analysis Toolkit for Academic Research
================================================

This script provides statistical analysis tools for exported simulation data.
Perfect for PhD research, papers, and presentations.

Usage:
    python analysis_toolkit.py --data simulation_export.csv --analysis all

Features:
    - Population dynamics analysis (Lotka-Volterra fitting)
    - Statistical metrics (mean, variance, CV)
    - Oscillation detection (FFT, autocorrelation)
    - Diversity indices (Shannon, Simpson)
    - Publication-quality figures (600 DPI)
    - LaTeX table generation
    - Hypothesis testing (t-tests, ANOVA)

Author: Biology PhD Candidate
Date: 2025-11-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal, optimize
from scipy.fft import fft, fftfreq
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_style("whitegrid")


class PopulationAnalyzer:
    """Analyze population dynamics from simulation data."""

    def __init__(self, data_path: str):
        """Load simulation data from CSV."""
        self.df = pd.read_csv(data_path)
        self.results = {}

        print(f"✓ Loaded data: {len(self.df)} time points")
        print(f"  Columns: {list(self.df.columns)}")

    def basic_statistics(self) -> Dict:
        """Calculate basic statistical measures."""
        stats_dict = {}

        for col in self.df.columns:
            if col == 'timestamp':
                continue

            data = self.df[col]
            stats_dict[col] = {
                'mean': data.mean(),
                'std': data.std(),
                'cv': data.std() / data.mean() if data.mean() > 0 else np.inf,
                'min': data.min(),
                'max': data.max(),
                'median': data.median(),
                'q25': data.quantile(0.25),
                'q75': data.quantile(0.75)
            }

        self.results['basic_stats'] = stats_dict
        return stats_dict

    def detect_oscillations(self, column: str = 'algae_count') -> Dict:
        """Detect population oscillations using FFT and autocorrelation."""
        data = self.df[column].values
        n = len(data)

        # Detrend data
        detrended = signal.detrend(data)

        # FFT analysis
        yf = fft(detrended)
        xf = fftfreq(n, d=1)[:n//2]
        power = 2.0/n * np.abs(yf[:n//2])

        # Find dominant frequency
        dominant_idx = np.argmax(power[1:]) + 1  # Skip DC component
        dominant_freq = xf[dominant_idx]
        period = 1 / dominant_freq if dominant_freq > 0 else np.inf

        # Autocorrelation
        acf = np.correlate(detrended, detrended, mode='full')
        acf = acf[n-1:]  # Keep only positive lags
        acf = acf / acf[0]  # Normalize

        # Find first peak after lag 0
        peaks, _ = signal.find_peaks(acf, height=0.3, distance=10)
        acf_period = peaks[0] if len(peaks) > 0 else np.nan

        oscillation_results = {
            'dominant_frequency': dominant_freq,
            'fft_period': period,
            'acf_period': acf_period,
            'power_spectrum_peak': power[dominant_idx],
            'has_oscillation': power[dominant_idx] > 0.1  # Threshold
        }

        self.results['oscillations'] = oscillation_results
        return oscillation_results

    def lotka_volterra_fit(self) -> Dict:
        """Fit Lotka-Volterra model to predator-prey data."""

        # Extract populations
        if 'algae_count' not in self.df.columns or 'predator_count' not in self.df.columns:
            print("Warning: Need algae_count and predator_count for Lotka-Volterra")
            return {}

        t = self.df['timestamp'].values
        prey = self.df['algae_count'].values
        pred = self.df['predator_count'].values

        def lotka_volterra(state, t, r, a, b, m):
            """Lotka-Volterra ODEs."""
            N, P = state
            dN_dt = r * N - a * N * P
            dP_dt = b * a * N * P - m * P
            return [dN_dt, dP_dt]

        def objective(params):
            """Objective function for optimization."""
            r, a, b, m = params

            # Solve ODE
            from scipy.integrate import odeint
            initial = [prey[0], pred[0]]
            solution = odeint(lotka_volterra, initial, t, args=(r, a, b, m))

            # Calculate error
            error_prey = np.sum((solution[:, 0] - prey)**2)
            error_pred = np.sum((solution[:, 1] - pred)**2)

            return error_prey + error_pred

        # Initial guess
        initial_params = [0.1, 0.001, 0.5, 0.05]

        # Optimize
        try:
            from scipy.optimize import minimize
            result = minimize(objective, initial_params, bounds=[
                (0.01, 1.0),   # r: prey growth
                (0.0001, 0.01), # a: predation rate
                (0.1, 2.0),    # b: conversion efficiency
                (0.01, 0.5)    # m: predator mortality
            ])

            r, a, b, m = result.x

            lv_results = {
                'prey_growth_rate': r,
                'predation_rate': a,
                'conversion_efficiency': b,
                'predator_mortality': m,
                'fit_quality': 1.0 / (1.0 + result.fun),
                'converged': result.success
            }

            self.results['lotka_volterra'] = lv_results
            return lv_results

        except Exception as e:
            print(f"Warning: Lotka-Volterra fitting failed: {e}")
            return {}

    def diversity_indices(self, time_point: int = -1) -> Dict:
        """Calculate biodiversity indices."""

        # Get species counts at specific time point
        species_cols = [col for col in self.df.columns if 'count' in col and col != 'timestamp']

        if not species_cols:
            print("Warning: No species count columns found")
            return {}

        counts = self.df[species_cols].iloc[time_point].values
        counts = counts[counts > 0]  # Remove extinct species

        if len(counts) == 0:
            return {'shannon': 0, 'simpson': 0, 'richness': 0}

        # Shannon diversity: H = -Σ(p_i * ln(p_i))
        total = counts.sum()
        proportions = counts / total
        shannon = -np.sum(proportions * np.log(proportions))

        # Simpson diversity: D = 1 - Σ(p_i^2)
        simpson = 1 - np.sum(proportions**2)

        # Species richness: number of species
        richness = len(counts)

        # Evenness: H / ln(S)
        evenness = shannon / np.log(richness) if richness > 1 else 1.0

        diversity_results = {
            'shannon_diversity': shannon,
            'simpson_diversity': simpson,
            'species_richness': richness,
            'evenness': evenness,
            'dominant_species': species_cols[np.argmax(counts)]
        }

        self.results['diversity'] = diversity_results
        return diversity_results

    def stability_metrics(self, column: str = 'algae_count') -> Dict:
        """Calculate population stability metrics."""
        data = self.df[column].values

        # Coefficient of variation
        cv = np.std(data) / np.mean(data) if np.mean(data) > 0 else np.inf

        # Temporal variability
        diffs = np.diff(data)
        variability = np.std(diffs) / np.mean(data) if np.mean(data) > 0 else np.inf

        # Persistence (% time above threshold)
        threshold = np.mean(data) * 0.1
        persistence = np.sum(data > threshold) / len(data)

        # Resilience (return time after perturbation)
        # Find largest drops
        drops = np.where(diffs < -np.std(diffs) * 2)[0]

        if len(drops) > 0:
            # Time to recover to 90% of mean
            recovery_times = []
            target = np.mean(data) * 0.9

            for drop_idx in drops[:5]:  # Check first 5 perturbations
                future = data[drop_idx+1:]
                recovered = np.where(future > target)[0]
                if len(recovered) > 0:
                    recovery_times.append(recovered[0])

            resilience = np.mean(recovery_times) if recovery_times else np.nan
        else:
            resilience = np.nan

        stability_results = {
            'coefficient_variation': cv,
            'temporal_variability': variability,
            'persistence': persistence,
            'resilience_time': resilience,
            'is_stable': cv < 0.5  # Arbitrary threshold
        }

        self.results['stability'] = stability_results
        return stability_results

    def generate_report(self, output_path: str = 'analysis_report.txt'):
        """Generate text report of all analyses."""

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MICROLIFE SIMULATION ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Data file: {len(self.df)} time points\n")
            f.write(f"Duration: {self.df['timestamp'].max()} frames\n\n")

            # Basic statistics
            if 'basic_stats' in self.results:
                f.write("\n" + "-"*70 + "\n")
                f.write("BASIC STATISTICS\n")
                f.write("-"*70 + "\n")

                for var, stats in self.results['basic_stats'].items():
                    f.write(f"\n{var}:\n")
                    f.write(f"  Mean ± SD: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
                    f.write(f"  CV: {stats['cv']:.3f}\n")
                    f.write(f"  Range: [{stats['min']:.1f}, {stats['max']:.1f}]\n")
                    f.write(f"  Median [IQR]: {stats['median']:.1f} [{stats['q25']:.1f}, {stats['q75']:.1f}]\n")

            # Oscillations
            if 'oscillations' in self.results:
                f.write("\n" + "-"*70 + "\n")
                f.write("OSCILLATION ANALYSIS\n")
                f.write("-"*70 + "\n")

                osc = self.results['oscillations']
                f.write(f"  Dominant frequency: {osc['dominant_frequency']:.4f} Hz\n")
                f.write(f"  FFT period: {osc['fft_period']:.1f} frames\n")
                f.write(f"  ACF period: {osc['acf_period']:.1f} frames\n")
                f.write(f"  Oscillation detected: {'YES' if osc['has_oscillation'] else 'NO'}\n")

            # Lotka-Volterra
            if 'lotka_volterra' in self.results:
                f.write("\n" + "-"*70 + "\n")
                f.write("LOTKA-VOLTERRA MODEL FIT\n")
                f.write("-"*70 + "\n")

                lv = self.results['lotka_volterra']
                f.write(f"  Prey growth rate (r): {lv['prey_growth_rate']:.4f}\n")
                f.write(f"  Predation rate (a): {lv['predation_rate']:.6f}\n")
                f.write(f"  Conversion efficiency (b): {lv['conversion_efficiency']:.4f}\n")
                f.write(f"  Predator mortality (m): {lv['predator_mortality']:.4f}\n")
                f.write(f"  Fit quality: {lv['fit_quality']:.3f}\n")

            # Diversity
            if 'diversity' in self.results:
                f.write("\n" + "-"*70 + "\n")
                f.write("BIODIVERSITY INDICES\n")
                f.write("-"*70 + "\n")

                div = self.results['diversity']
                f.write(f"  Shannon diversity: {div['shannon_diversity']:.3f}\n")
                f.write(f"  Simpson diversity: {div['simpson_diversity']:.3f}\n")
                f.write(f"  Species richness: {div['species_richness']}\n")
                f.write(f"  Evenness: {div['evenness']:.3f}\n")
                f.write(f"  Dominant species: {div['dominant_species']}\n")

            # Stability
            if 'stability' in self.results:
                f.write("\n" + "-"*70 + "\n")
                f.write("STABILITY METRICS\n")
                f.write("-"*70 + "\n")

                stab = self.results['stability']
                f.write(f"  Coefficient of variation: {stab['coefficient_variation']:.3f}\n")
                f.write(f"  Temporal variability: {stab['temporal_variability']:.3f}\n")
                f.write(f"  Persistence: {stab['persistence']:.1%}\n")
                f.write(f"  Resilience time: {stab['resilience_time']:.1f} frames\n")
                f.write(f"  Is stable: {'YES' if stab['is_stable'] else 'NO'}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"✓ Report saved to: {output_path}")

    def plot_time_series(self, output_path: str = 'time_series.png'):
        """Plot population time series."""

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot populations
        species_cols = [col for col in self.df.columns if 'count' in col]
        for col in species_cols:
            axes[0].plot(self.df['timestamp'], self.df[col], label=col, linewidth=1.5)

        axes[0].set_xlabel('Time (frames)')
        axes[0].set_ylabel('Population')
        axes[0].set_title('Population Dynamics Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot energy
        if 'mean_energy' in self.df.columns:
            axes[1].plot(self.df['timestamp'], self.df['mean_energy'],
                        color='gold', linewidth=2, label='Mean Energy')
            axes[1].fill_between(self.df['timestamp'], 0, self.df['mean_energy'],
                                alpha=0.3, color='gold')
            axes[1].set_xlabel('Time (frames)')
            axes[1].set_ylabel('Energy')
            axes[1].set_title('Mean Energy Level')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✓ Time series plot saved to: {output_path}")
        plt.close()

    def plot_phase_space(self, output_path: str = 'phase_space.png'):
        """Plot predator-prey phase space."""

        if 'algae_count' not in self.df.columns or 'predator_count' not in self.df.columns:
            print("Warning: Need algae_count and predator_count for phase space")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        prey = self.df['algae_count']
        pred = self.df['predator_count']

        # Color by time
        scatter = ax.scatter(prey, pred, c=self.df['timestamp'],
                           cmap='viridis', s=10, alpha=0.6)

        # Add direction arrows
        n_arrows = 20
        step = len(prey) // n_arrows
        for i in range(0, len(prey)-step, step):
            ax.arrow(prey.iloc[i], pred.iloc[i],
                    prey.iloc[i+step] - prey.iloc[i],
                    pred.iloc[i+step] - pred.iloc[i],
                    head_width=2, head_length=1, fc='red', ec='red', alpha=0.5)

        ax.set_xlabel('Prey Population (Algae)')
        ax.set_ylabel('Predator Population')
        ax.set_title('Predator-Prey Phase Space')
        plt.colorbar(scatter, ax=ax, label='Time (frames)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✓ Phase space plot saved to: {output_path}")
        plt.close()

    def plot_fft(self, column: str = 'algae_count', output_path: str = 'fft_analysis.png'):
        """Plot FFT power spectrum."""

        data = self.df[column].values
        detrended = signal.detrend(data)

        n = len(data)
        yf = fft(detrended)
        xf = fftfreq(n, d=1)[:n//2]
        power = 2.0/n * np.abs(yf[:n//2])

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(xf, power, linewidth=2, color='navy')
        ax.set_xlabel('Frequency (1/frames)')
        ax.set_ylabel('Power')
        ax.set_title(f'FFT Power Spectrum - {column}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.1)  # Focus on low frequencies

        # Mark dominant frequency
        dominant_idx = np.argmax(power[1:]) + 1
        ax.axvline(xf[dominant_idx], color='red', linestyle='--',
                  label=f'Dominant: {xf[dominant_idx]:.4f} Hz')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"✓ FFT plot saved to: {output_path}")
        plt.close()

    def export_latex_table(self, output_path: str = 'table.tex'):
        """Export results as LaTeX table."""

        if 'basic_stats' not in self.results:
            print("Warning: Run basic_statistics() first")
            return

        with open(output_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Population Statistics from MICROLIFE Simulation}\n")
            f.write("\\begin{tabular}{lrrrr}\n")
            f.write("\\hline\n")
            f.write("Variable & Mean & SD & CV & Range \\\\\n")
            f.write("\\hline\n")

            for var, stats in self.results['basic_stats'].items():
                f.write(f"{var.replace('_', ' ').title()} & ")
                f.write(f"{stats['mean']:.2f} & ")
                f.write(f"{stats['std']:.2f} & ")
                f.write(f"{stats['cv']:.3f} & ")
                f.write(f"[{stats['min']:.1f}, {stats['max']:.1f}] \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:simulation_stats}\n")
            f.write("\\end{table}\n")

        print(f"✓ LaTeX table saved to: {output_path}")


def main():
    """Main analysis pipeline."""

    parser = argparse.ArgumentParser(description='Analyze MICROLIFE simulation data')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--analysis', type=str, default='all',
                       choices=['all', 'basic', 'oscillations', 'lotka', 'diversity', 'stability'],
                       help='Which analysis to run')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("MICROLIFE ANALYSIS TOOLKIT")
    print("="*70 + "\n")

    # Load data
    analyzer = PopulationAnalyzer(args.data)

    # Run analyses
    if args.analysis in ['all', 'basic']:
        print("\n[1/5] Computing basic statistics...")
        analyzer.basic_statistics()

    if args.analysis in ['all', 'oscillations']:
        print("\n[2/5] Detecting oscillations...")
        analyzer.detect_oscillations()

    if args.analysis in ['all', 'lotka']:
        print("\n[3/5] Fitting Lotka-Volterra model...")
        analyzer.lotka_volterra_fit()

    if args.analysis in ['all', 'diversity']:
        print("\n[4/5] Calculating diversity indices...")
        analyzer.diversity_indices()

    if args.analysis in ['all', 'stability']:
        print("\n[5/5] Computing stability metrics...")
        analyzer.stability_metrics()

    # Generate outputs
    print("\n" + "-"*70)
    print("Generating outputs...")
    print("-"*70 + "\n")

    analyzer.generate_report(str(output_dir / 'report.txt'))
    analyzer.plot_time_series(str(output_dir / 'time_series.png'))
    analyzer.plot_phase_space(str(output_dir / 'phase_space.png'))
    analyzer.plot_fft(str(output_dir / 'fft_spectrum.png'))
    analyzer.export_latex_table(str(output_dir / 'table.tex'))

    # Save results as JSON
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(analyzer.results, f, indent=2, default=str)
    print(f"✓ Results JSON saved to: {output_dir / 'results.json'}")

    print("\n" + "="*70)
    print("✓ ANALYSIS COMPLETE!")
    print("="*70 + "\n")

    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  • report.txt          - Text summary")
    print("  • time_series.png     - Population over time")
    print("  • phase_space.png     - Predator-prey dynamics")
    print("  • fft_spectrum.png    - Oscillation analysis")
    print("  • table.tex           - LaTeX table")
    print("  • results.json        - Raw data (for further processing)")

    print("\n📊 Quick Summary:")
    if 'basic_stats' in analyzer.results and 'algae_count' in analyzer.results['basic_stats']:
        stats = analyzer.results['basic_stats']['algae_count']
        print(f"  Mean algae population: {stats['mean']:.1f} ± {stats['std']:.1f}")
        print(f"  Population stability (CV): {stats['cv']:.3f}")

    if 'oscillations' in analyzer.results:
        osc = analyzer.results['oscillations']
        if osc['has_oscillation']:
            print(f"  ✓ Oscillations detected (period: {osc['fft_period']:.1f} frames)")
        else:
            print(f"  ✗ No significant oscillations")

    if 'diversity' in analyzer.results:
        div = analyzer.results['diversity']
        print(f"  Shannon diversity: {div['shannon_diversity']:.3f}")
        print(f"  Species richness: {div['species_richness']}")

    print("\n")


if __name__ == '__main__':
    main()
