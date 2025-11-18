#!/usr/bin/env python3
"""
Batch Experiment Runner for MICROLIFE Simulations
=================================================

Run multiple simulation replicates with different parameters.
Essential for PhD research requiring statistical power.

Usage:
    # Run 100 replicates with same config
    python batch_runner.py --config experiment_config.yaml --replicates 100

    # Parameter sweep: test different mutation rates
    python batch_runner.py --config base.yaml --sweep mutation_rate 0.05,0.10,0.15,0.20

    # Parallel execution (use all CPU cores)
    python batch_runner.py --config experiment_config.yaml --replicates 100 --parallel 8

Features:
    - Automated parameter sweeps
    - Parallel execution (multiprocessing)
    - Progress tracking with ETA
    - Automatic data aggregation
    - Statistical analysis of ensemble results
    - Publication-ready summary tables

Author: Biology PhD Candidate
Date: 2025-11-18
"""

import argparse
import subprocess
import multiprocessing as mp
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
from datetime import timedelta
import json
from itertools import product
import shutil

class BatchRunner:
    """Manage batch simulation experiments."""

    def __init__(self, config_path: str, output_dir: str = "batch_results"):
        """Initialize batch runner."""
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load base configuration
        with open(config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

        self.results = []
        self.failed_runs = []

        print(f"✓ Loaded configuration: {config_path}")
        print(f"✓ Output directory: {output_dir}")

    def generate_config(self, replicate_id: int, param_overrides: Dict = None) -> Path:
        """Generate a configuration file for a single replicate."""

        config = self.base_config.copy()

        # Update random seed for this replicate
        if config.get('random_seed', -1) != -1:
            config['random_seed'] = self.base_config['random_seed'] + replicate_id

        # Apply parameter overrides
        if param_overrides:
            for key, value in param_overrides.items():
                # Handle nested keys (e.g., "mutation.efficiency_mutation")
                keys = key.split('.')
                target = config
                for k in keys[:-1]:
                    target = target.setdefault(k, {})
                target[keys[-1]] = value

        # Update export paths
        replicate_name = f"replicate_{replicate_id:04d}"
        if param_overrides:
            # Add parameter values to name
            param_str = "_".join([f"{k}={v}" for k, v in param_overrides.items()])
            replicate_name = f"{replicate_name}_{param_str}"

        replicate_dir = self.output_dir / replicate_name
        replicate_dir.mkdir(exist_ok=True)

        config['export']['output_directory'] = str(replicate_dir)
        config['simulation']['headless'] = True  # No visualization for batch

        # Save config
        config_file = replicate_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        return config_file

    def run_single_replicate(self, args: tuple) -> Dict[str, Any]:
        """Run a single simulation replicate."""

        replicate_id, config_file, executable = args

        start_time = time.time()

        try:
            # Run simulation
            result = subprocess.run(
                [executable, '--config', str(config_file)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            runtime = time.time() - start_time

            if result.returncode == 0:
                # Success! Load and parse results
                replicate_dir = config_file.parent

                # Load time series data
                timeseries_file = replicate_dir / "population_timeseries.csv"
                if timeseries_file.exists():
                    df = pd.read_csv(timeseries_file)

                    # Calculate summary statistics
                    summary = {
                        'replicate_id': replicate_id,
                        'success': True,
                        'runtime_seconds': runtime,
                        'final_population': df.iloc[-1].sum() if len(df) > 0 else 0,
                        'mean_population': df.mean().sum() if len(df) > 0 else 0,
                        'population_cv': df.std().sum() / df.mean().sum() if len(df) > 0 else np.nan,
                        'extinction': df.iloc[-1].sum() == 0 if len(df) > 0 else True,
                        'config_file': str(config_file)
                    }

                    # Species-specific stats
                    for col in df.columns:
                        if 'count' in col:
                            summary[f'{col}_mean'] = df[col].mean()
                            summary[f'{col}_std'] = df[col].std()
                            summary[f'{col}_final'] = df[col].iloc[-1]

                    return summary
                else:
                    # No data file - simulation ran but didn't export
                    return {
                        'replicate_id': replicate_id,
                        'success': False,
                        'error': 'No data file generated',
                        'runtime_seconds': runtime
                    }
            else:
                # Simulation failed
                return {
                    'replicate_id': replicate_id,
                    'success': False,
                    'error': result.stderr[-500:] if result.stderr else 'Unknown error',
                    'runtime_seconds': runtime
                }

        except subprocess.TimeoutExpired:
            return {
                'replicate_id': replicate_id,
                'success': False,
                'error': 'Timeout (>10 minutes)',
                'runtime_seconds': time.time() - start_time
            }

        except Exception as e:
            return {
                'replicate_id': replicate_id,
                'success': False,
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }

    def run_batch(self, n_replicates: int, param_overrides: Dict = None,
                  executable: str = './MICROLIFE_ULTIMATE', n_parallel: int = 1):
        """Run batch of simulations."""

        print(f"\n{'='*70}")
        print(f"BATCH EXPERIMENT: {n_replicates} replicates")
        if param_overrides:
            print(f"Parameter overrides: {param_overrides}")
        print(f"Parallel workers: {n_parallel}")
        print(f"{'='*70}\n")

        # Generate all config files
        print("Generating configuration files...")
        configs = []
        for i in range(n_replicates):
            config_file = self.generate_config(i, param_overrides)
            configs.append((i, config_file, executable))

        print(f"✓ Generated {len(configs)} configurations\n")

        # Run simulations
        start_time = time.time()

        if n_parallel > 1:
            # Parallel execution
            print(f"Running {n_replicates} simulations in parallel ({n_parallel} workers)...")

            with mp.Pool(processes=n_parallel) as pool:
                # Use imap for progress tracking
                results = []
                for i, result in enumerate(pool.imap(self.run_single_replicate, configs)):
                    results.append(result)

                    # Progress update
                    completed = i + 1
                    elapsed = time.time() - start_time
                    eta = (elapsed / completed) * (n_replicates - completed)

                    print(f"  [{completed}/{n_replicates}] "
                          f"Success: {result['success']} | "
                          f"Runtime: {result['runtime_seconds']:.1f}s | "
                          f"ETA: {timedelta(seconds=int(eta))}", flush=True)

                    if not result['success']:
                        self.failed_runs.append(result)
        else:
            # Sequential execution
            print(f"Running {n_replicates} simulations sequentially...")

            results = []
            for i, config in enumerate(configs):
                result = self.run_single_replicate(config)
                results.append(result)

                completed = i + 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (n_replicates - completed)

                print(f"  [{completed}/{n_replicates}] "
                      f"Success: {result['success']} | "
                      f"Runtime: {result['runtime_seconds']:.1f}s | "
                      f"ETA: {timedelta(seconds=int(eta))}", flush=True)

                if not result['success']:
                    self.failed_runs.append(result)

        total_time = time.time() - start_time

        # Store results
        self.results.extend(results)

        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n{'='*70}")
        print(f"BATCH COMPLETE!")
        print(f"{'='*70}")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Successful runs: {successful}/{n_replicates} ({successful/n_replicates*100:.1f}%)")
        print(f"Failed runs: {len(self.failed_runs)}")
        print(f"Average runtime: {np.mean([r['runtime_seconds'] for r in results]):.1f}s")
        print(f"{'='*70}\n")

        return results

    def parameter_sweep(self, param_name: str, param_values: List[Any],
                       replicates_per_value: int = 10,
                       executable: str = './MICROLIFE_ULTIMATE',
                       n_parallel: int = 1):
        """Run parameter sweep across multiple values."""

        print(f"\n{'='*70}")
        print(f"PARAMETER SWEEP")
        print(f"{'='*70}")
        print(f"Parameter: {param_name}")
        print(f"Values: {param_values}")
        print(f"Replicates per value: {replicates_per_value}")
        print(f"Total simulations: {len(param_values) * replicates_per_value}")
        print(f"{'='*70}\n")

        sweep_results = {}

        for value in param_values:
            print(f"\n--- Testing {param_name} = {value} ---\n")

            param_overrides = {param_name: value}
            results = self.run_batch(
                replicates_per_value,
                param_overrides=param_overrides,
                executable=executable,
                n_parallel=n_parallel
            )

            sweep_results[value] = results

        # Analyze sweep results
        self.analyze_sweep(param_name, sweep_results)

        return sweep_results

    def multi_parameter_sweep(self, param_dict: Dict[str, List[Any]],
                             replicates_per_combination: int = 10,
                             executable: str = './MICROLIFE_ULTIMATE',
                             n_parallel: int = 1):
        """Run sweep across multiple parameters (full factorial design)."""

        # Generate all combinations
        param_names = list(param_dict.keys())
        param_value_lists = list(param_dict.values())
        combinations = list(product(*param_value_lists))

        print(f"\n{'='*70}")
        print(f"MULTI-PARAMETER SWEEP (Full Factorial Design)")
        print(f"{'='*70}")
        print(f"Parameters: {param_names}")
        print(f"Combinations: {len(combinations)}")
        print(f"Replicates per combination: {replicates_per_combination}")
        print(f"Total simulations: {len(combinations) * replicates_per_combination}")
        print(f"{'='*70}\n")

        sweep_results = {}

        for i, combination in enumerate(combinations):
            param_overrides = dict(zip(param_names, combination))

            print(f"\n--- Combination {i+1}/{len(combinations)}: {param_overrides} ---\n")

            results = self.run_batch(
                replicates_per_combination,
                param_overrides=param_overrides,
                executable=executable,
                n_parallel=n_parallel
            )

            sweep_results[str(combination)] = results

        return sweep_results

    def analyze_sweep(self, param_name: str, sweep_results: Dict):
        """Analyze parameter sweep results."""

        print(f"\n{'='*70}")
        print(f"PARAMETER SWEEP ANALYSIS: {param_name}")
        print(f"{'='*70}\n")

        # Extract metrics for each parameter value
        data = []
        for param_value, results in sweep_results.items():
            successful = [r for r in results if r['success']]

            if len(successful) > 0:
                data.append({
                    'param_value': param_value,
                    'n_replicates': len(results),
                    'n_successful': len(successful),
                    'success_rate': len(successful) / len(results),
                    'mean_population': np.mean([r['mean_population'] for r in successful]),
                    'std_population': np.std([r['mean_population'] for r in successful]),
                    'mean_cv': np.mean([r['population_cv'] for r in successful]),
                    'extinction_rate': sum(r['extinction'] for r in successful) / len(successful)
                })

        df = pd.DataFrame(data)

        # Print summary table
        print(df.to_string(index=False))
        print()

        # Save to CSV
        output_file = self.output_dir / f"sweep_{param_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"✓ Sweep results saved to: {output_file}\n")

        # Statistical test (ANOVA)
        if len(data) > 2:
            from scipy import stats

            # Group data by parameter value
            groups = []
            for param_value, results in sweep_results.items():
                successful = [r for r in results if r['success']]
                cvs = [r['population_cv'] for r in successful if not np.isnan(r['population_cv'])]
                if len(cvs) > 0:
                    groups.append(cvs)

            if len(groups) > 1:
                # One-way ANOVA
                f_stat, p_value = stats.f_oneway(*groups)

                print(f"One-way ANOVA: F = {f_stat:.3f}, p = {p_value:.4f}")

                if p_value < 0.05:
                    print(f"✓ Significant effect of {param_name} on population stability (p < 0.05)")
                else:
                    print(f"✗ No significant effect of {param_name} (p >= 0.05)")

                print()

    def generate_summary(self, output_file: str = "batch_summary.txt"):
        """Generate comprehensive summary of all batch results."""

        output_path = self.output_dir / output_file

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATCH SIMULATION SUMMARY\n")
            f.write("="*70 + "\n\n")

            f.write(f"Total runs: {len(self.results)}\n")
            f.write(f"Successful: {sum(r['success'] for r in self.results)}\n")
            f.write(f"Failed: {len(self.failed_runs)}\n\n")

            if len(self.failed_runs) > 0:
                f.write("-"*70 + "\n")
                f.write("FAILED RUNS\n")
                f.write("-"*70 + "\n")
                for fail in self.failed_runs:
                    f.write(f"Replicate {fail['replicate_id']}: {fail['error']}\n")
                f.write("\n")

            # Overall statistics (successful runs only)
            successful = [r for r in self.results if r['success']]

            if len(successful) > 0:
                f.write("-"*70 + "\n")
                f.write("OVERALL STATISTICS (Successful Runs)\n")
                f.write("-"*70 + "\n")

                mean_pops = [r['mean_population'] for r in successful]
                cvs = [r['population_cv'] for r in successful if not np.isnan(r['population_cv'])]
                extinctions = sum(r['extinction'] for r in successful)

                f.write(f"Mean population: {np.mean(mean_pops):.2f} ± {np.std(mean_pops):.2f}\n")
                f.write(f"Population CV: {np.mean(cvs):.3f} ± {np.std(cvs):.3f}\n")
                f.write(f"Extinction rate: {extinctions/len(successful)*100:.1f}%\n")
                f.write(f"Runtime: {np.mean([r['runtime_seconds'] for r in successful]):.1f}s ± ")
                f.write(f"{np.std([r['runtime_seconds'] for r in successful]):.1f}s\n")

            f.write("\n" + "="*70 + "\n")

        print(f"✓ Summary saved to: {output_path}")

    def export_aggregated_data(self, output_file: str = "aggregated_results.csv"):
        """Export all results to a single CSV file."""

        successful = [r for r in self.results if r['success']]

        if len(successful) == 0:
            print("Warning: No successful runs to export")
            return

        df = pd.DataFrame(successful)

        output_path = self.output_dir / output_file
        df.to_csv(output_path, index=False)

        print(f"✓ Aggregated data saved to: {output_path}")
        print(f"  {len(df)} rows × {len(df.columns)} columns")


def main():
    """Main batch runner."""

    parser = argparse.ArgumentParser(
        description='Batch runner for MICROLIFE simulations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 100 replicates
    python batch_runner.py --config experiment_config.yaml --replicates 100

    # Parameter sweep with parallel execution
    python batch_runner.py --config base.yaml --sweep mutation_rate 0.05,0.10,0.15 --parallel 4

    # Multi-parameter sweep (factorial design)
    python batch_runner.py --config base.yaml --multi-sweep "mutation_rate:0.1,0.2 temperature:15,20,25"
        """
    )

    parser.add_argument('--config', type=str, required=True,
                       help='Base configuration YAML file')
    parser.add_argument('--executable', type=str, default='./MICROLIFE_ULTIMATE',
                       help='Path to simulation executable')
    parser.add_argument('--output-dir', type=str, default='batch_results',
                       help='Output directory for all results')
    parser.add_argument('--replicates', type=int, default=10,
                       help='Number of replicates to run')
    parser.add_argument('--sweep', nargs=2, metavar=('PARAM', 'VALUES'),
                       help='Parameter sweep: PARAM "val1,val2,val3"')
    parser.add_argument('--multi-sweep', type=str,
                       help='Multi-parameter sweep: "param1:v1,v2 param2:v3,v4"')
    parser.add_argument('--parallel', type=int, default=1,
                       help='Number of parallel workers')

    args = parser.parse_args()

    # Check if executable exists
    if not Path(args.executable).exists():
        print(f"Error: Executable not found: {args.executable}")
        print("Compile first with: make -f Makefile.microlife")
        return 1

    # Create batch runner
    runner = BatchRunner(args.config, args.output_dir)

    # Run appropriate experiment type
    if args.multi_sweep:
        # Multi-parameter sweep
        param_dict = {}
        for param_spec in args.multi_sweep.split():
            param_name, values_str = param_spec.split(':')
            values = [float(v) if '.' in v else int(v) for v in values_str.split(',')]
            param_dict[param_name] = values

        runner.multi_parameter_sweep(
            param_dict,
            replicates_per_combination=args.replicates,
            executable=args.executable,
            n_parallel=args.parallel
        )

    elif args.sweep:
        # Single parameter sweep
        param_name, values_str = args.sweep
        values = [float(v) if '.' in v else int(v) for v in values_str.split(',')]

        runner.parameter_sweep(
            param_name,
            values,
            replicates_per_value=args.replicates,
            executable=args.executable,
            n_parallel=args.parallel
        )

    else:
        # Simple batch (no sweep)
        runner.run_batch(
            args.replicates,
            executable=args.executable,
            n_parallel=args.parallel
        )

    # Generate outputs
    print("\nGenerating summary outputs...")
    runner.generate_summary()
    runner.export_aggregated_data()

    print("\n" + "="*70)
    print("✓ BATCH PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")

    return 0


if __name__ == '__main__':
    exit(main())
