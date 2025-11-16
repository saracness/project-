"""
Command-Line Interface Module
==============================

CLI tool for analyzing Pixhawk/MAVLink flight data.
"""

import click
import os
import sys
import logging
from typing import Optional

from .data_loader import FlightDataLoader
from .data_processor import FlightDataProcessor
from .statistics import FlightAnalyzer
from .visualization import FlightVisualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    Pixhawk Flight Analyzer - Analyze and visualize drone flight data.

    Load .tlog or MAVLink files, process data, generate statistics,
    and create beautiful visualizations.
    """
    pass


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output CSV file for statistics')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(file_path: str, output: Optional[str], verbose: bool):
    """
    Analyze a flight log file and display statistics.

    Example:
        pixhawk-analyzer analyze flight.tlog
        pixhawk-analyzer analyze flight.tlog --output stats.csv
    """
    try:
        if verbose:
            logger.setLevel(logging.DEBUG)

        click.echo(f"\n📊 Analyzing flight log: {file_path}\n")

        # Load data
        loader = FlightDataLoader(file_path)
        data = loader.load()

        if not data:
            click.echo("❌ No data loaded from file", err=True)
            sys.exit(1)

        click.echo(f"✅ Loaded {len(data)} message types")

        # Analyze
        analyzer = FlightAnalyzer(data)
        stats = analyzer.get_statistics()

        # Display summary
        analyzer.print_summary(stats)

        # Export if requested
        if output:
            analyzer.export_statistics_to_csv(output, stats)
            click.echo(f"💾 Statistics exported to: {output}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            raise
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', help='Output directory for plots')
@click.option('--format', '-f', type=click.Choice(['png', 'html', 'both']), default='png',
              help='Output format')
@click.option('--plot-type', '-t',
              type=click.Choice(['2d', '3d', 'altitude', 'speed', 'attitude', 'dashboard', 'all']),
              default='all', help='Type of plot to generate')
@click.option('--show/--no-show', default=False, help='Display plots interactively')
def visualize(file_path: str, output_dir: str, format: str, plot_type: str, show: bool):
    """
    Generate visualizations from a flight log file.

    Example:
        pixhawk-analyzer visualize flight.tlog
        pixhawk-analyzer visualize flight.tlog --plot-type 3d --format html
        pixhawk-analyzer visualize flight.tlog --plot-type dashboard --show
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        click.echo(f"\n📈 Generating visualizations from: {file_path}\n")

        # Load data
        loader = FlightDataLoader(file_path)
        data = loader.load()

        if not data:
            click.echo("❌ No data loaded from file", err=True)
            sys.exit(1)

        visualizer = FlightVisualizer(data)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Generate requested plots
        if plot_type in ['2d', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_flight_2d.png')
                visualizer.plot_flight_path_2d(save_path=output_path, show=show)
                click.echo(f"✅ 2D flight path saved to: {output_path}")

        if plot_type in ['3d', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_flight_3d.png')
                visualizer.plot_flight_path_3d(save_path=output_path, show=show)
                click.echo(f"✅ 3D flight path (static) saved to: {output_path}")

            if format in ['html', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_flight_3d.html')
                visualizer.plot_flight_path_3d_interactive(save_path=output_path, show=show)
                click.echo(f"✅ 3D flight path (interactive) saved to: {output_path}")

        if plot_type in ['altitude', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_altitude.png')
                visualizer.plot_altitude_profile(save_path=output_path, show=show)
                click.echo(f"✅ Altitude profile saved to: {output_path}")

        if plot_type in ['speed', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_speed.png')
                visualizer.plot_speed_profile(save_path=output_path, show=show)
                click.echo(f"✅ Speed profile saved to: {output_path}")

        if plot_type in ['attitude', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_attitude.png')
                visualizer.plot_attitude(save_path=output_path, show=show)
                click.echo(f"✅ Attitude plot saved to: {output_path}")

        if plot_type in ['dashboard', 'all']:
            if format in ['png', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_dashboard.png')
                visualizer.plot_dashboard(save_path=output_path, show=show)
                click.echo(f"✅ Dashboard saved to: {output_path}")

            if format in ['html', 'both']:
                output_path = os.path.join(output_dir, f'{base_name}_dashboard.html')
                visualizer.create_interactive_dashboard(save_path=output_path, show=show)
                click.echo(f"✅ Interactive dashboard saved to: {output_path}")

        click.echo(f"\n✨ All visualizations saved to: {output_dir}/")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def info(file_path: str):
    """
    Display basic information about a flight log file.

    Example:
        pixhawk-analyzer info flight.tlog
    """
    try:
        click.echo(f"\n📄 File Information: {file_path}\n")

        # Validate file
        if not FlightDataLoader.is_valid_file(file_path):
            click.echo("❌ File does not appear to be a valid telemetry file", err=True)
            sys.exit(1)

        # Load minimal data
        loader = FlightDataLoader(file_path)
        data = loader.load()

        info = loader.get_basic_info()

        # Display info
        click.echo(f"File Size: {info['file_size_mb']:.2f} MB")
        click.echo(f"Duration: {info.get('duration_seconds', 'N/A')} seconds")
        click.echo(f"Message Types: {info['message_types_count']}")
        click.echo(f"\nAvailable Message Types:")
        for msg_type in info['message_types']:
            count = info.get(f'count_{msg_type}', 0)
            if count > 0:
                click.echo(f"  - {msg_type}: {count} messages")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default='./output', help='Output directory')
@click.option('--format', '-f', type=click.Choice(['png', 'html', 'both']), default='both',
              help='Output format')
def process(file_path: str, output_dir: str, format: str):
    """
    Complete analysis pipeline: load, analyze, and visualize.

    This command runs the full analysis pipeline and generates
    all statistics and visualizations.

    Example:
        pixhawk-analyzer process flight.tlog
        pixhawk-analyzer process flight.tlog --output-dir results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        click.echo(f"\n🚀 Running complete analysis pipeline\n")
        click.echo(f"Input:  {file_path}")
        click.echo(f"Output: {output_dir}/\n")

        # Load data
        click.echo("1️⃣  Loading flight data...")
        loader = FlightDataLoader(file_path)
        data = loader.load()

        if not data:
            click.echo("❌ No data loaded", err=True)
            sys.exit(1)

        click.echo(f"   ✅ Loaded {len(data)} message types\n")

        # Analyze
        click.echo("2️⃣  Analyzing flight data...")
        analyzer = FlightAnalyzer(data)
        stats = analyzer.get_statistics()
        analyzer.print_summary(stats)

        # Export statistics
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        stats_file = os.path.join(output_dir, f'{base_name}_statistics.csv')
        analyzer.export_statistics_to_csv(stats_file, stats)
        click.echo(f"   ✅ Statistics saved to: {stats_file}\n")

        # Generate visualizations
        click.echo("3️⃣  Generating visualizations...")
        visualizer = FlightVisualizer(data)

        plot_count = 0

        if format in ['png', 'both']:
            visualizer.plot_flight_path_2d(
                save_path=os.path.join(output_dir, f'{base_name}_flight_2d.png'),
                show=False
            )
            visualizer.plot_altitude_profile(
                save_path=os.path.join(output_dir, f'{base_name}_altitude.png'),
                show=False
            )
            visualizer.plot_speed_profile(
                save_path=os.path.join(output_dir, f'{base_name}_speed.png'),
                show=False
            )
            visualizer.plot_attitude(
                save_path=os.path.join(output_dir, f'{base_name}_attitude.png'),
                show=False
            )
            visualizer.plot_dashboard(
                save_path=os.path.join(output_dir, f'{base_name}_dashboard.png'),
                show=False
            )
            plot_count += 5

        if format in ['html', 'both']:
            visualizer.plot_flight_path_3d_interactive(
                save_path=os.path.join(output_dir, f'{base_name}_flight_3d.html'),
                show=False
            )
            visualizer.create_interactive_dashboard(
                save_path=os.path.join(output_dir, f'{base_name}_dashboard.html'),
                show=False
            )
            plot_count += 2

        click.echo(f"   ✅ Generated {plot_count} visualizations\n")

        click.echo(f"✨ Analysis complete! Results saved to: {output_dir}/")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
