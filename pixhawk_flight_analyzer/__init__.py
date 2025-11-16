"""
Pixhawk Flight Analyzer
========================

A comprehensive Python package for loading, processing, analyzing, and visualizing
Pixhawk/MAVLink telemetry data from drone flights.

Main modules:
- data_loader: Load .tlog and MAVLink files
- data_processor: Clean and process flight data
- statistics: Generate summary statistics
- visualization: Create 2D/3D visualizations
- cli: Command-line interface

Example usage:
    from pixhawk_flight_analyzer import FlightDataLoader, FlightAnalyzer

    # Load flight data
    loader = FlightDataLoader('flight.tlog')
    data = loader.load()

    # Analyze and visualize
    analyzer = FlightAnalyzer(data)
    stats = analyzer.get_statistics()
    analyzer.plot_flight_path_3d()
"""

__version__ = '1.0.0'
__author__ = 'Flight Analysis Team'
__all__ = [
    'FlightDataLoader',
    'FlightDataProcessor',
    'FlightAnalyzer',
    'FlightVisualizer',
]

from .data_loader import FlightDataLoader
from .data_processor import FlightDataProcessor
from .statistics import FlightAnalyzer
from .visualization import FlightVisualizer
