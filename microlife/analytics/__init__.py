"""
Scientific analytics and automatic reporting system.
"""
from .data_logger import DataLogger
from .scientific_plotter import ScientificPlotter
from .statistical_analyzer import StatisticalAnalyzer
from .report_generator import ReportGenerator

__all__ = [
    'DataLogger',
    'ScientificPlotter',
    'StatisticalAnalyzer',
    'ReportGenerator',
]
__version__ = '1.0.0'
