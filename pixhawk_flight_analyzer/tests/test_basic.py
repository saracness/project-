"""
Basic Unit Tests for Pixhawk Flight Analyzer
============================================

Run with: pytest tests/
"""

import pytest
import numpy as np
import pandas as pd
from pixhawk_flight_analyzer.data_processor import FlightDataProcessor
from pixhawk_flight_analyzer.statistics import FlightAnalyzer


class TestFlightDataProcessor:
    """Test cases for FlightDataProcessor."""

    def setup_method(self):
        """Set up test data."""
        # Create sample flight data
        self.sample_data = {
            'GPS': pd.DataFrame({
                'time_boot_ms': np.arange(0, 10000, 100),
                'Lat': np.linspace(37.7749, 37.7850, 100),
                'Lng': np.linspace(-122.4194, -122.4094, 100),
                'Alt': np.linspace(0, 100, 100) + np.random.randn(100) * 2
            }),
            'ATTITUDE': pd.DataFrame({
                'time_boot_ms': np.arange(0, 10000, 100),
                'roll': np.sin(np.linspace(0, 2*np.pi, 100)) * 0.1,
                'pitch': np.cos(np.linspace(0, 2*np.pi, 100)) * 0.1,
                'yaw': np.linspace(0, 2*np.pi, 100)
            })
        }

    def test_initialization(self):
        """Test processor initialization."""
        processor = FlightDataProcessor(self.sample_data)
        assert processor is not None
        assert len(processor.data) == 2

    def test_extract_flight_path(self):
        """Test flight path extraction."""
        processor = FlightDataProcessor(self.sample_data)
        path = processor.extract_flight_path()

        assert path is not None
        assert 'lat' in path.columns
        assert 'lon' in path.columns
        assert 'alt' in path.columns
        assert len(path) == 100

    def test_extract_attitude(self):
        """Test attitude data extraction."""
        processor = FlightDataProcessor(self.sample_data)
        attitude = processor.extract_attitude()

        assert attitude is not None
        assert 'roll' in attitude.columns
        assert 'pitch' in attitude.columns
        assert 'yaw' in attitude.columns
        assert len(attitude) == 100

        # Check conversion to degrees
        assert attitude['roll'].max() <= 360
        assert attitude['pitch'].max() <= 360

    def test_clean_data(self):
        """Test data cleaning functionality."""
        processor = FlightDataProcessor(self.sample_data)
        cleaned = processor.clean_data(remove_outliers=True, interpolate_gaps=True)

        assert len(cleaned) == 2
        assert 'GPS' in cleaned
        assert 'ATTITUDE' in cleaned

    def test_calculate_distance_traveled(self):
        """Test distance calculation."""
        processor = FlightDataProcessor(self.sample_data)
        distance = processor.calculate_distance_traveled()

        assert distance is not None
        assert distance > 0
        # Distance should be reasonable (not too small or too large)
        assert 1000 < distance < 10000  # meters


class TestFlightAnalyzer:
    """Test cases for FlightAnalyzer."""

    def setup_method(self):
        """Set up test data."""
        self.sample_data = {
            'GLOBAL_POSITION_INT': pd.DataFrame({
                'time_boot_ms': np.arange(0, 60000, 1000),
                'lat': (np.linspace(37.7749, 37.7850, 60) * 1e7).astype(int),
                'lon': (np.linspace(-122.4194, -122.4094, 60) * 1e7).astype(int),
                'alt': (np.linspace(0, 100, 60) * 1000).astype(int),  # mm
                'vx': np.random.randint(0, 500, 60),  # cm/s
                'vy': np.random.randint(0, 500, 60),
                'vz': np.random.randint(-100, 100, 60)
            }),
            'ATTITUDE': pd.DataFrame({
                'time_boot_ms': np.arange(0, 60000, 1000),
                'roll': np.random.randn(60) * 0.1,
                'pitch': np.random.randn(60) * 0.1,
                'yaw': np.linspace(0, 2*np.pi, 60)
            }),
            'GPS_RAW_INT': pd.DataFrame({
                'time_boot_ms': np.arange(0, 60000, 1000),
                'satellites_visible': np.random.randint(8, 15, 60),
                'eph': np.random.randint(50, 200, 60),
                'fix_type': np.full(60, 3)
            })
        }

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = FlightAnalyzer(self.sample_data)
        assert analyzer is not None

    def test_get_statistics(self):
        """Test statistics generation."""
        analyzer = FlightAnalyzer(self.sample_data)
        stats = analyzer.get_statistics()

        assert isinstance(stats, dict)
        assert len(stats) > 0

        # Check for expected statistics
        assert 'duration_seconds' in stats
        assert 'altitude_max_m' in stats
        assert 'altitude_min_m' in stats

    def test_time_stats(self):
        """Test time statistics calculation."""
        analyzer = FlightAnalyzer(self.sample_data)
        stats = analyzer.get_statistics()

        assert stats['duration_seconds'] is not None
        assert stats['duration_seconds'] > 0
        assert stats['duration_minutes'] == stats['duration_seconds'] / 60

    def test_altitude_stats(self):
        """Test altitude statistics."""
        analyzer = FlightAnalyzer(self.sample_data)
        stats = analyzer.get_statistics()

        assert stats['altitude_max_m'] is not None
        assert stats['altitude_min_m'] is not None
        assert stats['altitude_max_m'] >= stats['altitude_min_m']
        assert stats['altitude_mean_m'] is not None

    def test_gps_stats(self):
        """Test GPS statistics."""
        analyzer = FlightAnalyzer(self.sample_data)
        stats = analyzer.get_statistics()

        assert stats['gps_satellites_min'] >= 0
        assert stats['gps_satellites_max'] >= stats['gps_satellites_min']
        assert 0 <= stats['gps_satellites_mean'] <= 30

    def test_export_statistics(self, tmp_path):
        """Test statistics export to CSV."""
        analyzer = FlightAnalyzer(self.sample_data)
        stats = analyzer.get_statistics()

        output_file = tmp_path / "test_stats.csv"
        analyzer.export_statistics_to_csv(str(output_file), stats)

        assert output_file.exists()

        # Read back and verify
        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert 'duration_seconds' in df.columns


class TestDataValidation:
    """Test data validation functionality."""

    def test_empty_data(self):
        """Test handling of empty data."""
        processor = FlightDataProcessor({})
        path = processor.extract_flight_path()
        assert path is None

    def test_missing_columns(self):
        """Test handling of missing columns."""
        data = {
            'GPS': pd.DataFrame({
                'time_boot_ms': [1000, 2000, 3000]
                # Missing Lat, Lng, Alt
            })
        }
        processor = FlightDataProcessor(data)
        path = processor.extract_flight_path()
        assert path is None

    def test_invalid_data_types(self):
        """Test handling of invalid data types."""
        data = {
            'GPS': pd.DataFrame({
                'time_boot_ms': ['invalid', 'data', 'here'],
                'Lat': ['not', 'a', 'number'],
                'Lng': ['also', 'not', 'number'],
                'Alt': ['nope', 'nope', 'nope']
            })
        }
        processor = FlightDataProcessor(data)
        # Should not crash
        assert processor is not None


def test_package_imports():
    """Test that all main components can be imported."""
    from pixhawk_flight_analyzer import (
        FlightDataLoader,
        FlightDataProcessor,
        FlightAnalyzer,
        FlightVisualizer
    )

    assert FlightDataLoader is not None
    assert FlightDataProcessor is not None
    assert FlightAnalyzer is not None
    assert FlightVisualizer is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
