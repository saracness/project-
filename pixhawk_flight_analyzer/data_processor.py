"""
Data Processor Module
=====================

This module handles cleaning, processing, and transformation of flight data.
Includes filtering, interpolation, coordinate transformations, and data validation.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightDataProcessor:
    """
    Process and clean flight telemetry data.

    Features:
    - Remove outliers and invalid data points
    - Interpolate missing data
    - Apply filtering (low-pass, etc.)
    - Coordinate transformations
    - Data alignment and synchronization
    - Feature extraction

    Example:
        processor = FlightDataProcessor(data)
        cleaned_data = processor.clean_data()
        filtered_data = processor.apply_lowpass_filter('GPS', 'Alt', cutoff_freq=2.0)
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the processor with flight data.

        Args:
            data: Dictionary of DataFrames from FlightDataLoader
        """
        self.data = data.copy()
        self.processed_data = {}

    def clean_data(self, remove_outliers: bool = True,
                   interpolate_gaps: bool = True,
                   max_gap_size: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Clean the flight data by removing outliers and interpolating gaps.

        Args:
            remove_outliers: Whether to remove statistical outliers
            interpolate_gaps: Whether to interpolate missing values
            max_gap_size: Maximum gap size to interpolate (number of samples)

        Returns:
            Dictionary of cleaned DataFrames
        """
        logger.info("Cleaning flight data...")
        cleaned = {}

        for msg_type, df in self.data.items():
            df_clean = df.copy()

            # Remove completely empty columns
            df_clean = df_clean.dropna(axis=1, how='all')

            # Remove outliers using IQR method
            if remove_outliers:
                df_clean = self._remove_outliers_iqr(df_clean)

            # Interpolate small gaps
            if interpolate_gaps:
                df_clean = self._interpolate_gaps(df_clean, max_gap_size)

            cleaned[msg_type] = df_clean
            logger.info(f"  {msg_type}: {len(df)} -> {len(df_clean)} records")

        self.processed_data = cleaned
        return cleaned

    def _remove_outliers_iqr(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers using the Interquartile Range (IQR) method.

        Args:
            df: Input DataFrame
            threshold: IQR multiplier for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        df_out = df.copy()
        numeric_cols = df_out.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['time_boot_ms', 'timestamp']:
                continue  # Don't remove outliers from time columns

            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Replace outliers with NaN
            df_out.loc[(df_out[col] < lower_bound) | (df_out[col] > upper_bound), col] = np.nan

        # Remove rows where too many values are NaN
        df_out = df_out.dropna(thresh=len(df_out.columns) * 0.5)

        return df_out

    def _interpolate_gaps(self, df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
        """
        Interpolate small gaps in the data.

        Args:
            df: Input DataFrame
            max_gap: Maximum gap size to interpolate

        Returns:
            DataFrame with gaps interpolated
        """
        df_interp = df.copy()
        numeric_cols = df_interp.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col in ['time_boot_ms']:
                continue

            df_interp[col] = df_interp[col].interpolate(
                method='linear',
                limit=max_gap,
                limit_area='inside'
            )

        return df_interp

    def apply_lowpass_filter(self, msg_type: str, column: str,
                            cutoff_freq: float, fs: float = 10.0,
                            order: int = 4) -> pd.Series:
        """
        Apply a Butterworth low-pass filter to a signal.

        Args:
            msg_type: Message type (e.g., 'GPS')
            column: Column name to filter
            cutoff_freq: Cutoff frequency in Hz
            fs: Sampling frequency in Hz
            order: Filter order

        Returns:
            Filtered signal as pandas Series

        Example:
            filtered_alt = processor.apply_lowpass_filter('GPS', 'Alt', cutoff_freq=2.0)
        """
        if msg_type not in self.data:
            raise ValueError(f"Message type '{msg_type}' not found in data")

        df = self.data[msg_type]
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in {msg_type}")

        # Design filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Apply filter
        y = df[column].values
        y_filtered = signal.filtfilt(b, a, y)

        return pd.Series(y_filtered, index=df.index, name=f'{column}_filtered')

    def extract_flight_path(self) -> Optional[pd.DataFrame]:
        """
        Extract the 3D flight path (lat, lon, alt) from the data.

        Returns:
            DataFrame with columns: timestamp, lat, lon, alt, or None if not available

        Example:
            path = processor.extract_flight_path()
            print(path[['lat', 'lon', 'alt']])
        """
        # Try GLOBAL_POSITION_INT first
        if 'GLOBAL_POSITION_INT' in self.data:
            df = self.data['GLOBAL_POSITION_INT'].copy()
            if all(col in df.columns for col in ['lat', 'lon', 'alt']):
                # Convert from int32 (degrees * 1e7) to float
                df['lat'] = df['lat'] / 1e7
                df['lon'] = df['lon'] / 1e7
                df['alt'] = df['alt'] / 1000.0  # mm to m

                result = df[['lat', 'lon', 'alt']].copy()
                if 'time_boot_ms' in df.columns:
                    result['timestamp'] = df['time_boot_ms']
                return result

        # Try GPS as fallback
        if 'GPS' in self.data:
            df = self.data['GPS'].copy()
            if all(col in df.columns for col in ['Lat', 'Lng', 'Alt']):
                result = df[['Lat', 'Lng', 'Alt']].copy()
                result.columns = ['lat', 'lon', 'alt']
                if 'time_boot_ms' in df.columns:
                    result['timestamp'] = df['time_boot_ms']
                return result

        logger.warning("Could not extract flight path - no GPS data found")
        return None

    def extract_attitude(self) -> Optional[pd.DataFrame]:
        """
        Extract attitude data (roll, pitch, yaw).

        Returns:
            DataFrame with columns: timestamp, roll, pitch, yaw (in degrees)
        """
        if 'ATTITUDE' not in self.data:
            logger.warning("No ATTITUDE data found")
            return None

        df = self.data['ATTITUDE'].copy()

        # Convert from radians to degrees
        if all(col in df.columns for col in ['roll', 'pitch', 'yaw']):
            result = df[['roll', 'pitch', 'yaw']].copy()
            result['roll'] = np.degrees(result['roll'])
            result['pitch'] = np.degrees(result['pitch'])
            result['yaw'] = np.degrees(result['yaw'])

            if 'time_boot_ms' in df.columns:
                result['timestamp'] = df['time_boot_ms']

            return result

        return None

    def extract_velocity(self) -> Optional[pd.DataFrame]:
        """
        Extract velocity data (vx, vy, vz).

        Returns:
            DataFrame with velocity components in m/s
        """
        # Try LOCAL_POSITION_NED
        if 'LOCAL_POSITION_NED' in self.data:
            df = self.data['LOCAL_POSITION_NED'].copy()
            if all(col in df.columns for col in ['vx', 'vy', 'vz']):
                result = df[['vx', 'vy', 'vz']].copy()
                if 'time_boot_ms' in df.columns:
                    result['timestamp'] = df['time_boot_ms']
                return result

        # Try GLOBAL_POSITION_INT
        if 'GLOBAL_POSITION_INT' in self.data:
            df = self.data['GLOBAL_POSITION_INT'].copy()
            if all(col in df.columns for col in ['vx', 'vy', 'vz']):
                # Convert from cm/s to m/s
                result = df[['vx', 'vy', 'vz']].copy()
                result['vx'] = result['vx'] / 100.0
                result['vy'] = result['vy'] / 100.0
                result['vz'] = result['vz'] / 100.0

                if 'time_boot_ms' in df.columns:
                    result['timestamp'] = df['time_boot_ms']
                return result

        logger.warning("No velocity data found")
        return None

    def calculate_distance_traveled(self) -> Optional[float]:
        """
        Calculate total distance traveled in meters using GPS coordinates.

        Returns:
            Total distance in meters, or None if cannot be calculated
        """
        path = self.extract_flight_path()
        if path is None or len(path) < 2:
            return None

        # Calculate distances between consecutive points using Haversine formula
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371000  # Earth radius in meters
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            dphi = np.radians(lat2 - lat1)
            dlambda = np.radians(lon2 - lon1)

            a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

            return R * c

        total_distance = 0.0
        for i in range(len(path) - 1):
            dist = haversine(
                path.iloc[i]['lat'], path.iloc[i]['lon'],
                path.iloc[i+1]['lat'], path.iloc[i+1]['lon']
            )
            total_distance += dist

        return total_distance

    def resample_to_frequency(self, msg_type: str, target_freq_hz: float) -> pd.DataFrame:
        """
        Resample data to a specific frequency.

        Args:
            msg_type: Message type to resample
            target_freq_hz: Target frequency in Hz

        Returns:
            Resampled DataFrame
        """
        if msg_type not in self.data:
            raise ValueError(f"Message type '{msg_type}' not found")

        df = self.data[msg_type].copy()

        if 'time_boot_ms' not in df.columns:
            raise ValueError(f"No timestamp column found in {msg_type}")

        # Set timestamp as index
        df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')
        df = df.set_index('timestamp')

        # Resample
        period_ms = int(1000.0 / target_freq_hz)
        df_resampled = df.resample(f'{period_ms}ms').mean()

        return df_resampled

    def get_processed_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get the processed data.

        Returns:
            Dictionary of processed DataFrames
        """
        return self.processed_data if self.processed_data else self.data
