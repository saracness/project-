"""
Statistics Module
=================

This module calculates summary statistics and flight performance metrics.
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from .data_processor import FlightDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightAnalyzer:
    """
    Analyze flight data and generate comprehensive statistics.

    Features:
    - Flight duration and distance
    - Altitude statistics (min, max, average)
    - Speed statistics
    - Battery consumption
    - Flight mode analysis
    - Attitude statistics (max roll, pitch, yaw)
    - GPS quality metrics

    Example:
        analyzer = FlightAnalyzer(data)
        stats = analyzer.get_statistics()
        print(stats)
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the analyzer with flight data.

        Args:
            data: Dictionary of DataFrames from FlightDataLoader
        """
        self.data = data
        self.processor = FlightDataProcessor(data)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive flight statistics.

        Returns:
            Dictionary containing all flight statistics

        Example:
            stats = analyzer.get_statistics()
            print(f"Flight duration: {stats['duration_seconds']}s")
            print(f"Max altitude: {stats['altitude_max_m']}m")
        """
        logger.info("Calculating flight statistics...")

        stats = {}

        # Time statistics
        stats.update(self._calculate_time_stats())

        # Altitude statistics
        stats.update(self._calculate_altitude_stats())

        # Speed statistics
        stats.update(self._calculate_speed_stats())

        # Distance statistics
        stats.update(self._calculate_distance_stats())

        # Attitude statistics
        stats.update(self._calculate_attitude_stats())

        # GPS statistics
        stats.update(self._calculate_gps_stats())

        # Battery statistics
        stats.update(self._calculate_battery_stats())

        # Flight mode statistics
        stats.update(self._calculate_flight_mode_stats())

        return stats

    def _calculate_time_stats(self) -> Dict[str, Any]:
        """Calculate time-related statistics."""
        stats = {}

        # Find earliest and latest timestamps
        all_times = []
        for msg_type, df in self.data.items():
            if 'time_boot_ms' in df.columns:
                all_times.extend(df['time_boot_ms'].tolist())

        if all_times:
            min_time = min(all_times)
            max_time = max(all_times)
            duration = (max_time - min_time) / 1000.0  # Convert to seconds

            stats['duration_seconds'] = duration
            stats['duration_minutes'] = duration / 60.0
            stats['start_time_ms'] = min_time
            stats['end_time_ms'] = max_time
        else:
            stats['duration_seconds'] = None
            stats['duration_minutes'] = None

        return stats

    def _calculate_altitude_stats(self) -> Dict[str, Any]:
        """Calculate altitude statistics."""
        stats = {}
        path = self.processor.extract_flight_path()

        if path is not None and 'alt' in path.columns:
            alt = path['alt'].dropna()
            if len(alt) > 0:
                stats['altitude_min_m'] = float(alt.min())
                stats['altitude_max_m'] = float(alt.max())
                stats['altitude_mean_m'] = float(alt.mean())
                stats['altitude_std_m'] = float(alt.std())
                stats['altitude_range_m'] = float(alt.max() - alt.min())
            else:
                stats['altitude_min_m'] = None
                stats['altitude_max_m'] = None
                stats['altitude_mean_m'] = None
                stats['altitude_std_m'] = None
                stats['altitude_range_m'] = None
        else:
            stats['altitude_min_m'] = None
            stats['altitude_max_m'] = None
            stats['altitude_mean_m'] = None

        return stats

    def _calculate_speed_stats(self) -> Dict[str, Any]:
        """Calculate speed/velocity statistics."""
        stats = {}
        velocity = self.processor.extract_velocity()

        if velocity is not None:
            # Calculate ground speed (horizontal)
            if 'vx' in velocity.columns and 'vy' in velocity.columns:
                ground_speed = np.sqrt(velocity['vx']**2 + velocity['vy']**2)
                stats['speed_ground_max_ms'] = float(ground_speed.max())
                stats['speed_ground_mean_ms'] = float(ground_speed.mean())
                stats['speed_ground_max_kmh'] = float(ground_speed.max() * 3.6)
                stats['speed_ground_mean_kmh'] = float(ground_speed.mean() * 3.6)

            # Vertical speed
            if 'vz' in velocity.columns:
                vz = velocity['vz'].dropna()
                stats['speed_vertical_max_ms'] = float(abs(vz).max())
                stats['speed_vertical_mean_ms'] = float(abs(vz).mean())
                stats['climb_rate_max_ms'] = float(-vz.min())  # Negative vz is up
                stats['descent_rate_max_ms'] = float(vz.max())
        else:
            # Try VFR_HUD as alternative
            if 'VFR_HUD' in self.data and 'groundspeed' in self.data['VFR_HUD'].columns:
                gs = self.data['VFR_HUD']['groundspeed']
                stats['speed_ground_max_ms'] = float(gs.max())
                stats['speed_ground_mean_ms'] = float(gs.mean())
                stats['speed_ground_max_kmh'] = float(gs.max() * 3.6)

        return stats

    def _calculate_distance_stats(self) -> Dict[str, Any]:
        """Calculate distance traveled."""
        stats = {}
        distance = self.processor.calculate_distance_traveled()

        if distance is not None:
            stats['distance_traveled_m'] = float(distance)
            stats['distance_traveled_km'] = float(distance / 1000.0)
        else:
            stats['distance_traveled_m'] = None
            stats['distance_traveled_km'] = None

        return stats

    def _calculate_attitude_stats(self) -> Dict[str, Any]:
        """Calculate attitude (roll, pitch, yaw) statistics."""
        stats = {}
        attitude = self.processor.extract_attitude()

        if attitude is not None:
            for axis in ['roll', 'pitch', 'yaw']:
                if axis in attitude.columns:
                    data = attitude[axis].dropna()
                    stats[f'{axis}_max_deg'] = float(abs(data).max())
                    stats[f'{axis}_mean_deg'] = float(data.mean())
                    stats[f'{axis}_std_deg'] = float(data.std())
        else:
            for axis in ['roll', 'pitch', 'yaw']:
                stats[f'{axis}_max_deg'] = None

        return stats

    def _calculate_gps_stats(self) -> Dict[str, Any]:
        """Calculate GPS quality statistics."""
        stats = {}

        # Check GPS_RAW_INT for satellite and quality info
        if 'GPS_RAW_INT' in self.data:
            gps = self.data['GPS_RAW_INT']

            if 'satellites_visible' in gps.columns:
                sats = gps['satellites_visible'].dropna()
                stats['gps_satellites_min'] = int(sats.min())
                stats['gps_satellites_max'] = int(sats.max())
                stats['gps_satellites_mean'] = float(sats.mean())

            if 'eph' in gps.columns:  # Horizontal dilution of precision
                eph = gps['eph'].dropna()
                # eph is in cm, convert to m
                stats['gps_hdop_min_m'] = float(eph.min() / 100.0)
                stats['gps_hdop_max_m'] = float(eph.max() / 100.0)
                stats['gps_hdop_mean_m'] = float(eph.mean() / 100.0)

            if 'fix_type' in gps.columns:
                fix_types = gps['fix_type'].value_counts().to_dict()
                stats['gps_fix_types'] = {int(k): int(v) for k, v in fix_types.items()}

        return stats

    def _calculate_battery_stats(self) -> Dict[str, Any]:
        """Calculate battery statistics."""
        stats = {}

        if 'BATTERY_STATUS' in self.data:
            battery = self.data['BATTERY_STATUS']

            if 'voltages' in battery.columns:
                # voltages is typically an array, take first cell
                try:
                    voltage_data = battery['voltages'].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
                    voltage_data = voltage_data.dropna()
                    if len(voltage_data) > 0:
                        # Convert from mV to V
                        stats['battery_voltage_min_v'] = float(voltage_data.min() / 1000.0)
                        stats['battery_voltage_max_v'] = float(voltage_data.max() / 1000.0)
                        stats['battery_voltage_mean_v'] = float(voltage_data.mean() / 1000.0)
                except:
                    pass

            if 'current_battery' in battery.columns:
                current = battery['current_battery'].dropna()
                if len(current) > 0:
                    # Convert from cA to A
                    stats['battery_current_max_a'] = float(current.max() / 100.0)
                    stats['battery_current_mean_a'] = float(current.mean() / 100.0)

            if 'battery_remaining' in battery.columns:
                remaining = battery['battery_remaining'].dropna()
                if len(remaining) > 0:
                    stats['battery_remaining_start_pct'] = float(remaining.iloc[0])
                    stats['battery_remaining_end_pct'] = float(remaining.iloc[-1])
                    stats['battery_consumed_pct'] = float(remaining.iloc[0] - remaining.iloc[-1])

        # Alternative: SYS_STATUS
        elif 'SYS_STATUS' in self.data:
            sys_status = self.data['SYS_STATUS']

            if 'voltage_battery' in sys_status.columns:
                voltage = sys_status['voltage_battery'].dropna()
                if len(voltage) > 0:
                    stats['battery_voltage_min_v'] = float(voltage.min() / 1000.0)
                    stats['battery_voltage_max_v'] = float(voltage.max() / 1000.0)
                    stats['battery_voltage_mean_v'] = float(voltage.mean() / 1000.0)

            if 'current_battery' in sys_status.columns:
                current = sys_status['current_battery'].dropna()
                if len(current) > 0:
                    stats['battery_current_max_a'] = float(current.max() / 100.0)
                    stats['battery_current_mean_a'] = float(current.mean() / 100.0)

            if 'battery_remaining' in sys_status.columns:
                remaining = sys_status['battery_remaining'].dropna()
                if len(remaining) > 0:
                    stats['battery_remaining_start_pct'] = float(remaining.iloc[0])
                    stats['battery_remaining_end_pct'] = float(remaining.iloc[-1])

        return stats

    def _calculate_flight_mode_stats(self) -> Dict[str, Any]:
        """Calculate flight mode statistics."""
        stats = {}

        if 'HEARTBEAT' in self.data:
            heartbeat = self.data['HEARTBEAT']

            if 'custom_mode' in heartbeat.columns:
                modes = heartbeat['custom_mode'].value_counts().to_dict()
                stats['flight_modes'] = {int(k): int(v) for k, v in modes.items()}

            if 'base_mode' in heartbeat.columns:
                base_modes = heartbeat['base_mode'].value_counts().to_dict()
                stats['base_modes'] = {int(k): int(v) for k, v in base_modes.items()}

        return stats

    def print_summary(self, stats: Optional[Dict[str, Any]] = None):
        """
        Print a formatted summary of flight statistics.

        Args:
            stats: Statistics dictionary. If None, calculates statistics first.

        Example:
            analyzer.print_summary()
        """
        if stats is None:
            stats = self.get_statistics()

        print("\n" + "="*60)
        print("FLIGHT STATISTICS SUMMARY")
        print("="*60)

        # Time
        if stats.get('duration_minutes'):
            print(f"\n📅 Duration: {stats['duration_minutes']:.2f} minutes ({stats['duration_seconds']:.1f} seconds)")

        # Distance
        if stats.get('distance_traveled_km'):
            print(f"📏 Distance: {stats['distance_traveled_km']:.2f} km ({stats['distance_traveled_m']:.0f} m)")

        # Altitude
        if stats.get('altitude_max_m'):
            print(f"\n🔝 Altitude:")
            print(f"   Max: {stats['altitude_max_m']:.1f} m")
            print(f"   Min: {stats['altitude_min_m']:.1f} m")
            print(f"   Avg: {stats['altitude_mean_m']:.1f} m")

        # Speed
        if stats.get('speed_ground_max_kmh'):
            print(f"\n⚡ Speed (Ground):")
            print(f"   Max: {stats['speed_ground_max_kmh']:.1f} km/h ({stats['speed_ground_max_ms']:.1f} m/s)")
            print(f"   Avg: {stats['speed_ground_mean_kmh']:.1f} km/h ({stats['speed_ground_mean_ms']:.1f} m/s)")

        # Attitude
        if stats.get('roll_max_deg'):
            print(f"\n🔄 Attitude (Max):")
            print(f"   Roll:  {stats['roll_max_deg']:.1f}°")
            print(f"   Pitch: {stats['pitch_max_deg']:.1f}°")

        # GPS
        if stats.get('gps_satellites_mean'):
            print(f"\n🛰️  GPS:")
            print(f"   Satellites: {stats['gps_satellites_mean']:.1f} avg ({stats['gps_satellites_min']}-{stats['gps_satellites_max']})")

        # Battery
        if stats.get('battery_voltage_mean_v'):
            print(f"\n🔋 Battery:")
            print(f"   Voltage: {stats['battery_voltage_mean_v']:.2f} V avg ({stats['battery_voltage_min_v']:.2f}-{stats['battery_voltage_max_v']:.2f} V)")
            if stats.get('battery_consumed_pct'):
                print(f"   Consumed: {stats['battery_consumed_pct']:.1f}%")

        print("\n" + "="*60 + "\n")

    def export_statistics_to_csv(self, output_file: str, stats: Optional[Dict[str, Any]] = None):
        """
        Export statistics to a CSV file.

        Args:
            output_file: Output CSV file path
            stats: Statistics dictionary. If None, calculates statistics first.
        """
        if stats is None:
            stats = self.get_statistics()

        # Flatten nested dictionaries
        flat_stats = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flat_stats[f"{key}_{subkey}"] = subvalue
            else:
                flat_stats[key] = value

        df = pd.DataFrame([flat_stats])
        df.to_csv(output_file, index=False)
        logger.info(f"Statistics exported to {output_file}")
