"""
Data Loader Module
==================

This module handles loading and parsing of Pixhawk/MAVLink telemetry files.
Supports .tlog files and MAVLink binary logs.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pymavlink import mavutil
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightDataLoader:
    """
    Load and parse Pixhawk/MAVLink telemetry files.

    Supports:
    - .tlog files (telemetry logs)
    - .bin files (binary logs)
    - MAVLink streams

    Example:
        loader = FlightDataLoader('flight.tlog')
        data = loader.load()
        print(loader.get_message_types())
    """

    def __init__(self, file_path: str):
        """
        Initialize the data loader.

        Args:
            file_path: Path to the telemetry file (.tlog or .bin)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()
        self.mlog = None
        self.messages = {}
        self.message_types = set()

        if self.file_extension not in ['.tlog', '.bin']:
            logger.warning(f"File extension {self.file_extension} may not be supported. Attempting to load anyway.")

    def load(self, message_types: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Load telemetry data from the file.

        Args:
            message_types: List of specific MAVLink message types to load.
                          If None, loads common message types.

        Returns:
            Dictionary mapping message types to DataFrames containing the data

        Example:
            data = loader.load(['GPS', 'ATTITUDE', 'GLOBAL_POSITION_INT'])
        """
        logger.info(f"Loading telemetry file: {self.file_path}")

        try:
            self.mlog = mavutil.mavlink_connection(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to open MAVLink file: {e}")

        # Default message types if none specified
        if message_types is None:
            message_types = [
                'GPS', 'GPS_RAW_INT', 'GLOBAL_POSITION_INT',
                'ATTITUDE', 'LOCAL_POSITION_NED',
                'VFR_HUD', 'HEARTBEAT', 'SYS_STATUS',
                'BATTERY_STATUS', 'RC_CHANNELS',
                'SERVO_OUTPUT_RAW', 'MISSION_CURRENT',
                'NAV_CONTROLLER_OUTPUT', 'STATUSTEXT'
            ]

        # Initialize message storage
        for msg_type in message_types:
            self.messages[msg_type] = []

        # Parse all messages
        logger.info("Parsing messages...")
        message_count = 0

        while True:
            try:
                msg = self.mlog.recv_match(blocking=False)
                if msg is None:
                    break

                msg_type = msg.get_type()
                self.message_types.add(msg_type)

                if msg_type in self.messages:
                    # Convert message to dictionary
                    msg_dict = msg.to_dict()
                    self.messages[msg_type].append(msg_dict)
                    message_count += 1

            except Exception as e:
                logger.warning(f"Error parsing message: {e}")
                continue

        logger.info(f"Loaded {message_count} messages of {len(self.message_types)} different types")

        # Convert lists to DataFrames
        dataframes = {}
        for msg_type, msg_list in self.messages.items():
            if msg_list:
                df = pd.DataFrame(msg_list)
                # Convert timestamps if present
                if 'time_boot_ms' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time_boot_ms'], unit='ms')
                dataframes[msg_type] = df
                logger.info(f"  {msg_type}: {len(df)} messages")

        return dataframes

    def get_message_types(self) -> List[str]:
        """
        Get list of all message types found in the file.

        Returns:
            List of message type names
        """
        return sorted(list(self.message_types))

    def get_flight_duration(self) -> Optional[float]:
        """
        Get the total flight duration in seconds.

        Returns:
            Duration in seconds, or None if cannot be determined
        """
        if 'GPS' in self.messages and self.messages['GPS']:
            times = [msg.get('time_boot_ms', 0) for msg in self.messages['GPS']]
            if times:
                return (max(times) - min(times)) / 1000.0

        if 'GLOBAL_POSITION_INT' in self.messages and self.messages['GLOBAL_POSITION_INT']:
            times = [msg.get('time_boot_ms', 0) for msg in self.messages['GLOBAL_POSITION_INT']]
            if times:
                return (max(times) - min(times)) / 1000.0

        return None

    def get_basic_info(self) -> Dict[str, Any]:
        """
        Get basic information about the flight log.

        Returns:
            Dictionary with file info, message counts, duration, etc.
        """
        info = {
            'file_path': self.file_path,
            'file_size_mb': os.path.getsize(self.file_path) / (1024 * 1024),
            'message_types_count': len(self.message_types),
            'message_types': self.get_message_types(),
            'duration_seconds': self.get_flight_duration(),
        }

        # Add message counts
        for msg_type, msg_list in self.messages.items():
            info[f'count_{msg_type}'] = len(msg_list)

        return info

    @staticmethod
    def is_valid_file(file_path: str) -> bool:
        """
        Check if a file is a valid telemetry file.

        Args:
            file_path: Path to check

        Returns:
            True if file appears to be a valid telemetry file
        """
        if not os.path.exists(file_path):
            return False

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.tlog', '.bin']:
            return False

        try:
            mlog = mavutil.mavlink_connection(file_path)
            msg = mlog.recv_match(blocking=False)
            return msg is not None
        except:
            return False
