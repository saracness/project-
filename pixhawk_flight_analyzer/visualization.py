"""
Visualization Module
====================

This module provides 2D and 3D visualization capabilities for flight data.
Uses matplotlib for 2D plots and plotly for interactive 3D visualizations.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .data_processor import FlightDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightVisualizer:
    """
    Visualize flight data using matplotlib and plotly.

    Features:
    - 2D flight path visualization
    - 3D interactive flight path
    - Altitude profile
    - Speed profile
    - Attitude (roll, pitch, yaw) plots
    - Multi-parameter time series
    - GPS quality visualization

    Example:
        visualizer = FlightVisualizer(data)
        visualizer.plot_flight_path_2d(save_path='flight_path.png')
        visualizer.plot_flight_path_3d_interactive(save_path='flight_path_3d.html')
    """

    def __init__(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the visualizer with flight data.

        Args:
            data: Dictionary of DataFrames from FlightDataLoader
        """
        self.data = data
        self.processor = FlightDataProcessor(data)

    def plot_flight_path_2d(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot 2D flight path (top-down view) using matplotlib.

        Args:
            save_path: Path to save the plot. If None, doesn't save.
            show: Whether to display the plot

        Example:
            visualizer.plot_flight_path_2d(save_path='flight_2d.png')
        """
        path = self.processor.extract_flight_path()
        if path is None:
            logger.error("No flight path data available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot path with color gradient based on altitude
        scatter = ax.scatter(path['lon'], path['lat'],
                           c=path['alt'], cmap='viridis',
                           s=10, alpha=0.6)

        # Mark start and end
        ax.plot(path['lon'].iloc[0], path['lat'].iloc[0],
               'go', markersize=15, label='Start', zorder=5)
        ax.plot(path['lon'].iloc[-1], path['lat'].iloc[-1],
               'ro', markersize=15, label='End', zorder=5)

        # Colorbar for altitude
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Altitude (m)', rotation=270, labelpad=20)

        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title('Flight Path (2D - Top View)', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"2D flight path saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_flight_path_3d(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot 3D flight path using matplotlib.

        Args:
            save_path: Path to save the plot
            show: Whether to display the plot

        Example:
            visualizer.plot_flight_path_3d(save_path='flight_3d.png')
        """
        path = self.processor.extract_flight_path()
        if path is None:
            logger.error("No flight path data available")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize lat/lon to meters (approximate)
        lat_center = path['lat'].mean()
        lon_center = path['lon'].mean()

        # Convert to meters (approximation)
        lat_m = (path['lat'] - lat_center) * 111320  # 1 degree lat ≈ 111.32 km
        lon_m = (path['lon'] - lon_center) * 111320 * np.cos(np.radians(lat_center))

        # Plot path with color gradient
        scatter = ax.scatter(lon_m, lat_m, path['alt'],
                           c=path['alt'], cmap='viridis',
                           s=10, alpha=0.6)

        # Mark start and end
        ax.scatter(lon_m.iloc[0], lat_m.iloc[0], path['alt'].iloc[0],
                  color='green', s=200, marker='o', label='Start', zorder=5)
        ax.scatter(lon_m.iloc[-1], lat_m.iloc[-1], path['alt'].iloc[-1],
                  color='red', s=200, marker='o', label='End', zorder=5)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Altitude (m)', rotation=270, labelpad=20)

        ax.set_xlabel('East-West (m)')
        ax.set_ylabel('North-South (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('Flight Path (3D)', fontsize=16, fontweight='bold')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"3D flight path saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_flight_path_3d_interactive(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create interactive 3D flight path using plotly.

        Args:
            save_path: Path to save HTML file
            show: Whether to display the plot in browser

        Example:
            visualizer.plot_flight_path_3d_interactive(save_path='flight_3d.html')
        """
        path = self.processor.extract_flight_path()
        if path is None:
            logger.error("No flight path data available")
            return

        # Normalize lat/lon to meters
        lat_center = path['lat'].mean()
        lon_center = path['lon'].mean()

        lat_m = (path['lat'] - lat_center) * 111320
        lon_m = (path['lon'] - lon_center) * 111320 * np.cos(np.radians(lat_center))

        # Create 3D scatter plot
        fig = go.Figure()

        # Flight path
        fig.add_trace(go.Scatter3d(
            x=lon_m,
            y=lat_m,
            z=path['alt'],
            mode='lines+markers',
            marker=dict(
                size=3,
                color=path['alt'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Altitude (m)")
            ),
            line=dict(color='darkblue', width=2),
            name='Flight Path'
        ))

        # Start point
        fig.add_trace(go.Scatter3d(
            x=[lon_m.iloc[0]],
            y=[lat_m.iloc[0]],
            z=[path['alt'].iloc[0]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start'
        ))

        # End point
        fig.add_trace(go.Scatter3d(
            x=[lon_m.iloc[-1]],
            y=[lat_m.iloc[-1]],
            z=[path['alt'].iloc[-1]],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='End'
        ))

        fig.update_layout(
            title='Interactive 3D Flight Path',
            scene=dict(
                xaxis_title='East-West (m)',
                yaxis_title='North-South (m)',
                zaxis_title='Altitude (m)',
                aspectmode='data'
            ),
            width=1200,
            height=800
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive 3D plot saved to {save_path}")

        if show:
            fig.show()

        return fig

    def plot_altitude_profile(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot altitude vs time profile.

        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        path = self.processor.extract_flight_path()
        if path is None or 'timestamp' not in path.columns:
            logger.error("No altitude data with timestamps available")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        time_seconds = (path['timestamp'] - path['timestamp'].iloc[0]) / 1000.0

        ax.plot(time_seconds, path['alt'], linewidth=2, color='#2E86AB')
        ax.fill_between(time_seconds, path['alt'], alpha=0.3, color='#2E86AB')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Altitude (m)', fontsize=12)
        ax.set_title('Altitude Profile', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Altitude profile saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_speed_profile(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot speed vs time profile.

        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        velocity = self.processor.extract_velocity()
        if velocity is None or 'timestamp' not in velocity.columns:
            logger.error("No velocity data with timestamps available")
            return

        # Calculate ground speed
        ground_speed = np.sqrt(velocity['vx']**2 + velocity['vy']**2)
        time_seconds = (velocity['timestamp'] - velocity['timestamp'].iloc[0]) / 1000.0

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(time_seconds, ground_speed, linewidth=2, color='#A23B72', label='Ground Speed')
        ax.plot(time_seconds, abs(velocity['vz']), linewidth=2, color='#F18F01',
               alpha=0.7, label='Vertical Speed')

        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Speed (m/s)', fontsize=12)
        ax.set_title('Speed Profile', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Speed profile saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_attitude(self, save_path: Optional[str] = None, show: bool = True):
        """
        Plot roll, pitch, yaw vs time.

        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        attitude = self.processor.extract_attitude()
        if attitude is None or 'timestamp' not in attitude.columns:
            logger.error("No attitude data with timestamps available")
            return

        time_seconds = (attitude['timestamp'] - attitude['timestamp'].iloc[0]) / 1000.0

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Roll
        axes[0].plot(time_seconds, attitude['roll'], linewidth=1.5, color='#E63946')
        axes[0].set_ylabel('Roll (°)', fontsize=11)
        axes[0].set_title('Attitude Over Time', fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Pitch
        axes[1].plot(time_seconds, attitude['pitch'], linewidth=1.5, color='#F77F00')
        axes[1].set_ylabel('Pitch (°)', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Yaw
        axes[2].plot(time_seconds, attitude['yaw'], linewidth=1.5, color='#06A77D')
        axes[2].set_ylabel('Yaw (°)', fontsize=11)
        axes[2].set_xlabel('Time (seconds)', fontsize=12)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attitude plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_dashboard(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create a comprehensive dashboard with multiple plots.

        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Flight path 2D
        ax1 = fig.add_subplot(gs[0, 0])
        path = self.processor.extract_flight_path()
        if path is not None:
            scatter = ax1.scatter(path['lon'], path['lat'],
                               c=path['alt'], cmap='viridis', s=5)
            ax1.plot(path['lon'].iloc[0], path['lat'].iloc[0], 'go', markersize=10)
            ax1.plot(path['lon'].iloc[-1], path['lat'].iloc[-1], 'ro', markersize=10)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('Flight Path (2D)')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Alt (m)')

        # 2. Altitude profile
        ax2 = fig.add_subplot(gs[0, 1])
        if path is not None and 'timestamp' in path.columns:
            time_s = (path['timestamp'] - path['timestamp'].iloc[0]) / 1000.0
            ax2.plot(time_s, path['alt'], linewidth=2)
            ax2.fill_between(time_s, path['alt'], alpha=0.3)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Altitude (m)')
            ax2.set_title('Altitude Profile')
            ax2.grid(True, alpha=0.3)

        # 3. Speed profile
        ax3 = fig.add_subplot(gs[1, 0])
        velocity = self.processor.extract_velocity()
        if velocity is not None and 'timestamp' in velocity.columns:
            ground_speed = np.sqrt(velocity['vx']**2 + velocity['vy']**2)
            time_s = (velocity['timestamp'] - velocity['timestamp'].iloc[0]) / 1000.0
            ax3.plot(time_s, ground_speed, linewidth=2, label='Ground')
            ax3.plot(time_s, abs(velocity['vz']), linewidth=2, alpha=0.7, label='Vertical')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Speed (m/s)')
            ax3.set_title('Speed Profile')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Attitude - Roll & Pitch
        ax4 = fig.add_subplot(gs[1, 1])
        attitude = self.processor.extract_attitude()
        if attitude is not None and 'timestamp' in attitude.columns:
            time_s = (attitude['timestamp'] - attitude['timestamp'].iloc[0]) / 1000.0
            ax4.plot(time_s, attitude['roll'], linewidth=1.5, label='Roll', alpha=0.8)
            ax4.plot(time_s, attitude['pitch'], linewidth=1.5, label='Pitch', alpha=0.8)
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Angle (°)')
            ax4.set_title('Roll & Pitch')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # 5. GPS Satellites
        ax5 = fig.add_subplot(gs[2, 0])
        if 'GPS_RAW_INT' in self.data and 'satellites_visible' in self.data['GPS_RAW_INT'].columns:
            gps = self.data['GPS_RAW_INT']
            if 'time_boot_ms' in gps.columns:
                time_s = (gps['time_boot_ms'] - gps['time_boot_ms'].iloc[0]) / 1000.0
                ax5.plot(time_s, gps['satellites_visible'], linewidth=2)
                ax5.set_xlabel('Time (s)')
                ax5.set_ylabel('Satellites')
                ax5.set_title('GPS Satellites Visible')
                ax5.grid(True, alpha=0.3)

        # 6. Battery voltage
        ax6 = fig.add_subplot(gs[2, 1])
        if 'SYS_STATUS' in self.data and 'voltage_battery' in self.data['SYS_STATUS'].columns:
            sys_status = self.data['SYS_STATUS']
            if 'time_boot_ms' in sys_status.columns:
                time_s = (sys_status['time_boot_ms'] - sys_status['time_boot_ms'].iloc[0]) / 1000.0
                voltage_v = sys_status['voltage_battery'] / 1000.0
                ax6.plot(time_s, voltage_v, linewidth=2, color='#E63946')
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Voltage (V)')
                ax6.set_title('Battery Voltage')
                ax6.grid(True, alpha=0.3)

        fig.suptitle('Flight Data Dashboard', fontsize=20, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_interactive_dashboard(self, save_path: Optional[str] = None, show: bool = True):
        """
        Create an interactive dashboard using plotly.

        Args:
            save_path: Path to save HTML file
            show: Whether to open in browser
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Flight Path 2D', 'Altitude Profile',
                          'Speed Profile', 'Roll & Pitch',
                          'GPS Satellites', 'Battery Voltage'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Flight path 2D
        path = self.processor.extract_flight_path()
        if path is not None:
            fig.add_trace(
                go.Scatter(x=path['lon'], y=path['lat'],
                          mode='markers',
                          marker=dict(size=4, color=path['alt'], colorscale='Viridis'),
                          name='Path'),
                row=1, col=1
            )

        # 2. Altitude profile
        if path is not None and 'timestamp' in path.columns:
            time_s = (path['timestamp'] - path['timestamp'].iloc[0]) / 1000.0
            fig.add_trace(
                go.Scatter(x=time_s, y=path['alt'],
                          mode='lines', name='Altitude', line=dict(width=2)),
                row=1, col=2
            )

        # 3. Speed profile
        velocity = self.processor.extract_velocity()
        if velocity is not None and 'timestamp' in velocity.columns:
            ground_speed = np.sqrt(velocity['vx']**2 + velocity['vy']**2)
            time_s = (velocity['timestamp'] - velocity['timestamp'].iloc[0]) / 1000.0
            fig.add_trace(
                go.Scatter(x=time_s, y=ground_speed,
                          mode='lines', name='Ground Speed'),
                row=2, col=1
            )

        # 4. Attitude
        attitude = self.processor.extract_attitude()
        if attitude is not None and 'timestamp' in attitude.columns:
            time_s = (attitude['timestamp'] - attitude['timestamp'].iloc[0]) / 1000.0
            fig.add_trace(
                go.Scatter(x=time_s, y=attitude['roll'],
                          mode='lines', name='Roll'),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=time_s, y=attitude['pitch'],
                          mode='lines', name='Pitch'),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title_text="Interactive Flight Dashboard",
            showlegend=True,
            height=1000,
            width=1400
        )

        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")

        if show:
            fig.show()

        return fig
