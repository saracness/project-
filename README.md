# Pixhawk Flight Analyzer

A comprehensive Python package for loading, processing, analyzing, and visualizing Pixhawk/MAVLink telemetry data from drone flights.

## Features

- **Data Loading**: Load and parse .tlog and MAVLink binary log files
- **Data Processing**: Clean, filter, and process flight data
- **Statistical Analysis**: Generate comprehensive flight statistics
- **2D Visualization**: Create publication-quality plots with matplotlib
- **3D Interactive Visualization**: Generate interactive 3D flight paths with plotly
- **CLI Tool**: Easy-to-use command-line interface
- **Modular Design**: Use as a library or standalone tool

## Installation

### From Source

```bash
git clone https://github.com/yourusername/pixhawk-flight-analyzer.git
cd pixhawk-flight-analyzer
pip install -e .
```

### Using pip (when published)

```bash
pip install pixhawk-flight-analyzer
```

## Quick Start

### Command-Line Interface

#### Analyze a flight log

```bash
pixhawk-analyzer analyze flight.tlog
```

#### Generate visualizations

```bash
# Generate all visualizations
pixhawk-analyzer visualize flight.tlog

# Generate specific plot types
pixhawk-analyzer visualize flight.tlog --plot-type 3d --format html
pixhawk-analyzer visualize flight.tlog --plot-type dashboard --show
```

#### Complete analysis pipeline

```bash
# Run full analysis and generate all outputs
pixhawk-analyzer process flight.tlog --output-dir results
```

#### Get file information

```bash
pixhawk-analyzer info flight.tlog
```

### Python Library

```python
from pixhawk_flight_analyzer import (
    FlightDataLoader,
    FlightDataProcessor,
    FlightAnalyzer,
    FlightVisualizer
)

# Load flight data
loader = FlightDataLoader('flight.tlog')
data = loader.load()

# Process data
processor = FlightDataProcessor(data)
cleaned_data = processor.clean_data()

# Analyze
analyzer = FlightAnalyzer(data)
stats = analyzer.get_statistics()
analyzer.print_summary()

# Visualize
visualizer = FlightVisualizer(data)
visualizer.plot_flight_path_3d_interactive(save_path='flight_3d.html')
visualizer.plot_dashboard(save_path='dashboard.png')
```

## Detailed Usage

### Data Loading

```python
from pixhawk_flight_analyzer import FlightDataLoader

# Load telemetry file
loader = FlightDataLoader('flight.tlog')

# Load specific message types
data = loader.load(message_types=['GPS', 'ATTITUDE', 'BATTERY_STATUS'])

# Get basic file information
info = loader.get_basic_info()
print(f"Duration: {info['duration_seconds']} seconds")
print(f"Message types: {info['message_types']}")
```

### Data Processing

```python
from pixhawk_flight_analyzer import FlightDataProcessor

processor = FlightDataProcessor(data)

# Clean data (remove outliers, interpolate gaps)
cleaned_data = processor.clean_data()

# Extract flight path
path = processor.extract_flight_path()
print(path[['lat', 'lon', 'alt']].head())

# Extract attitude data
attitude = processor.extract_attitude()
print(attitude[['roll', 'pitch', 'yaw']].head())

# Apply low-pass filter
filtered_altitude = processor.apply_lowpass_filter('GPS', 'Alt', cutoff_freq=2.0)

# Calculate distance traveled
distance = processor.calculate_distance_traveled()
print(f"Total distance: {distance:.2f} meters")
```

### Statistical Analysis

```python
from pixhawk_flight_analyzer import FlightAnalyzer

analyzer = FlightAnalyzer(data)

# Get comprehensive statistics
stats = analyzer.get_statistics()

# Print formatted summary
analyzer.print_summary()

# Export to CSV
analyzer.export_statistics_to_csv('statistics.csv')

# Access specific statistics
print(f"Max altitude: {stats['altitude_max_m']} m")
print(f"Max speed: {stats['speed_ground_max_kmh']} km/h")
print(f"Flight duration: {stats['duration_minutes']} minutes")
```

### Visualization

```python
from pixhawk_flight_analyzer import FlightVisualizer

visualizer = FlightVisualizer(data)

# 2D flight path
visualizer.plot_flight_path_2d(save_path='flight_2d.png')

# 3D flight path (matplotlib)
visualizer.plot_flight_path_3d(save_path='flight_3d.png')

# Interactive 3D flight path (plotly)
visualizer.plot_flight_path_3d_interactive(save_path='flight_3d.html')

# Altitude profile
visualizer.plot_altitude_profile(save_path='altitude.png')

# Speed profile
visualizer.plot_speed_profile(save_path='speed.png')

# Attitude (roll, pitch, yaw)
visualizer.plot_attitude(save_path='attitude.png')

# Comprehensive dashboard
visualizer.plot_dashboard(save_path='dashboard.png')

# Interactive dashboard
visualizer.create_interactive_dashboard(save_path='dashboard.html')
```

## CLI Commands

### `analyze`

Analyze a flight log and display statistics.

```bash
pixhawk-analyzer analyze <file_path> [OPTIONS]

Options:
  -o, --output TEXT    Output CSV file for statistics
  -v, --verbose        Verbose output
```

### `visualize`

Generate visualizations from a flight log.

```bash
pixhawk-analyzer visualize <file_path> [OPTIONS]

Options:
  -o, --output-dir TEXT           Output directory [default: ./output]
  -f, --format [png|html|both]    Output format [default: png]
  -t, --plot-type [2d|3d|altitude|speed|attitude|dashboard|all]
                                  Type of plot [default: all]
  --show / --no-show              Display plots interactively [default: no-show]
```

### `process`

Complete analysis pipeline: load, analyze, and visualize.

```bash
pixhawk-analyzer process <file_path> [OPTIONS]

Options:
  -o, --output-dir TEXT           Output directory [default: ./output]
  -f, --format [png|html|both]    Output format [default: both]
```

### `info`

Display basic information about a flight log file.

```bash
pixhawk-analyzer info <file_path>
```

## Statistics Generated

The analyzer calculates the following statistics:

### Time Statistics
- Flight duration (seconds, minutes)
- Start and end timestamps

### Altitude Statistics
- Minimum, maximum, average altitude
- Altitude standard deviation
- Altitude range

### Speed Statistics
- Maximum and average ground speed (m/s and km/h)
- Maximum and average vertical speed
- Maximum climb and descent rates

### Distance Statistics
- Total distance traveled (meters, kilometers)

### Attitude Statistics
- Maximum roll, pitch, yaw angles
- Average and standard deviation for each axis

### GPS Statistics
- Satellite count (min, max, average)
- Horizontal dilution of precision (HDOP)
- GPS fix types

### Battery Statistics
- Voltage statistics (min, max, average)
- Current statistics (max, average)
- Battery consumption percentage

### Flight Mode Statistics
- Distribution of flight modes
- Base mode information

## Supported Message Types

The package supports all standard MAVLink message types, with special handling for:

- `GPS`, `GPS_RAW_INT`, `GLOBAL_POSITION_INT`: Position data
- `ATTITUDE`: Roll, pitch, yaw
- `LOCAL_POSITION_NED`: Local position and velocity
- `VFR_HUD`: Heads-up display data
- `BATTERY_STATUS`, `SYS_STATUS`: Battery information
- `HEARTBEAT`: Flight mode and status
- `RC_CHANNELS`: Radio control inputs
- And many more...

## Examples

See the `examples/` directory for complete usage examples:

- `example_usage.py`: Basic usage demonstration
- `advanced_analysis.py`: Advanced analysis techniques
- `custom_visualizations.py`: Creating custom plots

## Requirements

- Python >= 3.8
- pymavlink >= 2.4.36
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- plotly >= 5.0.0
- scipy >= 1.7.0
- click >= 8.0.0

## Project Structure

```
pixhawk-flight-analyzer/
├── pixhawk_flight_analyzer/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading functionality
│   ├── data_processor.py        # Data processing and cleaning
│   ├── statistics.py            # Statistical analysis
│   ├── visualization.py         # Visualization tools
│   └── cli.py                   # Command-line interface
├── examples/
│   └── example_usage.py         # Usage examples
├── tests/
│   └── test_basic.py            # Unit tests
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── LICENSE.txt                  # License information
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black pixhawk_flight_analyzer/
```

### Linting

```bash
flake8 pixhawk_flight_analyzer/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

## Acknowledgments

- [PyMAVLink](https://github.com/ArduPilot/pymavlink) for MAVLink protocol implementation
- [Matplotlib](https://matplotlib.org/) for static visualizations
- [Plotly](https://plotly.com/) for interactive visualizations
- ArduPilot and PX4 communities for documentation and support

## Support

For issues, questions, or contributions, please visit:
- GitHub Issues: https://github.com/yourusername/pixhawk-flight-analyzer/issues
- Documentation: https://pixhawk-flight-analyzer.readthedocs.io/

## Changelog

### Version 1.0.0 (Initial Release)

- Load and parse .tlog and MAVLink binary files
- Comprehensive data processing and cleaning
- Statistical analysis with 30+ metrics
- 2D and 3D visualizations
- Interactive dashboards
- Full CLI interface
- Python library API

---

**Happy Flying!** 🚁
