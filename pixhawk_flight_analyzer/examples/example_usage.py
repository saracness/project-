"""
Example Usage of Pixhawk Flight Analyzer
=========================================

This script demonstrates the basic usage of the pixhawk-flight-analyzer package.
"""

from pixhawk_flight_analyzer import (
    FlightDataLoader,
    FlightDataProcessor,
    FlightAnalyzer,
    FlightVisualizer
)


def main():
    # Replace with your actual flight log file
    flight_log_file = 'flight.tlog'

    print("="*60)
    print("Pixhawk Flight Analyzer - Example Usage")
    print("="*60)

    # ========================================
    # 1. Load Flight Data
    # ========================================
    print("\n1. Loading flight data...")
    loader = FlightDataLoader(flight_log_file)

    # Load all common message types
    data = loader.load()

    # Or load specific message types
    # data = loader.load(message_types=['GPS', 'ATTITUDE', 'BATTERY_STATUS'])

    print(f"   Loaded {len(data)} message types")
    print(f"   Available types: {loader.get_message_types()}")

    # Get basic file information
    info = loader.get_basic_info()
    print(f"   Duration: {info.get('duration_seconds', 'N/A')} seconds")

    # ========================================
    # 2. Process Data
    # ========================================
    print("\n2. Processing flight data...")
    processor = FlightDataProcessor(data)

    # Clean data (remove outliers and interpolate gaps)
    cleaned_data = processor.clean_data(
        remove_outliers=True,
        interpolate_gaps=True,
        max_gap_size=10
    )

    # Extract flight path
    flight_path = processor.extract_flight_path()
    if flight_path is not None:
        print(f"   Flight path: {len(flight_path)} points")
        print(f"   Lat: {flight_path['lat'].min():.6f} to {flight_path['lat'].max():.6f}")
        print(f"   Lon: {flight_path['lon'].min():.6f} to {flight_path['lon'].max():.6f}")
        print(f"   Alt: {flight_path['alt'].min():.1f} to {flight_path['alt'].max():.1f} m")

    # Extract attitude data
    attitude = processor.extract_attitude()
    if attitude is not None:
        print(f"   Attitude data: {len(attitude)} points")

    # Calculate distance traveled
    distance = processor.calculate_distance_traveled()
    if distance is not None:
        print(f"   Distance traveled: {distance:.2f} meters ({distance/1000:.2f} km)")

    # ========================================
    # 3. Statistical Analysis
    # ========================================
    print("\n3. Analyzing flight data...")
    analyzer = FlightAnalyzer(data)

    # Get all statistics
    stats = analyzer.get_statistics()

    # Print formatted summary
    analyzer.print_summary(stats)

    # Export statistics to CSV
    analyzer.export_statistics_to_csv('flight_statistics.csv', stats)
    print("   Statistics exported to: flight_statistics.csv")

    # Access specific statistics
    print("\n   Key Statistics:")
    if stats.get('duration_minutes'):
        print(f"   - Duration: {stats['duration_minutes']:.2f} minutes")
    if stats.get('altitude_max_m'):
        print(f"   - Max altitude: {stats['altitude_max_m']:.1f} m")
    if stats.get('speed_ground_max_kmh'):
        print(f"   - Max speed: {stats['speed_ground_max_kmh']:.1f} km/h")
    if stats.get('distance_traveled_km'):
        print(f"   - Distance: {stats['distance_traveled_km']:.2f} km")

    # ========================================
    # 4. Visualization
    # ========================================
    print("\n4. Generating visualizations...")
    visualizer = FlightVisualizer(data)

    # 2D flight path
    print("   - Creating 2D flight path...")
    visualizer.plot_flight_path_2d(
        save_path='flight_path_2d.png',
        show=False
    )

    # 3D flight path (static)
    print("   - Creating 3D flight path (static)...")
    visualizer.plot_flight_path_3d(
        save_path='flight_path_3d.png',
        show=False
    )

    # 3D flight path (interactive)
    print("   - Creating 3D flight path (interactive)...")
    visualizer.plot_flight_path_3d_interactive(
        save_path='flight_path_3d_interactive.html',
        show=False
    )

    # Altitude profile
    print("   - Creating altitude profile...")
    visualizer.plot_altitude_profile(
        save_path='altitude_profile.png',
        show=False
    )

    # Speed profile
    print("   - Creating speed profile...")
    visualizer.plot_speed_profile(
        save_path='speed_profile.png',
        show=False
    )

    # Attitude plot
    print("   - Creating attitude plot...")
    visualizer.plot_attitude(
        save_path='attitude_plot.png',
        show=False
    )

    # Comprehensive dashboard
    print("   - Creating comprehensive dashboard...")
    visualizer.plot_dashboard(
        save_path='flight_dashboard.png',
        show=False
    )

    # Interactive dashboard
    print("   - Creating interactive dashboard...")
    visualizer.create_interactive_dashboard(
        save_path='flight_dashboard_interactive.html',
        show=False
    )

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - flight_statistics.csv")
    print("  - flight_path_2d.png")
    print("  - flight_path_3d.png")
    print("  - flight_path_3d_interactive.html")
    print("  - altitude_profile.png")
    print("  - speed_profile.png")
    print("  - attitude_plot.png")
    print("  - flight_dashboard.png")
    print("  - flight_dashboard_interactive.html")
    print()


if __name__ == '__main__':
    main()
