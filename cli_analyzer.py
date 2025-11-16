#!/usr/bin/env python3
"""CLI Wrapper for Pixhawk Flight Analyzer"""
import sys
sys.path.insert(0, '/home/user/project-')

from pixhawk_flight_analyzer.cli import main

if __name__ == '__main__':
    main()
