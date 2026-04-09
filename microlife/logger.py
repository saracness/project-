"""
SimulationLogger: writes per-step stats and event logs to ./logs/<timestamp>/.

Two files are created for each run:

  stats.csv   -- one CSV row per simulation step (import into pandas/Excel)
  events.log  -- timestamped human-readable notable events

The ./logs/ directory is already covered by [Ll]ogs/ in .gitignore,
so log files are never accidentally committed.

Basic usage:

    from microlife.logger import SimulationLogger

    logger = SimulationLogger(base_dir="logs")
    logger.log_event("Simulation started", {"organisms": 20})
    for step in range(N):
        env.update()
        logger.log_step(env.get_statistics())   # one CSV row per step
    logger.close()                              # flush + close files

    # or as a context manager:
    with SimulationLogger("logs") as logger:
        ...
"""
import csv
import json
import os
from datetime import datetime


class SimulationLogger:
    """
    Per-run file logger for MicroLife simulations.

    Args:
        base_dir (str): Parent directory for log folders.  Created if absent.
        enabled  (bool): False disables all file I/O — useful for tests or
                         when the caller passes --no-logs.
    """

    _CSV_FIELDS = [
        "timestep", "population", "total_organisms", "food_count",
        "avg_energy", "avg_age", "seeking_count", "wandering_count",
        "temperature_zones", "obstacles",
    ]

    def __init__(self, base_dir="logs", enabled=True):
        self.enabled = enabled
        self.run_dir = None
        self._csv_fh = None
        self._csv_writer = None
        self._log_fh = None

        if not enabled:
            return

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Per-step CSV metrics
        csv_path = os.path.join(self.run_dir, "stats.csv")
        self._csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_fh,
            fieldnames=self._CSV_FIELDS,
            extrasaction="ignore",
        )
        self._csv_writer.writeheader()

        # Human-readable event log
        log_path = os.path.join(self.run_dir, "events.log")
        self._log_fh = open(log_path, "w", encoding="utf-8")
        self._log_fh.write(f"# MicroLife run — {run_id}\n\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_step(self, stats: dict) -> None:
        """Write one CSV row for the current timestep.

        Expects the dict returned by Environment.get_statistics().
        Extra keys are silently ignored via extrasaction='ignore'.
        """
        if not self.enabled or self._csv_writer is None:
            return
        self._csv_writer.writerow(stats)
        self._csv_fh.flush()

    def log_event(self, message: str, data: dict = None) -> None:
        """Append a timestamped event line to events.log and print to stdout.

        Args:
            message: Short description of the event.
            data:    Optional dict of structured details (serialised as JSON).
        """
        ts   = datetime.now().strftime("%H:%M:%S")
        body = ("  " + json.dumps(data)) if data else ""
        line = f"[{ts}] {message}{body}\n"

        print(line, end="")

        if not self.enabled or self._log_fh is None:
            return
        self._log_fh.write(line)
        self._log_fh.flush()

    def close(self) -> None:
        """Flush and close all open file handles."""
        for fh in (self._csv_fh, self._log_fh):
            if fh and not fh.closed:
                fh.close()

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
