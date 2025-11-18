"""
DataLogger - Background scientific data collection system
"""
import sqlite3
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np


class DataLogger:
    """
    Background data logger with automatic database storage.

    Features:
    - Non-blocking background logging
    - SQLite database storage
    - Auto-save graphs at intervals
    - Memory-efficient buffering
    - Thread-safe operations
    """

    def __init__(self,
                 db_path: str = 'outputs/data/metrics.db',
                 sampling_rate: int = 1,
                 buffer_size: int = 1000,
                 auto_save_graphs: bool = True,
                 graph_interval: int = 100):
        """
        Initialize data logger.

        Args:
            db_path: Database file path
            sampling_rate: Log every N steps (1 = every step)
            buffer_size: Buffer size before auto-flush
            auto_save_graphs: Automatically save graphs
            graph_interval: Save graphs every N episodes
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.auto_save_graphs = auto_save_graphs
        self.graph_interval = graph_interval

        # Initialize database
        self._init_database()

        # Data buffer (thread-safe)
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_lock = threading.Lock()

        # Counters
        self.step_count = 0
        self.last_graph_save = 0

        # Background thread
        self.running = False
        self.flush_thread = None

        # Plotter (lazy initialization)
        self.plotter = None

        print(f"✅ DataLogger initialized: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database with schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Main metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                episode INTEGER,
                step INTEGER,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                metadata TEXT
            )
        ''')

        # Population snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS population_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                episode INTEGER,
                organism_count INTEGER,
                total_energy REAL,
                avg_age REAL,
                birth_count INTEGER,
                death_count INTEGER,
                food_count INTEGER
            )
        ''')

        # Organism details table (sampled)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS organism_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                episode INTEGER,
                organism_id INTEGER,
                x REAL,
                y REAL,
                energy REAL,
                age INTEGER,
                brain_type TEXT,
                alive INTEGER
            )
        ''')

        # Experiment metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiment_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL,
                end_time REAL,
                total_episodes INTEGER,
                config TEXT,
                notes TEXT
            )
        ''')

        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_episode ON metrics(episode)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric_name ON metrics(metric_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)')

        conn.commit()
        conn.close()

    def start(self):
        """Start background flush thread."""
        if not self.running:
            self.running = True
            self.flush_thread = threading.Thread(target=self._background_flush, daemon=True)
            self.flush_thread.start()
            print("📊 DataLogger background thread started")

    def stop(self):
        """Stop background thread and flush remaining data."""
        if self.running:
            self.running = False
            if self.flush_thread:
                self.flush_thread.join(timeout=5.0)
            self.flush()
            print("📊 DataLogger stopped")

    def log_step(self, episode: int, organisms: List[Any],
                 environment: Any = None, **kwargs):
        """
        Log simulation step data.

        Args:
            episode: Current episode number
            organisms: List of organisms
            environment: Environment object
            **kwargs: Additional metrics to log
        """
        self.step_count += 1

        # Sample based on sampling rate
        if self.step_count % self.sampling_rate != 0:
            return

        timestamp = time.time()

        # Collect population metrics
        if organisms:
            organism_count = len(organisms)
            total_energy = sum(getattr(org, 'energy', 0) for org in organisms)
            avg_energy = total_energy / organism_count if organism_count > 0 else 0
            avg_age = sum(getattr(org, 'age', 0) for org in organisms) / organism_count if organism_count > 0 else 0

            # Buffer data
            with self.buffer_lock:
                self.buffer.append({
                    'type': 'snapshot',
                    'timestamp': timestamp,
                    'episode': episode,
                    'organism_count': organism_count,
                    'total_energy': total_energy,
                    'avg_energy': avg_energy,
                    'avg_age': avg_age,
                    'food_count': len(getattr(environment, 'food_sources', [])) if environment else 0
                })

                # Log custom metrics
                for key, value in kwargs.items():
                    self.buffer.append({
                        'type': 'metric',
                        'timestamp': timestamp,
                        'episode': episode,
                        'step': self.step_count,
                        'name': key,
                        'value': float(value) if isinstance(value, (int, float, np.number)) else 0
                    })

        # Auto-flush if buffer is full
        if len(self.buffer) >= self.buffer_size * 0.9:
            threading.Thread(target=self.flush, daemon=True).start()

        # Auto-save graphs
        if self.auto_save_graphs and episode > 0 and episode % self.graph_interval == 0:
            if episode != self.last_graph_save:
                self.last_graph_save = episode
                threading.Thread(target=self._save_graphs, args=(episode,), daemon=True).start()

    def log_metric(self, episode: int, metric_name: str, value: float,
                   metadata: Optional[Dict] = None):
        """
        Log a single metric.

        Args:
            episode: Episode number
            metric_name: Metric name
            value: Metric value
            metadata: Optional metadata dict
        """
        timestamp = time.time()

        with self.buffer_lock:
            self.buffer.append({
                'type': 'metric',
                'timestamp': timestamp,
                'episode': episode,
                'step': self.step_count,
                'name': metric_name,
                'value': float(value),
                'metadata': json.dumps(metadata) if metadata else None
            })

    def flush(self):
        """Flush buffer to database."""
        with self.buffer_lock:
            if not self.buffer:
                return

            # Copy buffer and clear
            data_to_write = list(self.buffer)
            self.buffer.clear()

        # Write to database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            for item in data_to_write:
                if item['type'] == 'snapshot':
                    cursor.execute('''
                        INSERT INTO population_snapshots
                        (timestamp, episode, organism_count, total_energy, avg_age, birth_count, death_count, food_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        item['timestamp'],
                        item['episode'],
                        item['organism_count'],
                        item['total_energy'],
                        item['avg_age'],
                        0,  # birth_count (can be tracked separately)
                        0,  # death_count
                        item['food_count']
                    ))
                elif item['type'] == 'metric':
                    cursor.execute('''
                        INSERT INTO metrics
                        (timestamp, episode, step, metric_name, metric_value, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        item['timestamp'],
                        item['episode'],
                        item['step'],
                        item['name'],
                        item['value'],
                        item.get('metadata')
                    ))

            conn.commit()
        except Exception as e:
            print(f"❌ Error flushing data: {e}")
        finally:
            conn.close()

    def _background_flush(self):
        """Background thread for periodic flushing."""
        while self.running:
            time.sleep(5.0)  # Flush every 5 seconds
            self.flush()

    def _save_graphs(self, episode: int):
        """Save auto-graphs at interval."""
        try:
            # Lazy import and initialize plotter
            if self.plotter is None:
                from .scientific_plotter import ScientificPlotter
                self.plotter = ScientificPlotter(self.db_path)

            # Generate and save graphs
            output_dir = Path('outputs/graphs')
            output_dir.mkdir(parents=True, exist_ok=True)

            # Population dynamics
            self.plotter.plot_population_dynamics(
                save_path=output_dir / f'population_ep{episode}.png'
            )

            # Energy over time
            self.plotter.plot_metric_timeseries(
                'avg_energy',
                save_path=output_dir / f'energy_ep{episode}.png'
            )

            print(f"📊 Auto-saved graphs at episode {episode}")

        except Exception as e:
            print(f"⚠️  Could not save graphs: {e}")

    def query(self, metric_name: str, episode_start: int = 0,
              episode_end: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Query metric data.

        Args:
            metric_name: Metric to query
            episode_start: Start episode
            episode_end: End episode (None = all)

        Returns:
            List of (episode, value) tuples
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        if episode_end is None:
            cursor.execute('''
                SELECT episode, metric_value
                FROM metrics
                WHERE metric_name = ? AND episode >= ?
                ORDER BY episode
            ''', (metric_name, episode_start))
        else:
            cursor.execute('''
                SELECT episode, metric_value
                FROM metrics
                WHERE metric_name = ? AND episode >= ? AND episode <= ?
                ORDER BY episode
            ''', (metric_name, episode_start, episode_end))

        results = cursor.fetchall()
        conn.close()

        return results

    def get_population_history(self) -> List[Dict[str, Any]]:
        """Get population snapshot history."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT episode, organism_count, total_energy, avg_age, food_count
            FROM population_snapshots
            ORDER BY episode
        ''')

        results = []
        for row in cursor.fetchall():
            results.append({
                'episode': row[0],
                'organism_count': row[1],
                'total_energy': row[2],
                'avg_age': row[3],
                'food_count': row[4]
            })

        conn.close()
        return results

    def export_csv(self, output_path: str = 'outputs/data/export.csv'):
        """Export all data to CSV."""
        import pandas as pd

        conn = sqlite3.connect(str(self.db_path))

        # Read all metrics
        df = pd.read_sql_query('SELECT * FROM metrics', conn)
        df.to_csv(output_path, index=False)

        conn.close()
        print(f"✅ Exported data to {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM metrics')
        total_records = cursor.fetchone()[0]

        cursor.execute('SELECT MAX(episode) FROM metrics')
        max_episode = cursor.fetchone()[0] or 0

        cursor.execute('SELECT COUNT(DISTINCT metric_name) FROM metrics')
        unique_metrics = cursor.fetchone()[0]

        conn.close()

        return {
            'total_records': total_records,
            'max_episode': max_episode,
            'unique_metrics': unique_metrics,
            'database_path': str(self.db_path)
        }

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
