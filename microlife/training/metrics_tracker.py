"""
MetricsTracker - Training metrics collection and management
"""
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
from pathlib import Path


class MetricsTracker:
    """
    Track and manage training metrics with history.

    Features:
    - Automatic metric collection
    - Moving average smoothing
    - Statistical analysis
    - Export to CSV/JSON
    - Checkpoint save/load
    """

    def __init__(self, max_history: int = 10000, smooth_window: int = 20):
        """
        Initialize metrics tracker.

        Args:
            max_history: Maximum number of records to keep
            smooth_window: Window size for moving average
        """
        self.max_history = max_history
        self.smooth_window = smooth_window

        # Metrics storage
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.smooth_metrics: Dict[str, List[float]] = defaultdict(list)

        # Episode tracking
        self.episode = 0
        self.step = 0

        # Metadata
        self.metadata = {
            'start_time': None,
            'end_time': None,
            'total_episodes': 0,
            'total_steps': 0,
        }

    def record(self, **kwargs):
        """
        Record metrics for current step/episode.

        Args:
            **kwargs: Metric name-value pairs

        Example:
            tracker.record(loss=0.5, accuracy=0.8, reward=15.0)
        """
        # Update episode/step
        if 'episode' in kwargs:
            self.episode = kwargs['episode']
        if 'step' in kwargs:
            self.step = kwargs['step']

        # Record all metrics
        for key, value in kwargs.items():
            # Append to history
            self.metrics[key].append(float(value))

            # Maintain max history
            if len(self.metrics[key]) > self.max_history:
                self.metrics[key].pop(0)

            # Calculate smoothed value
            smooth_value = self._smooth(key)
            self.smooth_metrics[key].append(smooth_value)

            if len(self.smooth_metrics[key]) > self.max_history:
                self.smooth_metrics[key].pop(0)

        # Update metadata
        self.metadata['total_episodes'] = max(self.metadata['total_episodes'], self.episode)
        self.metadata['total_steps'] = max(self.metadata['total_steps'], self.step)

    def _smooth(self, metric_name: str) -> float:
        """Calculate moving average for metric."""
        values = self.metrics[metric_name]
        if not values:
            return 0.0

        # Take last N values
        window = min(self.smooth_window, len(values))
        recent = values[-window:]

        return sum(recent) / len(recent)

    def get(self, metric_name: str, last: Optional[int] = None,
            smoothed: bool = False) -> List[float]:
        """
        Get metric history.

        Args:
            metric_name: Name of metric
            last: Return only last N values (None = all)
            smoothed: Return smoothed values

        Returns:
            List of metric values
        """
        source = self.smooth_metrics if smoothed else self.metrics

        if metric_name not in source:
            return []

        values = source[metric_name]

        if last is not None:
            return values[-last:]

        return values.copy()

    def get_latest(self, metric_name: str, smoothed: bool = False) -> Optional[float]:
        """
        Get latest value of metric.

        Args:
            metric_name: Name of metric
            smoothed: Return smoothed value

        Returns:
            Latest value or None
        """
        values = self.get(metric_name, smoothed=smoothed)
        return values[-1] if values else None

    def get_statistics(self, metric_name: str, last: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate statistics for metric.

        Args:
            metric_name: Name of metric
            last: Calculate for last N values (None = all)

        Returns:
            Dictionary with mean, std, min, max
        """
        values = self.get(metric_name, last=last)

        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }

        values_array = np.array(values)

        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(values)
        }

    def get_all_metrics(self) -> List[str]:
        """Get list of all tracked metric names."""
        return list(self.metrics.keys())

    def get_metrics_dict(self, smoothed: bool = False) -> Dict[str, List[float]]:
        """
        Get all metrics as dictionary.

        Args:
            smoothed: Return smoothed values

        Returns:
            Dictionary of metric name -> values
        """
        source = self.smooth_metrics if smoothed else self.metrics
        return {k: v.copy() for k, v in source.items()}

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.smooth_metrics.clear()
        self.episode = 0
        self.step = 0
        self.metadata['total_episodes'] = 0
        self.metadata['total_steps'] = 0

    def export_csv(self, filepath: Union[str, Path]):
        """
        Export metrics to CSV file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        # Get all metric names
        metric_names = self.get_all_metrics()

        if not metric_names:
            print("⚠️  No metrics to export")
            return

        # Find max length
        max_len = max(len(self.metrics[name]) for name in metric_names)

        # Write CSV
        with open(filepath, 'w') as f:
            # Header
            f.write(','.join(metric_names) + '\n')

            # Data rows
            for i in range(max_len):
                row = []
                for name in metric_names:
                    values = self.metrics[name]
                    if i < len(values):
                        row.append(str(values[i]))
                    else:
                        row.append('')
                f.write(','.join(row) + '\n')

        print(f"✅ Exported metrics to {filepath}")

    def export_json(self, filepath: Union[str, Path]):
        """
        Export metrics to JSON file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        data = {
            'metrics': self.get_metrics_dict(),
            'smooth_metrics': self.get_metrics_dict(smoothed=True),
            'metadata': self.metadata,
            'config': {
                'max_history': self.max_history,
                'smooth_window': self.smooth_window,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Exported metrics to {filepath}")

    def save_checkpoint(self, filepath: Union[str, Path]):
        """
        Save tracker state to checkpoint.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)

        checkpoint = {
            'metrics': dict(self.metrics),
            'smooth_metrics': dict(self.smooth_metrics),
            'episode': self.episode,
            'step': self.step,
            'metadata': self.metadata,
            'max_history': self.max_history,
            'smooth_window': self.smooth_window,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)

        print(f"✅ Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: Union[str, Path]):
        """
        Load tracker state from checkpoint.

        Args:
            filepath: Input file path
        """
        filepath = Path(filepath)

        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)

        self.metrics = defaultdict(list, checkpoint['metrics'])
        self.smooth_metrics = defaultdict(list, checkpoint['smooth_metrics'])
        self.episode = checkpoint['episode']
        self.step = checkpoint['step']
        self.metadata = checkpoint['metadata']
        self.max_history = checkpoint['max_history']
        self.smooth_window = checkpoint['smooth_window']

        print(f"✅ Loaded checkpoint from {filepath}")

    def summary(self) -> str:
        """
        Generate summary string of tracked metrics.

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("Training Metrics Summary")
        lines.append("=" * 60)

        # Metadata
        lines.append(f"Episodes: {self.metadata['total_episodes']}")
        lines.append(f"Steps: {self.metadata['total_steps']}")
        lines.append("")

        # Metrics
        for name in sorted(self.get_all_metrics()):
            stats = self.get_statistics(name)
            latest = self.get_latest(name)
            latest_smooth = self.get_latest(name, smoothed=True)

            lines.append(f"{name}:")
            lines.append(f"  Latest: {latest:.4f} (smoothed: {latest_smooth:.4f})")
            lines.append(f"  Mean:   {stats['mean']:.4f}")
            lines.append(f"  Std:    {stats['std']:.4f}")
            lines.append(f"  Min:    {stats['min']:.4f}")
            lines.append(f"  Max:    {stats['max']:.4f}")
            lines.append("")

        lines.append("=" * 60)

        return '\n'.join(lines)

    def print_summary(self):
        """Print summary to console."""
        print(self.summary())

    def __repr__(self) -> str:
        metric_names = self.get_all_metrics()
        return f"MetricsTracker(metrics={len(metric_names)}, episodes={self.episode}, steps={self.step})"


class EMATracker:
    """
    Exponential Moving Average tracker for real-time smoothing.

    More efficient than simple moving average for large histories.
    """

    def __init__(self, alpha: float = 0.1):
        """
        Initialize EMA tracker.

        Args:
            alpha: Smoothing factor (0-1, lower = smoother)
        """
        self.alpha = alpha
        self.values: Dict[str, float] = {}

    def update(self, metric_name: str, value: float) -> float:
        """
        Update EMA for metric.

        Args:
            metric_name: Name of metric
            value: New value

        Returns:
            Smoothed value
        """
        if metric_name not in self.values:
            # Initialize with first value
            self.values[metric_name] = value
            return value

        # EMA formula: smooth = alpha * new + (1 - alpha) * smooth
        smooth = self.alpha * value + (1 - self.alpha) * self.values[metric_name]
        self.values[metric_name] = smooth

        return smooth

    def get(self, metric_name: str) -> Optional[float]:
        """Get current smoothed value."""
        return self.values.get(metric_name)

    def reset(self, metric_name: Optional[str] = None):
        """Reset EMA values."""
        if metric_name is None:
            self.values.clear()
        elif metric_name in self.values:
            del self.values[metric_name]
