"""
StatisticalAnalyzer - Advanced statistical analysis
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from scipy import stats


class StatisticalAnalyzer:
    """
    Perform statistical analysis on collected data.

    Features:
    - Descriptive statistics
    - Hypothesis testing
    - Correlation analysis
    - Trend detection
    - Anomaly detection
    """

    def __init__(self, db_path: str):
        """
        Initialize statistical analyzer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)

    def get_descriptive_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Calculate descriptive statistics for a metric.

        Args:
            metric_name: Metric name

        Returns:
            Dictionary of statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            'SELECT metric_value FROM metrics WHERE metric_name = ?',
            conn,
            params=(metric_name,)
        )
        conn.close()

        if df.empty:
            return {}

        values = df['metric_value'].values

        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'var': float(np.var(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'skewness': float(stats.skew(values)),
            'kurtosis': float(stats.kurtosis(values)),
        }

    def detect_trend(self, metric_name: str, method: str = 'linear') -> Dict[str, float]:
        """
        Detect trend in metric over time.

        Args:
            metric_name: Metric name
            method: 'linear' or 'polynomial'

        Returns:
            Trend statistics
        """
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            'SELECT episode, metric_value FROM metrics WHERE metric_name = ? ORDER BY episode',
            conn,
            params=(metric_name,)
        )
        conn.close()

        if df.empty or len(df) < 3:
            return {}

        x = df['episode'].values
        y = df['metric_value'].values

        if method == 'linear':
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            return {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'std_err': float(std_err),
                'trend': 'increasing' if slope > 0 else 'decreasing',
                'significant': p_value < 0.05
            }

        return {}

    def correlation_analysis(self, metrics: List[str]) -> pd.DataFrame:
        """
        Calculate correlation matrix for metrics.

        Args:
            metrics: List of metric names

        Returns:
            Correlation matrix DataFrame
        """
        conn = sqlite3.connect(str(self.db_path))

        # Fetch all metrics
        data_dict = {}
        for metric in metrics:
            df = pd.read_sql_query(
                'SELECT episode, metric_value FROM metrics WHERE metric_name = ? ORDER BY episode',
                conn,
                params=(metric,)
            )
            if not df.empty:
                data_dict[metric] = df['metric_value'].values

        conn.close()

        if not data_dict:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        # Calculate correlation
        return df.corr()

    def t_test(self, metric1: str, metric2: str) -> Dict[str, float]:
        """
        Perform t-test comparing two metrics.

        Args:
            metric1: First metric name
            metric2: Second metric name

        Returns:
            Test results
        """
        conn = sqlite3.connect(str(self.db_path))

        df1 = pd.read_sql_query(
            'SELECT metric_value FROM metrics WHERE metric_name = ?',
            conn,
            params=(metric1,)
        )

        df2 = pd.read_sql_query(
            'SELECT metric_value FROM metrics WHERE metric_name = ?',
            conn,
            params=(metric2,)
        )

        conn.close()

        if df1.empty or df2.empty:
            return {}

        values1 = df1['metric_value'].values
        values2 = df2['metric_value'].values

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values1, values2)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_diff': float(np.mean(values1) - np.mean(values2))
        }

    def detect_anomalies(self, metric_name: str, threshold: float = 3.0) -> List[Tuple[int, float]]:
        """
        Detect anomalies using z-score method.

        Args:
            metric_name: Metric name
            threshold: Z-score threshold

        Returns:
            List of (episode, value) anomalies
        """
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql_query(
            'SELECT episode, metric_value FROM metrics WHERE metric_name = ? ORDER BY episode',
            conn,
            params=(metric_name,)
        )
        conn.close()

        if df.empty:
            return []

        values = df['metric_value'].values
        z_scores = np.abs(stats.zscore(values))

        anomalies = []
        for idx, z in enumerate(z_scores):
            if z > threshold:
                anomalies.append((int(df.iloc[idx]['episode']), float(df.iloc[idx]['metric_value'])))

        return anomalies

    def generate_report(self, metrics: Optional[List[str]] = None) -> str:
        """
        Generate text-based statistical report.

        Args:
            metrics: List of metrics to analyze (None = all)

        Returns:
            Report string
        """
        conn = sqlite3.connect(str(self.db_path))

        # Get all metrics if not specified
        if metrics is None:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT metric_name FROM metrics')
            metrics = [row[0] for row in cursor.fetchall()]

        conn.close()

        lines = []
        lines.append("=" * 70)
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        for metric in metrics[:10]:  # Limit to 10 metrics
            lines.append(f"Metric: {metric}")
            lines.append("-" * 70)

            # Descriptive stats
            stats_dict = self.get_descriptive_stats(metric)
            if stats_dict:
                lines.append(f"  Count:      {stats_dict['count']}")
                lines.append(f"  Mean:       {stats_dict['mean']:.4f}")
                lines.append(f"  Median:     {stats_dict['median']:.4f}")
                lines.append(f"  Std Dev:    {stats_dict['std']:.4f}")
                lines.append(f"  Min:        {stats_dict['min']:.4f}")
                lines.append(f"  Max:        {stats_dict['max']:.4f}")
                lines.append(f"  Q25-Q75:    {stats_dict['q25']:.4f} - {stats_dict['q75']:.4f}")

            # Trend analysis
            trend = self.detect_trend(metric)
            if trend:
                lines.append(f"  Trend:      {trend['trend']} (slope={trend['slope']:.4f})")
                lines.append(f"  R²:         {trend['r_squared']:.4f}")
                lines.append(f"  Significant: {'Yes' if trend['significant'] else 'No'} (p={trend['p_value']:.4f})")

            lines.append("")

        lines.append("=" * 70)

        return '\n'.join(lines)

    def export_summary_csv(self, output_path: str = 'outputs/reports/summary.csv'):
        """Export summary statistics to CSV."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT DISTINCT metric_name FROM metrics')
        metrics = [row[0] for row in cursor.fetchall()]

        conn.close()

        summaries = []
        for metric in metrics:
            stats_dict = self.get_descriptive_stats(metric)
            if stats_dict:
                stats_dict['metric_name'] = metric
                summaries.append(stats_dict)

        df = pd.DataFrame(summaries)
        df.to_csv(output_path, index=False)
        print(f"✅ Exported summary to {output_path}")
