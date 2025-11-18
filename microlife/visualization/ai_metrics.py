"""
AI Metrics Tracking System
Collects and aggregates AI training metrics for visualization
"""
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Optional


class AIMetrics:
    """
    Metrics for a single AI organism.
    """

    def __init__(self, organism_id, brain_type, window_size=100):
        self.organism_id = organism_id
        self.brain_type = brain_type
        self.window_size = window_size

        # Time series data
        self.timesteps = []
        self.rewards = []
        self.losses = []  # For neural network losses
        self.q_values = []  # Average Q-value per timestep
        self.actions = []  # Action indices
        self.epsilon_history = []

        # Moving averages
        self.reward_window = deque(maxlen=window_size)
        self.loss_window = deque(maxlen=window_size)

        # Aggregated stats
        self.total_reward = 0.0
        self.total_loss = 0.0
        self.survival_time = 0
        self.decision_count = 0
        self.food_consumed = 0
        self.deaths = 0

        # Action distribution
        self.action_counts = defaultdict(int)

        # Status
        self.is_alive = True
        self.birth_timestep = 0

    def record_step(self, timestep, reward, loss=None, q_value=None, action=None, epsilon=None):
        """Record metrics for one timestep."""
        self.timesteps.append(timestep)
        self.rewards.append(reward)
        self.total_reward += reward
        self.reward_window.append(reward)

        if loss is not None:
            self.losses.append(loss)
            self.total_loss += loss
            self.loss_window.append(loss)

        if q_value is not None:
            self.q_values.append(q_value)

        if action is not None:
            self.actions.append(action)
            self.action_counts[action] += 1

        if epsilon is not None:
            self.epsilon_history.append(epsilon)

        self.survival_time += 1

    def get_avg_reward(self, window=None):
        """Get average reward (recent window or all-time)."""
        if window and len(self.rewards) > window:
            return np.mean(self.rewards[-window:])
        elif self.reward_window:
            return np.mean(self.reward_window)
        return 0.0

    def get_avg_loss(self, window=None):
        """Get average loss (recent window or all-time)."""
        if not self.losses:
            return 0.0
        if window and len(self.losses) > window:
            return np.mean(self.losses[-window:])
        elif self.loss_window:
            return np.mean(self.loss_window)
        return 0.0

    def get_avg_q_value(self):
        """Get average Q-value."""
        return np.mean(self.q_values) if self.q_values else 0.0

    def get_action_distribution(self):
        """Get action distribution as percentages."""
        if not self.action_counts:
            return {}
        total = sum(self.action_counts.values())
        return {action: count / total for action, count in self.action_counts.items()}

    def mark_death(self, timestep):
        """Mark organism as dead."""
        self.is_alive = False
        self.deaths += 1
        self.survival_time = timestep - self.birth_timestep


class AIMetricsTracker:
    """
    Tracks metrics for all AI organisms in the simulation.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics: Dict[str, AIMetrics] = {}  # organism_id -> AIMetrics
        self.global_timestep = 0

        # Brain type aggregates
        self.brain_type_stats = defaultdict(lambda: {
            'total_reward': 0.0,
            'total_loss': 0.0,
            'count': 0,
            'alive_count': 0,
            'avg_survival': 0.0
        })

    def register_organism(self, organism_id, brain_type, timestep=0):
        """Register a new AI organism."""
        if organism_id not in self.metrics:
            self.metrics[organism_id] = AIMetrics(organism_id, brain_type, self.window_size)
            self.metrics[organism_id].birth_timestep = timestep
            self.brain_type_stats[brain_type]['count'] += 1
            self.brain_type_stats[brain_type]['alive_count'] += 1

    def record(self, organism_id, brain, timestep=None):
        """
        Record metrics from a brain.

        Args:
            organism_id: Unique organism identifier
            brain: Brain object with metrics
            timestep: Current simulation timestep
        """
        if timestep is None:
            timestep = self.global_timestep

        if organism_id not in self.metrics:
            self.register_organism(organism_id, brain.brain_type, timestep)

        metrics = self.metrics[organism_id]

        # Extract metrics from brain
        reward = getattr(brain, 'last_reward', 0.0)
        loss = getattr(brain, 'last_loss', None)
        q_value = getattr(brain, 'last_q_value', None)
        action = getattr(brain, 'last_action_idx', None)
        epsilon = getattr(brain, 'epsilon', None)

        # Record
        metrics.record_step(timestep, reward, loss, q_value, action, epsilon)

        # Update brain type stats
        stats = self.brain_type_stats[brain.brain_type]
        stats['total_reward'] = sum(m.total_reward for m in self.metrics.values() if m.brain_type == brain.brain_type)
        stats['total_loss'] = sum(m.total_loss for m in self.metrics.values() if m.brain_type == brain.brain_type)
        stats['alive_count'] = sum(1 for m in self.metrics.values() if m.brain_type == brain.brain_type and m.is_alive)

    def mark_death(self, organism_id, timestep=None):
        """Mark an organism as dead."""
        if timestep is None:
            timestep = self.global_timestep

        if organism_id in self.metrics:
            metrics = self.metrics[organism_id]
            metrics.mark_death(timestep)
            self.brain_type_stats[metrics.brain_type]['alive_count'] -= 1

    def update_timestep(self, timestep):
        """Update global timestep."""
        self.global_timestep = timestep

    def get_metrics_by_brain_type(self, brain_type):
        """Get all metrics for a specific brain type."""
        return [m for m in self.metrics.values() if m.brain_type == brain_type]

    def get_alive_metrics(self):
        """Get metrics for all alive organisms."""
        return [m for m in self.metrics.values() if m.is_alive]

    def get_brain_types(self):
        """Get list of all brain types being tracked."""
        return list(self.brain_type_stats.keys())

    def get_aggregate_stats(self, brain_type=None):
        """
        Get aggregated statistics.

        Args:
            brain_type: If specified, get stats for this brain type only

        Returns:
            Dict with aggregated stats
        """
        if brain_type:
            metrics_list = self.get_metrics_by_brain_type(brain_type)
        else:
            metrics_list = list(self.metrics.values())

        if not metrics_list:
            return {
                'count': 0,
                'alive': 0,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'total_loss': 0.0,
                'avg_loss': 0.0,
                'avg_survival': 0.0,
                'max_survival': 0
            }

        alive = [m for m in metrics_list if m.is_alive]
        total_reward = sum(m.total_reward for m in metrics_list)
        total_loss = sum(m.total_loss for m in metrics_list)
        survival_times = [m.survival_time for m in metrics_list]

        return {
            'count': len(metrics_list),
            'alive': len(alive),
            'total_reward': total_reward,
            'avg_reward': total_reward / len(metrics_list) if metrics_list else 0.0,
            'total_loss': total_loss,
            'avg_loss': total_loss / len(metrics_list) if metrics_list else 0.0,
            'avg_survival': np.mean(survival_times) if survival_times else 0.0,
            'max_survival': max(survival_times) if survival_times else 0
        }

    def get_reward_history(self, brain_type=None, moving_average=True):
        """
        Get reward history over time.

        Args:
            brain_type: Filter by brain type
            moving_average: If True, return moving average

        Returns:
            (timesteps, rewards) tuple
        """
        if brain_type:
            metrics_list = self.get_metrics_by_brain_type(brain_type)
        else:
            metrics_list = list(self.metrics.values())

        if not metrics_list:
            return [], []

        # Aggregate rewards by timestep
        reward_by_timestep = defaultdict(list)
        for metrics in metrics_list:
            for t, r in zip(metrics.timesteps, metrics.rewards):
                reward_by_timestep[t].append(r)

        timesteps = sorted(reward_by_timestep.keys())
        rewards = [np.mean(reward_by_timestep[t]) for t in timesteps]

        if moving_average and len(rewards) > 10:
            # Apply moving average
            window = min(20, len(rewards) // 5)
            rewards = np.convolve(rewards, np.ones(window) / window, mode='valid')
            timesteps = timesteps[window - 1:]

        return timesteps, rewards

    def get_loss_history(self, brain_type=None):
        """Get loss history over time (for neural networks)."""
        if brain_type:
            metrics_list = self.get_metrics_by_brain_type(brain_type)
        else:
            metrics_list = list(self.metrics.values())

        if not metrics_list:
            return [], []

        # Aggregate losses by timestep
        loss_by_timestep = defaultdict(list)
        for metrics in metrics_list:
            for t, loss in zip(metrics.timesteps, metrics.losses):
                if loss is not None:
                    loss_by_timestep[t].append(loss)

        if not loss_by_timestep:
            return [], []

        timesteps = sorted(loss_by_timestep.keys())
        losses = [np.mean(loss_by_timestep[t]) for t in timesteps]

        return timesteps, losses

    def get_action_distribution(self, brain_type=None):
        """Get aggregated action distribution."""
        if brain_type:
            metrics_list = self.get_metrics_by_brain_type(brain_type)
        else:
            metrics_list = list(self.metrics.values())

        action_counts = defaultdict(int)
        for metrics in metrics_list:
            for action, count in metrics.action_counts.items():
                action_counts[action] += count

        if not action_counts:
            return {}

        total = sum(action_counts.values())
        return {action: count / total for action, count in action_counts.items()}

    def get_summary(self):
        """Get summary of all tracked metrics."""
        lines = []
        lines.append("=" * 70)
        lines.append("AI METRICS SUMMARY")
        lines.append("=" * 70)

        lines.append(f"\nTotal Organisms Tracked: {len(self.metrics)}")
        lines.append(f"Currently Alive: {len(self.get_alive_metrics())}")
        lines.append(f"Brain Types: {len(self.get_brain_types())}")

        lines.append("\n" + "-" * 70)
        lines.append("BY BRAIN TYPE:")
        lines.append("-" * 70)

        for brain_type in sorted(self.get_brain_types()):
            stats = self.get_aggregate_stats(brain_type)
            lines.append(f"\n🧠 {brain_type}:")
            lines.append(f"  Count: {stats['count']} (Alive: {stats['alive']})")
            lines.append(f"  Avg Reward: {stats['avg_reward']:.2f}")
            lines.append(f"  Avg Loss: {stats['avg_loss']:.4f}")
            lines.append(f"  Avg Survival: {stats['avg_survival']:.1f} timesteps")
            lines.append(f"  Max Survival: {stats['max_survival']} timesteps")

        lines.append("\n" + "=" * 70)

        return "\n".join(lines)

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.brain_type_stats.clear()
        self.global_timestep = 0
