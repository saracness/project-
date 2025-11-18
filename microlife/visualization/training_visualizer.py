"""
Training Visualizer
Real-time visualization of AI training metrics
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from .ai_metrics import AIMetricsTracker


class TrainingVisualizer:
    """
    Real-time training graphs for AI metrics.
    """

    def __init__(self, metrics_tracker: AIMetricsTracker, update_interval=10):
        self.tracker = metrics_tracker
        self.update_interval = update_interval
        self.last_update = 0

        # Color mapping for brain types
        self.colors = {
            'Q-Learning': '#2ECC71',
            'DQN': '#3498DB',
            'Double-DQN': '#9B59B6',
            'CNN': '#E74C3C',
            'GA': '#F39C12',
            'NEAT': '#1ABC9C',
            'CMA-ES': '#E67E22',
            'Random': '#95A5A6'
        }

        self.fig = None
        self.axes = None
        self.initialized = False

    def initialize(self):
        """Initialize the visualization plots."""
        if self.initialized:
            return

        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.suptitle('AI Training Metrics - Real-Time', fontsize=16, fontweight='bold')

        # Create grid layout
        gs = GridSpec(2, 4, figure=self.fig, hspace=0.3, wspace=0.3)

        # Create axes
        self.ax_reward = self.fig.add_subplot(gs[0, 0:2])  # Reward curve (large)
        self.ax_loss = self.fig.add_subplot(gs[0, 2])      # Loss curve
        self.ax_survival = self.fig.add_subplot(gs[0, 3])  # Survival time
        self.ax_qvalue = self.fig.add_subplot(gs[1, 0])    # Q-value distribution
        self.ax_action = self.fig.add_subplot(gs[1, 1])    # Action distribution
        self.ax_epsilon = self.fig.add_subplot(gs[1, 2])   # Epsilon decay
        self.ax_stats = self.fig.add_subplot(gs[1, 3])     # Statistics table

        self.axes = {
            'reward': self.ax_reward,
            'loss': self.ax_loss,
            'survival': self.ax_survival,
            'qvalue': self.ax_qvalue,
            'action': self.ax_action,
            'epsilon': self.ax_epsilon,
            'stats': self.ax_stats
        }

        # Style
        for ax in [self.ax_reward, self.ax_loss, self.ax_survival, self.ax_qvalue, self.ax_epsilon]:
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        self.initialized = True
        plt.ion()  # Interactive mode

    def update(self, timestep=None):
        """Update all plots with current metrics."""
        if not self.initialized:
            self.initialize()

        if timestep is None:
            timestep = self.tracker.global_timestep

        # Check if we should update
        if timestep - self.last_update < self.update_interval:
            return

        self.last_update = timestep

        # Clear all axes
        for ax in self.axes.values():
            ax.clear()

        # Update each plot
        self._plot_reward_curve()
        self._plot_loss_curve()
        self._plot_survival_time()
        self._plot_qvalue_distribution()
        self._plot_action_distribution()
        self._plot_epsilon_decay()
        self._plot_statistics()

        # Refresh
        plt.pause(0.001)

    def _plot_reward_curve(self):
        """Plot reward over time for each brain type."""
        ax = self.ax_reward
        ax.set_title('Reward Over Time', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Average Reward')

        brain_types = self.tracker.get_brain_types()
        if not brain_types:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return

        for brain_type in brain_types:
            timesteps, rewards = self.tracker.get_reward_history(brain_type, moving_average=True)
            if timesteps:
                color = self.colors.get(brain_type, '#000000')
                ax.plot(timesteps, rewards, label=brain_type, color=color, linewidth=2, alpha=0.8)

        ax.legend(loc='best', framealpha=0.9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    def _plot_loss_curve(self):
        """Plot neural network loss over time."""
        ax = self.ax_loss
        ax.set_title('Neural Network Loss', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Loss')

        brain_types = self.tracker.get_brain_types()
        neural_types = [bt for bt in brain_types if 'DQN' in bt or 'CNN' in bt]

        if not neural_types:
            ax.text(0.5, 0.5, 'N/A\n(Neural Networks Only)', ha='center', va='center', transform=ax.transAxes)
            return

        for brain_type in neural_types:
            timesteps, losses = self.tracker.get_loss_history(brain_type)
            if timesteps:
                color = self.colors.get(brain_type, '#000000')
                ax.plot(timesteps, losses, label=brain_type, color=color, linewidth=2, alpha=0.8)

        if neural_types:
            ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.set_yscale('log')

    def _plot_survival_time(self):
        """Plot survival time comparison."""
        ax = self.ax_survival
        ax.set_title('Avg Survival Time', fontweight='bold')
        ax.set_xlabel('Brain Type')
        ax.set_ylabel('Timesteps')

        brain_types = self.tracker.get_brain_types()
        if not brain_types:
            return

        survival_times = []
        labels = []
        colors_list = []

        for brain_type in sorted(brain_types):
            stats = self.tracker.get_aggregate_stats(brain_type)
            survival_times.append(stats['avg_survival'])
            labels.append(brain_type.replace('-', '\n'))
            colors_list.append(self.colors.get(brain_type, '#000000'))

        x = np.arange(len(labels))
        bars = ax.bar(x, survival_times, color=colors_list, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)

        # Add value labels on bars
        for bar, value in zip(bars, survival_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=8)

    def _plot_qvalue_distribution(self):
        """Plot Q-value distribution."""
        ax = self.ax_qvalue
        ax.set_title('Q-Value Distribution', fontweight='bold')
        ax.set_xlabel('Q-Value')
        ax.set_ylabel('Frequency')

        brain_types = self.tracker.get_brain_types()
        q_learning_types = [bt for bt in brain_types if 'Q' in bt or 'DQN' in bt]

        if not q_learning_types:
            ax.text(0.5, 0.5, 'N/A\n(Q-Learning Only)', ha='center', va='center', transform=ax.transAxes)
            return

        all_q_values = []
        for brain_type in q_learning_types:
            for metrics in self.tracker.get_metrics_by_brain_type(brain_type):
                all_q_values.extend(metrics.q_values)

        if all_q_values:
            ax.hist(all_q_values, bins=30, alpha=0.7, color='#3498DB', edgecolor='black')
            ax.axvline(np.mean(all_q_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_q_values):.2f}')
            ax.legend(fontsize=8)

    def _plot_action_distribution(self):
        """Plot action distribution as pie chart."""
        ax = self.ax_action
        ax.set_title('Action Distribution', fontweight='bold')

        action_dist = self.tracker.get_action_distribution()

        if not action_dist:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            return

        # Action names
        action_names = {
            0: 'N', 1: 'NE', 2: 'E', 3: 'SE',
            4: 'S', 5: 'SW', 6: 'W', 7: 'NW', 8: 'Stay'
        }

        labels = [action_names.get(action, f'A{action}') for action in sorted(action_dist.keys())]
        sizes = [action_dist[action] * 100 for action in sorted(action_dist.keys())]

        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie,
                                            autopct='%1.1f%%', startangle=90)

        for text in texts:
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_fontsize(7)
            autotext.set_color('black')

    def _plot_epsilon_decay(self):
        """Plot epsilon decay over time."""
        ax = self.ax_epsilon
        ax.set_title('Epsilon Decay', fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Epsilon')

        brain_types = self.tracker.get_brain_types()
        epsilon_types = [bt for bt in brain_types if 'Q' in bt or 'DQN' in bt]

        if not epsilon_types:
            ax.text(0.5, 0.5, 'N/A\n(ε-greedy Only)', ha='center', va='center', transform=ax.transAxes)
            return

        for brain_type in epsilon_types:
            metrics_list = self.tracker.get_metrics_by_brain_type(brain_type)
            for metrics in metrics_list:
                if metrics.epsilon_history:
                    color = self.colors.get(brain_type, '#000000')
                    ax.plot(metrics.timesteps[:len(metrics.epsilon_history)],
                            metrics.epsilon_history, color=color, alpha=0.3, linewidth=1)

        ax.set_ylim(0, 1)
        ax.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Min ε=0.01')
        ax.legend(fontsize=8)

    def _plot_statistics(self):
        """Plot statistics table."""
        ax = self.ax_stats
        ax.axis('off')
        ax.set_title('Statistics', fontweight='bold')

        brain_types = self.tracker.get_brain_types()
        if not brain_types:
            return

        # Create table data
        table_data = []
        headers = ['Brain', 'Alive', 'Avg R', 'Max S']

        for brain_type in sorted(brain_types):
            stats = self.tracker.get_aggregate_stats(brain_type)
            row = [
                brain_type[:10],  # Truncate name
                str(stats['alive']),
                f"{stats['avg_reward']:.1f}",
                str(stats['max_survival'])
            ]
            table_data.append(row)

        # Create table
        if table_data:
            table = ax.table(cellText=table_data, colLabels=headers,
                            loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 2)

            # Style table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#3498DB')
                table[(0, i)].set_text_props(weight='bold', color='white')

            for i in range(1, len(table_data) + 1):
                for j in range(len(headers)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#ECF0F1')

    def show(self):
        """Show the visualization window."""
        if not self.initialized:
            self.initialize()
        plt.show(block=False)

    def save(self, filename='training_metrics.png'):
        """Save current visualization to file."""
        if self.initialized:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ Training metrics saved to {filename}")

    def close(self):
        """Close the visualization."""
        if self.initialized:
            plt.close(self.fig)
            self.initialized = False
