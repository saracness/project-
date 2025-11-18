#!/usr/bin/env python3
"""
AI Training Visualization Demo
Real-time visualization of neural network training
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import moderngl_window as mglw
import moderngl

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from microlife.training.metrics_tracker import MetricsTracker
from microlife.training.training_visualizer import TrainingVisualizer
from microlife.training.network_visualizer import NetworkVisualizer
from microlife.training.decision_boundary import DecisionBoundaryVisualizer
from microlife.ml.neural_network import SimpleNeuralNetwork


class TrainingVisualizationDemo(mglw.WindowConfig):
    """
    Interactive AI training visualization demo.
    """

    gl_version = (3, 3)
    title = "AI Training Visualization - Real-time Monitoring"
    window_size = (1920, 1080)
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("=" * 60)
        print("AI Training Visualization Demo")
        print("=" * 60)

        # ModernGL context
        self.ctx: moderngl.Context = self.ctx
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Create visualization components
        self.metrics_tracker = MetricsTracker(max_history=1000, smooth_window=20)
        self.training_viz = TrainingVisualizer(self.ctx, self.window_size)
        self.network_viz = NetworkVisualizer(self.ctx, self.window_size)
        self.boundary_viz = DecisionBoundaryVisualizer(self.ctx, self.window_size, resolution=50)

        # Configure training graph panels
        self.training_viz.add_panel(
            'loss_panel', 'loss',
            position=(20, 20), size=(400, 250),
            title='Training Loss'
        )
        self.training_viz.add_panel(
            'accuracy_panel', 'accuracy',
            position=(20, 290), size=(400, 250),
            title='Accuracy'
        )
        self.training_viz.add_panel(
            'reward_panel', 'reward',
            position=(20, 560), size=(400, 250),
            title='Episode Reward'
        )

        # Create simple neural network for demo
        self.model = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        # Set network structure for visualization
        self.network_viz.set_network([4, 8, 2])  # input, hidden, output

        # Training data (simple classification problem)
        self.X_train, self.y_train = self._generate_training_data()

        # Fit PCA for decision boundary
        self.boundary_viz.fit_projection(self.X_train)

        # Training state
        self.episode = 0
        self.training_active = True
        self.train_interval = 0.1  # Train every 0.1 seconds
        self.train_accumulator = 0.0

        # Performance tracking
        self.fps = 0.0
        self.frame_times = []
        import time
        self.last_time = time.time()

        print("✅ Training visualization initialized")
        print("\nControls:")
        print("  SPACE - Pause/Resume training")
        print("  R - Reset training")
        print("  S - Print summary")
        print("  E - Export metrics")
        print("  ESC - Exit")
        print()

    def _generate_training_data(self, n_samples: int = 500):
        """Generate synthetic training data."""
        # Simple 2-class classification problem
        # Class 0: cluster around (-1, -1)
        # Class 1: cluster around (1, 1)

        n_per_class = n_samples // 2

        # Class 0
        X_class0 = np.random.randn(n_per_class, 4) * 0.5
        X_class0[:, :2] -= 1.0
        y_class0 = np.zeros(n_per_class)

        # Class 1
        X_class1 = np.random.randn(n_per_class, 4) * 0.5
        X_class1[:, :2] += 1.0
        y_class1 = np.ones(n_per_class)

        # Combine
        X = np.vstack([X_class0, X_class1]).astype(np.float32)
        y = np.hstack([y_class0, y_class1]).astype(np.int64)

        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]

        return X, y

    def _train_step(self):
        """Perform one training step."""
        # Sample mini-batch
        batch_size = 32
        indices = np.random.choice(len(self.X_train), batch_size, replace=False)
        X_batch = torch.FloatTensor(self.X_train[indices])
        y_batch = torch.LongTensor(self.y_train[indices])

        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.model(X_batch)
        loss = self.criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_batch).float().mean().item()

        # Calculate reward (for RL-like metrics)
        reward = accuracy * 10.0

        # Record metrics
        self.metrics_tracker.record(
            episode=self.episode,
            loss=loss.item(),
            accuracy=accuracy,
            reward=reward,
            learning_rate=self.optimizer.param_groups[0]['lr']
        )

        # Update network visualization
        sample_input = torch.FloatTensor(self.X_train[:1])
        self.network_viz.update_from_pytorch(self.model, sample_input)

        self.episode += 1

    def render(self, time_elapsed: float, frame_time: float):
        """Render visualization."""
        # Clear screen
        self.ctx.clear(0.05, 0.05, 0.05)

        # Update training
        if self.training_active:
            self.train_accumulator += frame_time

            while self.train_accumulator >= self.train_interval:
                self._train_step()
                self.train_accumulator -= self.train_interval

        # Render training graphs
        self.training_viz.render(self.metrics_tracker)

        # Render neural network visualization
        self.network_viz.render(
            panel_x=440,
            panel_y=20,
            panel_width=700,
            panel_height=400
        )

        # Render decision boundary
        self.boundary_viz.render(
            self.model,
            panel_x=440,
            panel_y=440,
            panel_width=700,
            panel_height=400,
            X_samples=self.X_train,
            y_samples=self.y_train
        )

        # Render info text
        self._render_info()

        # Update FPS
        import time
        current_time = time.time()
        frame_time_actual = current_time - self.last_time
        self.last_time = current_time

        self.frame_times.append(frame_time_actual * 1000)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def _render_info(self):
        """Render info panel (simple text display)."""
        # For now, we'll just print to console on certain events
        # Full text rendering would require font atlas texture
        pass

    def key_event(self, key, action, modifiers):
        """Handle keyboard input."""
        if action == self.wnd.keys.ACTION_PRESS:
            # Space - Pause/Resume
            if key == self.wnd.keys.SPACE:
                self.training_active = not self.training_active
                status = "▶️  Resumed" if self.training_active else "⏸️  Paused"
                print(f"{status} training (Episode {self.episode})")

            # R - Reset training
            elif key == self.wnd.keys.R:
                self.model = SimpleNeuralNetwork(input_size=4, hidden_size=8, output_size=2)
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                self.metrics_tracker.clear()
                self.episode = 0
                print("🔄 Reset training")

            # S - Print summary
            elif key == self.wnd.keys.S:
                self.metrics_tracker.print_summary()

            # E - Export metrics
            elif key == self.wnd.keys.E:
                self.metrics_tracker.export_csv('training_metrics.csv')
                self.metrics_tracker.export_json('training_metrics.json')
                print("💾 Exported metrics")

            # F - Print FPS
            elif key == self.wnd.keys.F:
                print(f"📊 FPS: {self.fps:.1f} | Episode: {self.episode}")
                latest_loss = self.metrics_tracker.get_latest('loss', smoothed=True)
                latest_acc = self.metrics_tracker.get_latest('accuracy', smoothed=True)
                if latest_loss is not None and latest_acc is not None:
                    print(f"   Loss: {latest_loss:.4f} | Accuracy: {latest_acc:.4f}")

            # ESC - Exit
            elif key == self.wnd.keys.ESCAPE:
                self.wnd.close()

    def close(self):
        """Cleanup on exit."""
        print("\n🧹 Cleaning up...")

        # Cleanup visualizers
        if hasattr(self, 'training_viz'):
            self.training_viz.cleanup()
        if hasattr(self, 'network_viz'):
            self.network_viz.cleanup()
        if hasattr(self, 'boundary_viz'):
            self.boundary_viz.cleanup()

        # Print final summary
        if hasattr(self, 'metrics_tracker'):
            print("\n📊 Final Training Summary:")
            self.metrics_tracker.print_summary()

        print("✅ Cleanup complete")


def main():
    """Run the demo."""
    print("Starting AI Training Visualization Demo...")
    mglw.run_window_config(TrainingVisualizationDemo)


if __name__ == '__main__':
    main()
