"""
Real AI Models Runner for C++ Animation Integration
====================================================
Trains multiple real AI models simultaneously and writes their
learning progress to /tmp/real_ai_state.txt so the C++ SFML
animation can visualize them in real-time.

Models:
  1. Q-Learning    - tabular RL on XOR task
  2. Neural-Net    - backprop feedforward network
  3. Genetic-Alg   - evolutionary weights search
  4. DQN-Simple    - experience replay + epsilon-greedy

Run standalone:
  python ai_models_runner.py

Or alongside C++ animation:
  python ai_models_runner.py &
  ./neuron_learning_fast
"""

import numpy as np
import time
import os
import threading
import random
import signal
import sys

STATE_FILE = "/tmp/real_ai_state.txt"
UPDATE_INTERVAL = 0.3  # seconds between file writes


# ─── XOR task helpers ────────────────────────────────────────────────────────

XOR_INPUTS  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
XOR_TARGETS = np.array([[0],    [1],    [1],    [0]],    dtype=np.float32)


def xor_accuracy(predict_fn):
    correct = 0
    for inp, tgt in zip(XOR_INPUTS, XOR_TARGETS):
        pred = predict_fn(inp)
        correct += int(round(float(pred)) == int(tgt[0]))
    return correct / 4.0


# ─── Model 1: Q-Learning (table-based) ───────────────────────────────────────

class QLearningModel:
    name = "Q-Learning"

    def __init__(self):
        self.q_table = {}
        self.lr = 0.2
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.05
        self.episodes = 0
        self.recent_rewards = []
        self.accuracy = 0.0
        self.avg_reward = 0.0
        self.lock = threading.Lock()

    def _state_key(self, inp):
        return (int(round(inp[0])), int(round(inp[1])))

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [random.random() * 0.1, random.random() * 0.1]
        return self.q_table[state]

    def run(self):
        while not stop_event.is_set():
            idx = random.randint(0, 3)
            inp, tgt = XOR_INPUTS[idx], XOR_TARGETS[idx]
            state = self._state_key(inp)

            # Epsilon-greedy action (0=output_0, 1=output_1)
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                q = self._get_q(state)
                action = int(np.argmax(q))

            reward = 1.0 if action == int(tgt[0]) else -1.0

            # Q-update
            q = self._get_q(state)
            q[action] += self.lr * (reward - q[action])

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episodes += 1
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 200:
                self.recent_rewards.pop(0)

            if self.episodes % 50 == 0:
                def predict(x):
                    q = self._get_q(self._state_key(x))
                    return float(np.argmax(q))
                with self.lock:
                    self.accuracy = xor_accuracy(predict)
                    self.avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0

            time.sleep(0.0001)


# ─── Model 2: Neural Network (backprop) ──────────────────────────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

def sigmoid_deriv(s):
    return s * (1.0 - s)


class NeuralNetModel:
    name = "Neural-Net"

    def __init__(self):
        np.random.seed(42)
        self.W1 = np.random.randn(2, 4) * 0.5
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 1) * 0.5
        self.b2 = np.zeros((1, 1))
        self.lr = 0.1
        self.episodes = 0
        self.recent_losses = []
        self.accuracy = 0.0
        self.avg_reward = 0.0
        self.lock = threading.Lock()

    def forward(self, X):
        self.h = sigmoid(X @ self.W1 + self.b1)
        self.out = sigmoid(self.h @ self.W2 + self.b2)
        return self.out

    def backward(self, X, y):
        out = self.forward(X)
        loss = np.mean((out - y) ** 2)

        d_out = (out - y) * sigmoid_deriv(out)
        d_W2 = self.h.T @ d_out
        d_b2 = np.sum(d_out, axis=0, keepdims=True)

        d_h = d_out @ self.W2.T * sigmoid_deriv(self.h)
        d_W1 = X.T @ d_h
        d_b1 = np.sum(d_h, axis=0, keepdims=True)

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        return loss

    def run(self):
        while not stop_event.is_set():
            loss = self.backward(XOR_INPUTS, XOR_TARGETS)
            self.episodes += 1
            reward = max(0.0, 1.0 - loss)
            self.recent_losses.append(reward)
            if len(self.recent_losses) > 200:
                self.recent_losses.pop(0)

            if self.episodes % 20 == 0:
                preds = self.forward(XOR_INPUTS)
                correct = sum(int(round(float(p[0]))) == int(t[0])
                              for p, t in zip(preds, XOR_TARGETS))
                with self.lock:
                    self.accuracy = correct / 4.0
                    self.avg_reward = float(np.mean(self.recent_losses))

            time.sleep(0.0002)


# ─── Model 3: Genetic Algorithm ──────────────────────────────────────────────

class GeneticModel:
    name = "Genetic-Alg"

    def __init__(self, pop_size=30):
        self.pop_size = pop_size
        self.n_weights = 2 * 4 + 4 + 4 * 1 + 1  # same arch as NN
        self.pop = np.random.randn(pop_size, self.n_weights) * 0.5
        self.fitness = np.zeros(pop_size)
        self.generation = 0
        self.best_fitness = 0.0
        self.accuracy = 0.0
        self.avg_reward = 0.0
        self.recent_best = []
        self.lock = threading.Lock()

    def _decode(self, weights):
        W1 = weights[:8].reshape(2, 4)
        b1 = weights[8:12].reshape(1, 4)
        W2 = weights[12:16].reshape(4, 1)
        b2 = weights[16:17].reshape(1, 1)
        return W1, b1, W2, b2

    def _evaluate(self, weights):
        W1, b1, W2, b2 = self._decode(weights)
        h = sigmoid(XOR_INPUTS @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        loss = np.mean((out - XOR_TARGETS) ** 2)
        return 1.0 - loss

    def _best_accuracy(self):
        best_idx = np.argmax(self.fitness)
        W1, b1, W2, b2 = self._decode(self.pop[best_idx])
        h = sigmoid(XOR_INPUTS @ W1 + b1)
        out = sigmoid(h @ W2 + b2)
        correct = sum(int(round(float(p[0]))) == int(t[0])
                      for p, t in zip(out, XOR_TARGETS))
        return correct / 4.0

    def run(self):
        while not stop_event.is_set():
            # Evaluate
            for i in range(self.pop_size):
                self.fitness[i] = self._evaluate(self.pop[i])

            best_f = float(np.max(self.fitness))
            self.recent_best.append(best_f)
            if len(self.recent_best) > 100:
                self.recent_best.pop(0)

            # Selection + crossover + mutation
            sorted_idx = np.argsort(self.fitness)[::-1]
            elite = self.pop[sorted_idx[:5]].copy()
            new_pop = [elite[i] for i in range(5)]

            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(list(sorted_idx[:15]), 2)
                mask = np.random.rand(self.n_weights) > 0.5
                child = np.where(mask, self.pop[p1], self.pop[p2])
                mutation = np.random.randn(self.n_weights) * 0.1
                mutation_mask = np.random.rand(self.n_weights) < 0.15
                child += mutation * mutation_mask
                new_pop.append(child)

            self.pop = np.array(new_pop)
            self.generation += 1

            if self.generation % 5 == 0:
                with self.lock:
                    self.accuracy = self._best_accuracy()
                    self.avg_reward = float(np.mean(self.recent_best)) if self.recent_best else 0.0

            time.sleep(0.005)


# ─── Model 4: DQN-Simple (experience replay) ─────────────────────────────────

class DQNSimpleModel:
    name = "DQN-Simple"

    def __init__(self):
        np.random.seed(7)
        # Small Q-network: 2 inputs -> 8 hidden -> 2 outputs
        self.W1 = np.random.randn(2, 8) * 0.3
        self.b1 = np.zeros((1, 8))
        self.W2 = np.random.randn(8, 2) * 0.3
        self.b2 = np.zeros((1, 2))
        # Target network (updated periodically)
        self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
        self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()

        self.replay_buffer = []
        self.buffer_size = 500
        self.batch_size = 32
        self.lr = 0.01
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.05
        self.target_update_freq = 100
        self.steps = 0
        self.episodes = 0
        self.recent_rewards = []
        self.accuracy = 0.0
        self.avg_reward = 0.0
        self.lock = threading.Lock()

    def _q_forward(self, x, W1, b1, W2, b2):
        h = np.tanh(x @ W1 + b1)
        return h @ W2 + b2

    def _train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = [b[1] for b in batch]
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)

        # Current Q values
        h = np.tanh(states @ self.W1 + self.b1)
        q_vals = h @ self.W2 + self.b2

        # Target Q values (using target network)
        th = np.tanh(next_states @ self.tW1 + self.tb1)
        tq = th @ self.tW2 + self.tb2
        targets = q_vals.copy()
        for i, a in enumerate(actions):
            targets[i, a] = rewards[i] + self.gamma * float(np.max(tq[i]))

        # Backprop
        d_out = (q_vals - targets) * (2.0 / self.batch_size)
        d_W2 = h.T @ d_out
        d_b2 = np.sum(d_out, axis=0, keepdims=True)
        d_h = d_out @ self.W2.T * (1 - h ** 2)
        d_W1 = states.T @ d_h
        d_b1 = np.sum(d_h, axis=0, keepdims=True)

        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

    def run(self):
        while not stop_event.is_set():
            idx = random.randint(0, 3)
            inp, tgt = XOR_INPUTS[idx], XOR_TARGETS[idx]
            state = inp.reshape(1, 2)

            # Epsilon-greedy
            if random.random() < self.epsilon:
                action = random.randint(0, 1)
            else:
                q = self._q_forward(state, self.W1, self.b1, self.W2, self.b2)
                action = int(np.argmax(q))

            reward = 1.0 if action == int(tgt[0]) else -1.0
            next_idx = random.randint(0, 3)
            next_state = XOR_INPUTS[next_idx].reshape(1, 2)

            self.replay_buffer.append((inp, action, reward, XOR_INPUTS[next_idx]))
            if len(self.replay_buffer) > self.buffer_size:
                self.replay_buffer.pop(0)

            self._train_step()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.steps += 1
            self.episodes += 1
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 200:
                self.recent_rewards.pop(0)

            # Update target network
            if self.steps % self.target_update_freq == 0:
                self.tW1, self.tb1 = self.W1.copy(), self.b1.copy()
                self.tW2, self.tb2 = self.W2.copy(), self.b2.copy()

            if self.episodes % 50 == 0:
                def predict(x):
                    q = self._q_forward(x.reshape(1, 2),
                                        self.W1, self.b1, self.W2, self.b2)
                    return float(np.argmax(q))
                with self.lock:
                    self.accuracy = xor_accuracy(predict)
                    self.avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0.0

            time.sleep(0.0001)


# ─── State writer ─────────────────────────────────────────────────────────────

def write_state(models):
    """Write model stats to shared file for C++ to read."""
    lines = []
    for m in models:
        with m.lock:
            acc = m.accuracy
            rwd = m.avg_reward
            eps = getattr(m, 'episodes', getattr(m, 'generation', 0))
        lines.append(f"{m.name},{acc:.4f},{rwd:.4f},{eps}")
    try:
        with open(STATE_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


# ─── Entry point ─────────────────────────────────────────────────────────────

stop_event = threading.Event()


def _signal_handler(sig, frame):
    print("\nStopping AI models runner...")
    stop_event.set()
    try:
        os.remove(STATE_FILE)
    except Exception:
        pass
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    models = [QLearningModel(), NeuralNetModel(), GeneticModel(), DQNSimpleModel()]

    threads = []
    for m in models:
        t = threading.Thread(target=m.run, daemon=True)
        t.start()
        threads.append(t)

    print(f"Running {len(models)} real AI models...")
    print(f"Writing state to: {STATE_FILE}")
    print("Press Ctrl+C to stop.\n")

    try:
        while not stop_event.is_set():
            write_state(models)
            for m in models:
                with m.lock:
                    acc = m.accuracy
                    rwd = m.avg_reward
                    eps = getattr(m, 'episodes', getattr(m, 'generation', 0))
                print(f"  {m.name:12s}  acc={acc:.2f}  reward={rwd:+.3f}  ep={eps}")
            print()
            time.sleep(UPDATE_INTERVAL)
    except KeyboardInterrupt:
        _signal_handler(None, None)


if __name__ == "__main__":
    main()
