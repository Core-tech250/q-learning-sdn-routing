"""
dqn_agent.py - Deep Q-Learning (DQN) agent for SDN path selection.

Project : Q-Learning Based Routing in SDN  (extension of q_agent.py)
State   : congestion level of Path 1  -> 0 = Low, 1 = Medium, 2 = High
Actions : 0 = route main flow via Path 1, 1 = route main flow via Path 2

The tabular agent in q_agent.py stores one Q-value per (state, action)
pair in a table. That works because this project has only 3 states x 2
actions. A Deep Q-Network replaces the table with a small neural network
that *approximates* the Q-function: Q(state, action) -> expected reward.
The network can generalise to states it has never seen exactly, which is
why DQN is the version that scales to large networks (the idea used in
the review paper "AI-driven routing in SDN using deep Q-learning").

This implementation uses only NumPy (no deep-learning framework needed
for a network this small) and contains the three ingredients that make
it a DQN and not just "a neural network":

  1. Experience replay  - past (state, action, reward, next_state)
     transitions are stored in a buffer and re-trained on in random
     mini-batches, which breaks the correlation between consecutive
     episodes and reuses every measurement many times.
  2. A separate target network - the TD target is computed with a
     slowly-updated copy of the network, which stabilises learning.
  3. Epsilon-greedy exploration - same schedule as the tabular agent.

Reward (identical to q_agent.compute_reward):
    reward = 10 * (throughput / 10) - 5 * (delay_ms / 100) - 2 * loss_pct
"""

import numpy as np


class _MLP:
    """Two-hidden-layer perceptron: n_in -> hidden -> hidden -> n_out.

    Forward and backward passes are written out by hand so the whole
    learning mechanism is visible - there is no framework magic.
    """

    def __init__(self, n_in, n_out, hidden=32, rng=None):
        rng = rng or np.random.default_rng()
        # He initialisation (suited to ReLU activations).
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_out))
        self.b3 = np.zeros(n_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = \
            [p.copy() for p in params]

    def forward(self, x):
        """x: (batch, n_in) -> Q-values (batch, n_out). Caches for backprop."""
        self._x = x
        self._h1 = np.maximum(0.0, x @ self.W1 + self.b1)        # ReLU
        self._h2 = np.maximum(0.0, self._h1 @ self.W2 + self.b2)  # ReLU
        return self._h2 @ self.W3 + self.b3

    def backward(self, grad_out):
        """grad_out: dLoss/dQ (batch, n_out) -> gradients for all params."""
        gW3 = self._h2.T @ grad_out
        gb3 = grad_out.sum(axis=0)
        gh2 = grad_out @ self.W3.T
        gh2[self._h2 <= 0] = 0.0                                  # ReLU'
        gW2 = self._h1.T @ gh2
        gb2 = gh2.sum(axis=0)
        gh1 = gh2 @ self.W2.T
        gh1[self._h1 <= 0] = 0.0                                  # ReLU'
        gW1 = self._x.T @ gh1
        gb1 = gh1.sum(axis=0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]


class _Adam:
    """Adam optimiser (Kingma & Ba, 2015), the standard choice for DQN."""

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class DQNAgent:
    """Deep Q-Learning agent with replay buffer and target network."""

    def __init__(self, n_inputs=4, n_actions=2, hidden=32,
                 gamma=0.9, lr=0.001,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.95,
                 buffer_size=2000, batch_size=32, target_sync=100,
                 seed=42):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.rng = np.random.default_rng(seed)

        self.online = _MLP(n_inputs, n_actions, hidden, self.rng)
        self.target = _MLP(n_inputs, n_actions, hidden, self.rng)
        self.target.set_params(self.online.params())
        self.optim = _Adam(self.online.params(), lr=lr)

        self.buffer = []          # list of (s, a, r, s_next)
        self.buffer_size = buffer_size
        self.train_steps = 0

    # -- acting ------------------------------------------------------------
    def q_values(self, state_vec):
        return self.online.forward(np.atleast_2d(state_vec))[0]

    def best_action(self, state_vec):
        return int(np.argmax(self.q_values(state_vec)))

    def choose_action(self, state_vec):
        """Epsilon-greedy, same exploration rule as the tabular agent."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.best_action(state_vec)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)

    # -- learning ----------------------------------------------------------
    def remember(self, state_vec, action, reward, next_state_vec):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((np.asarray(state_vec, dtype=float), action,
                            reward, np.asarray(next_state_vec, dtype=float)))

    def train_step(self):
        """One gradient step on a random mini-batch from the replay buffer.

        Returns the batch loss, or None while the buffer is still filling.
        """
        if len(self.buffer) < self.batch_size:
            return None
        idx = self.rng.integers(0, len(self.buffer), self.batch_size)
        states = np.array([self.buffer[i][0] for i in idx])
        actions = np.array([self.buffer[i][1] for i in idx])
        rewards = np.array([self.buffer[i][2] for i in idx], dtype=float)
        next_states = np.array([self.buffer[i][3] for i in idx])

        # TD target from the *target* network:  y = r + gamma * max_a' Q'(s', a')
        q_next = self.target.forward(next_states)
        targets = rewards + self.gamma * q_next.max(axis=1)

        # Current Q-values; loss is MSE on the taken action only.
        q = self.online.forward(states)
        taken = q[np.arange(self.batch_size), actions]
        errors = taken - targets
        loss = float(np.mean(errors ** 2))
        grad_out = np.zeros_like(q)
        grad_out[np.arange(self.batch_size), actions] = \
            2.0 * errors / self.batch_size
        grads = self.online.backward(grad_out)
        self.optim.step(self.online.params(), grads)

        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target.set_params(self.online.params())
        return loss

    # -- persistence -------------------------------------------------------
    def save(self, path="dqn_model.npz"):
        np.savez(path, *[p for p in self.online.params()])
        print(f"saved DQN weights to {path}")

    def load(self, path="dqn_model.npz"):
        data = np.load(path)
        params = [data[f"arr_{i}"] for i in range(6)]
        self.online.set_params(params)
        self.target.set_params(params)
