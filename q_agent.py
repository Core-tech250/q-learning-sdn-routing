"""
q_agent.py - Tabular Q-Learning agent for SDN path selection.

Project : Q-Learning Based Routing in SDN
State   : congestion level of Path 1  -> 0 = Low, 1 = Medium, 2 = High
Actions : 0 = route main flow via Path 1, 1 = route main flow via Path 2

This file has NO Mininet/Ryu dependency. Run it anywhere to (a) verify the
learning loop with the built-in simulation demo, and (b) reuse the same
QLearningAgent class with real measurements from Mininet (Day 2).

Reward:  reward = 10 * (throughput / LINK_MBPS)
                  - 5 * (delay_ms / 100)
                  - 2 * loss_pct
Speed is rewarded, delay and packet loss are penalised.
"""

import json
import random

LINK_MBPS = 10.0  # capacity of each path in our topology

STATES = ["Low", "Medium", "High"]
ACTIONS = ["Path 1", "Path 2"]


def congestion_state(background_mbps):
    """Convert background traffic level on Path 1 into a state bucket."""
    if background_mbps < 4.0:
        return 0  # Low
    if background_mbps <= 7.5:
        return 1  # Medium
    return 2      # High


def compute_reward(throughput_mbps, delay_ms, loss_pct):
    """Reward from one measurement (iperf throughput + ping delay/loss)."""
    return (10.0 * (throughput_mbps / LINK_MBPS)
            - 5.0 * (delay_ms / 100.0)
            - 2.0 * loss_pct)


class QLearningAgent:
    def __init__(self, n_states=3, n_actions=2, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.95):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q-table: rows = states, columns = actions, all start at 0.
        self.q_table = [[0.0] * n_actions for _ in range(n_states)]

    def choose_action(self, state):
        """Epsilon-greedy: explore randomly, or exploit the best known action."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        row = self.q_table[state]
        return row.index(max(row))

    def best_action(self, state):
        row = self.q_table[state]
        return row.index(max(row))

    def update(self, state, action, reward, next_state):
        """Standard Q-Learning update rule."""
        old = self.q_table[state][action]
        best_next = max(self.q_table[next_state])
        self.q_table[state][action] = old + self.alpha * (
            reward + self.gamma * best_next - old)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.json"):
        with open(path, "w") as f:
            json.dump({"q_table": self.q_table}, f, indent=2)

    def load(self, path="q_table.json"):
        with open(path) as f:
            self.q_table = json.load(f)["q_table"]

    def print_table(self):
        print(f"{'State':<10} {'Path 1':>10} {'Path 2':>10}  Best")
        for s, row in enumerate(self.q_table):
            best = ACTIONS[self.best_action(s)]
            print(f"{STATES[s]:<10} {row[0]:>10.3f} {row[1]:>10.3f}  {best}")


# ---------------------------------------------------------------------------
# Simulation demo: a *model* of the network used ONLY to verify the agent
# learns the expected policy before we connect it to Mininet.
# Path 1 is fast when free but collapses under background load;
# Path 2 is slower (extra delay) but unaffected by the background flow.
# ---------------------------------------------------------------------------

def simulated_measurement(state, action):
    """Return (throughput, delay, loss) roughly like the real topology."""
    noise_t = random.uniform(-0.4, 0.4)
    noise_d = random.uniform(-2.0, 2.0)
    if action == 0:          # Path 1 (shared with background traffic)
        if state == 0:       # Low background
            return 9.2 + noise_t, 6 + noise_d, 0.0
        if state == 1:       # Medium background
            return 5.5 + noise_t, 35 + noise_d, 0.5
        return 1.8 + noise_t, 110 + noise_d, 3.0   # High background
    # Path 2 (free, but +10 ms per link and one extra hop)
    return 8.6 + noise_t, 28 + noise_d, 0.0


def run_simulation(episodes=90):
    agent = QLearningAgent()
    print("Training on simulated measurements "
          f"({episodes} episodes)...\n")
    episode_rewards = []
    for ep in range(episodes):
        state = ep % 3  # cycle Low / Medium / High so all states are learned
        action = agent.choose_action(state)
        tput, delay, loss = simulated_measurement(state, action)
        reward = compute_reward(tput, delay, loss)
        agent.update(state, action, reward, state)
        agent.decay_epsilon()
        episode_rewards.append(reward)
        if (ep + 1) % 20 == 0:
            recent = episode_rewards[-20:]
            print(f"  episodes {ep - 18:>2}-{ep + 1:>2}: "
                  f"avg reward = {sum(recent) / len(recent):.2f}, "
                  f"epsilon = {agent.epsilon:.2f}")
    print("\nLearned Q-table:")
    agent.print_table()
    ok = (agent.best_action(0) == 0 and agent.best_action(2) == 1)
    print("\nExpected policy learned (Low -> Path 1, High -> Path 2):", ok)
    agent.save("q_table_simulated.json")
    print("Saved simulated Q-table to q_table_simulated.json")
    return agent


if __name__ == "__main__":
    random.seed(42)  # reproducible demo; remove the seed for real training
    run_simulation()
