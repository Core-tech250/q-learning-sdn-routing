"""
train_dqn.py - Train the Deep Q-Learning (DQN) agent in simulation.

Extension of the tabular project (train_agent.py). The DQN agent from
dqn_agent.py is trained in EXACTLY the same built-in network model as
the tabular agent - same simulated_measurement(), same reward function,
same Low/Medium/High episode cycle - so the two agents can be compared
fairly on identical conditions.

    python train_dqn.py                 (300 episodes, reproducible)
    python train_dqn.py --episodes 500

Outputs:
    dqn_training_log.csv   - one row per episode (incl. training loss)
    dqn_model.npz          - the trained network weights
    dqn_learning_curve.png - reward per episode with moving average

IMPORTANT (honesty note, same as the rest of this project): the DQN is
trained and evaluated in SIMULATION only. The live Mininet testbed
validation in results.csv was performed with the tabular agent's
learned policy. The DQN is the scalability extension - it shows the
same routing policy being learned by a neural network, which is the
version that would scale to a large topology where a Q-table cannot.
"""

import argparse
import csv
import random

import numpy as np

from q_agent import STATES, ACTIONS, compute_reward, simulated_measurement
from train_agent import BACKGROUND_LEVELS
from dqn_agent import DQNAgent

UPDATES_PER_EPISODE = 4  # gradient steps per episode (replay makes this cheap)


def state_vector(state, background_mbps):
    """Network input: one-hot congestion state + normalised background rate.

    Giving the network the continuous background rate (not only the
    Low/Medium/High bucket) is what lets it *generalise*: it learns a
    smooth Q-function over congestion, not just three memorised rows.
    """
    vec = np.zeros(4)
    vec[state] = 1.0
    vec[3] = background_mbps / 10.0
    return vec


def plot_learning_curve(rewards, outfile="dqn_learning_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    window = max(1, len(rewards) // 10)
    smoothed = [sum(rewards[max(0, i - window + 1):i + 1]) /
                len(rewards[max(0, i - window + 1):i + 1])
                for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.3,
            label="Per episode")
    ax.plot(range(1, len(smoothed) + 1), smoothed, linewidth=2,
            label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Deep Q-Learning: Reward per Episode")
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")


def run(episodes):
    random.seed(42)  # same seed as the tabular simulation (reproducible)
    agent = DQNAgent()
    rewards = []
    with open("dqn_training_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["episode", "state", "action", "throughput_mbps",
                      "delay_ms", "loss_pct", "reward", "epsilon", "loss"])
        for ep in range(1, episodes + 1):
            state = (ep - 1) % 3  # cycle Low / Medium / High, as before
            bg = BACKGROUND_LEVELS[state]
            s_vec = state_vector(state, bg)
            action = agent.choose_action(s_vec)
            tput, delay, loss = simulated_measurement(state, action)
            reward = compute_reward(tput, delay, loss)
            # Each episode is one independent measurement, so - exactly
            # as in the tabular agent - the "next state" is this state.
            agent.remember(s_vec, action, reward, s_vec)
            losses = [agent.train_step() for _ in range(UPDATES_PER_EPISODE)]
            losses = [l for l in losses if l is not None]
            agent.decay_epsilon()
            rewards.append(reward)
            log.writerow([ep, STATES[state], ACTIONS[action],
                          f"{tput:.2f}", f"{delay:.1f}", f"{loss:.1f}",
                          f"{reward:.3f}", f"{agent.epsilon:.3f}",
                          f"{np.mean(losses):.4f}" if losses else ""])
            if ep % 50 == 0:
                recent = rewards[-50:]
                print(f"  episodes {ep - 49:>3}-{ep:>3}: "
                      f"avg reward = {sum(recent) / len(recent):.2f}, "
                      f"epsilon = {agent.epsilon:.2f}")

    agent.save("dqn_model.npz")
    plot_learning_curve(rewards)

    print("\nLearned policy (DQN, greedy):")
    print(f"{'State':<10} {'Path 1':>10} {'Path 2':>10}  Best")
    policy = {}
    for s in range(3):
        q = agent.q_values(state_vector(s, BACKGROUND_LEVELS[s]))
        best = int(np.argmax(q))
        policy[s] = best
        print(f"{STATES[s]:<10} {q[0]:>10.3f} {q[1]:>10.3f}  {ACTIONS[best]}")

    expected = {0: 0, 1: 1, 2: 1}  # the tabular agent's learned policy
    print("\nMatches the tabular agent's policy "
          "(Low -> Path 1, Medium/High -> Path 2):", policy == expected)
    first = sum(rewards[:30]) / 30
    last = sum(rewards[-30:]) / 30
    print(f"Avg reward, first 30 episodes: {first:.2f}")
    print(f"Avg reward, last 30 episodes:  {last:.2f}")
    print("\nFiles written: dqn_training_log.csv, dqn_model.npz, "
          "dqn_learning_curve.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    run(args.episodes)
