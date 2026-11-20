"""
train_agent.py - Training driver for the Q-Learning routing agent.

Two modes:

1) SIMULATION (run anywhere, e.g. Windows + VS Code - do this FIRST):
       python train_agent.py
   Trains the agent on the built-in network model, writes training_log.csv,
   q_table.json and learning_curve.png, and prints the learned policy.
   This proves the learning code works before Mininet is involved.

2) LIVE (run inside the Mininet VM, next to controller_q.py):
       python3 train_agent.py --live --episodes 30
   For each episode the driver:
     a. picks a path with the agent and writes it to current_action.txt
        (the Ryu controller reads this file for the h1 -> h2 flow),
     b. WAITS while you, in the Mininet CLI:
          - clear old flows:   sh ovs-ofctl del-flows s1
          - run background:    h3 iperf -c 10.0.0.4 -u -b <level>M -t 15 &
          - run main iperf:    h1 iperf -c 10.0.0.2 -t 10
          - run ping:          h1 ping -c 20 10.0.0.2
          (iperf servers must already be running:  h2 iperf -s &  /  h4 iperf -s -u &)
     c. asks you for the 3 numbers (throughput, avg delay, loss),
        computes the reward, updates the Q-table, and logs the episode.
   Episodes cycle through Low / Medium / High background levels so every
   state is trained. Set the background level it shows you each episode.

Either mode leaves q_table.json in this folder - that file IS the learned
policy you report, and the source for the final comparison runs.
"""

import argparse
import csv
import random
import sys

from q_agent import (QLearningAgent, STATES, ACTIONS, congestion_state,
                     compute_reward, simulated_measurement)

BACKGROUND_LEVELS = [2.0, 6.0, 9.0]  # Mbps on Path 1 -> Low / Medium / High


def plot_learning_curve(rewards, outfile="learning_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    window = max(1, len(rewards) // 10)
    smoothed = [sum(rewards[max(0, i - window + 1):i + 1]) /
                len(rewards[max(0, i - window + 1):i + 1])
                for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.3, label="Per episode")
    ax.plot(range(1, len(smoothed) + 1), smoothed, linewidth=2,
            label="Moving average")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Q-Learning: Reward per Episode")
    ax.grid(linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")


def run_simulation(episodes):
    agent = QLearningAgent()
    rewards = []
    with open("training_log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["episode", "state", "action",
                      "throughput_mbps", "delay_ms", "loss_pct", "reward"])
        for ep in range(1, episodes + 1):
            state = (ep - 1) % 3
            action = agent.choose_action(state)
            tput, delay, loss = simulated_measurement(state, action)
            reward = compute_reward(tput, delay, loss)
            agent.update(state, action, reward, state)
            agent.decay_epsilon()
            rewards.append(reward)
            log.writerow([ep, STATES[state], ACTIONS[action],
                          f"{tput:.2f}", f"{delay:.1f}", f"{loss:.1f}",
                          f"{reward:.3f}"])
    agent.save("q_table.json")
    plot_learning_curve(rewards)
    print("\nLearned policy (simulation):")
    agent.print_table()
    print("\nFiles written: training_log.csv, q_table.json, learning_curve.png")


def ask_float(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  please type a number, e.g. 8.6")
        except EOFError:
            print("\nInput closed - stopping training early.")
            sys.exit(1)


def run_live(episodes):
    try:
        agent = QLearningAgent()
        agent.load("q_table.json")
        print("Loaded existing q_table.json - continuing training.")
    except Exception:
        agent = QLearningAgent()
        print("No q_table.json found - starting from a fresh Q-table.")
    rewards = []
    with open("training_log.csv", "a", newline="") as f:
        log = csv.writer(f)
        for ep in range(1, episodes + 1):
            bg = BACKGROUND_LEVELS[(ep - 1) % 3]
            state = congestion_state(bg)
            action = agent.choose_action(state)
            with open("current_action.txt", "w") as af:
                af.write(str(action))
            print(f"\n--- Episode {ep}/{episodes} ---")
            print(f"State: {STATES[state]}  (set background h3->h4 to {bg} Mbps)")
            print(f"Agent chose: {ACTIONS[action]}  "
                  f"(written to current_action.txt)")
            input("Run the iperf + ping commands (see file header), "
                  "then press Enter here...")
            tput = ask_float("  iperf throughput h1->h2 (Mbps): ")
            delay = ask_float("  ping average delay (ms): ")
            loss = ask_float("  ping packet loss (%): ")
            reward = compute_reward(tput, delay, loss)
            agent.update(state, action, reward, state)
            agent.decay_epsilon()
            agent.save("q_table.json")
            rewards.append(reward)
            log.writerow([ep, STATES[state], ACTIONS[action],
                          tput, delay, loss, f"{reward:.3f}"])
            f.flush()
            print(f"  reward = {reward:.2f}   epsilon now {agent.epsilon:.2f}")
    print("\nTraining session done. Current policy:")
    agent.print_table()
    if rewards:
        plot_learning_curve(rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="train with real Mininet measurements")
    parser.add_argument("--episodes", type=int, default=90)
    args = parser.parse_args()
    random.seed(42)
    if args.live:
        run_live(args.episodes)
    else:
        run_simulation(args.episodes)
