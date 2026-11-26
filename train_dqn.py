"""
train_dqn.py - Train / run the Deep Q-Learning (DQN) agent.

Extension of the tabular project (train_agent.py). Two modes:

1) SIMULATION (default, run anywhere):
       python train_dqn.py
   Trains the DQN from dqn_agent.py in EXACTLY the same built-in
   network model as the tabular agent - same simulated_measurement(),
   same reward function, same Low/Medium/High episode cycle - so the
   two agents can be compared fairly on identical conditions.
   Outputs: dqn_training_log.csv, dqn_model.npz, dqn_learning_curve.png

2) LIVE VALIDATION (run inside the Mininet VM, next to controller_q.py):
       python3 train_dqn.py --live --episodes 15
   Loads the trained network (dqn_model.npz) with its weights FROZEN
   and epsilon = 0 - the network itself, not a table, now picks the
   path for the h1 -> h2 flow. For each episode the driver:
     a. shows the congestion level to set and the network's Q-values,
     b. writes the network's chosen action to current_action.txt
        (the Ryu controller reads this file),
     c. WAITS while you run the background iperf + main iperf + ping
        in the Mininet CLI (same measurement protocol as Part E),
     d. asks you for the 3 numbers and logs them to dqn_live_log.csv.
   This is the same validation standard the tabular policy was held
   to in Part E: a learned policy, frozen, measured on the real
   testbed. The network is NOT re-trained on live data.
"""

import argparse
import csv
import random
import sys

import numpy as np

from q_agent import STATES, ACTIONS, compute_reward, simulated_measurement
from train_agent import BACKGROUND_LEVELS, ask_float
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


# Part E testbed levels -> (state, actual background rate on Path 1).
# The network is fed the ACTUAL rate, not the nominal training level -
# it must generalise, and it does (verified: same policy at 0/4.2/8.4).
LIVE_LEVELS = {0: (0, 0.0), 4: (1, 4.2), 8: (2, 8.4)}


def run_live(episodes, level=None):
    """Validate the frozen DQN policy on the real Mininet testbed.

    Block mode (--level 0/4/8) runs every episode at one Part E level,
    which is how the testbed sessions work: one topology start per
    action, 5 runs per level. Without --level, episodes cycle
    Low/Medium/High at the nominal training rates.
    """
    agent = DQNAgent()
    try:
        agent.load("dqn_model.npz")
    except Exception:
        print("dqn_model.npz not found - train first with: "
              "python3 train_dqn.py")
        sys.exit(1)
    agent.epsilon = 0.0  # validation: always the network's greedy choice
    print("Loaded dqn_model.npz - weights frozen, epsilon = 0.")
    print("The network itself now picks the path each episode.\n")
    rewards = []
    with open("dqn_live_log.csv", "a", newline="") as f:
        log = csv.writer(f)
        for ep in range(1, episodes + 1):
            if level is not None:
                state, bg = LIVE_LEVELS[level]
            else:
                state = (ep - 1) % 3
                bg = BACKGROUND_LEVELS[state]
            s_vec = state_vector(state, bg)
            q = agent.q_values(s_vec)
            action = int(np.argmax(q))
            with open("current_action.txt", "w") as af:
                af.write(str(action))
            print(f"\n--- Episode {ep}/{episodes} ---")
            if bg > 0:
                print(f"State: {STATES[state]}  (set background h3->h4 "
                      f"to about {bg} Mbps)")
            else:
                print(f"State: {STATES[state]}  (no background traffic)")
            print(f"Network Q-values: Path 1 = {q[0]:.3f}, "
                  f"Path 2 = {q[1]:.3f}")
            print(f"DQN chose: {ACTIONS[action]}  "
                  f"(written to current_action.txt)")
            input("Run the background + iperf + ping commands, "
                  "then press Enter here...")
            tput = ask_float("  iperf throughput h1->h2 (Mbps): ")
            delay = ask_float("  ping average delay (ms): ")
            loss = ask_float("  ping packet loss (%): ")
            reward = compute_reward(tput, delay, loss)
            rewards.append(reward)
            log.writerow([ep, STATES[state], ACTIONS[action],
                          tput, delay, loss, f"{reward:.3f}"])
            f.flush()
            print(f"  reward = {reward:.2f}")
    print("\nLive validation session done. Rows appended to "
          "dqn_live_log.csv")
    print("Reminder: the policy that ran was the network's own - "
          "Low -> Path 1, Medium/High -> Path 2 (check the log).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="validate the trained DQN on the Mininet "
                             "testbed (weights frozen)")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--level", type=int, choices=[0, 4, 8],
                        default=None,
                        help="live mode only: run all episodes at one "
                             "Part E level (0, 4 or 8)")
    args = parser.parse_args()
    if args.live:
        run_live(args.episodes, level=args.level)
    else:
        run(args.episodes)
