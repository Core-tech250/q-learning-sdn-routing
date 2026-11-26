# Q-Learning Based Routing in SDN

Can a central SDN controller **learn** — by trial and error — when to shift traffic
away from the shortest path? This project answers that with a working testbed:
a tabular Q-Learning agent that routes traffic in a Mininet network controlled
by a Ryu SDN controller, compared head-to-head against fixed shortest-path routing
over 30 real measurements. A **Deep Q-Learning (DQN) extension** then replaces the
Q-table with a neural network and learns the same policy in simulation.

## The problem

In a normal network, traffic always takes the shortest path. When that path gets
crowded, delay explodes and throughput collapses — while a second, longer path
sits idle. Fixed routing never adapts. This project tests whether a learned
policy does better.

## Testbed

- **Topology (Mininet):** diamond — hosts h1–h4, switches s1–s4 (Open vSwitch)
  - **Path 1** (s1–s2–s4): short, 1 ms per link — best when free
  - **Path 2** (s1–s3–s4): longer, 10 ms per link — idle, but slower
  - All links 10 Mbps
- **Main flow:** h1 → h2 (the flow the agent routes and we measure)
- **Background flow:** h3 → h4, always on Path 1 — this creates the congestion
- **Controller:** Ryu (Python, OpenFlow). It installs fixed next-hop flows and
  routes the main flow according to the agent's decision in `current_action.txt`
- **Measurement:** `iperf` (throughput) + `ping` (delay, loss), 5 runs per setting

## The Q-Learning design

| Item | Value |
|---|---|
| **State** | Congestion on Path 1: Low / Medium / High |
| **Actions** | 0 = main flow via Path 1, 1 = main flow via Path 2 |
| **Reward** | `10 × (throughput/10) − 5 × (delay_ms/100) − 2 × loss%` |
| **Learning rate α** | 0.1 |
| **Discount γ** | 0.9 |
| **Exploration ε** | 1.0, decaying ×0.95 per episode (min 0.05) |

A Q-table is used for the measured experiment because the state space is only
3 states × 2 actions — a table represents the Q-function exactly and can be
inspected entry by entry. (The project is based on the core Q-Learning principle
of the review paper *"AI-driven routing in SDN using deep Q-learning"*.) The
Deep Q-Learning extension below exists for the scale case, where a table would
no longer work.

## What the agent learned

| Path 1 congestion | Q – Path 1 | Q – Path 2 | Policy |
|---|---|---|---|
| Low | **22.471** | 0.870 | Path 1 |
| Medium | 0.794 | **17.487** | Path 2 |
| High | 0.059 | **18.247** | Path 2 |

**Policy in one line:** stay on Path 1 while it is free; shift to Path 2 as soon
as congestion builds.

## Results (5-run means, real testbed measurements)

| Background load | Shortest path | Q-Learning policy |
|---|---|---|
| Level 0 (none) | 8.30 Mbps · 13.96 ms · 0% loss | 9.40 Mbps · 13.20 ms · 0% loss |
| Level 4 (~4.2 Mbps) | 6.92 Mbps · 7.25 ms · 0% loss | 9.23 Mbps · 46.23 ms · 0% loss |
| Level 8 (~8 Mbps) | 3.29 Mbps · 626.41 ms · 1.0% loss | **9.26 Mbps · 43.46 ms · 0% loss** |

**Headline result (Level 8):** under heavy congestion the shortest path collapses
by *bufferbloat* — its queue fills, delay becomes bistable (calm ~10 ms runs vs
~1000 ms queue-drain runs) and packets are lost. The learned policy removes the
collapse entirely: **throughput ×2.8, delay ÷14, loss 1% → 0%**. The honest
trade-off is at Level 4: +33% throughput is bought with higher delay, because
Path 2's links are longer.

## Deep Q-Learning extension

`dqn_agent.py` replaces the Q-table with a small neural network that
*approximates* the Q-function — the version of the method that scales to large
topologies, where a table cannot. It is a genuine DQN, written in pure NumPy
(hand-written forward/backward passes, no deep-learning framework):

- **Network:** 4 → 32 → 32 → 2 multilayer perceptron (1,282 parameters).
  Input = one-hot congestion state + the continuous background rate, so the
  network learns a smooth Q-function over congestion instead of three
  memorised rows.
- **Experience replay:** 2,000-transition buffer, random mini-batches of 32,
  4 gradient steps per episode.
- **Target network:** separate copy, synchronised every 100 gradient steps.
- **Optimiser:** Adam, learning rate 0.001. Same reward, γ = 0.9 and
  ε-greedy schedule as the tabular agent.

Trained for **300 episodes in exactly the same simulation model** as the
tabular agent (`python train_dqn.py`, seed 42 — fully reproducible), it learned
**the same policy**:

| Path 1 congestion | DQN Q – Path 1 | DQN Q – Path 2 | DQN policy |
|---|---|---|---|
| Low | **59.757** | 57.677 | Path 1 |
| Medium | 43.589 | **48.073** | Path 2 |
| High | 30.213 | **47.523** | Path 2 |

Average episode reward rose from **4.62** (first 30 episodes) to **7.21**
(last 30). The DQN's Q-value *magnitudes* are not comparable with the table's —
replay and bootstrapping put them on a different scale; only the ordering
(the policy) carries meaning. Note the narrow Low margin: shared weights pull
the Path 2 value at Low upward — generalisation made visible.

**Scope, stated honestly:** the DQN was trained and evaluated in **simulation
only** and was not deployed in the Mininet testbed. The live results above are
the tabular policy's. Outputs: `dqn_training_log.csv` (per-episode log),
`dqn_model.npz` (trained weights, generated locally by the training run) and
`dqn_learning_curve.png` (generated locally by the training run).

**Live validation driver:** `train_dqn.py --live` loads the trained weights
(frozen, ε = 0) inside the Mininet VM and lets the network itself choose the
path for the h1 → h2 flow, under the same measurement protocol as the tabular
validation — e.g. `python3 train_dqn.py --live --level 8 --episodes 5`. The
network was also queried at the testbed's actual background rates
(0 / 4.2 / 8.4 Mbps) and gives the same policy there. The live runs themselves
are the remaining step; until they are recorded, the scope note above stands.


## Files

| File | What it does |
|---|---|
| `q_agent.py` | Tabular Q-Learning agent + reward function |
| `train_agent.py` | Training driver — simulation mode and `--live` mode |
| `dqn_agent.py` | Deep Q-Network agent — NumPy MLP, experience replay, target network |
| `train_dqn.py` | DQN driver — simulation training (log, weights, learning curve); `--live` mode runs the frozen policy on the testbed |
| `dqn_training_log.csv` | DQN per-episode training log (300 episodes, seed 42) |
| `topology.py` | Mininet diamond topology (runs in the VM) |
| `controller_q.py` | Ryu controller — fixed next-hop flows + agent action for h1→h2 |
| `measure_utils.py` | Helpers for parsing iperf/ping measurements |
| `plot_results.py` | Mean ± std table and bar graphs with error bars from `results.csv` |
| `results.csv` | All 30 real measurements (15 baseline + 15 learned-policy) |
| `SETUP_GUIDE.md` | Full step-by-step setup and run guide (Parts A–E) |

## Quick start

- **On Windows:** `py train_agent.py` (simulation training) ·
  `py train_dqn.py` (Deep Q-Learning extension) ·
  `py plot_results.py` (graphs from `results.csv`)
- **In the Mininet VM:** start the Ryu controller, then `sudo python3 topology.py`
  — see `SETUP_GUIDE.md` for the complete procedure. For the DQN live
  validation: `python3 train_dqn.py --live --level 8 --episodes 5`
  (needs `numpy` and `dqn_model.npz` in the VM's project folder).
