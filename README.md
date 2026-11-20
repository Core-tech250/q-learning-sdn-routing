# Q-Learning Based Routing in SDN

Can a central SDN controller **learn** — by trial and error — when to shift traffic
away from the shortest path? This project answers that with a working testbed:
a tabular Q-Learning agent that routes traffic in a Mininet network controlled
by a Ryu SDN controller, compared head-to-head against fixed shortest-path routing
over 30 real measurements.

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

A Q-table is used instead of a Deep Q-Network because the state space is only
3 states × 2 actions — a neural network would add cost with no benefit at this
scale. (The project is based on the core Q-Learning principle of the review
paper *"AI-driven routing in SDN using deep Q-learning"*.)

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

## Method note (stated honestly)

The Q-table was **trained in simulation** (90 episodes on Windows, on a model
parameterised from the testbed and the measured baseline, after 2 pilot live
episodes) and the learned policy was then **validated with 15 live runs** on the
real Mininet testbed. No Deep Q-Learning was implemented.

## Files

| File | What it does |
|---|---|
| `q_agent.py` | Tabular Q-Learning agent + reward function |
| `train_agent.py` | Training driver — simulation mode and `--live` mode |
| `topology.py` | Mininet diamond topology (runs in the VM) |
| `controller_q.py` | Ryu controller — fixed next-hop flows + agent action for h1→h2 |
| `measure_utils.py` | Helpers for parsing iperf/ping measurements |
| `plot_results.py` | Mean ± std table and bar graphs with error bars from `results.csv` |
| `results.csv` | All 30 real measurements (15 baseline + 15 learned-policy) |
| `SETUP_GUIDE.md` | Full step-by-step setup and run guide (Parts A–E) |

## Quick start

- **On Windows:** `py train_agent.py` (simulation training) ·
  `py plot_results.py` (graphs from `results.csv`)
- **In the Mininet VM:** start the Ryu controller, then `sudo python3 topology.py`
  — see `SETUP_GUIDE.md` for the complete procedure.

## Author

**Nishant Goel** — B.Tech CSE (Semester 5), Vivekananda Institute of
Professional Studies (VIPS), GGSIPU. Computer Networks course project, 2026.
