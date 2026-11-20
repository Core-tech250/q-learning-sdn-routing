"""
plot_results.py - Graphs for the SDN Q-Learning project.

Input: results.csv with header
    case,run,throughput_mbps,avg_delay_ms,loss_pct
where case is 'shortest' (Case A baseline) or 'qlearning' (Case B policy).
Collect 5 runs per case, per congestion level is also fine - this script
compares the two cases overall; group by level in the report table.

Output: throughput_graph.png and delay_graph.png (bars = mean, error bars =
standard deviation), plus a printed mean +/- std table for the report.
"""

import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CASES = [("shortest", "Shortest Path"), ("qlearning", "Q-Learning (SDN)")]


def load(path):
    data = {key: {"throughput": [], "delay": []} for key, _ in CASES}
    with open(path) as f:
        for row in csv.DictReader(f):
            case = row["case"].strip()
            if case not in data:
                continue
            data[case]["throughput"].append(float(row["throughput_mbps"]))
            data[case]["delay"].append(float(row["avg_delay_ms"]))
    return data


def mean_std(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def bar_graph(data, metric, ylabel, title, outfile):
    labels, means, stds = [], [], []
    for key, label in CASES:
        m, s = mean_std(data[key][metric])
        labels.append(label)
        means.append(m)
        stds.append(s)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=stds, capsize=6,
           color=["#c0392b", "#27ae60"], edgecolor="black")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"saved {outfile}")


def main(path="results.csv"):
    data = load(path)
    print(f"{'Case':<22} {'Throughput (Mbps)':>20} {'Delay (ms)':>16}")
    for key, label in CASES:
        tm, ts = mean_std(data[key]["throughput"])
        dm, ds = mean_std(data[key]["delay"])
        n = len(data[key]["throughput"])
        print(f"{label:<22} {tm:>10.2f} +/- {ts:<7.2f} {dm:>8.2f} +/- {ds:<5.2f}"
              f"   (n={n})")
    bar_graph(data, "throughput", "Throughput (Mbps)",
              "Throughput: Shortest Path vs Q-Learning Routing",
              "throughput_graph.png")
    bar_graph(data, "delay", "Average Delay (ms)",
              "Delay: Shortest Path vs Q-Learning Routing",
              "delay_graph.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
