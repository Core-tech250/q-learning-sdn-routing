"""
measure_utils.py - Parse real `ping` and `iperf` (iperf2, as bundled with
Mininet) output into numbers for results.csv / the training driver.

Tested with sample outputs in the project test run.
Keep the raw terminal output of every run in your notes as well - these
parsers only extract the three numbers the project needs.
"""

import re


def parse_ping(text):
    """
    Return (avg_delay_ms, loss_pct) from Linux ping output, e.g.
        20 packets transmitted, 20 received, 0% packet loss, time 19037ms
        rtt min/avg/max/mdev = 1.234/28.456/55.001/12.345 ms
    Returns (None, loss) if no reply was received at all.
    """
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", text)
    loss = float(loss_match.group(1)) if loss_match else 0.0
    rtt_match = re.search(
        r"(?:rtt|round-trip) min/avg/max/(?:mdev|stddev) = "
        r"[\d.]+/([\d.]+)/", text)
    avg_delay = float(rtt_match.group(1)) if rtt_match else None
    return avg_delay, loss


def parse_iperf(text):
    """
    Return throughput in Mbps from iperf2 client output - the last
    '[SUM]' or single-stream summary line, e.g.
        [  3]  0.0-10.0 sec  10.5 MBytes  8.81 Mbits/sec
    Handles Mbits/sec and Kbits/sec. Returns None if not found.
    """
    value = None
    for line in text.splitlines():
        if "sec" not in line:
            continue
        m = re.search(r"([\d.]+)\s+([MK])bits/sec", line)
        if m:
            v = float(m.group(1))
            value = v if m.group(2) == "M" else v / 1000.0
    return value


if __name__ == "__main__":
    sample_ping = """
PING 10.0.0.2 (10.0.0.2) 56(84) bytes of data.
--- 10.0.0.2 ping statistics ---
20 packets transmitted, 19 received, 5% packet loss, time 19037ms
rtt min/avg/max/mdev = 20.101/107.654/180.202/40.111 ms
"""
    sample_iperf = """
[  3] local 10.0.0.1 port 5001 connected with 10.0.0.2 port 54321
[ ID] Interval       Transfer     Bandwidth
[  3]  0.0-10.0 sec  2.50 MBytes  2.10 Mbits/sec
"""
    print("ping  ->", parse_ping(sample_ping), " (expect (107.654, 5.0))")
    print("iperf ->", parse_iperf(sample_iperf), " (expect 2.1)")
