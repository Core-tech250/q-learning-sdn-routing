# SETUP GUIDE — do the parts in order, tick them off

## Part A — Windows / VS Code (do this first, ~15 minutes)
You need: this `sdm` folder open in VS Code, Python installed.

- [ ] Open a terminal in VS Code (Terminal → New Terminal), make sure you are inside the `sdm` folder: `cd` to it if needed.
- [ ] Install the one Windows-side library: `pip install -r requirements.txt`
- [ ] Run the agent demo: `python q_agent.py`
      Expected ending: a Q-table with **Low → Path 1, Medium → Path 2, High → Path 2** and the line `Expected policy learned ... : True`
- [ ] Test the output parsers: `python measure_utils.py`
      Expected: `ping -> (107.654, 5.0)` and `iperf -> 2.1`
- [ ] Run a full simulated training: `python train_agent.py`
      Expected files appear next to the code: `training_log.csv`, `q_table.json`, `learning_curve.png`
- [ ] **Send a screenshot of the `python q_agent.py` output to Muse.** Part A is done when Muse confirms it.

> If `python` is not found, try `py q_agent.py`. If matplotlib is missing, re-run the pip step.

## Part B — Mininet VM (Day 1 setup, do once)
Goal: one Linux VM with Mininet + Ryu. Two routes — take Route 1 if possible.

**Route 1 (recommended): ready Mininet VM image**
- [ ] Install VirtualBox (virtualbox.org → Windows hosts).
- [ ] Download the official Mininet VM image (mininet.org → "Download" → VM image, ~64-bit). It already contains Mininet; login is usually `mininet / mininet`.
- [ ] Import the image in VirtualBox (File → Import Appliance). Give the VM **4 GB RAM** (2 GB minimum), 2 CPUs. Before starting it, in Windows Task Manager → Performance confirm **Virtualization: Enabled**; if Disabled, enable VT-x in BIOS or Part B will not work.
- [ ] Start the VM, log in, test: `sudo mn --test pingall` → all pings must pass, then `sudo mn -c` to clean up.

**Route 2 (only if Route 1 fails): Ubuntu VM + install script**
- [ ] Ubuntu 20.04 VM in VirtualBox (4 GB RAM), then in the VM:
      `git clone https://github.com/mininet/mininet` → `cd mininet` → `util/install.sh -a`
      (this takes a long time — that's normal for Route 2 only).

**Ryu install (both routes), inside the VM:**
- [ ] `pip3 install ryu` — if pip complains about eventlet/gevent versions on newer Python, note the exact error and send it to Muse before trying random fixes.
- [ ] Check: `which ryu-manager` prints a path.

**Copy this project into the VM** (pick ONE):
- [ ] VirtualBox Shared Folder pointing at your Windows `sdm` folder (VM Settings → Shared Folders, then it appears under `/media/sf_...`), OR
- [ ] `git` (push from Windows / clone in VM), OR a zip on a pen drive / Google Drive download inside the VM.
- In the VM, `cd` to the copied `sdm` folder and confirm: `ls` shows all the `.py` files.

## Part C — Baseline runs in the VM (Day 1 finish)
- [ ] Terminal 1 (VM): `ryu-manager controller_q.py` — leave it running. If it errors, copy the full error to Muse.
- [ ] Terminal 2 (VM): `sudo python3 topology.py`
- [ ] In the Mininet CLI: `pingall` → 100% must pass.
- [ ] Verify s1's uplink ports (the controller depends on them):
      `sh ovs-ofctl dump-ports s1` and compare with `controller_q.py`: Path 1 = port 3 (to s2), Path 2 = port 4 (to s3). If different, edit only those two constants and restart the controller.
- [ ] Start servers: `h2 iperf -s &` and `h4 iperf -s -u &`
- [ ] Set the agent to the shortest path and keep it there for all of Part C: the file `current_action.txt` must contain `0`.
- [ ] Baseline (case `shortest`), at each background level 0 / 4 / 8 Mbps, 5 runs each:
      `h3 iperf -c 10.0.0.4 -u -b 8M -t 15 &`  (use 0 = skip, 4M, 8M)
      `h1 iperf -c 10.0.0.2 -t 10`             → note throughput
      `h1 ping -c 20 10.0.0.2`                 → note avg delay + loss
      Append one row per run to `results.csv` (copy `results_template.csv` first):
      `shortest,1,8.9,6.2,0.0` = case, run no, throughput Mbps, delay ms, loss %

## Part D — Training with real measurements (Day 2)
- [ ] Keep controller + topology running as in Part C (servers running too).
- [ ] Terminal 3 (VM), inside the `sdm` folder: `python3 train_agent.py --live --episodes 30`
- [ ] For every episode: set the background level the driver shows you, clear flows (`sh ovs-ofctl del-flows s1`), run the iperf + ping pair, type the 3 numbers into the driver. It updates `q_table.json` after every episode.
- [ ] Done when the driver prints a final Q-table. Expected shape: Low → Path 1, High → Path 2 (Medium follows your real numbers — either is a valid result).

## Part E — Final comparison + graphs (Day 3)
- [ ] Case `qlearning`: for each run, read the learned best action for the run's congestion level from `q_table.json`, write it to `current_action.txt`, clear flows on s1, run the same iperf + ping pair as Part C, append rows with case `qlearning`. 5 runs per background level, same levels as Part C.
- [ ] Copy `results.csv` back to Windows, then in the `sdm` folder: `python plot_results.py`
      → prints the mean ± std table and saves `throughput_graph.png` + `delay_graph.png`.
- [ ] Write the report using `PROJECT_REPORT_GUIDE.md` (structure + viva lines are ready there). Use only numbers from YOUR `results.csv`.

## If something breaks
Send Muse: (1) which Part/step, (2) the full error text or a screenshot,
(3) the last command you ran. Do not reinstall things randomly — most SDN
errors are one-line fixes once the exact error is visible.
