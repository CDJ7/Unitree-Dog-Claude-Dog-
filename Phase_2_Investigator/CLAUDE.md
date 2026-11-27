# Phase 2 Investigator: notes for Claude Code (terminal, Linux laptop)

**Start with [`docs/STATUS.md`](docs/STATUS.md):** where everything stands, the working agreement with the user,
what comes next, and the lessons and gotchas learned the hard way. The results and method are in `REPORT.md`.

The crime-scene search dog for a Unitree Go2:
- **Stage A, simulation:** MuJoCo, spatter physics, scenarios, navigation, inspect, playback.
- **Stage R, the real dog:** it maps the lab and tags zones itself (`mapping/`, `real_robot/`). Done in
  simulation; the first real-dog session follows `LAB_PLAN.md`.
- **Stage C, the crime-scene search:** stain detector, evidence map, keep-out, close-ups and stain analysis
  (`scripts/investigate.py`). Done in simulation only.

Plans: `docs/plan.md` (Stage A/B), `docs/real_robot_plan.md` (Stage R and crime scene C1-C6).
Lab sessions with the real dog follow `LAB_PLAN.md` (Part 1: mapping + go-to-zone, no stain prints; Part 2 later).

## Where things are
- `navigation/`: occupancy grid, frontiers, planner, safety layer, `Navigator` (explorer.py).
  - It talks to a *session*: `sim_env.runtime.SimSession` (simulator) or `real_robot.session.RealGo2Session`
    (real dog, or `SimLink` = the simulator presented like the dog).
- `mapping/`: `zone_map.py` (zone votes per place, flood fill, regions), `mapper.py` (explore + look-arounds +
  viewpoint coverage), `localize.py` (scan matching on a saved map), `evaluate.py` (sim scoring).
- `perception/zone_classifier.py`: loads Test 2's ResNet18 checkpoints.
  - simulation: `../Test2/results/q1/resnet18/best.pt` (in git);
  - lab: `../Test2/results/real_q1/resnet18/best.pt` (not in git).
- `perception/stain_detector.py` + `perception/models/stain_rf_{depth,image}.pkl` (committed): the stain detector.
- `mapping/evidence_map.py`: detections -> evidence clusters, keep-out disks, GT scoring, stains walked over.
- `navigation/closeups.py`: the close-up tour. `analysis/closeup.py`: per-stain measurement.
  `analysis/scene_report.py`: groups -> mechanism (`perception/models/mechanism_gb_simclose.pkl`) -> origin -> report.
- `scripts/`:
  - `real_check.py`, `map_zones.py`, `goto_zone.py` (Stage R);
  - `investigate.py`, `reanalyse.py` (Stage C);
  - plus the Stage A scripts (`run_sim.sh`, ...).
- `training/`: `train_stain_detector.py`, `train_mechanism_simclose.py`, Phase 2's COCO and mechanism training.
- `report/build_pdf.sh`: `REPORT.md` -> `Phase2_Report.pdf`.
- Outputs go to `outputs/` (gitignored). Logs of robot runs go in `logs/` (gitignored): `cmd 2>&1 | tee logs/<name>.log`.

## Rules
- **Git:** work on `claude/eng6010-test2-crime-scene-0rst3r`. When the user asks, open a PR into `main`.
  **They merge; never merge yourself.** After a merge, restart the branch from `origin/main`.
- **Test 2 (`../Test2/`) is submitted and frozen: never edit it.** Read or import from it only.
- **Do not edit Test 1 (`../go2_stream/`)** unless the user says so; Phase 2 imports its modules.
- **Never run anything that moves the real dog.** That means `map_zones.py --backend real` or
  `goto_zone.py --backend real` without `--link sim`, Test 1's `main.py` / teleop, or any SportClient
  call. Write the command for the user instead.
- These are safe to run yourself:
  - `scripts/real_check.py` (sensors only, no SportClient);
  - anything with `--link sim` or `--backend sim`;
  - `python -m pytest`;
  - `ping 192.168.123.161`, `ip -br addr`, `nmcli con show`;
  - `python ../go2_stream/preflight.py`.
- Every number reported must come from a file a script wrote (`outputs/**/report.json`,
  `outputs/goto/*.json`, `docs/media/**`). Never invent results, and never quote an overwritten run.
  Copy result files that docs quote into `docs/media/` (`outputs/` is gitignored).
- After code changes: `python -m pytest -q` (about 2 minutes; the zone-map, localize and real-session
  tests take seconds).

## Shared laptop
The laptop also runs a TurtleBot / ROS 2 terminal (senior design). Keep Phase 2 light: `PHASE2_THREADS` (default 2)
caps numpy / OpenCV / PyTorch threads. Never kill or change other processes. Don't change `ROS_DOMAIN_ID` or the
TurtleBot's setup; just suggest a separate ROS 2 domain (e.g. 30) if both robots' DDS traffic mixes.

## Real-dog facts (from Test 1 / Test 2 lab sessions)
- **Ethernet:** the laptop is 192.168.123.222/24 and the dog is 192.168.123.161. The interface is
  auto-detected (`real_robot.session.pick_iface`). The network profile is
  `../go2_stream/setup_network.sh nmcli <iface>`.
- **LiDAR:** use `rt/utlidar/cloud_deskewed`, which is in the odometry frame and has the body removed.
  `rt/utlidar/cloud` is in the upside-down sensor frame: the floor looks like an obstacle there.
- The floor is about 0.31 m below the body origin (`--body-height`; `real_check.py` measures it).
- At small rates the Go2 barely turns on the spot. Navigation turns at 0.5 rad/s.
- **Stopping:** any key in the terminal (`TerminalKeys`), the Unitree remote, or a Test 1 teleop key ends
  an autonomous run. The map so far is saved.
