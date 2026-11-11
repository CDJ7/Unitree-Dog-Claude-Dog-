# Test 2 — notes for Claude Code

ENG 6010 Test 2: a Unitree Go2 project.
- `q2_data_simulation/`, `q1_supervised/`, `q3_rl/`: MuJoCo simulation work.
- `real_go2/`: the same Q1 pipeline on the real dog in the lab, plus "go to zone" autonomy with LiDAR
  obstacle avoidance.
- `REPORT.md` / `Test2_Report.pdf`: the report. Section 7 covers the real Go2.

## How the user works
- Windows + WSL2 (mirrored networking) + VS Code. The venv is `~/go2venv`. Run commands from `Test2/`.
- Dog over Ethernet: the laptop is 192.168.123.222/24, the dog 192.168.123.161. The interface name comes
  from `ip -br addr | grep 192.168.123`.
- Wi-Fi alternative: `--source webrtc --ip <dog IP>`, with `GO2_AES_KEY` set. It has no LiDAR.
- Terminal logs of robot runs are in `Test2/logs_real/*.log` (made with `| tee`). Read them when
  debugging.
- The step-by-step plan is `real_go2/LAB_PLAN.md`. The script reference is `real_go2/README.md`.

## Safety rules (always)
- **Never run anything that makes the dog move.** That means `--source sdk` or `--source webrtc` with
  `record_real.py`, `live_zone.py`, `goto_zone.py`, Test 1's `main.py` / `run_detection.py` with teleop or
  missions, or any `SportClient` call. The user runs these with a hand on SPACE. Write the command for
  them instead.
- These are safe to run yourself:
  - `ip`, `ping`, `python ../go2_stream/preflight.py --iface <name>`;
  - anything with `--source fake`;
  - `python -m pytest -q tests`;
  - `real_go2/nav_sim.py`, `summarize_real.py`, `evaluate_real.py`;
  - `run_detection.py ... --mission-dry-run` (prints only).
- `goto_zone.py --lidar-check` sends no autonomous motion, but it starts teleop. Let the user run it.
- **Do not edit Test 1 files** (`../go2_stream/`, including `config/behaviors.yaml`) unless the user
  explicitly says to. Propose the change instead.
- After any code change, run `python -m pytest -q tests` (23 tests, no robot needed).

## Results and honesty
- Every number in the report must come from a file in `results/` written by the scripts. Never invent or
  estimate results.
- `data_real/`, `*.mp4`, `best.pt` and `logs_real/` are not committed (`.gitignore`).

## Useful facts when debugging
- The connections are in `real_go2/sources.py`:
  - `SdkLink` uses Test 1's `go2_stream` modules (camera, state, teleop);
  - `WebRTCLink` uses `unitree_webrtc_connect` 2.2.0;
  - `FakeLink` replays `data/sample` frames.
- The LiDAR free space comes from `goto_zone.LidarProfile`. It uses Test 1's `LidarSource`, then
  `perception.pipeline.cloud_to_body`, then `free_profile`, with the corridor from
  `../go2_stream/config/behaviors.yaml` `safety.corridor`.
- The navigator thresholds are in `real_go2/real_config.yaml` under `navigation:`.
- Teleop (Test 1, `go2_stream/teleop.py`):
  - autonomy commands lapse after 0.3 s;
  - any manual key cancels autonomy;
  - without pynput / WSLg the keys don't work (`echo $DISPLAY` should be `:0`).
