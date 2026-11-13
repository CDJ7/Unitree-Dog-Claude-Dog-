# Q1 on the real Unitree Go2: zone recognition and "go to zone" in the lab

The simulated Q1 taught a ResNet18 the zones of the MuJoCo house. Here the same pipeline runs on the
**real dog in the real lab**, in the same data format and with the same training code. On top of it, a
small autonomous behaviour lets you **pick a zone and the dog walks there**.

1. Record labelled camera frames while you drive the dog.
2. Retrain with the unchanged Q1 `train.py`.
3. Evaluate on a recording session the model never saw.
4. Run live recognition and record the video.
5. Go to zone: press a number and the dog finds and walks to that zone.

The lab is split into four zones, set in [`real_config.yaml`](real_config.yaml). The number key selects
the zone everywhere:

| Key | Zone | What it is |
|---|---|---|
| 1 | `lab_1` | the desks on one side |
| 2 | `lab_2` | the desks on the other side |
| 3 | `open_area` | the open floor between them |
| 4 | `shelves` | the shelves with the wires |

> **Status (12 Nov 2026):**
> - **Done on the dog over Ethernet:** 5 recording sessions (7,346 labelled frames), training and
>   evaluation. The held-out session scores **0.814** accuracy and macro-F1 0.803 (`results/real_q1/`,
>   REPORT.md section 7.6).
> - **LiDAR check run on the dog.** It led to two fixes: the deskewed LiDAR topic (the raw cloud is upside
>   down, so the floor looked like an obstacle) and a minimum turn rate.
> - **Still to do:** the live-demo video and the go-to-zone trials on the dog.

| Script | Step |
|---|---|
| [`record_real.py`](record_real.py) | Drive the dog and record labelled frames → `data_real/run_XXX/` |
| `q1_supervised/train.py --config real_go2/real_config.yaml` | Retrain ResNet18 on the lab (same recipe as Q1) |
| [`evaluate_real.py`](evaluate_real.py) | Accuracy on the held-out last session, per-zone scores, confusion matrix |
| [`live_zone.py`](live_zone.py) | Live recognition on the dog's camera, with an on-screen panel, recorded to MP4 |
| [`goto_zone.py`](goto_zone.py) | Pick a zone (keys 1–4); the dog localises on the zone map and walks there |
| [`zone_map.py`](zone_map.py) | Map of the lab from the recorded odometry + localiser + route planner (`--tune`, `--eval`) |
| [`sources.py`](sources.py) | Connections: `sdk` (Ethernet), `webrtc` (Wi-Fi), `fake` (no robot) |

Driving and safety reuse Test 1's teleop ([`go2_stream/teleop.py`](../../go2_stream/teleop.py)): speed
limits, a watchdog, and manual override of anything autonomous.

---

> **Doing it for the first time?** Follow [`LAB_PLAN.md`](LAB_PLAN.md). It gives the sessions in order,
> what you should see at each step, what to do if you don't, and how to use Claude Code in VS Code for
> debugging.

## Part A — Get it running in WSL from VS Code (once)

### A1. Open the project in VS Code connected to WSL
1. In VS Code, install the **WSL** extension (Microsoft), if you don't have it.
2. Open a **WSL terminal** (Ubuntu from the Start menu, or VS Code → Terminal → New Terminal, with WSL
   shown bottom-left) and get the latest code:
   ```bash
   cd ~/Unitree-Dog-Claude-Dog-        # your clone from Test 1
   git fetch origin
   git checkout main && git pull       # after this work is merged; before that, use the branch it is on
   code .                              # opens VS Code attached to WSL
   ```
   VS Code's terminals now run inside Ubuntu. Run all the commands below there.

### A2. Use Test 1's Python environment and add the Test 2 packages
Test 1's venv (`~/go2venv`) already has the Unitree SDK, CycloneDDS, OpenCV and pynput.
```bash
source ~/go2venv/bin/activate
cd ~/Unitree-Dog-Claude-Dog-/Test2
pip install -r real_go2/requirements.txt     # PyTorch etc. (a few minutes the first time)
python -m pytest -q tests/test_real_go2.py tests/test_goto_zone.py    # 12 tests, no robot needed
```
In VS Code, choose the interpreter with Ctrl+Shift+P → *Python: Select Interpreter* →
`~/go2venv/bin/python`, so its terminals activate the venv automatically.

### A3. Turn on WSL "mirrored" networking (needed for Ethernet)
With WSL2's default networking, Linux sits behind a virtual network, and the Unitree SDK's device
discovery cannot reach the dog. Mirrored mode (Windows 11 22H2 or later) gives WSL the real adapters.
1. In Windows, open Notepad and save this as `C:\Users\<your user>\.wslconfig`:
   ```
   [wsl2]
   networkingMode=mirrored
   ```
2. In PowerShell: `wsl --shutdown`. Then reopen VS Code (it reconnects to WSL).
3. Check in the WSL terminal: `ip -br addr` should now list the same adapters as Windows.

### A4. Connect the dog by Ethernet (recommended: needed for go-to-zone's LiDAR safety)
1. Plug the laptop into the Go2 EDU's Ethernet port. Power on the dog and let it stand up by itself.
2. Windows → Settings → Network & internet → Ethernet → *IP assignment* → **Edit** → Manual, IPv4 on:
   - IP address **192.168.123.222**
   - Subnet mask **255.255.255.0** (in Windows 11 it may ask for a prefix length: **24**)
   - Gateway and DNS left empty

   With mirrored networking, WSL inherits this address. It replaces Test 1's `setup_network.sh`.
3. In the WSL terminal:
   ```bash
   ip -br addr | grep 192.168.123        # first word = the interface name, e.g. eth1
   ping -c 3 192.168.123.161             # the dog must answer
   cd ~/Unitree-Dog-Claude-Dog-/go2_stream
   python preflight.py --iface eth1      # Test 1's check: robot state, LiDAR, camera (PASS/FAIL list)
   ```
   If preflight passes, the connection works. To avoid typing `--iface` every time, put the name under
   `network: iface:` in `go2_stream/config.yaml`.
4. If the dog answers ping but preflight sees no data, Windows' firewall is blocking WSL. In an
   **administrator** PowerShell, run this, then `wsl --shutdown`:
   ```
   Set-NetFirewallHyperVVMSetting -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -DefaultInboundAction Allow
   ```

### A4 (alternative). Connect by Wi-Fi (no cable: you can stay away from the dog)
Recording, the live demo and go-to-zone work over Wi-Fi, with the camera (about 14 fps) and the dog's IMU heading,
but **no LiDAR**: go-to-zone only runs with `--allow-no-lidar` (walking capped at 0.15 m/s, closely supervised).
Checked on our dog: it needs **no AES key** (its handshake is the older type), and the Wi-Fi code connects in about 2 s.

**Option 1: the dog's own hotspot (simplest).**
1. In the Unitree app, set the dog's Wi-Fi mode to **AP** (hotspot). Note the network name and password.
2. **Close the Unitree app** (the dog accepts one Wi-Fi client), then join that network from Windows.
   While on it, the laptop has **no internet** (Claude Code and pip won't work until you switch back).
3. Unplug the Ethernet cable and run with `--source webrtc --ap`:
   ```bash
   python real_go2/record_real.py --source webrtc --ap --speed medium
   python real_go2/goto_zone.py --source webrtc --ap --allow-no-lidar --model results/q1/resnet18/best.pt
   ```

**Option 2: dog and laptop on the same Wi-Fi network (keeps the laptop's internet).**
1. In the Unitree app, connect the dog to the network (STA mode) and note the IP it gets.
   Public / campus networks often block devices from talking to each other; if it won't connect, use Option 1.
2. Close the Unitree app, then use `--source webrtc --ip <dog IP>`.

If the dog's firmware is later updated to 1.1.15 or newer, it will ask for a key (`AesKeyRequiredError`):
`unitree-fetch-aes-key --email <your Unitree account> --device-type Go2`, then `export GO2_AES_KEY=<key>`.

### A5. Dry run without the dog (2 minutes)
```bash
cd ~/Unitree-Dog-Claude-Dog-/Test2
python real_go2/record_real.py --source fake --max-frames 100    # a window opens; press 1-4, then ESC
rm -r data_real                                                   # remove the test recording
```
This checks the window, keys and saving before you take the dog out. In WSL, windows open through WSLg.

---

## Part B — Collect the training data (about 20 minutes with the dog)

### B1. Before you start
- Clear a path through all four zones. Keep cables on the shelves out of the dog's reach and the floor
  free of loose wires.
- Have someone stand by. **SPACE** stops the dog immediately; the remote is a backup.
- Charge the dog. Recording uses the battery mostly for walking.

### B2. Record one session
```bash
cd ~/Unitree-Dog-Claude-Dog-/Test2
python real_go2/record_real.py --source sdk --iface eth1 --max-speed 0.25
```

**Driving** (Test 1's keys; they work even when the terminal has focus):

| Key | Action |
|---|---|
| R | stand up (once at the start) |
| W / S | forward / back |
| Q / E | turn left / right |
| A / D | step sideways |
| SPACE | stop |
| F | lie down |
| ESC | end the session |

**Labelling** (click the *Go2 zone recorder* window first):
- **1–4** the zone the dog is **in** right now: 1 lab_1, 2 lab_2, 3 open_area, 4 shelves.
- **0** while walking between zones: those frames are not saved, so boundaries are not mislabelled.
- **P** pause / resume.

The window shows the zone, frames per zone, and the part of the image that is saved (orange box).

### B3. What to do in each session (3–4 minutes)
1. Press R. Walk into **lab_1** and press **1**.
2. Slowly walk around inside lab_1, approaching the desks from different sides and distances.
3. **Turn on the spot a full circle** (hold Q). This matters for go-to-zone, which recognises where it
   is by looking around.
4. Press **0**, walk to the next zone, press its number, and repeat for **lab_2**, **open_area** and
   **shelves**.
5. Aim for roughly equal counts per zone (shown at the bottom of the window), then ESC.

Each session becomes one folder, `data_real/run_000`, `run_001`, … in the same layout as the simulated
dataset.

### B4. Record 5 sessions, and vary them
Run B2–B3 five times. Between sessions, change what you can: start from a different zone, take another
route, try another time of day or lights on/off, and leave chairs and people as they naturally are.

The **last session is the test set** and the **second-to-last is validation**. Your accuracy number
therefore measures recognition on a walk the model never saw. Make the last session a normal, complete
tour.

A good total is 3,000–5,000 frames, about 750–1,250 per zone. Recording at 5 Hz, that's 2.5–4 minutes of
labelled walking per zone across all sessions.

---

## Part C — Train and evaluate (no dog needed)
```bash
cd ~/Unitree-Dog-Claude-Dog-/Test2
python q1_supervised/train.py --config real_go2/real_config.yaml   # ~15-30 min on a laptop CPU
python real_go2/evaluate_real.py
python real_go2/evaluate_real.py --split val     # validation predictions (to tune the localiser)
python real_go2/zone_map.py                      # zone map for go-to-zone (rebuild after new sessions)
```
Results go to `results/real_q1/`:
- `metrics.json`: test accuracy, macro-F1, per-zone precision/recall/F1;
- `confusion_resnet18.png`: which zones get confused;
- `curves_resnet18.png`: the training curves;
- `resnet18/best.pt`: the model.

Training uses ImageNet ResNet18 exactly like the simulated Q1. A faster option is Colab with a GPU:
upload `data_real/`, then run `!Q1_CONFIG=real_go2/real_config.yaml python q1_supervised/train.py`.

**Rule of thumb:** if two zones are often confused (for example lab_1 and lab_2, which may look alike),
record more sessions that include the distinguishing details of each: different desk layouts, the walls,
windows and doors behind them.

---

## Part D — Live recognition video
```bash
python real_go2/live_zone.py --source sdk --iface eth1
```
**What the window shows:**
- the dog's camera;
- the **predicted zone** and the confidence for each zone;
- a 1-second majority vote;
- the **true zone** you key in, and a running **live accuracy**.

**Keys:**

| Key | Action |
|---|---|
| V | start / stop the MP4 |
| 1–4 | the true zone right now |
| 0 | true zone unknown |
| ESC | quit |

Driving is as in recording. The output, `results/real_q1/live_<time>.mp4`, `.csv` and `.json`, includes
the live accuracy.

## Part E — Go to zone (autonomous)
```bash
python real_go2/goto_zone.py --source sdk --iface eth1 --lidar-check   # first: LiDAR only, no autonomy
python real_go2/goto_zone.py --source sdk --iface eth1
```
**`--lidar-check`** shows the live free distance in every direction, and whether the navigator would
stop, slow down or continue. It sends no autonomous motion. Use it before the first mission to confirm
that an empty floor reads "clear ahead" (see `LAB_PLAN.md` 1.4).

1. Press **R** (stand).
2. Click the *Go2 go to zone* window and press **1–4** to choose a zone. For example, **4** sends the dog
   to the shelves.
3. The dog then:
   - **looks around:** turns once on the spot, classifying every frame *and* reading the LiDAR. That gives
     two maps over 24 directions: how likely the target is in each direction (camera), and how far the
     dog can walk in each direction before something is in the way (LiDAR, a 0.6 m-wide corridor between
     knee and chest height, so the floor is ignored). The zone most views belong to is *where it is*. If
     the target already holds at least half of the views, it has **arrived** and waves.
   - **chooses a path that avoids obstacles:** each direction scores *target likelihood × free space*.
     A direction with a desk 0.5 m away scores near zero even if the target is right behind it. If the
     target's own direction is blocked, it **detours**: it picks the open direction (at least 1.1 m free)
     closest to the target and keeps detouring on the same side, so it doesn't swing back and forth
     around a desk. Directions that just blocked it are remembered and penalised for the next 3 legs.
   - **explores** if the target isn't visible at all, for example hidden behind shelves. It walks toward
     open space, preferring to keep going the same way and to pass the edges of obstacles, where a hidden
     zone tends to come into view. It explores for at most 10 legs.
   - **turns** to the chosen direction and **walks one leg** of up to 6 s (10 s while exploring to
     localise). While walking,
     the LiDAR is checked every scan. If something is within **1.28 m** ahead (at medium speed), it slows to 40 % and steers
     toward the freer side. If something is within **0.78 m** (at medium speed), it stops, marks that direction as blocked
     and looks around again. The leg also ends early when the target is recognised steadily, or when it
     fades from view.
   - **looks around again**, and repeats, for at most 20 legs. It gives up with "boxed in" when no
     direction has at least 1.1 m free.

   The panel shows the target, the state, what the camera sees, and a radar. On the radar, the colour
   shows where the target looked likely, grey shows the free space the LiDAR measured in each direction,
   and the white line is the chosen path. While it is steering around something, the panel says
   "avoiding left/right".
4. **Any manual key** (SPACE, W/A/S/D/Q/E) cancels the mission at once. **C** cancels too, **V**
   records the MP4, and **ESC** quits.

Without the LiDAR (Wi-Fi, or `--allow-no-lidar`), there is no obstacle avoidance. The dog then walks at
most 0.15 m/s for 2 s legs. Use this only with a clear floor and a hand on SPACE.

**Tested in a 2-D simulated room, not yet on the real dog.** `python real_go2/nav_sim.py` runs the
navigator closed-loop in a 6 × 4 m room. The robot follows the commands exactly, the LiDAR profile is
ray-cast, and the "camera" sees the target zone when facing it with nothing in the way. Each layout is
run from 4 starting points × 4 starting headings, and the result is saved in
`results/real_q1/nav_sweep.json`:

| Layout | Reached the zone | Closest approach to an obstacle (from the dog's centre) |
|---|---|---|
| Low desk blocks the walk, not the view | 16/16 | 0.25 m |
| Tall shelf hides the zone from the start | 3/16 | 0.26 m |

(The closest approach is in any direction, usually sideways while passing the obstacle.) The tall shelf
leaves 0.8 m gaps to the walls. Since the larger stop distances below, a gap that narrow counts as too
tight, so most of those runs no longer get round it (10/16 with the old 0.5 m stop distance). With 1.0 m gaps
it arrives from every start tried (`tests/test_goto_zone.py`).

**Stop distances (changed 13 Nov 2026, after the dog touched things in the lab).** The LiDAR measures
from the dog's centre, and its nose is about 0.35 m ahead of that. The dog now stops at
`stop_obstacle_m + reaction_s × speed` = 0.60 + 0.5 × 0.35 ≈ 0.78 m at medium speed, which leaves about
0.4 m in front of the nose before braking, and slows down from 1.10 m + the same speed term. The walking
corridor now starts 11 cm above the floor (Test 1's starts at 16 cm), so lower obstacles are seen too.

Without the zone map, start where the target zone is in view, or walk it part of the
way. `tests/test_goto_zone.py` checks the avoidance rules and both layouts.

### E2. The zone map: localisation and planned routes
The classifier alone mixes up zones (open_area is often taken for lab_2 or shelves), and from one spot
it can't tell which way a zone is. The recordings also hold the dog's **odometry**, so
`python real_go2/zone_map.py` aligns the 5 sessions into one map of the lab
(`results/real_q1/zone_map.png`: lab_1 at one end, open_area in the middle, lab_2 and shelves beyond).
`goto_zone.py` uses it automatically when it exists (Ethernet or Wi-Fi: both give odometry; `--no-map`
turns it off):
- **Localising.** All the time, also while you drive by hand, the camera's zone guesses are matched against
  the map along the path the odometry measures. Single frames are often wrong, but a path through
  several zones fits only one place on the map. The mini-map (top right of the camera view) shows the dog
  and says **"map: here <zone>"** once it is sure, or **"localising – drive around"** until then.
  **Tip:** before pressing 1–4, drive the dog by hand through two or three zones (about a minute). It
  then starts the mission already localised and goes straight there.
- **Routes.** Once localised, it plans the shortest path over ground walked while recording, to at least
  0.75 m inside the target zone, and follows it (re-planning every 3 s), with the same LiDAR
  slow/steer/stop. A blocked spot is avoided in the next plan.
- **Arrival needs the map too.** It arrives only when it is localised **and** the map puts it in the
  target zone (≥ 60 %), **and** the camera agrees in some of the views. If the camera says "here" but the
  map disagrees, it keeps exploring instead of stopping in the wrong zone.
- Not localised yet: the camera-only search above, with longer exploring legs, until it localises.

**Checked offline on real data:**
- `python real_go2/zone_map.py --eval`: the held-out test session (`run_004`), with its real classifier
  outputs, on a map built from the other 4 sessions → `results/real_q1/localization_eval.json` / `.png`.
- `python real_go2/nav_sim.py --map`: closed loop on the real map, every zone → every other zone from a
  cold start. The camera is the validation session's real classifier outputs, replayed per zone →
  `results/real_q1/nav_map_sweep.json` / `nav_map_paths.png`. The simulated camera doesn't depend on
  direction, which makes the camera-only baseline weaker than on the real dog.
- The localiser settings were chosen on the validation session (`--tune`), not the test session.

After the trials, `python real_go2/summarize_real.py` gathers the dataset size, test metrics, live
accuracy and every go-to-zone request into `results/real_q1/summary.md` for the report.

All thresholds are in `real_config.yaml` under `navigation:`. Each run writes
`results/real_q1/goto_<time>.json`, the event log with every state change and its reason, plus the MP4.

**How well will it work?** It's coarse visual navigation: it finds a zone it can see, room by room, not
an exact spot. It works best when the zones look clearly different and the target is visible from where
the dog starts. Expect lab_1 / lab_2 mix-ups if they look alike. The JSON log shows exactly why it
stopped (arrived, blocked, boxed in, not seen, out of legs), so you can report successes and failures
honestly. Try each zone from a few starting points.

## Filming checklist
1. A phone films the dog. Run `live_zone.py` and `goto_zone.py` with **V** to record the screen view.
2. **Live clip:** one slow tour through all four zones, pressing the true zone keys, so the video shows
   predicted vs true and the live accuracy.
3. **Go-to-zone clip:** from the open area, send it to shelves, then to lab_2, then to lab_1. Include a
   case where it fails, if one happens, and show its reason in the panel.
4. **For the report:**
   - `metrics.json` and the confusion matrix (test session);
   - `live_<time>.json` (live accuracy);
   - the `goto_<time>.json` logs: count arrivals and failures;
   - the comparison with the simulated Q1's 0.882.

## Safety
- Clear floor, no loose cables, people out of the path. Start with `--max-speed 0.2`.
- **SPACE** stops at once. **ESC** stops and quits, and always sends StopMove. Test 1's watchdog stops the
  dog if the control loop stalls.
- Autonomous walking runs only with the LiDAR check over Ethernet. Commands lapse within 0.3 s if the
  program hangs, and any manual key takes over.
- Test 1 notes its LiDAR corridor check is not yet tuned on the real robot. If the dog refuses to walk
  ("obstacle … ahead" with nothing there), floor points may be triggering it. Check with Test 1's
  `python run_detection.py --iface eth1 --teleop --mission chair --mission-dry-run` (prints only, no motion)
  and adjust `safety.corridor` in `go2_stream/config/behaviors.yaml`.
- Never use X then Y (damp) while standing: the dog collapses.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No network interface 'enp0s31f6'` | Pass `--iface` with the name from `ip -br addr \| grep 192.168.123`. |
| No 192.168.123.222 in `ip -br addr` | Mirrored networking is not on (A3), or the Windows static IP is not set (A4). |
| Ping works, no camera / state | Firewall rule in A4 step 4; run `preflight.py`. |
| `pynput unavailable` / keys don't drive | WSLg must be running (`echo $DISPLAY` shows `:0`). Press R first. |
| Wi-Fi: `AesKeyRequiredError` / `AesKeyRejectedError` | Fetch or check the key (A4 alternative, step 3). |
| Wi-Fi: `RobotBusyError` | Close the Unitree app. |
| `need at least 3 recorded sessions` | Record more sessions (B4). |
| Go-to-zone: "not seen in any direction" | The target isn't visible from here. Drive closer and retry, or record more varied data. |
