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

> **Status:** everything was tested end to end with a *fake* camera (replayed simulator frames):
> recording, training, evaluation, the live demo, go-to-zone and the MP4s. The navigator logic is also
> covered by unit tests. The connections to the real dog (`--source sdk`, `--source webrtc`) have **not
> yet been run against the robot**. The first session is the real test; send any error message.

| Script | Step |
|---|---|
| [`record_real.py`](record_real.py) | Drive the dog and record labelled frames → `data_real/run_XXX/` |
| `q1_supervised/train.py --config real_go2/real_config.yaml` | Retrain ResNet18 on the lab (same recipe as Q1) |
| [`evaluate_real.py`](evaluate_real.py) | Accuracy on the held-out last session, per-zone scores, confusion matrix |
| [`live_zone.py`](live_zone.py) | Live recognition on the dog's camera, with an on-screen panel, recorded to MP4 |
| [`goto_zone.py`](goto_zone.py) | Pick a zone (keys 1–4); the dog looks around, turns and walks there |
| [`sources.py`](sources.py) | Connections: `sdk` (Ethernet), `webrtc` (Wi-Fi), `fake` (no robot) |

Driving and safety reuse Test 1's teleop ([`go2_stream/teleop.py`](../../go2_stream/teleop.py)): speed
limits, a watchdog, and manual override of anything autonomous.

---

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
python -m pytest -q tests/test_real_go2.py tests/test_goto_zone.py    # 6 tests, no robot needed
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

### A4 (alternative). Connect by Wi-Fi
Recording and the live demo work over Wi-Fi. Go-to-zone needs the LiDAR, which this connection doesn't
provide, so it only runs with `--allow-no-lidar` (slower, closely supervised).
1. `sudo apt install -y portaudio19-dev && pip install unitree_webrtc_connect`
2. In the Unitree app, connect the dog to the lab Wi-Fi (STA mode) and note its IP. The laptop must be on
   the same network.
3. If the app shows firmware **1.1.15 or newer**, fetch the dog's key once:
   `unitree-fetch-aes-key --email <your Unitree account> --device-type Go2`. It asks for the password.
   Then run `export GO2_AES_KEY=<key>`.
4. **Close the Unitree app.** The dog accepts only one Wi-Fi client.
5. Use `--source webrtc --ip <dog IP>` instead of `--source sdk --iface eth1` in the commands below.

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
python real_go2/goto_zone.py --source sdk --iface eth1
```
1. Press **R** (stand).
2. Click the *Go2 go to zone* window and press **1–4** to choose a zone. For example, **4** sends the dog
   to the shelves.
3. The dog then:
   - **looks around:** turns once on the spot, classifying every frame. The zone most views belong to is
     *where it is*. The direction where the target looked most likely is *where to go*. If the target
     already holds at least half of the views, it has **arrived** and waves.
   - **turns** to that direction and **walks one leg** of up to 3 s at 0.25 m/s. The leg ends early if
     the LiDAR sees something within 0.5 m ahead, if the target is recognised steadily, or if it fades
     from view.
   - **looks around again**, and repeats, for at most 8 legs.

   The panel shows the target, the state, what the camera sees, and a radar of where the target looked
   likely during the last look-around.
4. **Any manual key** (SPACE, W/A/S/D/Q/E) cancels the mission at once. **C** cancels too, **V**
   records the MP4, and **ESC** quits.

All thresholds are in `real_config.yaml` under `navigation:`. Each run writes
`results/real_q1/goto_<time>.json`, the event log with every state change and its reason, plus the MP4.

**How well will it work?** It's coarse visual navigation: it finds a zone it can see, room by room, not
an exact spot. It works best when the zones look clearly different and the target is visible from where
the dog starts. Expect lab_1 / lab_2 mix-ups if they look alike. The JSON log shows exactly why it
stopped, so you can report successes and failures honestly. Try each zone from a few starting points.

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
