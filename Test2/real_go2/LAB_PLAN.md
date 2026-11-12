# Lab plan: connect to the Go2 and finish the real-world Q1 + go-to-zone

This is the order of work, with go/no-go checks at each step. [`README.md`](README.md) explains each
script in detail. This plan says **what to do when**, **what you should see**, and **what to do if you
don't**.

| Session | Where | Dog needed | Time | Result |
|---|---|---|---|---|
| 0. Prepare | home | no | 30 min | software installed, dry runs pass |
| 1. Connect + check | lab | yes | 30 min | preflight PASS, LiDAR check done |
| 2. Record | lab | yes | 30–45 min | 5 sessions in `data_real/` |
| 3. Train + evaluate | home / Colab | no | 30–60 min | `metrics.json`, confusion matrix |
| 4. Demos + go-to-zone trials | lab | yes | 45–60 min | live MP4, goto MP4s and logs |
| 5. Report | home | no | 15 min | push results; section 7.6 filled in |

Battery: a full Go2 battery lasts roughly 1–2 h of walking. Sessions 1 + 2 fit one charge, and session 4
fits another.

All commands run in a **WSL terminal inside VS Code**, with the venv active:
```bash
source ~/go2venv/bin/activate
cd ~/Unitree-Dog-Claude-Dog-/Test2
mkdir -p logs_real          # terminal logs for debugging (not in git)
```
**Tip:** pipe every robot command through `tee` so a log exists when something goes wrong, for example
`... 2>&1 | tee logs_real/record_1.log`. Claude Code can then read the log file directly (see the last
section).

---

## Session 0 — Prepare at home (no dog)

1. **Get the code:** `cd ~/Unitree-Dog-Claude-Dog- && git checkout main && git pull`
2. **Packages** (README A2):
   ```bash
   pip install -r Test2/real_go2/requirements.txt
   cd Test2 && python -m pytest -q tests/test_real_go2.py tests/test_goto_zone.py
   ```
   ✅ `12 passed`. ❌ If an import fails, run `pip install` for the missing package. Then ask Claude Code
   with the error.
3. **Mirrored networking** (README A3). Afterwards, check that `ip -br addr` lists your Windows adapters
   (Ethernet, Wi-Fi).
4. **Dry run of the recorder:**
   `python real_go2/record_real.py --source fake --max-frames 100`. A window with replayed frames should
   open. Press 1–4, then ESC, then `rm -r data_real`.
   ❌ No window: check `echo $DISPLAY` (it should be `:0`); run `wsl --update` in PowerShell for WSLg.
5. **Check the keyboard:** the teleop keys (R, W, SPACE…) come from `pynput`. In the dry run, pressing W
   should log a `[dry-run] Move` line.
   ❌ Nothing: WSLg's X server isn't receiving keys. Click the window first; see README troubleshooting.
6. **Pack:** an Ethernet cable (and a USB-C/USB-A Ethernet adapter if the laptop has no port), the laptop
   charger, the dog's remote, and tape to mark start points on the floor.

---

## Session 1 — Connect and check (dog on, standing in the open area)

### 1.1 Ethernet link (README A4)
1. Plug the cable into the dog's Ethernet port (on the back, under the cover on the EDU). Power on the dog
   and wait until it stands.
2. Set the Windows Ethernet adapter to **192.168.123.222 / 24**, with no gateway.
3. In WSL:
   ```bash
   ip -br addr | grep 192.168.123        # note the interface name, e.g. eth1
   ping -c 3 192.168.123.161
   ```
   ✅ 3 replies. ❌ No address: redo the Windows IP setting, run `wsl --shutdown`, reopen VS Code. ❌ No
   ping replies: check the cable or port, and that the dog is fully booted (about 1 min).

### 1.2 Test 1 preflight
```bash
cd ~/Unitree-Dog-Claude-Dog-/go2_stream
python preflight.py --iface eth1 2>&1 | tee ../Test2/logs_real/preflight.log
cd ../Test2
```
✅ PASS for robot state, LiDAR and camera. ❌ Ping works but there's no data: add the firewall rule (README
A4 step 4) and run `wsl --shutdown`. ❌ Only the LiDAR fails: go-to-zone can't run, but recording and the
live demo still can. Continue with session 2 and debug the LiDAR afterwards.

**Save the interface name** under `network: iface:` in `go2_stream/config.yaml`, or keep typing
`--iface eth1`. The commands below use `eth1`; replace it with yours.

### 1.3 Drive check
```bash
python real_go2/record_real.py --source sdk --iface eth1 --max-speed 0.2 2>&1 | tee logs_real/drive_check.log
```
1. Press **R** (stand), tap **W** (it walks slowly), **Q**/**E** (turns), then **SPACE** (stop).
2. Check that the window shows the dog's camera.
3. Press ESC, then `rm -r data_real` (this was only a test).

✅ The dog moves and stops on SPACE, and the camera shows. ❌ The keys do nothing: click the window, check
`echo $DISPLAY`, and see whether the log shows `Move` lines. ❌ No camera: check preflight's camera line.

### 1.4 LiDAR check (important before any autonomy)
Test 1 noted that its LiDAR corridor is not tuned on the real dog. Check it now. This mode sends **no**
autonomous motion; you only drive by hand. The script loads a classifier even though this check doesn't
use it. Your lab model doesn't exist yet, so point it to the simulated Q1 model that is in git:
```bash
python real_go2/goto_zone.py --source sdk --iface eth1 --lidar-check --model results/q1/resnet18/best.pt 2>&1 | tee logs_real/lidar_check.log
```
The panel shows the free distance in every direction, and whether the navigator *would* stop, slow or
continue. Check:

| Test | Expected |
|---|---|
| Dog standing in open floor | "clear ahead", ahead ≥ 2 m (lookahead 2.5) |
| Walk it slowly (W) toward a desk | "would SLOW" at about 1.0 m, "would STOP" at about 0.5 m |
| Box or bag on the floor, 30 cm tall | it is detected (z from −0.15 to +0.45 m around the body) |
| Floor only, nothing ahead | **not** detected. If it says STOP on an empty floor, floor points are leaking in. |

❌ **Stops on an empty floor:** raise `z_min` in `go2_stream/config/behaviors.yaml` → `safety.corridor`
(for example from −0.15 to −0.05). That is a Test 1 file, so it's your decision to change. Note the
change in the report. You can also cross-check with Test 1's
`python run_detection.py --iface eth1 --teleop --mission chair --mission-dry-run`.
❌ **Ahead is always 0.00 and "NO LiDAR DATA":** the LiDAR topic isn't arriving. Preflight's LiDAR line
will say the same. Ask Claude Code with `logs_real/lidar_check.log`.

---

## Session 2 — Record the training data (README Part B)

**Before:** clear the floor of loose cables, tell people what you're doing, and have someone with the
remote nearby.

Run this 5 times, one session each:
```bash
python real_go2/record_real.py --source sdk --iface eth1 --max-speed 0.25 2>&1 | tee logs_real/record_$(date +%H%M).log
```
In each session:
1. Press **R**, then click the recorder window.
2. Walk into a zone and press its number (1 lab_1, 2 lab_2, 3 open_area, 4 shelves).
3. Move around inside the zone and **turn a full circle with Q**. Go-to-zone recognises zones while
   turning.
4. Press **0** while crossing to the next zone.
5. Visit all four zones and keep the counts per zone roughly equal, then press ESC.

**Vary the sessions:** use a different start zone and route each time. Leave the lights, chairs and people
as they naturally are. Make the **5th session a normal, complete tour**, because it becomes the test set.

✅ After each session, `ls data_real/` shows a new `run_00X`, and its `states.csv` grows. Aim for about
3,000–5,000 frames in total.
❌ "0 frames" or images not saving: the camera stream isn't arriving. Check the camera line of preflight,
and ask Claude Code with the log.

**Back up the data** before leaving the lab (it isn't in git), for example:
`cp -r data_real /mnt/c/Users/<you>/Documents/go2_data_real`.

---

## Session 3 — Train and evaluate (README Part C, no dog)

```bash
python q1_supervised/train.py --config real_go2/real_config.yaml 2>&1 | tee logs_real/train.log
python real_go2/evaluate_real.py 2>&1 | tee logs_real/eval.log
```
Training takes 15–30 min on a laptop CPU. On Colab with a GPU it takes a few minutes: upload
`data_real/`, then run `!Q1_CONFIG=real_go2/real_config.yaml python q1_supervised/train.py`, and download
`results/real_q1/`.

**Go / no-go before session 4:**
- Test accuracy ≥ 0.80: good, continue.
- 0.60–0.80: continue, but look at `confusion_resnet18.png`. If two zones are confused (likely lab_1 and
  lab_2), record 2 more sessions that show what distinguishes them, then retrain.
- < 0.60: check the labels first. Open a few images from `data_real/run_000/images/` against `states.csv`.
  A wrong key pressed for a whole zone shows up here.

---

## Session 4 — Demos and go-to-zone trials

### 4.1 Live recognition video (README Part D)
```bash
python real_go2/live_zone.py --source sdk --iface eth1 --max-speed 0.25 2>&1 | tee logs_real/live.log
```
Press **V** to record, drive one slow tour through all four zones, and press the true zone (1–4) as you
enter each. The panel shows a live accuracy. Press ESC.

### 4.2 LiDAR check with the trained model
If you changed `safety.corridor` in session 1, run the 1.4 check again first (without `--model`, it now
uses your lab model). Continue only when an empty floor reads "clear ahead".

### 4.3 Go-to-zone trials
```bash
python real_go2/goto_zone.py --source sdk --iface eth1 --max-speed 0.25 2>&1 | tee logs_real/goto_$(date +%H%M).log
```
Press **R**, click the window and press **V** to record. Then press a zone number. Keep a hand over
**SPACE**: any driving key cancels the mission at once.

**Trial protocol** (gives the report a success rate):

| # | Start (tape mark) | Requests, in order | Notes |
|---|---|---|---|
| A | middle of open_area | 4 shelves → 2 lab_2 → 1 lab_1 → 3 open_area | target in view from the start |
| B | inside lab_1 | 2 lab_2 → 4 shelves | crosses the room |
| C | inside lab_2 | 1 lab_1 → 4 shelves | |
| D | open_area, with a chair placed between the dog and the target | 4 shelves | the obstacle-avoidance case for the video |

- **Repeat** A–D twice if the battery allows: that is about 16 requests.
- **Cancelled requests:** a request you cancel with SPACE is logged as "cancelled" and is not counted as
  a success or a failure. Note why in your notes.
- **Starting point:** start with the target zone in view. Finding a zone it can't see is the weak point;
  it was 10/16 in simulation.

Each run writes `results/real_q1/goto_<time>.json`. It lists every request with its result and reason
(arrived, blocked, boxed in, not seen, out of legs), plus the MP4.

**What to film for the professor:**
1. The recording session, briefly.
2. The live recognition tour.
3. Trials A and D. D shows the dog slowing, steering round the chair, and the "avoiding" text on the
   panel.
4. One failure with its reason, if one happens. An honest failure is a good result.

---

## Session 5 — Results into the report

```bash
python real_go2/summarize_real.py         # -> results/real_q1/summary.md
git checkout -b real-go2-results
git add results/real_q1/*.json results/real_q1/*.png results/real_q1/*.csv results/real_q1/summary.md
git commit -m "Real Go2 results" && git push -u origin real-go2-results
```
MP4s, `best.pt` and `data_real/` stay out of git (`.gitignore`). Upload the videos separately. Then tell
me (in a Claude Code session on claude.ai) that the results are pushed. Section 7.6 of `REPORT.md` and
the PDF get filled in from those files only.

---

## Debugging with Claude Code in VS Code

Your local Claude Code runs on the same machine as the dog connection, so it can do what this cloud
session can't: run `ip`, `ping` and `preflight.py`, and read your logs.

**Start it** in the VS Code WSL terminal from `Test2/`:
```bash
cd ~/Unitree-Dog-Claude-Dog-/Test2 && claude
```
It reads `Test2/CLAUDE.md`, which explains this project, the safety rules and where the logs are.

**Good prompts:**
- "Preflight fails on LiDAR. Read logs_real/preflight.log and go2_stream/lidar.py and tell me why. Don't
  run anything that moves the dog."
- "record_real.py saves 0 frames. Read logs_real/record_1432.log and real_go2/sources.py and find the
  cause."
- "Run `ip -br addr` and `ping -c 2 192.168.123.161` and tell me if the Ethernet link is set up right."
- "The LiDAR check says STOP on an empty floor. Read logs_real/lidar_check.log and suggest a
  safety.corridor change. Don't edit Test 1 files, just tell me the values."
- "Training accuracy is 0.55. Read results/real_q1/metrics.json and the confusion matrix numbers. Which
  zones are confused, and what should I record next?"

**Rules to keep (also in CLAUDE.md):**
- Let it run read-only and no-motion commands: `ip`, `ping`, `preflight.py`, `--source fake`, `pytest`,
  `--lidar-check`.
- **You** run anything that moves the dog, with a hand on SPACE.
- Don't let it change Test 1 files (`go2_stream/`) without you deciding.
- If it changes code, run `python -m pytest -q tests` before using the dog again.
