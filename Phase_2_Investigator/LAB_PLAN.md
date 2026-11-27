# Lab plan: the real Go2 in the lab (Phase 2)

This is the step-by-step walkthrough for working with the real dog. It is current as of 27 Nov 2026 and
matches the code on `main`.

It has two parts:

| Part | What | Blood-spatter prints needed? |
|---|---|---|
| **Part 1** | The dog explores the lab on its own, builds its map, tags the zones (lab_1, lab_2, open_area, shelves), and walks to zones you ask for | **No. Do everything in Part 1 first.** |
| **Part 2** | The crime-scene search with printed stain patterns | Yes, but **only when you reach the red line below** |

Everything runs in a terminal on the Linux laptop (Claude Code in the terminal can help, see the end).
The older `Test2/real_go2/LAB_PLAN.md` was for Test 2 (Windows/WSL, recording training data). Test 2 is
finished, so use this file.

**Commands run from `~/Unitree-Dog-Claude-Dog-/Phase_2_Investigator` with the venv active:**
```bash
source ~/go2venv/bin/activate
cd ~/Unitree-Dog-Claude-Dog-/Phase_2_Investigator
mkdir -p logs
```
Every robot command below ends with `2>&1 | tee logs/<name>.log`, so there is a log for debugging.

---

# PART 1: map the lab and go to zones (no blood spatter)

| Step | What | Dog moves? | Time |
|---|---|---|---|
| 0 | Laptop setup (once) | No | 30 min |
| 1 | Connect the dog | No | 10 min |
| 2 | Sensor check | **No** | 5 min |
| 3 | First short mapping run | Yes | 10 min |
| 4 | Full mapping run | Yes | 15–20 min |
| 5 | Go-to-zone trials, same boot | Yes | 20 min |
| 6 | Relocalisation trials, after a reboot | Yes | 20 min |
| 7 | Send me the results | No | 5 min |

Battery: steps 3–6 are about an hour of walking. Start with a full battery.

## Step 0: laptop setup (once)

Skip what is already done. The full details are in [`real_robot/README.md`](real_robot/README.md),
section 1.

```bash
cd ~/Unitree-Dog-Claude-Dog- && git checkout main && git pull
cd Phase_2_Investigator
./scripts/setup.sh                                   # Python packages (+ the simulator, for rehearsals)
pip install -r ../go2_stream/requirements.txt        # Test 1 modules used for the real dog (Unitree SDK, CycloneDDS)
python -m pytest -q tests/test_zone_map.py tests/test_localize.py tests/test_real_session.py
```
**✅ Expect:** all tests pass.

**The lab zone model:** copy `best.pt` from the old laptop to
**`~/Unitree-Dog-Claude-Dog-/Test2/results/real_q1/resnet18/best.pt`**. It isn't in git, and the
mapping step stops with a clear message if it's missing.

**Rehearsal without the dog (optional, 3 min):** the same code path through the simulator:
```bash
python scripts/real_check.py --link sim --seconds 4
python scripts/map_zones.py --backend real --link sim --max-time 120
```

**If the laptop also runs the TurtleBot:** put the TurtleBot's terminal on its own ROS 2 domain
(`export ROS_DOMAIN_ID=30`). Phase 2 uses at most 2 CPU threads (`PHASE2_THREADS`).

## Step 1: connect the dog

1. Clear the floor: no loose cables or bags. Tell people in the room.
2. Power on the dog and wait until it stands (about 1 min).
3. Plug in the Ethernet cable between the dog and the laptop.
4. Set up the network:
   ```bash
   ../go2_stream/setup_network.sh detect             # shows the wired interface, e.g. enp0s31f6
   ../go2_stream/setup_network.sh nmcli enp0s31f6    # FIRST TIME ONLY: saves a profile "go2-direct"
   nmcli con up go2-direct                           # every session
   ping -c 3 192.168.123.161
   python ../go2_stream/preflight.py 2>&1 | tee logs/preflight.log
   ```

**✅ Expect:** 3 ping replies, and preflight PASS on state, LiDAR and camera.

**❌ If not:**

| Problem | Fix |
|---|---|
| No ping reply | Check the cable; is the dog fully booted? |
| Ping works but preflight sees no data | Check the firewall (`sudo ufw status`); if it's on, run `sudo ufw allow from 192.168.123.0/24` |

## Step 2: sensor check (the dog does NOT move)

```bash
python scripts/real_check.py --seconds 15 2>&1 | tee logs/real_check.log
```

While it runs, walk the dog 2–3 m with the **remote** and turn it once, so the map snapshot shows some of
the room. It prints a PASS / WARN / FAIL list and saves `outputs/real_check/<time>/` (`map.png`,
`frame.jpg`, `report.json`).

| Line | ✅ Good | ❌ If not |
|---|---|---|
| odometry pose | poses arriving | preflight; is `rt/sportmodestate` publishing? |
| LiDAR scans | most of the 360 directions have a return | is the LiDAR on? It must be `rt/utlidar/cloud_deskewed` |
| floor height | "floor 0.31 m below the body", or within 0.06 m of it | write down the number and add **`--body-height <number>`** to every command in steps 3–6 |
| occupancy map | `map.png` shows walls around the path | a filled blob means the floor counts as an obstacle: check the floor-height line |
| camera | a frame | preflight's camera line |
| zone classifier | the zone the dog is really in | is `best.pt` at the path in step 0? |

**Open `map.png` and `frame.jpg` and look at them.** Don't go on until this check is all PASS (a WARN on
the floor height is fine once you've added `--body-height`).

## Step 3: first short mapping run (the dog moves on its own)

**Safety, every run:**
- Hold the remote.
- **Any key in the terminal stops the dog** and saves the map so far. The remote also takes over at once.
- Start the dog in the **open area**, with about 1 m of free space around it.

```bash
python scripts/map_zones.py --backend real --max-time 300 --video 2>&1 | tee logs/map_short.log
```

**What it does:**
1. Stands up and turns a full circle.
2. Drives to unexplored openings (`frontier N: go to [x, y]`), turning a full circle at each.
3. Visits viewpoints about 1 m apart (`viewpoint N`), turning at each.

It drives at ≤ 0.3 m/s, slows from 0.9 m, and stops 0.4 m from obstacles.

**Watch for:**
- it stopping far from obstacles, or refusing to move: the floor is being seen as an obstacle, so stop
  and recheck step 2;
- it going near the shelf cables: stop it.

**Output:** `outputs/zone_map/real_<time>/`:
- `zone_map.png`: the map, zones in colour;
- `zones.json`: each zone's area, goal point and confidence;
- `report.json`: path length, look-arounds, frames tagged;
- `map_building.mp4`: the run.

**✅ Look at `zone_map.png`.** Do the walls look like the lab? Are the zones roughly where they really
are? If yes, go on. If not, send me the folder and `logs/map_short.log` before step 4.

## Step 4: full mapping run (this is the map used in steps 5 and 6)

Start in the open area again (same as step 3, longer):
```bash
python scripts/map_zones.py --backend real --max-time 900 --video 2>&1 | tee logs/map_full.log
```

It ends by itself when every reachable area has had a look-around, or after 15 min. Then:
```bash
ls -d outputs/zone_map/real_*                        # the newest one is this run: call it MAP below
python scripts/goto_zone.py --map outputs/zone_map/real_<time> --list
```

**✅ Expect:** all four zones listed, each with an area and a goal point.

**Write down while you look at `zone_map.png`:** is each zone's coloured area really that part of the
lab? Answer yes / mostly / no per zone, in the sheet below.

**If a zone is missing:** the dog never looked around there. Run step 4 again, starting nearer that
zone.

## Step 5: go-to-zone trials (same boot: do NOT turn the dog off)

**Leave the dog powered on after step 4,** so its odometry still matches the map. From a starting spot
in each zone, send it to the other zones, one command per trial:
```bash
python scripts/goto_zone.py --map outputs/zone_map/real_<time> --backend real --zone lab_2 2>&1 | tee logs/goto_5a.log
```

**What it does:**
1. Turns a circle to find itself on the map: `[relocalise] … score … OK`.
2. Plans a path to the zone's goal point and walks it.
3. Turns a circle on arrival.

It prints `ARRIVED` / `NOT ARRIVED`, what the map says, and what the camera's look-around says. Each run
writes `outputs/goto/goto_<time>.json` and a PNG.

**Trial protocol** (12 trials; do as many as the battery allows):

| Start in | Send to (one trial each) |
|---|---|
| open_area | lab_1, lab_2, shelves |
| lab_1 | lab_2, shelves, open_area |
| lab_2 | lab_1, shelves, open_area |
| shelves | lab_1, lab_2, open_area |

**For each trial, write down whether the dog really ended in the right zone,** judged by you, not by
the dog. This is the ground truth for the report.

## Step 6: relocalisation trials (after a reboot)

The dog's odometry restarts at every power-up, so the saved map has to be found again from the LiDAR.

1. Sit the dog down and power it off and on. Wait until it stands, then `nmcli con up go2-direct` if
   needed.
2. Put it somewhere **different** from where you started step 4, in the middle of a zone.
3. Run the same command as step 5, for example:
   ```bash
   python scripts/goto_zone.py --map outputs/zone_map/real_<time> --backend real --zone shelves 2>&1 | tee logs/goto_6a.log
   ```

**✅ Expect:** `[relocalise] … score … OK`, then it walks to the zone. The score is the share of LiDAR points that
land on the map's walls: it must be at least 0.35, and higher is better. Note it in the sheet.

**If it says `Could not find the dog on this map`:** it refuses to drive rather than drive wrong.
- Move it to a more open spot, away from chairs that have moved, and try again.
- Write down each failure. It's a result too.

Do 4–6 trials from different start spots.

## Step 7: results sheet and what to send

**Results sheet:** copy this table into a text file and fill it in.
```
date: ____   floor height used: ____   map folder: outputs/zone_map/real_________

Zone map check (step 4): lab_1 __  lab_2 __  open_area __  shelves __   (yes / mostly / no)

trial | step (5 or 6) | start zone | target | dog said ARRIVED? | really in target? (Y/N) | notes
------+---------------+------------+--------+-------------------+-------------------------+------
  1   |               |            |        |                   |                         |
```

**Send me** (zip, or push to a branch):
- `outputs/real_check/*/report.json`, `map.png`, `frame.jpg`;
- the `outputs/zone_map/real_*` folders (the MP4s can go to YouTube instead);
- `outputs/goto/` (all the `goto_*.json` and PNGs);
- `logs/`;
- the results sheet.

From these I'll tune the settings, add the real-dog results to `REPORT.md` (sections 5.5 and 8), and
fix anything that went wrong.

**Troubleshooting**

| Symptom | Do this |
|---|---|
| `No interface with a 192.168.123.x address` | Step 1: `nmcli con up go2-direct`, then `ip -br addr` |
| `No pose or LiDAR data from the dog after 15 s` | Run `preflight.py`. Is the LiDAR on? Is `CYCLONEDDS_HOME` set? |
| `Zone classifier not found` | Copy `best.pt` (step 0) |
| The dog won't move, or stops at nothing | The floor is seen as an obstacle: step 2's floor-height line, then `--body-height` |
| Doubled or blurred walls after a long run | Odometry drift. Use the shorter step-3 map, or map in two shorter runs |
| The far end of the lab is missing from the map | The map is 30 × 30 m around the start: start nearer the middle, or add `--map-size 40` |
| `stuck at … backing off` many times | Tight spot. Stop it, move it to open floor, and send me the log |
| The terminal key doesn't stop it | Use the remote. Then tell me, with the log |

---

<br>

# 🟥🟥🟥 STOP. PART 1 ENDS HERE. 🟥🟥🟥

## ⛔ DO NOT PRINT ANY BLOOD-SPATTER PATTERNS BEFORE THIS LINE.

**Print the stain patterns only after:**
1. Part 1 is finished: the zone map works and go-to-zone trials are done;
2. you've sent me the Part 1 results;
3. I've sent you the printable stain kit. **It doesn't exist yet.** Tell me when Part 1 is done, and I'll
   generate it from the simulator's spatter physics, at true scale, with the ground truth for every
   stain.

---

# PART 2: crime-scene search with printed stains (LATER, not today)

An outline only, so you know what's coming. Each step will get full instructions here when its software
is ready.

1. **Print the stain kit** (from me, see above): non-graphic red patterns on paper at true scale
   (impact spatter, a cast-off line, a drip trail), plus a few red non-blood objects as distractors.
   Tape them to walls and the floor in different zones.
2. **Photos for the detector:** drive the dog past the prints while recording. The laptop's GPU then
   fine-tunes the stain detector (RF-DETR-Seg) on simulated images plus these photos.
3. **The search on the real dog:**
   - the dog explores and finds the prints;
   - it puts them on its zone map ("stain cluster in lab_2 at x, y") and walks around floor prints;
   - this needs depth for the camera from the floor plane and LiDAR, which I'll add first.
4. **Close-ups for analysis:** a phone or other high-resolution camera mounted on the dog. The Go2's own
   camera is too coarse for single drops.

---

## Using Claude Code in the terminal

From `Phase_2_Investigator/`, run `claude`. It reads `CLAUDE.md`, which has the project rules: it never
runs anything that moves the dog, and it never edits Test 1 or Test 2.

**Good prompts:**
- "real_check says FAIL on LiDAR. Read logs/real_check.log and tell me why."
- "Look at outputs/zone_map/real_<time>/zone_map.png and report.json: did the mapping run go well?"
- "The dog stopped with nothing in front of it. Read logs/map_short.log and suggest a --body-height."
- "Summarise outputs/goto/*.json into the trial table."
