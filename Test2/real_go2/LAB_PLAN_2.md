# Lab plan 2: obstacle memory, more localisation data, and the next go-to-zone trials

This follows [`LAB_PLAN.md`](LAB_PLAN.md) (sessions 0–5: connect, record 5 sessions, train, go-to-zone, report).
That plan is done: 0.814 accuracy on the held-out session and 6/6 go-to-zone arrivals with the zone map
(REPORT 7.6–7.8). This plan does the next three things:

1. **The dog forgets close objects.** It stops about 0.5 m from an object and turns to look around. The object
   then drops out of the LiDAR's view (too close, too low, or behind the sensor). The dog treats it as gone and
   can walk back into it. **New:** an obstacle memory (`obstacle_memory.py`) keeps obstacles in the odometry
   frame. Before turning next to a remembered obstacle, the dog backs up. YOLO labels can be added (`--objects`).
2. **The localiser needs more and better pictures.** The held-out session confused open_area with lab_2
   (83/465 frames) and with shelves (70/465), and shelves with lab_2 (70/268). Only 3 sessions trained the
   model. **New:** `coverage.py` shows where and from which directions the pictures are thin and writes a
   recording plan. The **T** key in the recorder does a recorded survey turn on the spot.
3. **Measure both on the real dog**, with the old behaviour (`--no-memory`, old model) as the baseline.

| Session | Where | Dog | Time | Result |
|---|---|---|---|---|
| 0. Prepare | home | no | 30 min | code pulled, tests pass, recording plan printed |
| 1. Connect + blind-zone check | lab | yes | 30 min | preflight PASS; LiDAR blind distance measured |
| 2. Memory check (no autonomy) | lab | yes | 15 min | memory keeps a box through a turn |
| 3. Survey recording | lab | yes | 45 min | 3 survey sessions + 1 new test tour |
| 4. Retrain + compare | home | no | 45 min | new model vs old on BOTH test sessions |
| 5. Go-to-zone A/B trials | lab | yes | 60 min | memory on vs off; new vs old model |
| 6. Report | home | no | 30 min | results pushed; report section 7.9 |

Battery: sessions 1–3 fit one charge, session 5 another.

**Safety, as before:** anything that moves the dog (`--source sdk` with `record_real.py`, `goto_zone.py`, and now
the **T** survey turn) is run by **you**, with a hand on SPACE. Any driving key or SPACE cancels autonomy and
survey turns at once. Claude Code only runs the no-motion commands (`--source fake`, tests, `nav_sim.py`,
`coverage.py`, `zone_map.py`, training).

---

## Session 0 — Prepare at home (no dog)

```bash
source ~/go2venv/bin/activate
cd ~/Unitree-Dog-Claude-Dog- && git pull
cd Test2 && python -m pytest -q tests                 # all pass (includes tests/test_obstacle_memory.py)
cd ../go2_stream && python -m pytest -q tests && cd ../Test2
python real_go2/nav_sim.py --blind                    # memory off vs on in simulation -> results/real_q1/nav_blind_*
python real_go2/coverage.py                           # needs data_real/ (copy it back from your backup)
```
- `coverage.py` writes `results/real_q1/recording_plan.md` and `coverage.png`: numbered **survey spots** on the
  lab map, and the directions each spot is missing. **Print `coverage.png` or keep it on your phone.** It is the
  map for session 3.
- ❌ "No recordings in data_real": copy the backup back first (`cp -r /mnt/c/Users/<you>/Documents/go2_data_real
  data_real`).

**What changed in Test 1 (`go2_stream/`)**, synced from Test 2's lab fixes:
- `config.yaml` now uses `rt/utlidar/cloud_deskewed`;
- `cloud_to_body` rejects the upside-down sensor-frame cloud;
- the corridor `z_min` is -0.20 and `stop_if_closer_m` is 0.7;
- the interface is auto-detected when WSL renames it.

If you have local edits to `go2_stream/config.yaml`, keep only your `network: iface:`.

---

## Session 1 — Connect, then measure the LiDAR's blind zone

1. Connect as in LAB_PLAN 1.1 and run preflight (LAB_PLAN 1.2). The LiDAR line should now say
   `data on rt/utlidar/cloud_deskewed`.
2. **Blind-zone measurement.** Put a low box (about 30 cm tall, like the storage box) on open floor. Then run:
   ```bash
   python real_go2/goto_zone.py --source sdk --iface eth1 --lidar-check --model results/real_q1/resnet18/best.pt 2>&1 | tee logs_real/blind_check.log
   ```
   - Walk slowly (W) straight at the box from 2 m.
   - Note the **"ahead"** reading at the moment it stops tracking the box and jumps back to a large value. That
     distance is the **blind distance**: closer than it, the LiDAR doesn't see this box.
   - Repeat with a chair, and with a bag on the floor.

   | Object | Blind distance (ahead jumps up at) |
   |---|---|
   | box 30 cm | |
   | chair | |
   | bag | |

   Set `navigation.memory.blind_m` in `real_config.yaml` to the **largest** value plus 0.05 m (default 0.45).
   If it is bigger than `stop_obstacle_m` (0.60), the dog stops inside the blind zone. That is the lab
   failure, and the memory is what keeps the object then.

---

## Session 2 — Check the memory with no autonomy (10 min)

Same `--lidar-check` window. The orange text at the bottom of the camera view is the memory. Do these by hand:

| Test | Expected |
|---|---|
| Open floor, nothing near | "memory: nothing remembered nearby" |
| Walk (W) up to the box until "ahead" shows < 0.8 m | "remembered: obstacle 0.7 m ahead" |
| Keep walking until the LiDAR loses it (session 1's distance) | still "remembered: obstacle 0.5 m ahead" |
| Turn 90° with Q | "remembered: obstacle 0.5 m right": it turned, the box didn't move |
| Turn the rest of the way round (Q) | still remembered, with the direction following round |
| Walk away 2 m, turn back to face it | the live LiDAR sees it again; the memory and the LiDAR agree |
| **Pick the box up and take it away** while the dog faces the spot from 1.5 m | gone within a few seconds (the LiDAR sees the floor/wall through where it was) |

❌ It forgets the box during the turn: `blind_m` is too small (session 1), or odometry is missing. The log says
"Obstacle memory on" at the start; without it there is no odometry (Wi-Fi).
❌ It never forgets a box that was taken away: the far side isn't visible (no wall within 2 m behind it). Fine:
unseen cells fade after `forget_s` (120 s).

Optional: add `--objects` (YOLO). The text then says "remembered: chair 0.6 m left".

---

## Session 3 — Survey recording: study the room

Goal: pictures of every part of the lab from every direction, focused on open_area, lab_2 and shelves.
Record **3 survey sessions + 1 normal tour** (the new test set).

```bash
python real_go2/record_real.py --source sdk --iface eth1 --max-speed 0.25 2>&1 | tee logs_real/survey_$(date +%H%M).log
```
In each survey session:
1. Press **R**, then click the recorder window.
2. Drive to the next numbered spot on `coverage.png`. Press its **zone number** (1–4).
3. Press **T**. The dog turns once on the spot, slowly (about 18 s), recording all the way round. The window
   shows `SURVEY TURN 120/360 deg`. Keep a hand over SPACE: any driving key stops it.
4. While driving between spots, keep the zone key current (or press **0** on borders, so nothing is saved).
5. In about half the spots, do a second turn **0.5 m to one side**. Nearby spots from the same direction are
   what teach "where exactly".
6. Vary the sessions: different order of spots, lights as they are, chairs where people left them.

**Session 4 of this block is a normal tour** (like LAB_PLAN session 2), with no survey turns. It is the **new test
set**. Keep the old test session `run_004` as well, so old and new models compare on both.

✅ After each session, `metadata.json` lists `"surveys"` with `completed: true` for each spot.
Aim: 8–10 completed turns per survey session, about 2,500–3,500 new frames in total.

Before leaving: back up `data_real/` (it is not in git).

---

## Session 4 — Retrain and compare (home, no dog)

The current split (`split: by_order` in `real_config.yaml`) makes the **last** session the test set and the
second-to-last the validation set. With the new sessions recorded in the order above, the new tour is last
(test) and the third survey is validation. Old `run_004` becomes a training session. To also report on the old
test session, run the evaluation on it separately (step 3).

```bash
python real_go2/zone_map.py                                    # rebuild the map with all sessions
python real_go2/coverage.py                                    # gaps closed? (share of cells seen from all 8 directions)
cp -r results/real_q1/resnet18 results/real_q1/resnet18_v1     # keep the old model for the comparison
python q1_supervised/train.py --config real_go2/real_config.yaml 2>&1 | tee logs_real/train_v2.log
python real_go2/evaluate_real.py 2>&1 | tee logs_real/eval_v2.log
python real_go2/zone_map.py --tune && python real_go2/zone_map.py --eval
python real_go2/nav_sim.py --map                               # go-to-zone in simulation with the NEW classifier outputs
```

| Check | Old model (v1) | New model (v2) |
|---|---|---|
| Accuracy, new test tour | (run evaluate with the v1 model) | from `metrics.json` |
| open_area recall (was 0.66) | 0.66 | |
| shelves recall (was 0.71) | 0.71 | |
| Map localisation: time to "localised" (`localization_eval.json`) | | |
| `nav_map_sweep.json`: right-zone arrivals | | |

**Go / no-go:** continue with v2 if accuracy on the new test tour is at least v1's and open_area recall
improved. If not, look at the confusion matrix: a zone still confused means more survey spots there. Run
`coverage.py` again and record one more survey session.

---

## Session 5 — Go-to-zone A/B trials on the real dog

Use the same trial table as LAB_PLAN 4.3 (A–D, from tape marks), with **one change**: put the **low box 0.5–1.0 m
in front of the start mark of trials A and C**, in the dog's way. This is the case that failed.

Run each trial in two modes, alternating so the battery and lighting don't favour one mode:
```bash
# memory ON (default) with the new model
python real_go2/goto_zone.py --source sdk --iface eth1 --max-speed 0.25 2>&1 | tee logs_real/goto_mem_$(date +%H%M).log
# memory OFF (the old behaviour), same model
python real_go2/goto_zone.py --source sdk --iface eth1 --max-speed 0.25 --no-memory 2>&1 | tee logs_real/goto_nomem_$(date +%H%M).log
```
Press **V** to film every run. For each request, write down in your notes:
- **touches**: the dog's body touched an object (watch the video afterwards);
- **backoffs**: the panel said "too close to turn ... backing up";
- the result: arrived / failed / cancelled (the JSON has it).

| Mode | Requests | Arrived (right zone) | Touches | Backoffs | Median time |
|---|---|---|---|---|---|
| memory OFF | | | | 0 | |
| memory ON | | | | | |

Each run's JSON (`results/real_q1/goto_<time>.json`) records `"obstacle_memory": true/false`, so the runs can be
split into the two modes afterwards (ask Claude Code to add the split to `summarize_real.py` with your files).

**Film for the professor:**
1. One memory-OFF run where the dog stops at the box, turns, and heads back into it (stop with SPACE before it
   touches).
2. The same start with memory ON: it stops, backs up, turns, and goes round.
3. A survey turn in the recorder.

---

## Session 6 — Report

```bash
python real_go2/summarize_real.py
git checkout -b real-go2-results-2
git add results/real_q1/*.json results/real_q1/*.png results/real_q1/*.md results/real_q1/*.csv
git commit -m "Real Go2: obstacle memory trials, survey data, retrained model" && git push -u origin real-go2-results-2
```
New report section 7.9, written only from files in `results/real_q1/`:
- the blind-zone table (session 1);
- the memory check (session 2);
- coverage before and after (`coverage.json`);
- v1 vs v2 accuracy;
- the A/B trial table;
- the simulation numbers from `nav_blind_sweep.json`.

---

## After this: the crime-scene next steps

Once the dog can reliably go to a zone without touching things, the Phase 2 crime-scene work comes back in:
- **Evidence markers as objects:** label stains or evidence markers with the detector and keep them in the
  obstacle memory's object list. The dog then knows "evidence 0.8 m behind-left" after turning, and can return
  to it for a close-up photo (Phase 2's `evidence_frame`).
- **Don't step on evidence:** a remembered floor stain becomes a no-go cell for the route planner
  (`ZoneMap.block`). This is the Phase 2 safety layer's "don't walk through floor stains", on the real dog.
- **Search policy (Q3) on the real dog:** the PPO policy chose which zone to search next in simulation. With
  go-to-zone working, its zone choices can drive real `goto_zone` requests.
