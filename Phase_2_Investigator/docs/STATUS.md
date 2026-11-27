# Phase 2 status and handoff (updated 27 Nov 2026)

Read this first when picking the project up: where everything stands, what was learned the hard way,
and what comes next. `CLAUDE.md` has the rules; `REPORT.md` has the results and method.

## 1. Working agreement with the user
- **Git:**
  - Develop on branch `claude/eng6010-test2-crime-scene-0rst3r`.
  - The user always asks "open a PR into main"; they merge it themselves. **Never merge.**
  - After a merge, restart the branch from `origin/main`: `git checkout -B <branch> origin/main`.
  - Check `git log origin/main` before saying something is or isn't merged. A stale fetch once caused a
    wrong "not merged" claim.
- **Content:** every number in docs and reports must come from a result file written by a script. If a
  run's output was overwritten, don't quote it. Report failures honestly, alongside the successes.
- **Freezes:** Test 2 (`Test2/`) is submitted and frozen: read and import only. Test 1 (`go2_stream/`):
  don't edit without asking.
- **The user's setup:**
  - a native Linux laptop, terminal only (Claude Code in the terminal), **with a good GPU**;
  - a TurtleBot / ROS 2 terminal for senior design also runs on it, so keep CPU use low
    (`PHASE2_THREADS`, default 2) and suggest `ROS_DOMAIN_ID=30` for the TurtleBot;
  - the cloud container has **no GPU**.
- **Style:** the user writes briefly ("its merged whats next", "go, do X"). Reply with a short summary
  table and clear next options, then act.

## 2. Where things stand
| Area | State | Key results (source file) |
|---|---|---|
| Stage A simulation | Done (Oct) | `docs/stage_a_checklist.md` |
| BPA on ideal data | Done | angle 1.9°, origin 0.12 m, mechanism 0.94 (`evaluation/results/*.json`) |
| Stage R zone map, go-to-zone, relocalisation | Done in sim; **real dog not yet run** | 4/4 zone goals, go-to 4/4, relocalised 2/2 within 2 cm / 2° (`docs/media/stage_r/`) |
| C2 stain detector | Done in sim | test precision 0.92 / recall 0.97, distractors flagged 4.8 % (`docs/media/stage_c/detector_metrics.json`) |
| C3 evidence map, C4 keep-out | Done in sim | S1 floor stains walked over 24/84 with keep-out vs 56/84 without |
| C5 close-ups and stain analysis | Done in sim | close-up classifier test 0.967 vs 0.700; missions S1, S2, S3, S5 mechanism 5/5, origin 0.40–0.68 m (`docs/media/stage_c/`) |
| Phase 2 report | Done | `REPORT.md`, `Phase2_Report.pdf` (`bash report/build_pdf.sh`) |
| Lab plan | Done | `LAB_PLAN.md`: Part 1 today, Part 2 after the red stop line |

## 3. Next steps, in order
1. **The user's lab session (`LAB_PLAN.md` Part 1).** They will send:
   - `outputs/real_check/*`;
   - the `outputs/zone_map/real_*` folders;
   - `outputs/goto/*`;
   - `logs/`;
   - a results sheet (whether the dog *really* ended in the right zone).

   Then:
   - tune `--body-height`, the scan band (`RealGo2Session(scan_band=...)`, default 0.12–0.60 m), speeds
     and safety distances from what happened;
   - add the real results to `REPORT.md` sections 5.5 and 8 and rebuild the PDF.
2. **The printable stain kit (C1),** generated **only after Part 1**: true-scale, non-graphic red patterns
   from `sim_env/spatter` (impact spatter, a cast-off line, a drip trail), each sheet with its ground
   truth (angles, directions, origin), as PDFs to print and tape up. Then fill in `LAB_PLAN.md` Part 2.
3. **The real-dog stain detector:** fine-tune RF-DETR-Seg on the laptop GPU, using the simulator COCO set
   (`training/build_coco_dataset.py`) plus the dog's photos of the printed kit, and compare it with
   `perception/models/stain_rf_image.pkl`.
4. **Real-dog detection geometry:** the Go2 camera has no depth, so region positions must come from the
   floor plane (camera height and pitch) or from LiDAR points projected into the image.
   `scripts/investigate.py` is simulation-only until then.
5. **Close-ups on the real dog:** ask whether a phone or high-resolution camera can be mounted. The Go2's
   camera is about 2 mm per pixel at 1 m, too coarse for droplets.
6. **Still unanswered by the user:** what stain props are allowed in the lab, and whether a close-up
   camera can be mounted.

## 4. Lessons and gotchas (each one cost time)
**Simulation and scenes**
- **Random scenes:** `--random SEED` *regenerates* `sim_env/scenarios/generated/rSEED.yaml` from the seed
  with the current generator, and picks a house from the seed. Test 2's scenes were made with
  `write_random(seed, env="scenesmith_house_185")`. `map_zones.py` forces house 185 (`--house`). r1854 is
  the Test 2 Q1 *test* scene. Generated scenarios and `sim_env/build/` are gitignored.
- **Video:** `imageio` has no ffmpeg plugin in the cloud container, so videos use `cv2.VideoWriter`
  (mp4v).
- **EGL errors at exit** (`eglDestroyContext`) are harmless renderer teardown noise.
- **The house-185 zone classifier** is Test 2's `results/q1/resnet18/best.pt`, with classes
  victim_area, bed_area, lounge_area, desk_entry_area and bathroom. Ground-truth zone polygons are in
  `mapping/config/house185_zones.yaml`.

**Navigation and safety** (`navigation/local.py`, `explorer.py`)
- **Escape moves:** points already inside the 3 cm margin used to make every escape move "hit", which
  wedged the dog in the S1 doorway. Straight and sideways moves now count only points they bring closer.
  **Turns keep the strict rule:** the legs stick out past the body box, and the relaxed rule let a leg
  scrape a wall (the corner test in `tests/test_navigation.py`).
- **`free_maneuver`** keeps the best-scoring move unless the safety filter would leave it unable to move
  the robot (S5 corner wedge). Don't require moves to pass the filter unchanged: the filter legitimately
  strips the turn from a sideways-plus-turn move, and the corner test then fails.
- **`Navigator.escape()`** runs after `back_off()` fails to move the robot.
- **Look-around side-steps were reverted:** they ignore keep-out, and S1 walked over 47 instead of 24
  floor stains.
- **Coverage viewpoints** use the normal planning radius (0.32 m). The tight radius (0.24 m) wedged the
  dog (`docs/media/stage_r/sim_r1854_tight_radius_report.json`). This is why the r1854 bathroom is never
  entered.
- **Planning-only layers:** keep-out is a cost (+60), not a wall. Viewpoints keep 0.4 m from floor
  evidence.

**Real dog** (`real_robot/session.py`)
- **LiDAR:** use `rt/utlidar/cloud_deskewed` (odometry frame, body removed). The raw `rt/utlidar/cloud`
  is upside down, so the floor looks like an obstacle.
- **Scans:** sparse-cloud directions with no return are **NaN = no information** (the occupancy grid
  skips them), not inf. Otherwise free space would be carved through walls.
- **Safety:** `real_safety_config()` stops at a 0.40 m nose gap and slows from 0.90 m (from Test 2's lab
  runs, where 0.50 m from the centre touched things).
- **Map size:** real maps are 30 × 30 m around the start (`--map-size`). The default 14 × 10 m was
  smaller than the lab.
- **Package clash:** Test 1's `perception` package clashes with Phase 2's `perception`, so the session
  decodes PointCloud2 with Test 1's `lidar.pointcloud2_to_xyz` and never imports Test 1's `perception`.
- **Stopping:** the Teleop keyboard (pynput) may not see keys on Wayland, so `TerminalKeys` (any key in
  the terminal) is the stop path.

**Stain detection and analysis**
- **Detector resolution:** at 320×240, stains are mostly sub-pixel. The detector is trained and run on
  1280×720 (the Go2 stream size, 100° field of view).
- **Mechanism from close-ups:** Phase 2's original classifier called S1's impact "projected" because of
  coverage bias (the close-ups miss small edge stains). A "degraded synthetic" retrain was worse. What
  worked was training on measurements from **simulated close-ups**
  (`training/train_mechanism_simclose.py`, about 1 h, cached in `outputs/mechanism_simclose/groups.pkl`).
- **Grouping:** group all stains, usable or not, as training does. `frac_round` and `frac_usable` are
  features.
- **`analysis/bpa/ellipse.py`:** masks seen almost edge-on project to near infinity, and the slice loop
  ran for hours. Extents over 3 m or 20,000 slices are rejected; real smears (up to ~1 m) still pass.
- **Model files:** Phase 2's `.gitignore` ignores `*.pkl`, with an exception for `perception/models/*.pkl`
  (the stain detectors and the close-up classifier, committed). `analysis/classifier/models/mechanism_gb.pkl`
  is gitignored and retrains in about 100 s.
- **S4:** the bed-top cast-off is above the camera (0.41 m) and can't be seen. A 1 m mast
  (`--camera-height 1.0`) only raises cluster recall from 1/19 to 4/19.

**Running long jobs in the cloud container**
- **Timings:**
  - a full mission (`investigate.py`) takes about 10–15 min;
  - training the detector takes about 25 min;
  - the close-up classifier takes about 1 h;
  - a 720p render takes about 0.3 s, a 1080p close-up about 0.6 s.
- **Restarts:** the container sometimes restarts and kills background jobs. Commit work in progress
  often, and rerun killed jobs.
- **Sim loops:** foreground `sleep` is blocked, so wait for a background job with an `until` loop in a
  background command.
- **The user's stop hook** asks to commit and push uncommitted changes before a turn ends.

## 5. Re-running the evidence
```bash
python scripts/map_zones.py --random 1854 --video                       # Stage R sim (writes outputs/zone_map/r1854)
python scripts/investigate.py --scenario N [--no-keepout] [--camera-height 1.0] [--video]
python scripts/reanalyse.py outputs/investigate/* --model ideal|simclose --table <json>
python training/train_stain_detector.py [--reuse]
python training/train_mechanism_simclose.py [--reuse]
bash report/build_pdf.sh
python -m pytest -q                                                     # 99 tests, about 2 min
```
Result files quoted in the README and report are copied into `docs/media/stage_r/` and
`docs/media/stage_c/`, because `outputs/` is gitignored.
