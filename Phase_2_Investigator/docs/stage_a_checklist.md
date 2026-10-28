# Stage A checklist (simulation): evidence

Status 29 Oct 2026: all items built and run in the cloud container (CPU only, headless). The
windowed modes (MuJoCo viewer, sensor viewer) were also tested on your WSL machine for A1. The
newer windowed options (`--viewer` for explore/playback) still need a check on your machine.
**Stage B starts only after you confirm this list.**

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | One command launches any scenario | done | `./scripts/run_sim.sh --scenario N [--mode ...]` (modes below) |
| 2 | Go2 walks; teleop works; all sensors verified | done | `--mode check`: 15/15 checks in both houses (`media/a1_check.json`); teleop and sensor viewer confirmed by you on WSL; `media/a1_*` |
| 3 | All 5 spatter mechanisms generate and render, unit tests pass | done | `tests/test_spatter_in_sim.py`, `tests/test_spatter_physics.py`; `media/a3_*.jpg` |
| 4 | 3-5 scenarios with stains, mannequins, obstacles, distractors | done (5) | `sim_env/scenarios/configs/`, `media/a3_*.jpg`, `--random SEED` |
| 5 | Robot explores each scenario autonomously, no collisions, inspects stain locations | done, with limits (below) | `media/a4_*_map.jpg`, `media/a4_*_closeup_*.jpg`, `media/a4_*_explore.mp4`; table below |
| 6 | Event playback works for every scenario | done | `media/a5_*` (videos and contact sheets), `--mode playback` |
| 7 | Run recording and COCO dataset export work | done | `docs/recording_format.md`, `tests/test_playback_and_data.py`, `media/a6_coco_samples.jpg`; numbers below |
| 8 | Demo video/screenshots in docs/media | done | `docs/media/` |

## Commands

```bash
./scripts/run_sim.sh --scenario 1                            # teleop (MuJoCo viewer)
./scripts/run_sim.sh --scenario 1 --mode sensors             # sensor viewer
./scripts/run_sim.sh --scenario 1 --mode explore --viewer    # watch the robot explore
./scripts/run_sim.sh --scenario 1 --mode inspect             # explore + patrol + inspect (report, map, close-ups)
./scripts/run_sim.sh --scenario 1 --mode playback --viewer   # reenactment (MP4 without --viewer)
./scripts/run_sim.sh --scenario 2 --mode record              # recorded run in runs/
./scripts/run_sim.sh --mode dataset                          # COCO dataset in datasets/csi_coco/
./scripts/run_sim.sh --mode check                            # A1 sensor/walking check
```

## A2: spatter in the simulator

- Droplets fly with gravity + quadratic drag. Every flight step is ray-cast against the visible
  scene, so stains land on the real walls, floors and furniture meshes.
- Stains are drawn as flat ellipsoids with the ground-truth width, length (W/L = sin α) and
  direction, plus tails, satellites and streaking smears.
- Tests show:
  - landing within 1 mm and 0.5 deg of the analytic reference;
  - rendered stain shapes match their angle (W/L = sin α within 0.03 at 20-70 deg; orientation within 4 deg);
  - all five mechanisms generate and render.

## A3: scenarios

| # | Scenario | Events | Weapon | Stains | Clusters |
|---|---|---|---|---|---|
| 1 | living room (house 185) | impact + cast-off | blunt (pipe) | 504 | 7 |
| 2 | desk corner + bathroom (185) | projected + drip trail | edged (knife) | 155 | 4 |
| 3 | bedroom NE corner (186) | low impact + drip + hand smear | none (fall) | 158 | 6 |
| 4 | bedroom desk (186) | stab impact + two cast-off swings | edged (knife) | 313 | 19 |
| 5 | simple room (our package) | impact + drag smear + drip | blunt (bat) | 470 | 7 |

## A4: autonomous runs (`--mode inspect`, latest run per scenario)

| Scenario | Path | Camera coverage (surfaces / floor) | Collisions | Inspection: views reached, share of each target's stains seen |
|---|---|---|---|---|
| S1 | 53 m | 73% / 81% | 0 | 5/5 targets; 98%, 100%, 77%, 100%, 100% |
| S2 | 14 m | 60% / 78% | 0 | 2/2 targets; 94%, 100% |
| S3 | 9 m | 64% / 79% | 0 | 2/2 targets; 29%, 38% (stains under/behind the sitting victim) |
| S4 | 13 m | 54% / 70% | 0 | 2/4 targets; south-wall impact: no reachable view; bed cast-off 13% |
| S5 | 45 m | 59% / 54% | 0 | 6/6 targets; 99%, 100%, 100%, 100%, 100%, and the floor smear (3 views) |

What happens in a run:
- **Explore:** frontier exploration with the LiDAR.
- **Patrol:** viewpoints chosen so the camera sees as much surface as possible.
- **Inspect:** each stain region is viewed from up to 3 angles with line of sight. Each view saves a 320x240 sensor frame and a 1920x1080 close-up with depth, K and pose.
- **Safety:** every command goes through cmd_vel and the safety layer. The run fails if any collision is logged.

Known limits:
- **House 186:** the colliders are conservative boxes, so the robot has to pass 0.5 m gaps. S4's south-wall pattern sits behind such a gap, with no reachable standoff in line of sight.
- **Low obstacles:** the MID-360's lowest ray (-7 deg) misses low things up close. The front depth camera now covers this; it was the cause of the S3 collisions before the fix.

## A5: event playback

Contact sheets and videos in `media/a5_*`:
- The attacker's arm and weapon follow the strike / cast-off arc.
- Droplets fly their exact trajectories; each stain appears when its droplet lands.
- Smears grow behind a hand marker.
- The victim walks or collapses; the attacker walks away.
- `--from-descriptions` rebuilds the event from mechanism/origin/parameters only (the Stage B path; `media/a5_s5_simple_room_bat_descriptions.mp4`).

## A6: data export

- **Recorded run** (S2, 33 s of exploration): 162 camera frames, 324 LiDAR scans, 324 robot states and the ground truth, 19 MB. It reads back with `RunReader` (round-trip test in `tests/test_playback_and_data.py`).
- **COCO dataset:** see `datasets/csi_coco/stats.json` after `--mode dataset`. Sample labels are in `media/a6_coco_samples.jpg`. Quick build (`--random 10 --per-scenario 30`, 17 MB, ~25 min on CPU):

| Split | Images | Regions | Frames without stains | impact | castoff | projected | drip | transfer |
|---|---|---|---|---|---|---|---|---|
| train (8 random) | 240 | 545 | 59 | 423 | 52 | 48 | 22 | 0 |
| valid (2 random) | 60 | 139 | 19 | 122 | 11 | 0 | 6 | 0 |
| test (5 hand-made) | 150 | 215 | 51 | 106 | 51 | 17 | 39 | 2 |

Class imbalance to fix in Stage B before training:
- impact dominates;
- the random generator makes no transfer smears.

Plan: a larger build (`--random 40 --per-scenario 80`, fast on your GPU), with the random generator balancing mechanisms and adding smears.

## Tests

`python -m pytest`: 77 passed (2 min on CPU; simulator tests skip if the submodule is missing).
