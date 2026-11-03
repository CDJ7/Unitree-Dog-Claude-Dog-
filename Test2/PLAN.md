# ENG 6010 Test 2 plan: Go2 crime-scene search dog

Status: **decisions recorded 3 Nov 2026 (section 6); waiting for your OK on the zone boundaries before Q2 code starts.**

## 1. What already exists (Step 0 findings)

### Where the crime scene lives
- `main` has only Stage A step **A1** of `Phase_2_Investigator/`: the simulator session, teleop and sensors.
- The full crime-scene simulation is on branch **`phase2-csi`** (7 commits ahead of `main`, "Stage A complete", 77 tests):
  - scenarios with stains, mannequins, obstacles and distractors;
  - navigation (exploration, A*, safety layer);
  - event playback;
  - run recorder and COCO export.
- My working branch (`claude/eng6010-test2-crime-scene-0rst3r`) was cut from `main`. **Decision:** `origin/phase2-csi` is merged into it unchanged, so Test 2 builds on the complete crime scene.

### Simulator
- **Simulator:** `syswonder/robot-unitree-go2-mujoco`, a pinned git submodule at `Phase_2_Investigator/third_party/` (519 MB).
- **Wrapper:** `Phase_2_Investigator/sim_env/runtime.py`, class `SimSession`. Used in-process, with no ROS or Docker.
- **Go2 model:** the official `unitree_mujoco` `go2.xml`, unmodified (BSD-3), with 12 torque-driven joints and real contact physics.

### Crime scenes
Five scenarios are defined in `sim_env/scenarios/configs/`, plus `--random SEED` for more.

| # | Environment | Rooms (floor extent) | Event |
|---|---|---|---|
| S1 | `scenesmith_house_185` | living room 6.0 x 4.5 m, x 0-6, y 0-4.5; bathroom 2.2 x 2.4 m around (7.1, 2.25) | pipe: impact + cast-off |
| S2 | house 185 | same | knife: projected + drip trail |
| S3 | `scenesmith_house_186` | bedroom 5.2 x 4.2 m, x 0-5.2, y 0-4.2; bathroom 2.4 x 2.6 m around (6.4, 2.1) | fall: low impact, drip, smear |
| S4 | house 186 | same | knife: stab + 2 cast-off swings |
| S5 | `csi_simple_room` (our package) | one room, 6 x 5 m | bat: impact + drag smear + drip |

Objects and their naming in the XML:
- Each house comes with about 100 textured furniture meshes (sofa, bed, tables, lamps, etc.):
  - visual geoms are group 1;
  - colliders are group 3 boxes named `<room>_<object>_static_collision`;
  - room shells are `room_geometry_<room>_..._floor/wall_*`.
- `scenario_scene.py` adds these bodies:
  - `mq_*`: mannequins (victim, attacker);
  - `prop_*`: obstacles (storage box, laundry basket);
  - `distractor_*`: red cushion, red book and red mug (not blood);
  - `stain_*`: one flat ellipsoid per bloodstain (group 1, no contacts).
- **Evidence:** the ground-truth JSON lists **stain clusters** (S1: 7 clusters, `c0..c6`). Each cluster has:
  - a centre and normal;
  - a surface kind (`wall` / `floor` / `furniture`);
  - a radius and a bbox.

  Floor clusters (S1: c1, c2, c5) are the ones a dog could step on.

### How the Go2 is controlled
- `SimSession.cmd_vel(vx, vy, wz)` sets the commanded velocity, using `Go2Controller` from the simulator. The command expires after 0.5 s, like a dead-man.
- The controller has two modes. I tested both here (simple room, 3 s per command):

| Mode | Command | Measured |
|---|---|---|
| scripted physical trot (default) | vx 0.4 | 0.31 m/s, upright 0.999 |
| **learned "MoE Rough" Go2 RL policy (ONNX, joint-position targets at 50 Hz)**, loaded with `controller.command({"type":"policy","action":"load"})` | vx 0.4 / vy 0.3 / wz 0.6 | 0.43 m/s / 0.16 m/s / 0.52 rad/s, upright 1.000 |

- Physics runs about 7x faster than real time on this CPU.
- The `phase2-csi` navigation stack (A*, local safety layer, frontier explorer) is built on `cmd_vel`.

### Sensors that already exist
All of these are in `SimSession`:
- front RGB-D camera (320x240, 60 deg vertical FOV);
- MID-360-style 3D LiDAR plus a 720-ray 2D scan;
- IMU (gyro, accelerometer, quaternion);
- base pose and velocity;
- contacts (with the floor filtered out, which gives collision detection).

The `phase2-csi` recorder (`sim_env/recorder.py`, documented in `docs/recording_format.md`) already writes RGB, depth, `poses.jsonl` and LiDAR `.npz`.

Missing for Test 2:
- joint position, velocity and torque logging;
- foot-contact flags;
- zone labels;
- lighting and camera-noise randomisation;
- CSV/Parquet state tables.

### Reference repos checked
- **`unitree_rl_gym`** ships pretrained deploy policies only for **G1, H1 and H1_2**. There is **no Go2 `motion.pt`**: `deploy/pre_train/` has no go2 folder, and `deploy_mujoco/configs/` has only g1/h1/h1_2. The preferred option in the brief therefore cannot be used as written. Training a Go2 policy with it needs Isaac Gym and an NVIDIA GPU.

  **Proposal:** use the learned Go2 RL policy (MoE Rough, ONNX) that already ships in the simulator, behind `set_velocity_command()`. This is a pretrained RL locomotion policy turning `(vx, vy, yaw_rate)` into joint targets, which is what the brief asked for. The scripted physical trot stays as the fallback, and the README explains why.
- **`unitree_go2_create_dataset`** is a ROS 2 C++ recorder: `record_data.cpp`, `control_speed.cpp` and a `RobotState.msg` with IMU, motor q/dq/tau, foot force and velocity command. I will mirror its fields in `states.csv`.

### Rendering in this container
- CPU only; EGL is not available, so OSMesa is used.
- An RGB frame at 224x224 takes **0.18 s**. 5,000 frames is about 15-20 min of rendering, so this is feasible here.
- Opening a scenario takes 8 s (simple room) to 36 s (house 185).

### Reuse from earlier work
**Directly (imported, not copied):**
- `SimSession`, `open_scenario`, scenario configs and ground truth;
- `navigation/planner.py` (A* on the occupancy grid) and `navigation/local.py` (safety layer), used for collision-free waypoint following;
- `navigation/occupancy.py` and `plots.py` for the top-down maps.

**Adapted:**
- the recorder format, extended for Test 2;
- the Phase 1 top-down plotting style.

**Linked from the README:**
- the Test 1 / Phase 1 results (`docs/phase1_summary.md`);
- the Phase 2 Stage A checklist.

## 2. Design

### Q2: data and simulation (built first)
- `go2sim/go2_env.py`, class `CrimeSceneGo2`: wraps `open_scenario(...)` and loads the learned policy. Its interface:
  - `set_velocity_command(vx, vy, yaw_rate)` (clipped to 0.4 m/s, 0.15 m/s strafe, 0.5 rad/s);
  - `step(dt)`;
  - `sensors()`, returning RGB, IMU, joints (q, dq, tau from `qfrc_actuator`), foot contacts, base pose and velocity, and the command.
- **Front camera:** I keep the simulator's Go2 head camera, check its mount against the Go2 spec (front of the head, about 0.28 m forward and 0.06 m up of the base), and document it.
- `go2sim/waypoint_follower.py` follows waypoints in this order:
  1. A* path on the inflated occupancy grid (reused);
  2. pure pursuit to get `cmd_vel`;
  3. the safety layer.

  Routes are defined in `config/routes.yaml`:
  - 4-6 hand-made routes per scenario, with different starts, orders and speeds (0.2-0.4 m/s);
  - N random routes, each a random sequence of free points sampled per zone, to keep the classes balanced.
- `config/zones.yaml` holds the zone polygons for house 185 (section 6). The label comes from the base (x, y) with a point-in-polygon test.
- **Scenes used for data:** S1 and S2, plus seeded random scenarios placed in house 185. Stains, props and lighting differ between them, so the classifier must learn the room layout and not one fixed set of props.
- **Recorder.** Output goes to `data/run_XXX/`:
  - `images/000123.jpg` at 224x224 and 5 Hz;
  - `states.csv` at 50 Hz, with a `frame` column that links each image to its state row;
  - `metadata.json`: scenario, route, seed, lighting, noise and camera intrinsics.
- **Randomisation per run:**
  - light intensity, colour and direction (by editing `model.light_*` before each run);
  - headlight ambient;
  - camera noise (Gaussian, brightness/contrast, slight blur, JPEG quality).
- **Analysis script** (`q2_data_simulation/analyze.py`) produces:
  - per-zone counts;
  - trajectory plots over the top-down render;
  - sample frames per zone;
  - IMU and joint plots;
  - a GIF (chase camera and front camera side by side).
- **Target size:** about 6,000+ frames over about 40 runs. The full set is gitignored. A small sample (1 run, about 100 frames) plus `collect_dataset.py --all` is committed to regenerate the rest.

### Q1: zone classifier
- Split **by run**: about 70/15/15. Test runs are whole routes the model never saw, ideally including held-out random-route seeds and lighting.
- **Model:** ResNet18, ImageNet weights, with a new fully connected layer. Augmentation: ColorJitter, RandomResizedCrop(0.8-1), rotation ±5°. AdamW, about 15 epochs on CPU.
- **Baselines:**
  1. RGB colour histogram + logistic regression;
  2. HSV colour histogram + logistic regression;
  3. a small CNN trained from scratch.
- **Evaluation:**
  - accuracy;
  - per-class precision, recall and F1;
  - confusion matrix;
  - loss and accuracy curves;
  - Grad-CAM for 2-3 test frames per zone;
  - predicted vs true zone over time for a held-out run, as a plot and a GIF.
- **Files:** `train.py`, `evaluate.py`, `gradcam.py`, `demo.py`, and `Q1_colab.ipynb`.

### Q3: RL evidence search
- **`CrimeSceneSearchEnv` (Gymnasium).** Its 2D map is exported from the compiled MuJoCo scene by `export_layout.py`, which writes `config/layout_<scenario>.json` with:
  - collider boxes (walls, furniture, props, mannequins) as oriented rectangles;
  - evidence = the stain-cluster centres from the ground truth;
  - "contamination" discs for **floor** clusters (radius = cluster radius + 0.1 m).

  Wall and furniture evidence is "found" when the dog is within the detection radius (about 0.8 m) with line of sight inside the camera FOV.
- **Dynamics:** a unicycle/holonomic base with first-order velocity lag fitted to the Q2 logs (time constant measured from the real policy's step response), at 5 Hz control.
- **Observation:**
  - pose (x, y, sin/cos of yaw);
  - body velocity;
  - 24 LiDAR rays;
  - distance and bearing to the nearest unfound evidence;
  - a found flag per evidence item (padded to K max).
- **Action:** `(vx, vy, wz)` in [-1, 1], scaled to the Go2 limits.
- **Rewards:** one function each in `rewards.py`, weights in `config/rl_config.yaml`:
  - find evidence: +;
  - progress: shaping toward the nearest unfound item;
  - contamination: -;
  - collision: -;
  - time: - per step;
  - all found: + bonus.
- **Episode end:**
  - terminate when all evidence is found, or on a severe collision (penetration or more than N collisions);
  - truncate at max steps.
- **Randomisation:** start pose, and optionally evidence jitter within free floor or wall space and a random scenario layout.
- **Training:**
  - SB3 PPO, 8 `SubprocVecEnv` workers;
  - TensorBoard logging, `CheckpointCallback` and `EvalCallback`;
  - YAML hyperparameters;
  - `Q3_colab.ipynb`, which saves to Google Drive and can resume from the last checkpoint.
- **Evaluation:** 50+ episodes each for the trained policy, a random policy and greedy nearest-evidence A*. Metrics:
  - success rate;
  - time to find all evidence;
  - contamination count;
  - collision count.
- **Deployment:** `deploy_mujoco.py` runs the policy in the full MuJoCo scene. It builds the observation from MuJoCo state plus ray casts, sends the policy's commands through `set_velocity_command`, and the learned Go2 policy walks. It produces trajectory plots and a GIF.
- I can train here on CPU (a 2D environment runs at thousands of steps/s, so 1-3 M steps takes about 1-2 h). The Colab notebook is for longer runs.

## 3. Folder layout

```
Test2/
  README.md  PLAN.md  requirements.txt  .gitignore
  config/      zones.yaml, routes.yaml, collect.yaml, q1_train.yaml, rl_config.yaml, layout_*.json
  go2sim/      paths.py (finds Phase_2_Investigator + submodule), go2_env.py, waypoint_follower.py,
               randomization.py, render.py
  q2_data_simulation/  collect_dataset.py, analyze.py, make_gif.py
  q1_supervised/       dataset.py, models.py, train.py, baselines.py, evaluate.py, gradcam.py, demo.py, Q1_colab.ipynb
  q3_rl/               export_layout.py, env.py, rewards.py, baselines.py, train.py, evaluate.py,
                       deploy_mujoco.py, Q3_colab.ipynb
  tests/               env_checker, zone labelling, reward terms, dataset split (no leakage)
  data/                sample run + README (how to regenerate)
  results/             q1/, q2/, q3/ figures, metrics JSON, GIFs, small checkpoints (<25 MB)
```

- Scripts run from `Test2/`.
- `go2sim/paths.py` adds `../Phase_2_Investigator` to `sys.path`. Nothing outside `Test2/` is edited, except the one "Test 2" section you asked for in the root `README.md`.

## 4. Milestones (I commit at each, and show you results before the next)
1. Plan approved; branch set up (Q-A).
2. **Q2:** environment, controller and follower; one run checked visually; then the full dataset, statistics and figures.
3. **Q1:** training, baselines, evaluation, Grad-CAM and demo.
4. **Q3:** environment (passes `env_checker`), training, evaluation against baselines, MuJoCo deployment and GIF.
5. README write-up (every number read from `results/*.json`), and the root README link.

## 5. Reproducibility
- Seeds are fixed everywhere (numpy, torch, SB3, and route/lighting randomisation seeded per run).
- Every script writes its config and hyperparameters next to its outputs.

## 6. Decisions (answered 3 Nov 2026)

1. **Branch:** `origin/phase2-csi` merged into the Test 2 branch, with no edits to its files.
2. **Zones:** one crime scene, house 185, with **5 zones**. The bathroom doorway at x = 6.0, y ≈ 1.6-2.5 is about 0.9 m wide, so the dog fits through.

   Proposed boundaries, in world metres; the map is in `results/q2/step0_house185_colliders.png`:

   | id | zone | polygon (axis-aligned box) | landmarks | S1 evidence |
   |---|---|---|---|---|
   | 0 | `victim_area` | x 0.0-1.8, y 0.0-4.5 | west wall, mirror, mannequin | c0 wall, c3 furniture |
   | 1 | `bed_area` | x 1.8-4.4, y 2.4-4.5 | bed, nightstand | c6, c1 (floor) |
   | 2 | `lounge_area` | x 1.8-4.4, y 0.0-2.4 | sofa, rug, coffee table | none |
   | 3 | `desk_entry_area` | x 4.4-6.0, y 0.0-4.5 | writing desk, chair, laundry basket | c2 (floor), c4 |
   | 4 | `bathroom` | x 6.0-8.2, y 1.05-3.45 | toilet, bathtub, vanity | c5 (floor) |

   The bathroom is the smallest zone. Routes spend dwell time there, and per-zone sampling keeps the classes balanced.
3. **Locomotion:** the simulator's learned Go2 RL policy (MoE Rough, ONNX) behind `set_velocity_command()`. The scripted physical trot is the documented fallback. The README explains why there is no `unitree_rl_gym` Go2 policy to use.
4. **Pushing:** push to `claude/eng6010-test2-crime-scene-0rst3r` after each milestone. No PR or merge into `main` without asking.
