# Recorded runs and the synthetic dataset (A6)

Stage B reads these two formats. Both are produced in the simulator and are gitignored
(`runs/`, `datasets/`).

## 1. Recorded run

Made by `./scripts/run_sim.sh --scenario N --mode record` (exploration + camera patrol while
recording) or by adding `sim_env.recorder.RunRecorder` as a `Navigator` hook.

```
runs/<scenario>_<YYYYmmdd_HHMMSS>/
  meta.json            format version, git revision, scenario, environment, rates, camera intrinsics
  frames.jsonl         one line per camera frame (5 Hz)
  frames/NNNNNN_rgb.png      320x240 RGB (the Go2's front camera)
  frames/NNNNNN_depth.png    320x240 uint16, millimetres along the optical axis, 0 = no return
  scans.jsonl          one line per LiDAR scan (10 Hz)
  lidar/NNNNNN.npz     ranges, angle_min, angle_increment, range_min, range_max,
                       points_lidar (N,3), T_world_lidar (4,4)
  states.jsonl         robot state at 10 Hz
  ground_truth.json    the scenario's full ground truth (evaluation only)
  nav.json             navigation report: path, goals, collisions, camera coverage
```

`frames.jsonl` line:
```json
{"index": 12, "t": 4.6, "rgb": "frames/000012_rgb.png", "depth": "frames/000012_depth.png",
 "T_world_cam": [[...4x4...]], "robot_position": [x, y, z], "robot_yaw": 1.57}
```

`states.jsonl` line:
```json
{"t": 4.6, "position": [x, y, z], "quat_wxyz": [...], "yaw": 1.57, "lin_vel_world": [...],
 "ang_vel_body": [...], "cmd_vel": [vx, vy, wz],
 "imu": {"gyro": [...], "accel": [...], "quat_wxyz": [...]}, "upright": 0.999}
```

Conventions (same as `sim_env/interfaces.py`):
- World frame: MuJoCo world, z up, metres.
- Camera: OpenCV frame (x right, y down, z forward); `T_world_cam` maps camera coordinates to
  world coordinates; `K` in `meta.json` (`fx = fy = 207.8`, `cx = 159.5`, `cy = 119.5` for 320x240,
  60 deg vertical FOV); no distortion.
- A pixel (u, v) with depth z maps to `T_world_cam @ [(u-cx)/fx*z, (v-cy)/fy*z, z, 1]`.
- LiDAR scan: ROS LaserScan convention in the LiDAR frame; `inf` = no return.

Reading it back:
```python
from sim_env.recorder import RunReader
run = RunReader("runs/s2_living_room_projected_20261029_101500")
for frame in run.frames():        # interfaces.CameraFrame: rgb, depth_m, K, T_world_cam
    ...
scan = run.scan(0)                # interfaces.LidarScan
run.states[0], run.ground_truth
```

Sim vs real: the simulated LiDAR is a MID-360 model (vertical -7 to 52 deg); the real Go2 has an
L1. Poses are simulator ground truth (no odometry drift). The real Go2's front camera streams
1920x1080; the simulator's is 320x240, so the analysis also uses evidence close-ups (below).

## 2. Evidence close-ups (inspection)

`--mode inspect` saves, for each inspected stain region and view, in
`outputs/explore/<scenario>/evidence/`:
- `tK_vJ_sensor.png` (+ `_depth.png`, `_meta.json`): the 320x240 sensor frame;
- `tK_vJ_closeup.jpg` (+ `_depth.png`, `_meta.json`): 1920x1080, 40 deg FOV, from the robot's
  camera position, pitched toward the target (emulating the Go2's body pitch; the sim gait cannot
  pitch). `_meta.json` holds `K` and `T_world_cam`; depth is uint16 millimetres.

## 3. COCO dataset of stain regions

Made by `./scripts/run_sim.sh --mode dataset` (`training/build_coco_dataset.py`).

```
datasets/csi_coco/
  train/ valid/ test/       *.jpg (640x480) + _annotations.coco.json   (RF-DETR / Roboflow layout)
  stats.json                images, annotations, negatives and per-class counts per split
```

- Categories: `0 bloodstain` (supercategory, as in Roboflow exports), `1 impact`, `2 castoff`,
  `3 projected`, `4 drip`, `5 transfer`.
- One annotation per stain *region*: the visible stains of one ground-truth cluster, grouped when
  closer than ~5% of the image diagonal; `segmentation` is the polygon of the group's convex hull
  (padded 2 px), `bbox` and `area` follow. Extra fields: `cluster` (ground-truth cluster id),
  `stain_pixels` (how many pixels are actually stain).
- Labels come from MuJoCo's segmentation render, so they are exact; stains smaller than 6 pixels
  are not labelled.
- Images: camera at 0.30-0.55 m height (robot camera range), 0.4-3.0 m from a cluster (65%), at a
  red distractor (10%, hard negative) or anywhere (25%); fovy 45/52/60 deg; randomised headlight,
  room lights and wall/floor tints. Each image entry also stores `K`, `T_world_cam`, `scenario`,
  `kind`.
- Split by scenario: the five hand-made scenarios are `test`; of the random scenarios every fifth
  goes to `valid` and the rest to `train`. No scenario appears in two splits.
