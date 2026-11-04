# Q2 dataset

- `data/sample/` is committed. It holds 3 complete runs (24 MB) in the same format as the full set; they are
  the data of the Colab quick check (`Q1_colab.ipynb` option C, `config/q1_sample.yaml`).
- `data/full/` holds the full dataset. It is about 36 runs and too large for GitHub, so it is gitignored. Regenerate it from `Test2/`:

```bash
git submodule update --init --depth 1 ../Phase_2_Investigator/third_party/robot-unitree-go2-mujoco
pip install -r requirements.txt
python q2_data_simulation/collect_dataset.py --workers 4      # data/full/, all runs in config/collect.yaml
python q2_data_simulation/analyze.py                          # statistics and figures -> results/q2/
```

Notes on regeneration:
- Everything is seeded: `config/collect.yaml` `seed`, plus one seed per run.
- Re-running gives the same routes, start poses, lighting and camera noise.
- Physics and rendering can differ in the last digits between machines and MuJoCo builds.
- Runs can be collected in parts, e.g. `--scenes s1 s2`. `runs.csv` is rebuilt from every run folder present.

## Layout

```
data/full/
  collect_config.yaml        the config used
  runs.csv                   one row per run: scene, route, speed, frames, duration, distance, goals, frames per zone
  run_XXX/
    images/000000.jpg ...    Go2 head camera, 320x240 RGB, 5 Hz sim time, per-run camera noise applied
    states.csv               50 Hz rows (see below)
    metadata.json            scene, route, seed, start pose, goals, lighting, camera noise, camera mount/intrinsics,
                             controller, per-zone frame counts, follower events
```

## `states.csv` columns

| Group | Columns |
|---|---|
| time | `t` (s since the run started, sim time) |
| base pose | `base_x, base_y, base_z` (m, world); `base_qw..qz` (orientation); `base_yaw` |
| base velocity | `base_v{x,y,z}_world`, `base_v{x,y,z}_body` (m/s); `base_w{x,y,z}_body` (rad/s) |
| IMU | `imu_qw..qz` (framequat), `imu_gyro_{x,y,z}` (rad/s), `imu_acc_{x,y,z}` (m/s², includes gravity); `upright` (cos of body tilt) |
| command | `cmd_vx, cmd_vy, cmd_yaw_rate` (the `set_velocity_command` input) |
| joints | for the 12 joints `{FL,FR,RL,RR}_{hip,thigh,calf}`: `q_*` (rad), `dq_*` (rad/s), `tau_*` (N·m, motor torque) |
| feet | `foot_force_{FL,FR,RL,RR}` (N, normal contact force); `foot_contact_*` (1 if more than 5 N) |
| labels | `zone_id`, `zone` (from the base x, y and `config/zones.yaml`; -1 / `outside` in doorways off the map) |
| bookkeeping | `collisions` (robot contacts other than feet on the floor), `goal_index`, `frame` (image index, or -1 for state-only rows) |

To get the image/label pairs, select the rows with `frame >= 0`. The image is `images/{frame:06d}.jpg` and its label is `zone_id`.

## Collection log (3 Nov 2026)

- 36 runs and 17,123 labelled frames. `results/q2/dataset_stats.json` has every count.
- The first collection hung in one scene (r1853) because of a bug in the follower: after a replan it could loop without stepping the simulation. The bug was fixed in `go2sim/waypoint_follower.py` (replan at most once per second, always step), and the 6 r1853 runs (run_024 to run_029) were collected again with the fixed code.
- The other 30 runs were collected before the fix. The fix only changes behaviour in the situation that used to hang, and none of those 30 runs hung.
- `data/sample/run_024`, `run_025` and `run_027` are copies of those full-set runs, all from scene r1853:
  `clockwise_tour` (578 frames), `counter_tour` (447) and `fast_sweep` (241), all 5 zones each.
  `config/q1_sample.yaml` uses them as test, validation and training run respectively.
