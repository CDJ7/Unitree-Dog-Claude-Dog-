# Go2 camera + LiDAR + keyboard teleop

Streams the Go2 front camera and LiDAR point cloud to your laptop and lets you drive the dog with the keyboard.
**Requires a Go2 EDU** (full SDK/DDS access over Ethernet). On AIR/PRO use the WebRTC route instead
(`go2_webrtc_connect`); this project will not work.

> Status: camera streaming and keyboard teleop were run on a real Go2 EDU, and LiDAR streaming was reported working.
> YOLO detection has been tested on still images only (a live run with the robot was not completed). The MuJoCo
> simulation, object mapping, `preflight.py` and `--map` were added later and are described in sections 9-12. Anything
> marked "untested" has not been run on the real robot. Unit tests: `python -m pytest -q tests`.
> The architecture and results write-up for the class project is in `../docs/phase1_summary.md`.

```
main.py      UI loop (main thread): camera window, 3D LiDAR, overlay, shutdown handling
camera.py    CameraThread: VideoClient (JPEG) or multicast H.264 via GStreamer -> latest-frame buffer
lidar.py     DDS subscribers (rt/utlidar/cloud, rt/sportmodestate, rt/lowstate), PointCloud2 parser, viewers
teleop.py    pynput key state + 20 Hz control thread + watchdog thread (SportClient)
config.yaml  all tunables and safety limits
setup_network.sh  detect interface / set static IP / ping
```

## 1. Network (your interface: `enp0s31f6`)

```bash
./setup_network.sh detect                      # = ip -br link ; ip -br addr
# Option A - temporary (lost on reboot / link reset), changes nothing permanent:
./setup_network.sh temp enp0s31f6              # sudo ip addr add 192.168.123.222/24 dev enp0s31f6
# Option B - persistent NetworkManager profile (asks for confirmation, no gateway, autoconnect off):
./setup_network.sh nmcli enp0s31f6             # undo with: ./setup_network.sh remove enp0s31f6
ping -c 4 192.168.123.161                      # or ./setup_network.sh ping
```
Do not continue until the ping works.

## 2. Install CycloneDDS + SDK (versions from the SDK README)

The SDK pins `cyclonedds==0.10.2` and builds against the `releases/0.10.x` branch. Use Python 3.8-3.12
(Open3D / cyclonedds wheels lag newest Python) in a venv.

```bash
sudo apt install -y python3-venv python3-pip cmake build-essential git
cd ~ && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
export CYCLONEDDS_HOME=~/cyclonedds/install            # add to ~/.bashrc

python3 -m venv ~/go2venv && source ~/go2venv/bin/activate
cd ~ && git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
pip install -r /path/to/go2_stream/requirements.txt
```

Confirm DDS works with the SDK's own examples **before** using this project:
```bash
cd ~/unitree_sdk2_python
python3 example/go2/front_camera/camera_opencv.py enp0s31f6     # camera window, ESC exits
python3 example/go2/high_level/go2_sport_client.py enp0s31f6    # DON'T run yet: it moves the dog
```

## 3. Staged bring-up (do these in order)

1. **Camera only:** `python main.py --iface enp0s31f6 --no-lidar --no-control`
2. **LiDAR only:** `python main.py --iface enp0s31f6 --no-camera --no-control`
   (if no points: the LiDAR may be off; the app publishes `"ON"` to `rt/utlidar/switch` at start - `lidar.switch_on_start`)
3. **Teleop logic with no robot:** `python teleop.py --dry-run` (prints the commands it would send).
4. **Teleop on the robot, dog lying down or lifted on a stand/held:** `python teleop.py --iface enp0s31f6 --max-speed 0.2`.
   Press and release W: you should see no motion while lying down (Move is rejected) and log lines only.
   Verify: release stops immediately; hold W then unplug/alt-tab -> stops within ~0.3-0.75 s; SPACE stops; ESC exits with StopMove.
5. **Full:** clear space around the dog, then `python main.py --iface enp0s31f6`, press **R** to stand, then drive.

## 4. Controls

| Key | Action |
|---|---|
| W / S | forward / backward (vx) |
| A / D | strafe left / right (vy) |
| Q / E | rotate left / right (vyaw) |
| Space | StopMove, ignore held keys until all are released |
| R | StopMove, StandUp, BalanceStand |
| F | StopMove, StandDown |
| + / - | speed scale up / down (0.2-1.0 of the limits) |
| X then Y (within 3 s) | **Damp()** - the dog goes limp and collapses. Never automatic |
| Esc | StopMove, then exit |

Overlay on the video: FPS, current command velocity, speed scale, status, battery %, position and gait/mode.

## 5. Flags

`--iface`, `--no-camera`, `--no-lidar`, `--no-control` (view-only: teleop never created), `--max-speed M_S`
(max vx; vy/vyaw scale proportionally; always clamped by `hard_cap_*` in config), plus `--camera-mode multicast`,
`--viewer topdown`, `--config`.

## 6. Safety design

- Defaults vx 0.4 m/s, vy 0.3 m/s, vyaw 0.8 rad/s; hard caps in `config.yaml` that no flag can exceed.
- Velocity is sent only while a key is held; release sends `StopMove()` once. Opposing keys cancel.
- **Deadman:** a held key expires `deadman_s` (0.3 s) after its last key event. Because X11 auto-repeat waits ~0.5 s
  before repeating, the *first* press gets `initial_repeat_grace_s` (0.75 s) so a held key doesn't stop at 0.3 s.
  Worst case for a lost key-release is therefore 0.75 s of motion. If auto-repeat is disabled on your system the dog
  stops 0.75 s after the press; check with `xset q` / `xset r on`, tune with `xset r rate 250 40`.
- A **watchdog thread** sends `StopMove()` if the control loop is >0.25 s late (not while a blocking stand/sit action runs).
- Any exception in the control loop, Ctrl+C, SIGTERM, window close, or Esc ends in `StopMove()`.
- Damp is never automatic.
- `Move` is a no-reply RPC, so a dropped packet is not detected by the client; the watchdog covers local stalls only.
  Keep the Unitree remote controller within reach as the real emergency stop.

## 7. Known limitations / things to check

- **Keyboard capture is global** (pynput hooks all key events, even when another window has focus). Needs an
  X11 session; under Wayland pynput does not see keys - log in with "GNOME on Xorg"/an X11 session.
- **Multicast camera mode** needs OpenCV built with GStreamer (the pip `opencv-python` wheel is not; use the distro
  `python3-opencv` + `gstreamer1.0-plugins-*`/`gstreamer1.0-libav`). The default `230.1.1.1:1720` RTP/H.264 pipeline in
  `config.yaml` is from memory and **I could not verify it**; check what your robot sends:
  `sudo tcpdump -i enp0s31f6 -n 'udp and dst net 224.0.0.0/4'`. `multicast-iface` must be the NIC on 192.168.123.x
  (`enp0s31f6`). Use it only if VideoClient's frame rate is too low (VideoClient is typically ~10-15 fps).
- The LiDAR cloud is expressed in the robot's odometry frame (`rt/utlidar/cloud`); a drifting origin is normal.
  `rt/utlidar/cloud_deskewed` may be a better topic - set `lidar.topic`.
- `rt/lowstate` arrives at ~500 Hz and is only used for battery SOC; if CPU is high set `state.low_topic` to a
  low-frequency variant if your firmware has one, or remove battery from the overlay.
- Camera, LiDAR and state callbacks run on DDS/camera threads and only store the latest value; both GUI windows
  (OpenCV + Open3D) are driven from the main thread, so a slow display can never block the control thread.
- If Open3D cannot open a window the app falls back to a top-down OpenCV view automatically.
- Topics that deliver nothing for 5 s print a warning naming the topic and what to check.

## 8. Object detection with distance (YOLO + LiDAR)

`run_detection.py` runs Ultralytics YOLO on the front camera and labels each object with its distance and 3D position.
**The Go2 front camera is RGB only (no depth channel)**, so distance comes from projecting the LiDAR point cloud into the
image and taking a robust median of the points that fall on each box. It is perception only: nothing moves the robot unless
you pass `--teleop`. `main.py` and `teleop.py` are unchanged.

```
perception/detector.py       YOLO wrapper (GPU if available, else CPU; class filter)
perception/depth_fusion.py   pinhole projection, frame transforms, robust per-box distance (pure numpy)
perception/pipeline.py       detection thread + fusion + CSV/JSONL logger
perception/visualization.py  boxes, labels, LiDAR overlay
run_detection.py             entry point
config/detection.yaml        model, thresholds, class filter, camera model, extrinsics
tests/test_depth_fusion.py   unit tests (synthetic arrays)
```

Install (inside the venv): `pip install ultralytics` (pulls in PyTorch; for an NVIDIA GPU install a CUDA build of torch first).
The first run downloads `yolov8n.pt`.

```bash
python -m pytest -q tests                                       # depth math tests, no robot needed
python run_detection.py --image some_photo.jpg                  # 1. detector alone (writes some_photo_det.jpg)
python run_detection.py --iface enp0s31f6 --no-lidar            # 2. live camera, distances n/a
python run_detection.py --iface enp0s31f6                       # 3. live camera + LiDAR distances
python run_detection.py --iface enp0s31f6 --classes person chair bottle --log det.csv
python run_detection.py --iface enp0s31f6 --teleop              # 4. optional: WASD alongside (global key capture)
```
Flags: `--model`, `--conf`, `--classes`, `--device`, `--no-display`, `--log FILE` (`.csv` or `.jsonl`), `--no-lidar`,
`--image`, `--teleop`. Key `L` toggles the LiDAR overlay; Esc quits (with `--teleop` it also stops the dog first).

Each detection (`perception.detector.Detection`) has `name, conf, bbox, distance_m, xyz_cam, xyz_body, n_depth_points`.
`distance_m` is the straight-line range; `xyz_body` is x forward / y left / z up (metres) from the robot body origin.
`distance_m is None` renders as "depth: n/a".

### Check that distances are accurate (do this before trusting them)
1. Start the app and press `L`. LiDAR dots (blue near, red far) should sit on the edges of real objects such as a wall, a
   door frame or a box. If they are shifted, tune `camera_model.hfov_deg` and `extrinsics` in `config/detection.yaml`.
   The defaults are **guesses**: the camera's field of view and its offset from the LiDAR were not measured.
2. Put a large flat object (a box or a person) at a **tape-measured 1.0 m, 2.0 m and 3.0 m** straight ahead of the dog's
   chest and compare `distance_m`. Expect a constant offset if the extrinsics are off, and a scale error if the field of
   view is wrong. Note that the distance is measured from the camera position set in `extrinsics`, not from the dog's nose.
3. Check that the sign of `xyz_body` is right: step the object to the dog's left and `y` should go positive.

### Limitations
- The LiDAR cloud is sparse: small or distant objects may get fewer than `fusion.min_points` points and show "depth: n/a".
- The camera frame and the cloud are matched by "latest of each", not by timestamp, so a moving object or a turning dog will
  have some error.
- `rt/utlidar/cloud` may be published in an odom (world) frame. With `fusion.lidar_frame: auto` the app checks the message's
  `header.frame_id` (logged once at start) and, if it contains "odom", transforms into the body frame using the pose from
  `rt/sportmodestate`. Tell me what the log line shows if distances look wrong.
- Not yet run on a real robot. Unit tests, the detector on a sample image, and the threaded pipeline with synthetic LiDAR
  data were run.


## 9. Simulation (no robot needed)

Everything in this section works without the dog. Use it for perception and mapping work.

**Why a custom sim mode?** Unitree's official MuJoCo simulator (`unitree_mujoco`, DDS domain 1 on the `lo` interface) only
accepts low-level joint commands on `rt/lowcmd`. It has no `SportClient` service (no `Move`/`StandUp`), no camera and no
LiDAR. So `sim/world.py` runs the same Go2 model in a **kinematic mode**: keyboard velocity moves the robot's base directly
and the legs blend between a lying and a standing pose. It is for perception tests, **not** for testing walking.
`SimSportClient` has the same methods as the real `SportClient`, so `teleop.py` is reused unchanged (same limits, deadman,
watchdog). The sim camera exposes the same `.buffer` / `.fps` as `camera.CameraThread`, so the detector can't tell them apart.

One-time setup (inside your venv):
```bash
./sim/setup_sim.sh                  # clones ~/unitree_mujoco (Go2 model) and installs mujoco + matplotlib
python run_sim.py                   # camera view + top-down map; R stands, WASD drives, Esc quits
python run_detection.py --sim       # live YOLO on the sim camera, with distances from the sim depth image
python run_detection.py --sim --gt  # GROUND TRUTH labels from the sim instead of YOLO (clearly labelled, magenta boxes)
python run_detection.py --sim --map # also build an object map; saved to results/ on exit
python run_mapping_eval.py --gt     # scripted survey of the room + accuracy report vs the true positions (about 90 s)
```
`--no-render` (any of the sim commands) uses a flat-colour ray-cast view instead of OpenGL. It needs no GPU/display
libraries but YOLO will not recognise it. Use it only for headless logic tests.

The room is `sim/scene_room.xml` (walls, a table, 3 chairs, a box, a backpack, 2 bottles, a cup). Edit it freely: any body
named `obj_<class>_<n>` (e.g. `obj_chair_4`) is a labelled object used for ground truth and the accuracy check.
Angles in the file are in radians.

**Making YOLO recognise sim objects:** YOLO is trained on real photos, so plain coloured boxes may be ignored.
`python sim/fetch_photos.py` downloads sample photos into `sim/photos/`; list them under `billboards:` in `config/sim.yaml`
and they are pasted on thin boards in the room. If detection is still weak, use `--gt`. On the author's laptop YOLO (`yolov8n`)
recognised the chairs, table, bottles and cup in the plain-shape room but not the cardboard box or the backpack; billboards
have not been tried yet.

## 10. Object map and accuracy check

`mapping/` turns detections into a persistent map. Per frame, each detection's 3D position comes from the sim depth image
(median depth of the nearest half of the valid pixels in the central 40% of the box, deprojected with the camera
intrinsics, moved to world coordinates with the robot pose). Repeated sightings of the same class within 0.5 m are merged
by a median; an object only appears after 5 sightings; sightings beyond 4 m, cut off by the image edge, or mostly hidden
behind a nearer detection are skipped. Depth gives a point on the object's *surface*, so a per-class half-depth prior moves
it towards the centre (`config/mapping.yaml`); both errors are reported.

`run_mapping_eval.py` writes `results/object_map.json`, `map_topdown.png` (room, robot path, true vs estimated positions),
`error_per_object.png` and `accuracy.csv`. Committed example results are in `../docs/phase1_results/`.

**Real robot (untested):** `python run_detection.py --iface <iface> --map` uses the LiDAR-based distances instead, converted
to the world (odom) frame with the pose from `rt/sportmodestate` (`mapping/real_adapter.py`) and saved as
`results/object_map_real.json`. The camera-LiDAR calibration (`camera_model:` and `extrinsics:` in `config/detection.yaml`)
holds **placeholders** (assumed 100 degree field of view, zero offset and tilt). **Measure them before trusting real positions.**

## 11. Preflight check before using the real robot

```bash
python preflight.py --iface enp0s31f6
```
Prints PASS/FAIL/WARN, each failure with a fix hint: interface has a 192.168.123.x address, ping 192.168.123.161, the
`cyclonedds` version matches what the SDK pins, `ufw` status, data on `rt/lowstate` and `rt/sportmodestate` within 5 s, one
camera frame, and LiDAR data on `rt/utlidar/cloud`. It never moves the robot, and ends by printing the launch command.
`--no-dds` runs only the network/software checks.

## 12. Key bindings

Keyboard teleop (global key capture via pynput; same in `main.py`, `teleop.py`, `run_sim.py`, and `--teleop` / `--sim`):

| Key | Action |
|---|---|
| W / S | forward / backward |
| A / D | strafe left / right |
| Q / E | rotate left / right |
| Space | stop and ignore held keys until released |
| R | stop, stand up, balance stand |
| F | stop, stand down |
| + / - | speed scale up / down |
| X then Y (within 3 s) | Damp (dog goes limp; never automatic) |
| Esc | stop and exit |

Extra keys in the detection window (`run_detection.py`): **L** toggles the LiDAR overlay (real robot), **P** saves a
screenshot of the detection view and the map to `results/screenshots/` (sim), **G** switches between YOLO and GROUND TRUTH
labels (sim). Esc quits. Avoid these letters in custom teleop keys.

Real-robot launch (after `preflight.py` passes; dog lying down, 2 m clear around it, remote controller in reach):
```bash
python main.py --iface enp0s31f6 --max-speed 0.3                     # camera + LiDAR + teleop
python run_detection.py --iface enp0s31f6 --teleop --max-speed 0.3   # YOLO + LiDAR distances + teleop
```

## 13. Approach an object and act (autonomous mission)

The dog distinguishes objects by YOLO class, walks up to the chosen one, stops at a safe distance and performs a small
action: **chair -> sit**, **person -> hello (greeting / handshake wave)**, **bottle -> content**. The class -> action ->
stop distance table is `config/behaviors.yaml`. Only `sit`, `hello`, `stretch` and `content` exist; flips, jumps,
handstands and Damp can never be requested.

```bash
python run_detection.py --sim --mission chair                  # try it in the simulator first (teleop is on by default)
python run_detection.py --sim --gt --mission chair             # same with ground-truth labels
python run_detection.py --iface enp0s31f6 --teleop --mission chair --mission-dry-run    # robot: prints, nothing moves
python run_detection.py --iface enp0s31f6 --teleop --mission chair --allow-autonomy --max-speed 0.3   # robot: moves
python behavior/try_action.py --iface enp0s31f6 sit            # one action on its own, asks you to type YES
```

**The go-ahead sequence:** R (stand, so the program knows the dog is standing) -> wait until the video shows
`MISSION READY: chair 2.3 m -> sit @ 0.90 m [B = go]` -> press **B in the video window**. Nothing moves before B.
`C` confirms the action if `confirm_before_action: true`; `V` cancels.

**What stops it, always:** any teleop key (WASD/QE, Space, R, F, +/-, Esc), V, the target being lost for 1.5 s, detections
older than 0.7 s (it stands still), the path blocked for 3 s, the approach taking longer than 45 s, the dog no longer
standing, or the program exiting. Autonomous velocity requests expire after 0.3 s, so if the mission thread dies the dog
stops; speeds are 0.25 m/s forward and 0.5 rad/s turning (never strafing or backing up), and Teleop's limits still apply.

**Safety limits you should know:**
- The stop distance is what the *sensors* read, which was accurate to about 12% in a quick test (1 m read 1.12 m, 2 m read
  1.88 m, 3 m read about 3 m). Per-class stop distances are raised to at least 0.6 m.
- The walking-corridor check uses the LiDAR (or the sim depth image). It reliably sees large surfaces (walls, furniture
  fronts) but **not thin obstacles such as chair legs**, and it is untested on the real LiDAR. It is a safety net, not
  obstacle avoidance. Keep the path clear yourself. If it cannot see the LiDAR it reports "blocked" and the dog stays put.
- `person -> hello` makes the robot approach people. Use it only with a willing, informed adult and keep children and pets
  clear. `Sit`, `RiseSit` and `Hello` behaviour on the real Go2 (does it return to standing by itself? how long does it take?)
  is unverified: try each with `behavior/try_action.py` first.

**Staged test on the real robot (do not skip steps):**
1. `python behavior/try_action.py --iface enp0s31f6 sit` (and `hello`) with the dog standing and the remote in your hand.
2. Dry run: `--mission chair --mission-dry-run`, press R then B with a chair 2-3 m in front. The dog must not move. Check the
   printed commands: forward `vx > 0` when facing the chair, and `vyaw > 0` (turn left) when the chair is to its left; move
   the chair to the other side and the sign must flip. The printed range should match a tape measure within about 15%.
3. Stand 1 m from a wall, run the dry run again: it should report `path blocked`.
4. Live, with `--allow-autonomy --max-speed 0.3`: 3 m of clear floor, chair straight ahead at about 2.5 m, remote in hand,
   a helper ready to hit Space. Press B, watch, abort at the slightest doubt. Then an off-centre chair, then a person.
