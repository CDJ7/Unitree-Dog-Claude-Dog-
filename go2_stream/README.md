# Go2 camera + LiDAR + keyboard teleop

Streams the Go2 front camera and LiDAR point cloud to your laptop and lets you drive the dog with the keyboard.
**Requires a Go2 EDU** (full SDK/DDS access over Ethernet). On AIR/PRO use the WebRTC route instead
(`go2_webrtc_connect`); this project will not work.

> Status: written against the current `unitree_sdk2_python` sources (APIs and topic names read from the repo).
> The teleop safety logic is tested in simulation (fake client). **Nothing here has been run against a real robot yet** —
> follow the staged bring-up below.

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
