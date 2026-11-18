# Real Go2: the dog maps the lab and tags its zones itself (Stage R)

The dog explores the lab on its own and builds a LiDAR map. Every place on the map collects the camera
zone classifier's votes (lab_1, lab_2, open_area, shelves) from all the directions it was seen from.
The result is a map split into zones that you can send it to: "go to lab_2" plans a path on the map.
Why this replaces Test 2's camera-only go-to-zone, and how it fits the crime-scene dog: see
[`docs/real_robot_plan.md`](../docs/real_robot_plan.md).

Everything runs from a **terminal on the Linux laptop**: no VS Code, no WSL, no windows needed. Results
are files (PNG map, JSON, MP4) that you, or Claude Code in the terminal, can open.

| Command | What it does | Dog moves? |
|---|---|---|
| `python scripts/real_check.py` | Sensor check: pose, LiDAR, floor height, camera, classifier, map snapshot | **No** |
| `python scripts/map_zones.py --backend real` | Explore the lab autonomously, look around at each stop, build the zone map | **Yes** |
| `python scripts/goto_zone.py --map <dir> --list` | List the zones on a map | No |
| `python scripts/goto_zone.py --map <dir> --backend real --zone lab_2` | Find itself on the map, then walk to the zone | **Yes** |

Every one of these also runs against the **simulator presented as the dog** (`--link sim`), so you can
rehearse a session without the robot.

---

## 1. Set up the Linux laptop (once)

Ubuntu 22.04 or 24.04, Python 3.10–3.12. If Test 1's setup is already on this laptop (`~/go2venv`,
`~/cyclonedds`, `~/unitree_sdk2_python`), skip to step 1.3.

```bash
# 1.1 CycloneDDS 0.10.x + the Unitree SDK (exactly as Test 1, go2_stream/README.md)
sudo apt install -y git cmake build-essential python3-venv
cd ~ && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir -p build install && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
echo 'export CYCLONEDDS_HOME=~/cyclonedds/install' >> ~/.bashrc && source ~/.bashrc
python3 -m venv ~/go2venv && source ~/go2venv/bin/activate
cd ~ && git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && pip install -e ~/unitree_sdk2_python

# 1.2 This repository
cd ~ && git clone https://github.com/CDJ7/Unitree-Dog-Claude-Dog-.git
cd ~/Unitree-Dog-Claude-Dog-/Phase_2_Investigator
./scripts/setup.sh                                  # simulator submodule + Python packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # zone classifier (CPU is enough)
pip install -r ../go2_stream/requirements.txt       # Test 1 modules used for the real dog

# 1.3 The lab zone classifier (not in git): copy best.pt from the old laptop to
#     ~/Unitree-Dog-Claude-Dog-/Test2/results/real_q1/resnet18/best.pt
python -m pytest -q tests/test_zone_map.py tests/test_localize.py tests/test_real_session.py
```

**Each new terminal:** `source ~/go2venv/bin/activate && cd ~/Unitree-Dog-Claude-Dog-/Phase_2_Investigator`.

## 2. Connect the dog (each session)

1. Plug the Ethernet cable into the dog (Go2 EDU) and the laptop. Power on the dog and wait until it stands.
2. Give the laptop 192.168.123.222 on that port. Test 1's script does it:
   ```bash
   ../go2_stream/setup_network.sh detect                  # find the wired interface, e.g. enp0s31f6
   ../go2_stream/setup_network.sh nmcli enp0s31f6         # once: a saved profile "go2-direct"
   nmcli con up go2-direct                                # later sessions
   ping -c 3 192.168.123.161                              # the dog must answer
   ```
3. If the firewall is on (`sudo ufw status`), allow the dog's subnet: `sudo ufw allow from 192.168.123.0/24`.
4. Test 1's preflight: `python ../go2_stream/preflight.py`. It needs PASS on state, LiDAR and camera.

The scripts find the interface on 192.168.123.x themselves, so `--iface` is optional.

## 3. Sensor check (the dog does not move)

```bash
python scripts/real_check.py --seconds 15
```
It prints PASS / WARN / FAIL and writes `outputs/real_check/<time>/` (`map.png`, `frame.jpg`,
`report.json`). While it runs, walk the dog a few metres with the remote, so the map shows some of the
lab.

| Line | Look for | If not |
|---|---|---|
| odometry pose | poses arriving | preflight; check the iface and `rt/sportmodestate` |
| LiDAR scans | most of the 360 directions have a return | is `rt/utlidar/cloud_deskewed` publishing? the L1 must be on |
| floor height | floor about 0.31 m below the body | pass the measured value as `--body-height` to the other scripts |
| occupancy map | `map.png` shows the walls around the dog, not a filled blob | if the floor shows up as obstacles, raise `--body-height` slightly |
| zone classifier | the zone it should be | the model file (step 1.3) |

## 4. Map the lab (the dog moves on its own)

**Before you start:**
- Clear the floor of cables and bags.
- Tell people in the room.
- Hold the remote.

**Stopping it:** press **any key in the terminal** (SPACE, Enter, q). The dog stops and the map built so
far is saved. The Unitree remote and Test 1's teleop keys also take over at once.

```bash
python scripts/map_zones.py --backend real --max-time 600 --video
```

**What the dog does:**
1. Stands up and turns a full circle.
2. Repeatedly drives to the nearest unexplored opening in its LiDAR map (frontier exploration), turning
   a full circle at each one.
3. Visits any free area that still has no zone votes.

**How it drives:**
- at most 0.3 m/s and 0.5 rad/s;
- Phase 2's safety layer slows down from 0.6 m and stops at 0.2 m from any obstacle;
- if it gets stuck, it backs out.

The terminal prints each goal and the zones found so far.

**Output in `outputs/zone_map/real_<time>/`:**
- `zone_map.png`: the map with the zones in colour;
- `zones.json`: each zone's area, centre, goal point and confidence;
- `map.npz`: everything, for go-to-zone;
- `report.json`: path length, look-arounds, frames tagged;
- `map_building.mp4`, with `--video`.

## 5. Go to a zone

```bash
python scripts/goto_zone.py --map outputs/zone_map/real_<time> --list
python scripts/goto_zone.py --map outputs/zone_map/real_<time> --backend real --zone lab_2 shelves
```

**What it does:**
1. **Relocalises.** It turns a full circle and matches its LiDAR to the saved map. The dog's odometry
   restarts at every boot, so this finds where it is on the map. A poor match stops the run with a
   message; it never drives on a map it doesn't fit.
2. **Walks to each zone in turn.** It plans a path on the map to the zone's goal point, the free point
   deepest inside the zone. The map keeps updating from the live LiDAR, so new obstacles are avoided.
3. **Confirms on arrival.** It looks around once, and the camera's majority vote is compared with the
   map's zone at that spot.

Results go to `outputs/goto/goto_<time>.json` (and a PNG). `--update-map` also adds the new camera votes
to the map.

## 6. Rehearse without the dog

```bash
python scripts/real_check.py --link sim
python scripts/map_zones.py --backend real --link sim --max-time 300
python scripts/goto_zone.py --map outputs/zone_map/r1854 --backend real --link sim --odom-offset 1.5 -2 40 --zone bathroom
```
`--link sim` presents the MuJoCo Go2 to the real-robot code exactly like the dog. It gives an odometry
pose, an odometry-frame point cloud, camera frames and velocity commands.
`--odom-offset` pretends the dog booted somewhere else, which tests relocalisation.

## Running next to other robot software (TurtleBot / ROS 2)

- **CPU:** Phase 2 uses at most `PHASE2_THREADS` threads (default 2) for numpy, OpenCV and PyTorch. The
  zone classifier needs about 30 ms per frame on one thread. On a laptop that is doing nothing else, you
  can raise it: `export PHASE2_THREADS=4`.
- **DDS:** the Go2 speaks DDS on domain 0, and so does ROS 2 by default. A ROS 2 / TurtleBot terminal on
  domain 0 can discover the dog's topics (and the other way round), which floods both. Put the TurtleBot
  on its own domain, e.g. `export ROS_DOMAIN_ID=30` (the TurtleBot3 guides use 30) in its terminal and
  `~/.bashrc`, and leave the Go2 terminals on 0. The Unitree SDK here also binds to the Ethernet port
  only (`ChannelFactoryInitialize(0, <iface>)`).
- **Network:** the dog's cable gives the laptop a second network (192.168.123.x, no gateway), so the
  TurtleBot's Wi-Fi network and internet are not affected.

## Safety

- The dog moves only through Test 1's teleop autonomy hook:
  - speed caps (here 0.3 m/s);
  - a 0.3 s dead-man, so if the program stalls the dog stops;
  - StopMove on every exit;
  - manual input wins.
- The session also stops the run if the odometry pose goes stale for more than 1 s.
- Not covered by the code: stairs and drop-offs (the map only knows obstacles), glass walls, and people
  walking into the dog's path faster than it reacts. Supervise every run.
- Never press X then Y (damp) while the dog stands: it collapses.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No interface with a 192.168.123.x address` | Section 2, step 2. Check `ip -br addr`. |
| `No pose or LiDAR data from the dog after 15 s` | Run `preflight.py`. Is the LiDAR on? Is `CYCLONEDDS_HOME` set? Check `ufw`. |
| The map is a filled blob around the dog | The floor counts as an obstacle: check the floor-height line in `real_check`, and pass `--body-height`. |
| Doubled walls after a long run | Odometry drift. Map in shorter runs (`--max-time 300`). Drift correction is plan step R7. |
| `Could not find the dog on this map` | Start where the map is clear, for example the middle of a zone, away from moved chairs. |
| Arrives but the camera says another zone | The camera looks into the next zone (Test 2's finding). The map decides; the camera check is a second opinion. |

When something fails, run the command again with `2>&1 | tee logs/<name>.log` and ask Claude Code in
the terminal to read the log. `CLAUDE.md` tells it the rules for this project.
