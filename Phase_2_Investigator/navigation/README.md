# Navigation (A4)

Everything moves the robot through `velocity.py` (`cmd_vel`: body-frame vx, vy, wz), with project
limits 0.3 m/s forward, 0.15 m/s backward, 0.15 m/s sideways, 0.5 rad/s, and a dead-man (commands
expire after 0.5 s unless re-sent).

| Module | What it does |
|---|---|
| `occupancy.py` | 5 cm log-odds grid from the LiDAR. The 2D scan carves free space and marks walls. Points of the 3D cloud between 4 and 70 cm high go into a persistent low-obstacle layer (lying mannequins, boxes, table edges the scan plane passes over). |
| `planner.py` | Minimum-cost path (scikit-image MCP, Dijkstra) on a costmap: obstacles inflated by 0.24 m, plus a soft penalty out to 0.6 m that keeps paths centred. |
| `frontier.py` | Frontier clusters (free cells next to unknown space). The goal is the reachable cell near a frontier with 0.5 m of clearance; attempted goals are never repeated. |
| `coverage.py` | Camera coverage: which wall, furniture and floor cells the front camera has seen (76 deg FOV, line of sight). Then a greedy patrol of viewpoints that see the most unseen surface. |
| `local.py` | Pure pursuit plus the safety layer (details below). |
| `explorer.py` | `Navigator` combines these at 10 Hz: `explore()`, `patrol()`, `go_to()`, `turn_to()`, collision log. |
| `inspect.py` | Inspect point: standoff views at 0 and ±25 deg with line of sight, sensor frame plus 1920x1080 close-up per view. |

The safety layer works independently of the map:
- **Obstacle points:** the current scan, the last second of low cloud points, and occupied map cells within 1.2 m.
- **Forward and backward motion:** slows from 0.6 m of clearance in the travel direction and stops at 0.2 m.
- **Turning:** refused if the body footprint, moved by the next 0.3 s of motion, would cover any point.
- **When turning is refused:** a short collision-free manoeuvre (turn, arc, straight or sideways step), chosen to maximise clearance.

A run never needs a human. If it gets stuck (no progress for 8 s), it backs off and gives up on that goal.

## Where simulation and the real Go2 differ

| | Simulation | Real Go2 (Phase 1) |
|---|---|---|
| LiDAR | MID-360 model: 1536 rays, -7 to 52 deg, 10 Hz, plus a 720-ray planar scan | Unitree L1 (4D LiDAR, wide vertical FOV); a planar scan would come from its point cloud |
| Low obstacles | The lowest MID-360 ray is 7 deg down, so 0.2 m-tall things are only seen beyond ~2 m (they stay in the map) | The L1 sees the ground close to the robot |
| Pose | Simulator ground truth | Odometry / SLAM, with drift |
| Gait | Scripted trot, clamps to 0.4 m/s, 0.15 m/s sideways, 0.5 rad/s; cannot pitch the body | Sport mode `Move(vx, vy, wz)`; can pitch the body to look up/down |
| Camera | 320x240 at 5 Hz, 60 deg; close-ups rendered at 1920x1080 with an emulated pitch | 1920x1080 stream (Phase 1) |
| Contacts | Any robot contact except feet on the floor is logged as a collision | Would need foot-force / IMU heuristics |

To drive the real robot, implement a `VelocityInterface` whose `_apply` calls Phase 1's
`SportClient.Move(vx, vy, wz)` (same limits, re-sent at 10 Hz), and feed `OccupancyGrid.update` from
the L1 point cloud.
