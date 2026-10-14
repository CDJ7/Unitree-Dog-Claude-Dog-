"""LiDAR point cloud (rt/utlidar/cloud) + robot state subscribers, PointCloud2 parsing, and viewers."""
import logging
import threading
import time

import cv2
import numpy as np

log = logging.getLogger("lidar")

# sensor_msgs/PointField datatype -> numpy dtype
_PF_DTYPES = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2_to_xyz(msg):
    """Parse a PointCloud2_ IDL message into an (N, 3) float32 array of x, y, z (NaN/inf rows dropped)."""
    byteorder = ">" if msg.is_bigendian else "<"
    offsets = {f.name: (f.offset, _PF_DTYPES[f.datatype]) for f in msg.fields if f.datatype in _PF_DTYPES}
    if not all(k in offsets for k in "xyz"):
        raise ValueError(f"PointCloud2 has no x/y/z fields (fields: {[f.name for f in msg.fields]})")
    dt = np.dtype({"names": ["x", "y", "z"],
                   "formats": [byteorder + offsets[k][1] for k in "xyz"],
                   "offsets": [offsets[k][0] for k in "xyz"],
                   "itemsize": msg.point_step})
    n = msg.width * msg.height
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8, count=n * msg.point_step)
    rec = raw.view(dt)
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


class Latest:
    """Latest-value holder fed from a DDS callback thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self.value, self.seq, self.stamp = None, 0, 0.0

    def put(self, v):
        with self._lock:
            self.value, self.seq, self.stamp = v, self.seq + 1, time.monotonic()

    def get(self):
        with self._lock:
            return self.value, self.seq, self.stamp


class LidarSource:
    def __init__(self, cfg):
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
        self.cfg, self.latest = cfg, Latest()
        self._warn_t = time.monotonic()
        if cfg.get("switch_on_start"):
            pub = ChannelPublisher(cfg["switch_topic"], String_)
            pub.Init()
            msg = std_msgs_msg_dds__String_()
            msg.data = "ON"
            time.sleep(0.5)  # let DDS discovery match the writer to the robot's reader
            pub.Write(msg)
            log.info("Sent 'ON' to %s", cfg["switch_topic"])
        self._sub = ChannelSubscriber(cfg["topic"], PointCloud2_)
        self._sub.Init(self.latest.put, 0)   # handler runs on the DDS thread: just store the message
        log.info("Subscribed to %s (sensor_msgs.msg.dds_.PointCloud2_)", cfg["topic"])

    def check_stale(self):
        """Warn (rate-limited) when nothing has arrived on the topic for no_data_warn_s."""
        _, seq, stamp = self.latest.get()
        now = time.monotonic()
        age = now - (stamp if seq else self._warn_t)
        if age > self.cfg["no_data_warn_s"] and now - self._warn_t > self.cfg["no_data_warn_s"]:
            self._warn_t = now
            log.warning("No data on %s for >%.0fs. Check: the LiDAR is switched on (%s 'ON'; try "
                        "`utlidar` service in the app), `ping 192.168.123.161`, correct --iface, "
                        "CYCLONEDDS_HOME/cyclonedds==0.10.2, and no firewall blocking UDP.",
                        self.cfg["topic"], self.cfg["no_data_warn_s"], self.cfg["switch_topic"])


class StateSource:
    """sportmodestate (position/gait/mode) + lowstate (battery SOC)."""

    def __init__(self, cfg):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_
        self.cfg = cfg
        self.sport, self.low = Latest(), Latest()
        self._warned = {"sport": time.monotonic(), "low": time.monotonic()}
        s = ChannelSubscriber(cfg["sport_topic"], SportModeState_)
        s.Init(self.sport.put, 0)
        l = ChannelSubscriber(cfg["low_topic"], LowState_)
        l.Init(lambda m: self.low.put(int(m.bms_state.soc)), 0)
        self._subs = (s, l)
        log.info("Subscribed to %s and %s", cfg["sport_topic"], cfg["low_topic"])

    def check_stale(self):
        now = time.monotonic()
        for key, src, topic in (("sport", self.sport, self.cfg["sport_topic"]),
                                ("low", self.low, self.cfg["low_topic"])):
            _, seq, stamp = src.get()
            if (not seq or now - stamp > self.cfg["no_data_warn_s"]) and \
                    now - self._warned[key] > self.cfg["no_data_warn_s"]:
                self._warned[key] = now
                log.warning("No data on %s for >%.0fs. Check the robot is on, `ping 192.168.123.161`, and "
                            "the --iface name.", topic, self.cfg["no_data_warn_s"])

    def lines(self):
        out = []
        s, _, _ = self.sport.get()
        soc, _, _ = self.low.get()
        out.append(f"battery {soc}%" if soc is not None else "battery --")
        if s is not None:
            out.append(f"pos {s.position[0]:+.2f} {s.position[1]:+.2f} {s.position[2]:+.2f}  "
                       f"mode {s.mode} gait {s.gait_type}  h {s.body_height:.2f}")
        return out


def height_colors(z, z_min, z_max):
    """Map heights to BGR colours (uint8, N x 3) via a turbo colormap."""
    t = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0, 1)
    img = (t * 255).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(img, cv2.COLORMAP_TURBO).reshape(-1, 3)


class TopDownViewer:
    """2D fallback: x forward = up, y left = left. Rendered into an OpenCV window."""

    def __init__(self, cfg, window="go2_lidar_topdown", size=640):
        self.cfg, self.window, self.size = cfg, window, size
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        self._img = np.zeros((size, size, 3), np.uint8)

    def update(self, xyz, info=()):
        r, size = self.cfg["topdown_range_m"], self.size
        px = size / (2 * r)
        img = np.zeros((size, size, 3), np.uint8)
        for ring in range(1, int(r) + 1):
            cv2.circle(img, (size // 2, size // 2), int(ring * px), (50, 50, 50), 1)
        if len(xyz):
            colors = height_colors(xyz[:, 2], self.cfg["z_min"], self.cfg["z_max"])
            u = (size / 2 - xyz[:, 1] * px).astype(np.int32)
            v = (size / 2 - xyz[:, 0] * px).astype(np.int32)
            m = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            img[v[m], u[m]] = colors[m]
        cv2.circle(img, (size // 2, size // 2), 4, (255, 255, 255), -1)
        for i, t in enumerate(info):
            cv2.putText(img, t, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        self._img = img

    def pump(self):
        cv2.imshow(self.window, self._img)

    def close(self):
        cv2.destroyWindow(self.window)


class Open3DViewer:
    def __init__(self, cfg):
        import open3d as o3d
        self.o3d, self.cfg = o3d, cfg
        self.vis = o3d.visualization.Visualizer()
        if not self.vis.create_window("Go2 LiDAR", width=900, height=700):
            raise RuntimeError("Open3D could not create a window (no display / OpenGL?)")
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        self.vis.get_render_option().point_size = 2.0
        self._first = True

    def update(self, xyz, info=()):
        if not len(xyz):
            return
        o3d = self.o3d
        rgb = height_colors(xyz[:, 2], self.cfg["z_min"], self.cfg["z_max"])[:, ::-1] / 255.0
        self.pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        self.vis.update_geometry(self.pcd)
        if self._first:
            self.vis.reset_view_point(True)
            self._first = False

    def pump(self):
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self):
        self.vis.destroy_window()


def make_viewer(cfg):
    """Open3D if requested and it works, else the OpenCV top-down view."""
    if cfg.get("viewer", "open3d") == "open3d":
        try:
            return Open3DViewer(cfg)
        except Exception as e:
            log.warning("Open3D viewer unavailable (%s); falling back to top-down OpenCV view", e)
    return TopDownViewer(cfg)


def prepare_points(xyz, cfg, rng=np.random.default_rng()):
    """Clip to the configured height window and subsample for display."""
    xyz = xyz[(xyz[:, 2] >= cfg["z_min"]) & (xyz[:, 2] <= cfg["z_max"])]
    if len(xyz) > cfg["max_points"]:
        xyz = xyz[rng.choice(len(xyz), cfg["max_points"], replace=False)]
    return xyz
