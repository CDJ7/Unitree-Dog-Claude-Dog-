"""Zone map: heading-balanced tagging, smoothing, flood fill through free space, save/load."""
import numpy as np

from mapping.zone_map import ZoneMap
from navigation.occupancy import OccupancyGrid

CLASSES = ["lab_1", "lab_2", "open_area"]


def two_rooms():
    """6 x 4 m room split by a wall at x = 3 with a 1 m door at y 1.5-2.5. Left = lab_1, right = lab_2."""
    g = OccupancyGrid(origin=(0, 0), size=(6.0, 4.0), res=0.05)
    g.seen[:] = True
    g.logodds[:] = -2.0
    occ = np.zeros(g.shape, bool)
    occ[[0, -1], :] = True
    occ[:, [0, -1]] = True
    c = int(3.0 / g.res)
    occ[:, c:c + 2] = True
    r0, r1 = int(1.5 / g.res), int(2.5 / g.res)
    occ[r0:r1, c:c + 2] = False
    g.logodds[occ] = 3.0
    return g


def p(k, conf=0.8):
    v = np.full(len(CLASSES), (1 - conf) / (len(CLASSES) - 1))
    v[k] = conf
    return v


def test_heading_balanced_tagging_and_flood_fill():
    g = two_rooms()
    zm = ZoneMap(g, CLASSES)
    rng = np.random.default_rng(0)
    for _ in range(400):
        x, y = rng.uniform(0.4, 2.6), rng.uniform(0.4, 3.6)
        yaw = rng.uniform(-np.pi, np.pi)
        # facing the door (east) the camera sees the other room: a heading-dependent mistake
        zm.add((x, y), yaw, p(1) if abs(yaw) < np.pi / 4 else p(0))
    for _ in range(400):
        x, y = rng.uniform(3.4, 5.6), rng.uniform(0.4, 3.6)
        zm.add((x, y), rng.uniform(-np.pi, np.pi), p(1))
    lab, conf = zm.labels()
    left = lab[:, 5:55][g.free()[:, 5:55]]
    right = lab[:, 65:115][g.free()[:, 65:115]]
    assert (left == 0).mean() > 0.95 and (right == 1).mean() > 0.95
    assert lab[g.occupied()].max() == -1                      # walls are never labelled
    reg = zm.regions()
    assert set(reg) == {"lab_1", "lab_2"}
    assert reg["lab_1"]["goal"][0] < 3.0 < reg["lab_2"]["goal"][0]
    assert zm.zone_at((1.0, 2.0))[0] == 0 and zm.zone_at((5.0, 2.0))[0] == 1


def test_one_view_cannot_outvote_a_look_around():
    g = two_rooms()
    zm = ZoneMap(g, CLASSES, min_frames=1)
    for _ in range(50):                                      # stared east for a long time: "lab_2"
        zm.add((1.5, 2.0), 0.0, p(1, 0.9))
    for yaw in np.linspace(-np.pi, np.pi, 8, endpoint=False)[1:]:   # one look around: "lab_1"
        zm.add((1.5, 2.0), yaw, p(0, 0.9))
    assert zm.zone_at((1.5, 2.0))[0] == 0


def test_flood_fill_does_not_cross_walls():
    g = two_rooms()
    c = int(3.0 / g.res)
    g.logodds[:, c:c + 2] = 3.0                               # close the door
    zm = ZoneMap(g, CLASSES)
    for _ in range(10):
        zm.add((1.0, 2.0), 0.0, p(0))
    lab, _ = zm.labels()
    assert (lab[:, c + 2:][g.free()[:, c + 2:]] == -1).all()  # the closed-off room stays unlabelled


def test_save_load_roundtrip(tmp_path):
    g = two_rooms()
    zm = ZoneMap(g, CLASSES)
    for _ in range(10):
        zm.add((1.0, 2.0), 0.3, p(0))
        zm.add((5.0, 2.0), 0.3, p(1))
    zm.save(tmp_path, path_xy=[(1, 2), (5, 2)])
    zm2, path = ZoneMap.load(tmp_path)
    assert zm2.classes == CLASSES and len(path) == 2
    assert np.array_equal(zm2.labels()[0], zm.labels()[0])
    assert (tmp_path / "zone_map.png").exists() and (tmp_path / "zones.json").exists()
