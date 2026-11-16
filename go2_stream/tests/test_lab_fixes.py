"""Fixes taken over from Test 2's lab runs (12-13 Nov 2026): interface auto-detect, LiDAR frame guard, config."""
import os
import sys
import types

import numpy as np
import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from netutil import resolve_iface  # noqa: E402
from perception.pipeline import cloud_to_body  # noqa: E402


def test_resolve_iface_keeps_a_good_name_and_fixes_a_renamed_one():
    assert resolve_iface("eth1", found=["eth1"]) == "eth1"
    assert resolve_iface("eth1", found=["eth0"]) == "eth0"           # WSL renamed eth1 -> eth0
    assert resolve_iface("eth1", found=[]) == "eth1"                 # nothing on the subnet: caller reports it
    assert resolve_iface("eth1", found=["eth0", "eth2"]) == "eth1"   # ambiguous: don't guess


def _msg(frame_id):
    return types.SimpleNamespace(header=types.SimpleNamespace(frame_id=frame_id))


def _lidar_mod(pts):
    return types.SimpleNamespace(pointcloud2_to_xyz=lambda m: np.asarray(pts, np.float32))


def test_sensor_frame_cloud_is_refused_in_auto_mode():
    pts, why = cloud_to_body(_msg("utlidar_lidar"), None, "auto", _lidar_mod([[1.0, 0.0, 0.3]]))
    assert pts is None and "cloud_deskewed" in why


def test_body_frame_cloud_still_passes_and_explicit_mode_is_respected():
    pts, frame = cloud_to_body(_msg("base_link"), None, "auto", _lidar_mod([[1.0, 0.0, 0.3]]))
    assert pts is not None and frame == "base_link"
    pts, _ = cloud_to_body(_msg("utlidar_lidar"), None, "body", _lidar_mod([[1.0, 0.0, 0.3]]))
    assert pts is not None                                            # the user forced it: their call


def test_configs_have_the_lab_values():
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    assert cfg["lidar"]["topic"] == "rt/utlidar/cloud_deskewed"
    b = yaml.safe_load(open(os.path.join(HERE, "config/behaviors.yaml")))
    cor = b["safety"]["corridor"]
    assert cor["z_min"] == -0.20 and cor["stop_if_closer_m"] >= 0.7
