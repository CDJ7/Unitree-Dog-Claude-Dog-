#!/usr/bin/env python3
"""Real-robot preflight: checks everything the app needs and prints PASS / FAIL / WARN with a fix hint.

    python preflight.py --iface enp0s31f6
    python preflight.py --iface enp0s31f6 --no-dds     # network checks only (no SDK needed)

Does NOT move the robot. It only listens, pings, and asks for one camera frame. Exit code 1 if anything FAILed.
"""
import argparse
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOT_IP = "192.168.123.161"
WAIT_S = 5.0

_tty = sys.stdout.isatty()
COLORS = {"PASS": "32", "FAIL": "31", "WARN": "33", "SKIP": "36"}
results = []


def report(status, title, detail="", hint=""):
    tag = f"\033[{COLORS[status]}m{status}\033[0m" if _tty else status
    print(f"[{tag}] {title}" + (f": {detail}" if detail else ""))
    if hint and status in ("FAIL", "WARN"):
        for line in hint.splitlines():
            print(f"        fix: {line}")
    results.append(status)
    return status == "PASS"


def run(cmd, timeout=10):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, f"{cmd[0]}: command not found"
    except subprocess.TimeoutExpired:
        return 124, "timed out"


# ------------------------------------------------------------------------------------------ network
def check_interface(iface):
    if not os.path.isdir(f"/sys/class/net/{iface}"):
        avail = ", ".join(os.listdir("/sys/class/net"))
        return report("FAIL", f"Ethernet interface '{iface}' exists", f"not found (available: {avail})",
                      "run ./setup_network.sh detect and pass the wired name with --iface")
    state = open(f"/sys/class/net/{iface}/operstate").read().strip()
    code, out = run(["ip", "-4", "-o", "addr", "show", "dev", iface])
    if code == 127:
        return report("WARN", f"{iface} address check", out, "sudo apt install iproute2")
    addrs = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/(\d+)", out)
    ok_addr = [a for a, _ in addrs if a.startswith("192.168.123.")]
    if not ok_addr:
        return report("FAIL", f"{iface} has a 192.168.123.x address",
                      f"IPv4 addresses: {[a for a, _ in addrs] or 'none'}, link {state}",
                      f"./setup_network.sh temp {iface}      (this is also what causes 'does not match an available interface')\n"
                      f"or for a persistent profile: ./setup_network.sh nmcli {iface}")
    if state not in ("up", "unknown"):
        return report("FAIL", f"{iface} link is up", f"link {state} (address {ok_addr[0]} is set)",
                      "no cable signal: plug the Ethernet cable into the dog's port and power the dog on, then wait 1-2 min")
    return report("PASS", f"{iface} has a 192.168.123.x address", f"{ok_addr[0]}, link {state}")


def check_ping():
    code, out = run(["ping", "-c", "3", "-W", "1", ROBOT_IP], timeout=15)
    if code == 127:
        return report("WARN", f"ping {ROBOT_IP}", out, "install iputils-ping")
    m = re.search(r"(\d+) received", out)
    got = int(m.group(1)) if m else 0
    if got == 0:
        return report("FAIL", f"ping {ROBOT_IP}", "no replies",
                      "dog powered on and fully booted (give it 1-2 min)? cable in the dog's Ethernet port?\n"
                      "laptop IP present? (./setup_network.sh temp <iface>)  blinking red light = power-cycle the dog")
    rtt = re.search(r"= [\d.]+/([\d.]+)/", out)
    return report("PASS", f"ping {ROBOT_IP}", f"{got}/3 replies" + (f", avg {rtt.group(1)} ms" if rtt else ""))


# ------------------------------------------------------------------------------------------ software
def check_cyclonedds():
    try:
        installed = md.version("cyclonedds")
    except md.PackageNotFoundError:
        return report("FAIL", "cyclonedds Python package", "not installed",
                      "export CYCLONEDDS_HOME=~/cyclonedds/install && pip install cyclonedds==0.10.2")
    pinned = None
    try:
        for r in md.requires("unitree_sdk2py") or []:
            m = re.match(r"cyclonedds\s*==\s*([\w.]+)", r)
            if m:
                pinned = m.group(1)
    except md.PackageNotFoundError:
        return report("FAIL", "unitree_sdk2py installed", "not found",
                      "cd ~/unitree_sdk2_python && pip install -e .   (inside your venv)")
    pinned = pinned or "0.10.2"
    home = os.environ.get("CYCLONEDDS_HOME")
    ok = installed == pinned
    detail = f"installed {installed}, SDK needs {pinned}" + (f", CYCLONEDDS_HOME={home}" if home else ", CYCLONEDDS_HOME unset")
    if not ok:
        return report("FAIL", "cyclonedds version matches the SDK", detail,
                      f"pip install cyclonedds=={pinned}   (a mismatch gives \"'NoneType' object has no attribute 'SupportsBasic'\")")
    if not home:
        return report("WARN", "cyclonedds version matches the SDK", detail,
                      "add export CYCLONEDDS_HOME=~/cyclonedds/install to ~/.bashrc (only needed when (re)installing)")
    return report("PASS", "cyclonedds version matches the SDK", detail)


def check_firewall():
    if not shutil.which("ufw"):
        return report("PASS", "firewall (ufw)", "ufw not installed")
    code, out = run(["ufw", "status"])
    if "inactive" in out.lower():
        return report("PASS", "firewall (ufw)", "inactive")
    if "active" in out.lower():
        return report("WARN", "firewall (ufw)", "active - it can block the robot's DDS (UDP) traffic",
                      "sudo ufw allow in on <iface>      (or sudo ufw disable while testing)")
    return report("WARN", "firewall (ufw)", "could not read status (needs sudo)",
                  "run: sudo ufw status    - if active: sudo ufw allow in on <iface>")


# ------------------------------------------------------------------------------------------ DDS data
class _Count:
    def __init__(self):
        self.n, self.first = 0, None

    def __call__(self, msg):
        self.n += 1
        if self.first is None:
            self.first = msg


def wait_for(counter, seconds=WAIT_S):
    t = time.monotonic()
    while time.monotonic() - t < seconds and counter.n == 0:
        time.sleep(0.05)
    return counter.n > 0


def check_topic(topic, msg_type, hint):
    from unitree_sdk2py.core.channel import ChannelSubscriber
    c = _Count()
    try:
        sub = ChannelSubscriber(topic, msg_type)
        sub.Init(c, 0)
    except Exception as e:
        return report("FAIL", f"data on {topic}", f"could not subscribe: {e}",
                      "cyclonedds version problem? see the cyclonedds check above"), None
    t0 = time.monotonic()
    if wait_for(c):
        return report("PASS", f"data on {topic}", f"first message after {time.monotonic() - t0:.1f} s"), c.first
    return report("FAIL", f"data on {topic}", f"nothing within {WAIT_S:.0f} s", hint), None


def check_camera():
    import cv2
    import numpy as np
    from unitree_sdk2py.go2.video.video_client import VideoClient
    client = VideoClient()
    client.SetTimeout(3.0)
    client.Init()
    code = None
    for _ in range(3):
        code, data = client.GetImageSample()
        if code == 0:
            img = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return report("FAIL", "front camera frame", "got bytes but OpenCV could not decode them")
            return report("PASS", "front camera frame", f"{img.shape[1]}x{img.shape[0]} JPEG decoded")
        time.sleep(0.5)
    return report("FAIL", "front camera frame", f"VideoClient code {code} (3102 = no answer from the dog's video service)",
                  "other app/phone streaming video? close it. Dog booted >1 min? Run the SDK example example/go2/front_camera/camera_opencv.py\n"
                  "power-cycle the dog if it still fails")


def check_lidar(cfg):
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_
    topic = cfg["lidar"]["topic"]
    c = _Count()
    try:
        sub = ChannelSubscriber(topic, PointCloud2_)
        sub.Init(c, 0)
    except Exception as e:
        return report("FAIL", f"data on {topic}", f"could not subscribe: {e}", "cyclonedds version problem? see the check above")
    if wait_for(c, 3.0):
        return report("PASS", f"data on {topic}", f"{c.first.width * c.first.height} points in the first cloud")
    # the app switches the LiDAR on at start-up; do the same here and wait again
    try:
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_
        pub = ChannelPublisher(cfg["lidar"]["switch_topic"], String_)
        pub.Init()
        msg = std_msgs_msg_dds__String_()
        msg.data = "ON"
        time.sleep(0.5)
        pub.Write(msg)
    except Exception as e:
        return report("FAIL", f"data on {topic}", f"no data, and could not send the LiDAR ON switch: {e}",
                      "cyclonedds version problem? see the check above")
    if wait_for(c, WAIT_S):
        return report("PASS", f"data on {topic}", f"arrived after sending 'ON' to {cfg['lidar']['switch_topic']} "
                                                f"({c.first.width * c.first.height} points)")
    return report("FAIL", f"data on {topic}", f"nothing within {WAIT_S:.0f} s even after the ON switch",
                  "LiDAR cover clean / not blocked? Dog fully booted? Try the Unitree app's LiDAR switch\n"
                  "or lidar.topic: rt/utlidar/cloud_deskewed in config.yaml")


def main():
    ap = argparse.ArgumentParser(description="Go2 real-robot preflight")
    ap.add_argument("--iface")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--no-dds", action="store_true", help="network + software checks only")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    iface = args.iface or cfg["network"]["iface"]
    print(f"Go2 preflight  iface={iface}  robot={ROBOT_IP}\n")

    net_ok = check_interface(iface)
    ping_ok = check_ping() if net_ok else report("SKIP", f"ping {ROBOT_IP}", "needs the interface/address first")
    sw_ok = check_cyclonedds()
    check_firewall()

    if args.no_dds:
        report("SKIP", "DDS data checks", "--no-dds")
    elif not (net_ok and ping_ok and sw_ok):
        why = "fix the FAILs above first"
        for t in ("data on rt/lowstate", "data on rt/sportmodestate", "front camera frame", "data on rt/utlidar/cloud"):
            report("SKIP", t, why)
    else:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, SportModeState_
        try:
            ChannelFactoryInitialize(0, iface)
        except Exception as e:
            report("FAIL", "DDS initialisation", str(e),
                   f"{iface} needs an IPv4 address on 192.168.123.x: ./setup_network.sh temp {iface}")
        else:
            hint = "dog booted and not in a fault state (blinking red = power-cycle)? --iface correct? firewall? cyclonedds==0.10.2?"
            check_topic("rt/lowstate", LowState_, hint)
            _, st = check_topic(cfg["state"]["sport_topic"], SportModeState_, hint)
            if st is not None and getattr(st, "error_code", 0):
                report("WARN", "robot error_code", str(st.error_code), "check the Unitree app for a fault before driving")
            check_camera()
            check_lidar(cfg)

    bad = results.count("FAIL")
    # a SKIP means a check could not run (an earlier one failed); that must never read as "passed"
    skipped = results.count("SKIP") - (1 if args.no_dds else 0)
    print()
    if bad:
        print(f"NOT READY: {bad} check(s) FAILED" + (f" and {skipped} could not run" if skipped else "")
              + f". Fix them, then re-run: python preflight.py --iface {iface}")
        sys.exit(1)
    if skipped > 0:
        print(f"NOT READY: {skipped} check(s) could not run. Re-run: python preflight.py --iface {iface}")
        sys.exit(1)
    warns = results.count("WARN")
    if args.no_dds:
        print("Network and software checks passed" + (f" ({warns} warning(s) above)" if warns else "")
              + ". The robot itself was NOT checked: run again without --no-dds with the dog on.")
        return
    print("All checks passed" + (f" ({warns} warning(s) above)." if warns else "."))
    print("\nLaunch the full real-robot app (camera + LiDAR + keyboard teleop):")
    print(f"    python main.py --iface {iface} --max-speed 0.3")
    print("With YOLO detection and LiDAR distances:")
    print(f"    python run_detection.py --iface {iface} --teleop --max-speed 0.3")
    print("Dog starts lying down: clear ~2 m around it, press R to stand, keep the Unitree remote in reach.")

if __name__ == "__main__":
    main()
