"""Keyboard teleoperation for the Go2 via SportClient.

Key-down/up state comes from a pynput listener (terminal input cannot see key releases).
A dedicated control thread sends Move() at a fixed rate while a key is held, and a watchdog
thread sends StopMove() if that loop stalls. Run standalone for a teleop-only test:

    python teleop.py --iface enp0s31f6            # real robot (dog lying down / on a stand first!)
    python teleop.py --dry-run                    # prints commands, no robot, no DDS
"""
import argparse
import logging
import threading
import time
from collections import deque

log = logging.getLogger("teleop")

# key name -> (axis, sign). Axes: 0=vx, 1=vy, 2=vyaw
MOVE_KEYS = {
    "w": (0, +1), "s": (0, -1),
    "a": (1, +1), "d": (1, -1),   # +vy = left in the Go2 body frame
    "q": (2, +1), "e": (2, -1),   # +vyaw = counter-clockwise (left)
}

HELP = """\
Keys:  W/S fwd/back   A/D strafe L/R   Q/E turn L/R   +/- speed scale
       SPACE stop   R stand up + balance   F stand down   X then Y damp (collapses!)
       ESC stop and exit"""


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class FakeSportClient:
    """Stand-in used by --dry-run and tests: logs instead of talking to the robot."""

    def __init__(self):
        self.calls = []

    def _rec(self, name, *a):
        self.calls.append((time.monotonic(), name, a))
        log.info("[dry-run] %s%s", name, a if a else "")
        return 0

    def Move(self, vx, vy, vyaw): return self._rec("Move", round(vx, 3), round(vy, 3), round(vyaw, 3))
    def StopMove(self): return self._rec("StopMove")
    def StandUp(self): return self._rec("StandUp")
    def BalanceStand(self): return self._rec("BalanceStand")
    def StandDown(self): return self._rec("StandDown")
    def Damp(self): return self._rec("Damp")


class Teleop:
    def __init__(self, cfg, client=None, max_speed=None):
        """cfg: the `control` section of config.yaml. client: SportClient-like (None -> create real one)."""
        self.cfg = cfg
        self._client = client
        # Effective limits: CLI --max-speed sets vx; vy/vyaw scale proportionally; hard caps always win.
        vx, vy, wz = cfg["max_vx"], cfg["max_vy"], cfg["max_vyaw"]
        if max_speed is not None:
            ratio = max_speed / vx
            vx, vy, wz = max_speed, vy * ratio, wz * ratio
        self.limits = (min(vx, cfg["hard_cap_vx"]),
                       min(vy, cfg["hard_cap_vy"]),
                       min(wz, cfg["hard_cap_vyaw"]))
        sc = cfg["speed_scale"]
        self.scale = clamp(sc["initial"], sc["min"], sc["max"])

        self._lock = threading.Lock()
        self._held = {}                 # key name -> expiry (monotonic)
        self._edge_down = set()         # non-movement keys currently down (de-bounces auto-repeat)
        self._inhibit = False           # set by SPACE until every key is released
        self._actions = deque()
        self._damp_armed_until = 0.0
        self._cmd = (0.0, 0.0, 0.0)     # last commanded velocity (for the UI)
        self.status = "idle"
        self._busy = False
        self._heartbeat = time.monotonic()
        self._stop = threading.Event()
        self.quit_event = threading.Event()   # set when ESC pressed; main watches this
        self._threads = []
        self._listener = None
        self._stall_reported = False

    # ---------------------------------------------------------------- lifecycle
    def start(self, use_keyboard=True):
        if self._client is None:
            from unitree_sdk2py.go2.sport.sport_client import SportClient
            c = SportClient()
            c.SetTimeout(self.cfg["client_timeout_s"])
            c.Init()
            self._client = c
            log.info("SportClient initialised (timeout %.1fs)", self.cfg["client_timeout_s"])
        if use_keyboard:
            self._start_listener()
        self._heartbeat = time.monotonic()
        for target, name in ((self._control_loop, "control"), (self._watchdog_loop, "watchdog")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        print(HELP)
        log.info("Limits vx=%.2f vy=%.2f vyaw=%.2f, scale=%.1f. Dog must be standing (press R) before it will walk.",
                 *self.limits, self.scale)

    def shutdown(self):
        """Always ends with StopMove. Safe to call more than once."""
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads.clear()
        if self._client is not None:
            for _ in range(2):
                try:
                    self._client.StopMove()
                    break
                except Exception as e:
                    log.error("StopMove during shutdown failed: %s", e)
        self.status = "stopped"

    # ---------------------------------------------------------------- input
    def _start_listener(self):
        try:
            from pynput import keyboard
        except Exception as e:  # ImportError, or no X display
            raise RuntimeError(
                f"pynput unavailable ({e}). Needs an X11 session (on Wayland log in with 'Xorg' "
                "or run under XWayland-free X11); see README.") from e

        def name_of(key):
            if key == keyboard.Key.space: return "space"
            if key == keyboard.Key.esc: return "esc"
            ch = getattr(key, "char", None)
            return ch.lower() if ch else None

        def on_press(key):
            n = name_of(key)
            if n: self.key_down(n)

        def on_release(key):
            n = name_of(key)
            if n: self.key_up(n)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        log.info("Keyboard listener started (global: keys are captured even if the window is unfocused)")

    def key_down(self, name):
        now = time.monotonic()
        if name in MOVE_KEYS:
            with self._lock:
                fresh = name not in self._held
                self._held[name] = now + (self.cfg["initial_repeat_grace_s"] if fresh
                                          else self.cfg["deadman_s"])
            return
        # Everything below is edge-triggered: ignore auto-repeat while the key stays down.
        with self._lock:
            if name in self._edge_down:
                return
            self._edge_down.add(name)
            if name == "space":
                self._inhibit = True
                self._held.clear()
                self._actions.clear()
                self._actions.append("stop")
            elif name in ("+", "="):
                self._set_scale(self.scale + self.cfg["speed_scale"]["step"])
            elif name in ("-", "_"):
                self._set_scale(self.scale - self.cfg["speed_scale"]["step"])
            elif name == "r":
                self._queue_once("stand")
            elif name == "f":
                self._queue_once("down")
            elif name == "x":
                self._damp_armed_until = now + self.cfg["damp_confirm_s"]
                self.status = f"DAMP armed: press Y within {self.cfg['damp_confirm_s']:.0f}s (dog will collapse)"
                log.warning(self.status)
            elif name == "y":
                if now <= self._damp_armed_until:
                    self._damp_armed_until = 0.0
                    self._queue_once("damp")
            elif name == "esc":
                self._held.clear()
                self._actions.clear()
                self.quit_event.set()

    def key_up(self, name):
        with self._lock:
            self._edge_down.discard(name)
            self._held.pop(name, None)
            if not self._held:
                self._inhibit = False

    def _queue_once(self, action):
        if action not in self._actions:
            self._actions.append(action)

    def _set_scale(self, v):
        sc = self.cfg["speed_scale"]
        self.scale = round(clamp(v, sc["min"], sc["max"]), 3)
        log.info("Speed scale %.1f", self.scale)

    # ---------------------------------------------------------------- command computation
    def _compute(self):
        now = time.monotonic()
        axes = [0.0, 0.0, 0.0]
        with self._lock:
            for k in [k for k, exp in self._held.items() if exp < now]:
                del self._held[k]          # deadman: key event too old -> treat as released
            if self._inhibit:
                return 0.0, 0.0, 0.0
            for k in self._held:
                ax, sign = MOVE_KEYS[k]
                axes[ax] += sign
        return tuple(clamp(a, -1, 1) * lim * self.scale for a, lim in zip(axes, self.limits))

    def snapshot(self):
        return {"vx": self._cmd[0], "vy": self._cmd[1], "vyaw": self._cmd[2],
                "scale": self.scale, "status": self.status}

    # ---------------------------------------------------------------- threads
    def _control_loop(self):
        period = 1.0 / self.cfg["rate_hz"]
        next_t = time.monotonic()
        moving = False
        last_err = 0.0
        while not self._stop.is_set():
            self._heartbeat = time.monotonic()
            try:
                if self._actions:
                    with self._lock:
                        action = self._actions.popleft() if self._actions else None
                    if action:
                        moving = False
                        self._run_action(action)
                        next_t = time.monotonic()
                vx, vy, wz = self._compute()
                if vx or vy or wz:
                    code = self._client.Move(vx, vy, wz)
                    moving = True
                    self.status = "moving"
                    if code not in (0, None) and time.monotonic() - last_err > 2.0:
                        last_err = time.monotonic()
                        log.warning("Move returned code %s (is the dog standing? press R)", code)
                elif moving:
                    self._client.StopMove()   # key released / deadman expired -> stop once
                    moving = False
                    self.status = "idle"
                self._cmd = (vx, vy, wz)
            except Exception as e:
                log.error("control loop error: %s -> StopMove", e)
                self._cmd = (0.0, 0.0, 0.0)
                moving = False
                try:
                    self._client.StopMove()
                except Exception:
                    pass
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()      # fell behind: don't burst-catch-up

    def _run_action(self, action):
        c = self._client
        self._busy = True
        self._cmd = (0.0, 0.0, 0.0)
        try:
            log.info("Action: %s", action)
            self.status = action
            if action == "stop":
                c.StopMove()
            elif action == "stand":
                c.StopMove(); c.StandUp(); c.BalanceStand()
            elif action == "down":
                c.StopMove(); c.StandDown()
            elif action == "damp":
                log.warning("Damp confirmed: dog will go limp")
                c.Damp()
            self.status = "idle"
        finally:
            self._busy = False
            self._heartbeat = time.monotonic()

    def _watchdog_loop(self):
        stall = self.cfg["loop_stall_s"]
        while not self._stop.wait(0.05):
            late = time.monotonic() - self._heartbeat
            if late > stall and not self._busy:
                if not self._stall_reported:
                    self._stall_reported = True
                    log.error("Control loop stalled %.2fs -> StopMove", late)
                    try:
                        self._client.StopMove()
                    except Exception as e:
                        log.error("watchdog StopMove failed: %s", e)
                    self._cmd = (0.0, 0.0, 0.0)
            else:
                self._stall_reported = False


def _main():
    import yaml, os, sys
    ap = argparse.ArgumentParser(description="Teleop-only test")
    ap.add_argument("--iface")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--max-speed", type=float)
    ap.add_argument("--dry-run", action="store_true", help="print commands, do not touch DDS/robot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(open(args.config))
    client = None
    if args.dry_run:
        client = FakeSportClient()
    else:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        ChannelFactoryInitialize(0, args.iface or cfg["network"]["iface"])
    t = Teleop(cfg["control"], client, args.max_speed)
    try:
        t.start()
        while not t.quit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        t.shutdown()


if __name__ == "__main__":
    _main()
