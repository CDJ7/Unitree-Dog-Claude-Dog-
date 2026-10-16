#!/usr/bin/env python3
"""Try ONE action on the robot by itself, before trusting it inside a mission.

    python behavior/try_action.py --iface enp0s31f6 sit        # asks you to type YES first
    python behavior/try_action.py --dry-run hello              # prints the calls, no robot

The dog must already be standing on open floor with the Unitree remote in your hand. Only these actions exist:
sit (Sit, hold, RiseSit, BalanceStand), hello, stretch, content.
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from behavior.control import ALLOWED_ACTIONS  # noqa: E402

log = logging.getLogger("try_action")


def main():
    ap = argparse.ArgumentParser(description="Run one allowlisted action")
    ap.add_argument("action", choices=ALLOWED_ACTIONS)
    ap.add_argument("--iface")
    ap.add_argument("--hold", type=float, default=4.0, help="sit: seconds to stay sitting")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.dry_run:
        from teleop import FakeSportClient
        c = FakeSportClient()
    else:
        if not args.iface:
            sys.exit("--iface is required (or use --dry-run)")
        print(f"About to make the REAL robot do: {args.action}. It must be standing, with clear floor around it.")
        if input("Type YES to continue: ").strip() != "YES":
            sys.exit("cancelled")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient
        ChannelFactoryInitialize(0, args.iface)
        c = SportClient()
        c.SetTimeout(10.0)
        c.Init()
    try:
        c.StopMove()
        if args.action == "sit":
            log.info("Sit -> code %s", c.Sit())
            time.sleep(args.hold)
            log.info("RiseSit -> code %s", c.RiseSit())
            log.info("BalanceStand -> code %s", c.BalanceStand())
        else:
            fn = {"hello": c.Hello, "stretch": c.Stretch, "content": c.Content}[args.action]
            log.info("%s -> code %s", args.action, fn())
            log.info("BalanceStand -> code %s", c.BalanceStand())
    finally:
        try:
            c.StopMove()
        except Exception:
            pass


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
