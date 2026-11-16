"""Find the laptop's Ethernet interface on the dog's subnet (192.168.123.x).

WSL renames adapters on restart (eth1 -> eth0), so a name saved in config.yaml or typed with --iface can go stale
between lab sessions. Found in Test 2's lab runs (Test2/real_go2/sources.py); shared here so every Test 1 script
gets the same fix.
"""
import logging
import subprocess

log = logging.getLogger("netutil")
DOG_PREFIX = "192.168.123."


def dog_ifaces(prefix=DOG_PREFIX):
    """Interfaces with an IPv4 address on the dog's subnet (from `ip -o -4 addr`)."""
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr"], capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001 - no `ip` tool: nothing found
        return []
    return [ln.split()[1] for ln in out.splitlines() if len(ln.split()) > 3 and ln.split()[3].startswith(prefix)]


def resolve_iface(iface, found=None):
    """The interface to use: `iface` if it is on the dog's subnet; otherwise the ONLY interface that is (with a
    warning); otherwise `iface` unchanged (the caller's own checks then explain what is wrong)."""
    found = dog_ifaces() if found is None else found
    if iface in found or len(found) != 1:
        return iface
    log.warning("Interface '%s' has no %sx address; using '%s', which does (WSL renames adapters on restart).",
                iface, DOG_PREFIX, found[0])
    return found[0]
