#!/usr/bin/env bash
# Configure a static IP on the Ethernet interface connected to the Go2.
# Usage:  ./setup_network.sh detect
#         ./setup_network.sh temp  <iface> [ip/24]   # not persistent; gone after reboot / link reset
#         ./setup_network.sh nmcli <iface> [ip/24]   # PERSISTENT NetworkManager profile
#         ./setup_network.sh remove <iface>          # delete the nmcli profile
#         ./setup_network.sh ping
set -euo pipefail
ROBOT_IP=192.168.123.161
DEFAULT_ADDR=192.168.123.222/24
CON=go2-direct

cmd=${1:-detect}; iface=${2:-}; addr=${3:-$DEFAULT_ADDR}

need_iface() { [[ -n "$iface" ]] || { echo "interface name required (try: $0 detect)"; exit 1; }; }

case "$cmd" in
  detect)
    ip -br link
    echo; ip -br addr
    echo; echo "Wired candidates:"; ls /sys/class/net | grep -E '^(en|eth)' || true
    ;;
  temp)
    need_iface
    sudo ip addr add "$addr" dev "$iface" || true
    sudo ip link set "$iface" up
    ip -br addr show "$iface"
    ;;
  nmcli)
    need_iface
    echo "This creates a PERSISTENT NetworkManager profile '$CON' on $iface ($addr, no gateway)."
    read -r -p "Continue? [y/N] " a; [[ "$a" == y ]] || exit 1
    sudo nmcli con add type ethernet ifname "$iface" con-name "$CON" \
         ipv4.method manual ipv4.addresses "$addr" ipv6.method disabled \
         ipv4.never-default yes connection.autoconnect no
    sudo nmcli con up "$CON"
    ;;
  remove)
    need_iface; sudo nmcli con delete "$CON"
    ;;
  ping)
    ping -c 4 -W 1 "$ROBOT_IP"
    ;;
  *) sed -n 2,9p "$0"; exit 1;;
esac
