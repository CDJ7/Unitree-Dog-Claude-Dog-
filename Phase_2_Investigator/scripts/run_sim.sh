#!/usr/bin/env bash
# One command to launch the simulator in any mode.
#
#   ./scripts/run_sim.sh --scenario 1                    # teleop in the MuJoCo viewer (default mode)
#   ./scripts/run_sim.sh --scenario 1 --mode sensors     # live sensor viewer, drive with the keys
#   ./scripts/run_sim.sh --mode teleop-cv                # teleop in an OpenCV window (if the viewer misbehaves)
#   ./scripts/run_sim.sh --mode check                    # headless A1 acceptance check
#   ./scripts/run_sim.sh --env scenesmith_house_186 --mode sensors
#
# Modes: teleop | teleop-cv | sensors | check   (explore, inspect, playback, record arrive in A4-A6)
# Other options (--scenario, --env, --spawn X Y YAW, --snapshot ...) are passed to the script.
# Environment: PYTHON=... to choose the interpreter (default: $VIRTUAL_ENV, else ~/go2venv, else python3);
# MUJOCO_GL=egl|glfw|osmesa to override the offscreen renderer.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

MODE=teleop
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --mode=*) MODE="${1#*=}"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -n "${PYTHON:-}" ]]; then PY="$PYTHON"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then PY="$VIRTUAL_ENV/bin/python"
elif [[ -x "$HOME/go2venv/bin/python" ]]; then PY="$HOME/go2venv/bin/python"
else PY=python3; fi

if [[ ! -f third_party/robot-unitree-go2-mujoco/sim/native/controller.py ]]; then
  echo "simulator submodule missing: run ./scripts/setup.sh first" >&2; exit 1
fi

case "$MODE" in
  teleop)    exec "$PY" scripts/teleop.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  teleop-cv) exec "$PY" scripts/teleop.py --cv "${ARGS[@]+"${ARGS[@]}"}" ;;
  sensors)   exec "$PY" scripts/sensor_viewer.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  check)     exec "$PY" scripts/check_sim.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  explore|inspect|playback|record)
    echo "mode '$MODE' is not built yet (Stage A items A4-A6, see docs/plan.md)" >&2; exit 2 ;;
  *) echo "unknown mode '$MODE' (teleop | teleop-cv | sensors | check)" >&2; exit 2 ;;
esac
