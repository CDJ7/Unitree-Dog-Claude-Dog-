#!/usr/bin/env bash
# One command to launch the simulator in any mode.
#
#   ./scripts/run_sim.sh --scenario 1                    # teleop in the MuJoCo viewer (default mode)
#   ./scripts/run_sim.sh --scenario 1 --mode sensors     # live sensor viewer, drive with the keys
#   ./scripts/run_sim.sh --scenario 1 --mode explore --viewer    # autonomous exploration, watch it live
#   ./scripts/run_sim.sh --scenario 1 --mode inspect     # explore + patrol + inspect the stain regions
#   ./scripts/run_sim.sh --scenario 1 --mode playback --viewer   # event reenactment (or MP4 without --viewer)
#   ./scripts/run_sim.sh --scenario 2 --mode record      # explore while recording camera/depth/LiDAR/poses
#   ./scripts/run_sim.sh --mode dataset                  # COCO dataset of stain regions
#   ./scripts/run_sim.sh --scenario 3 --mode render      # overview sheet of a scenario
#   ./scripts/run_sim.sh --random 7 --mode explore       # a seeded random scenario
#   ./scripts/run_sim.sh --mode check                    # headless A1 acceptance check
#
# Modes: teleop | teleop-simple | sensors | explore | inspect | record | playback | dataset | render | check
# Other options (--scenario, --random, --env, --spawn X Y YAW, --viewer, --max-time ...) go to the script.
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
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
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
  teleop-simple|teleop-cv) exec "$PY" scripts/teleop.py --simple "${ARGS[@]+"${ARGS[@]}"}" ;;
  sensors)   exec "$PY" scripts/sensor_viewer.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  check)     exec "$PY" scripts/check_sim.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  explore)   exec "$PY" scripts/explore.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  inspect)   exec "$PY" scripts/explore.py --inspect "${ARGS[@]+"${ARGS[@]}"}" ;;
  record)    exec "$PY" scripts/explore.py --record "${ARGS[@]+"${ARGS[@]}"}" ;;
  playback)  exec "$PY" scripts/playback.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  dataset)   exec "$PY" training/build_coco_dataset.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  render)    exec "$PY" scripts/render_scenario.py "${ARGS[@]+"${ARGS[@]}"}" ;;
  *) echo "unknown mode '$MODE' (teleop | teleop-simple | sensors | explore | inspect | record | playback | dataset | render | check)" >&2; exit 2 ;;
esac
