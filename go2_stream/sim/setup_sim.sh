#!/usr/bin/env bash
# One-time setup for the MuJoCo sim: get Unitree's Go2 model and the mujoco Python package.
# Run inside your venv:  source ~/go2venv/bin/activate && ./sim/setup_sim.sh
set -euo pipefail
DEST=${UNITREE_MUJOCO:-$HOME/unitree_mujoco}
if [[ ! -d "$DEST/.git" ]]; then
  git clone https://github.com/unitreerobotics/unitree_mujoco.git "$DEST"
else
  echo "unitree_mujoco already at $DEST"
fi
pip install mujoco matplotlib
echo
echo "Go2 model: $DEST/unitree_robots/go2/go2.xml"
python - <<PY
import mujoco; print("mujoco", mujoco.__version__)
PY
echo "Next: python run_sim.py"
