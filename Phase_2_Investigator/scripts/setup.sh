#!/usr/bin/env bash
# Phase 2 Investigator setup / check script. Never uses sudo.
#
#   ./scripts/setup.sh --check              read-only checks, changes nothing
#   ./scripts/setup.sh                      checks + simulator submodule + core Python packages
#   ./scripts/setup.sh --with-ml            ... + torch, RF-DETR, SAM, GUI packages
#   ./scripts/setup.sh --venv ~/go2venv     choose the venv (default: $VIRTUAL_ENV, else ~/go2venv)
#
# Anything that needs sudo (apt packages) is printed as a suggestion, not run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"     # Phase_2_Investigator/
REPO="$(cd "$HERE/.." && pwd)"
SIM_PATH="Phase_2_Investigator/third_party/robot-unitree-go2-mujoco"
SIM_COMMIT="d6e54647541fb7b048db5d9817457ee11d0e2cf9"
VENV="${VIRTUAL_ENV:-$HOME/go2venv}"
CHECK_ONLY=0
WITH_ML=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) CHECK_ONLY=1 ;;
    --with-ml) WITH_ML=1 ;;
    --venv) VENV="$2"; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; WARNINGS=$((WARNINGS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILS=$((FAILS+1)); }
WARNINGS=0; FAILS=0
SUGGEST=()

echo "== System"
. /etc/os-release 2>/dev/null && ok "$PRETTY_NAME" || warn "cannot read /etc/os-release"
[[ "${VERSION_ID:-}" == "22.04" ]] || warn "tested target is Ubuntu 22.04"
command -v git >/dev/null && ok "$(git --version)" || bad "git missing (sudo apt install git)"
IS_WSL=0
if grep -qi microsoft /proc/version 2>/dev/null || [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
  IS_WSL=1
  if [[ -d /run/WSL || "$(uname -r)" == *WSL2* || "$(uname -r)" == *microsoft-standard* ]]; then
    ok "WSL2 (${WSL_DISTRO_NAME:-unknown distro}, kernel $(uname -r))"
  else
    bad "this looks like WSL1: convert it in PowerShell with  wsl --set-version ${WSL_DISTRO_NAME:-Ubuntu-22.04} 2"
  fi
  case "$HERE" in
    /mnt/*) warn "repo is on the Windows filesystem ($HERE): very slow; clone it under ~ instead" ;;
    *) ok "repo is on the Linux filesystem" ;;
  esac
  if [[ -d /mnt/wslg && ( -n "${WAYLAND_DISPLAY:-}" || -n "${DISPLAY:-}" ) ]]; then
    ok "WSLg present (DISPLAY=${DISPLAY:-unset}): the native MuJoCo viewer can open windows"
  else
    warn "WSLg not detected: run 'wsl --update' in PowerShell, then 'wsl --shutdown'"
  fi
fi
for T in node uv docker; do
  if command -v $T >/dev/null; then ok "$T: $($T --version 2>&1 | head -1)"
  else warn "$T not installed (optional: only needed for the simulator's Docker/Robonix route)"; fi
done
if command -v docker >/dev/null; then
  docker compose version >/dev/null 2>&1 && ok "$(docker compose version | head -1)" || warn "docker compose plugin missing"
fi

echo "== GPU and OpenGL"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)"
elif [[ $IS_WSL -eq 1 ]]; then
  warn "nvidia-smi not working in WSL: update the NVIDIA driver on WINDOWS (never install a Linux NVIDIA driver inside WSL)"
else
  warn "nvidia-smi not working: RF-DETR training will be CPU-only (very slow)"
fi
if command -v glxinfo >/dev/null; then
  R="$(glxinfo -B 2>/dev/null | grep -m1 'OpenGL renderer' || true)"
  if [[ "$R" == *llvmpipe* || -z "$R" ]]; then warn "OpenGL renderer: ${R:-none} (software rendering)"; else ok "$R"; fi
else
  warn "glxinfo not installed (only needed for this check)"; SUGGEST+=("sudo apt install mesa-utils")
fi
if ldconfig -p 2>/dev/null | grep -q 'libEGL.so'; then ok "libEGL present (headless rendering)"
else warn "libEGL not found"; SUGGEST+=("sudo apt install libegl1 libgl1"); fi

echo "== Python ($VENV)"
PY="$VENV/bin/python"
if [[ -x "$PY" ]]; then
  V="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then ok "Python $V"
  else bad "Python $V: need 3.10 or newer (rfdetr)"; fi
else
  bad "no venv at $VENV (python3 -m venv $VENV)"
fi

echo "== Simulator submodule ($SIM_PATH)"
if [[ -f "$REPO/$SIM_PATH/sim/native/controller.py" ]]; then
  HEAD="$(git -C "$REPO/$SIM_PATH" rev-parse HEAD 2>/dev/null || echo unknown)"
  [[ "$HEAD" == "$SIM_COMMIT" ]] && ok "present at pinned commit ${SIM_COMMIT:0:7}" || warn "present at $HEAD (pinned ${SIM_COMMIT:0:7})"
elif [[ $CHECK_ONLY -eq 1 ]]; then
  warn "not fetched yet (run ./scripts/setup.sh without --check)"
else
  echo "  fetching (shallow, ~880 MB working tree)..."
  git -C "$REPO" submodule update --init --depth 1 "$SIM_PATH" \
    || git -C "$REPO" submodule update --init "$SIM_PATH"
  ok "fetched at $(git -C "$REPO/$SIM_PATH" rev-parse --short HEAD)"
fi

if [[ $CHECK_ONLY -eq 0 && -x "$PY" ]]; then
  echo "== Python packages"
  "$PY" -m pip install --upgrade pip >/dev/null
  "$PY" -m pip install -r "$HERE/requirements.txt"
  # versions pinned by the simulator (sim/bridge/requirements.txt)
  "$PY" -m pip install "mujoco==3.3.7" "onnxruntime==1.23.2" websockets
  if [[ $WITH_ML -eq 1 ]]; then
    "$PY" -m pip install torch torchvision
    "$PY" -m pip install "rfdetr[train,loggers]" segment-anything pycocotools gdown opencv-python \
      fastapi "uvicorn[standard]"
  fi
  ok "packages installed"
fi

if [[ -x "$PY" ]] && "$PY" -c "import torch" 2>/dev/null; then
  T="$("$PY" -c 'import torch; print(torch.__version__, "CUDA" if torch.cuda.is_available() else "no CUDA", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
  [[ "$T" != *"no CUDA"* ]] && ok "PyTorch $T" || warn "PyTorch $T (Stage B training needs CUDA)"
fi

if [[ -x "$PY" ]] && "$PY" -c "import mujoco" 2>/dev/null; then
  echo "== MuJoCo rendering smoke test"
  for GL in egl glfw osmesa; do
    if MUJOCO_GL=$GL "$PY" - <<'EOF' 2>/dev/null; then ok "MUJOCO_GL=$GL renders offscreen"; break; else warn "MUJOCO_GL=$GL failed"; fi
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 2'/><geom type='sphere' size='.1'/></worldbody></mujoco>")
r = mujoco.Renderer(m, 120, 160)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r.update_scene(d)
assert r.render().mean() > 0
r.close()
EOF
  done
fi

echo
if [[ ${#SUGGEST[@]} -gt 0 ]]; then
  echo "Suggested (needs sudo, not run):"
  printf '  %s\n' "${SUGGEST[@]}"
fi
echo "Done: $FAILS failure(s), $WARNINGS warning(s)."
[[ $FAILS -eq 0 ]]
