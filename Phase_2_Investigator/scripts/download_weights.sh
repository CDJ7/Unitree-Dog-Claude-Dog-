#!/usr/bin/env bash
# Download model weights into Phase_2_Investigator/weights/ (gitignored).
#
#   ./scripts/download_weights.sh            bloodstain SAM (fine-tuned vit_b) + stock SAM vit_b fallback
#
# The fine-tuned bloodstain SAM checkpoint (github.com/Zdong104/Bloodstain-Analysis-AI-Tool)
# is only hosted on Google Drive, with no published hash. We record the sha256 we
# got so later downloads can be compared. If Drive refuses (quota, login), the
# stock Meta SAM vit_b checkpoint is used instead (perception/bloodstain_sam.py
# picks whichever exists).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$HERE/weights/sam"
mkdir -p "$OUT"
PY="${VIRTUAL_ENV:-$HOME/go2venv}/bin/python"

TUNED="$OUT/bloodstain_sam_vit_b.pth"
STOCK="$OUT/sam_vit_b_01ec64.pth"
DRIVE_ID="1mvh81GEjU7Z2ExLVQyUotMUV0l84UpO0"
STOCK_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if [[ ! -s "$TUNED" ]]; then
  echo "Fetching fine-tuned bloodstain SAM from Google Drive..."
  if "$PY" -m gdown --version >/dev/null 2>&1; then
    "$PY" -m gdown "$DRIVE_ID" -O "$TUNED" || { echo "  Drive download failed"; rm -f "$TUNED"; }
  else
    echo "  gdown not installed (./scripts/setup.sh --with-ml); skipping"
  fi
fi
if [[ ! -s "$STOCK" ]]; then
  echo "Fetching stock SAM vit_b (fallback, ~375 MB)..."
  curl -L --fail -o "$STOCK" "$STOCK_URL"
fi
( cd "$OUT" && sha256sum ./*.pth > SHA256SUMS && cat SHA256SUMS )
echo "Weights in $OUT"
