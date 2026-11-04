#!/usr/bin/env bash
# Install the official adapter source beside the frozen adapter; keep Dora at 0.4.1.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${SKILL_SERVER_VENV:-$HERE/../.venvs/piper-dual-tabletop-skill-server}/bin/python"
ROOT="${CAMERA_NODES_DIR:-$HOME/.PhyAgentOS/camera-nodes}"
mkdir -p "$ROOT"
ARCHIVE="$ROOT/orbbec-source-v1.1.0.tar.gz"
curl --fail --location --output "$ARCHIVE" https://codeload.github.com/PhyAgentOS/adapter-camera-orbbec/tar.gz/refs/tags/v1.1.0
printf '%s  %s\n' f7fe4ffcce336d4a39ab22d2146ebe5a9bfd22a02ae5374efa862262a9e4e1b9 "$ARCHIVE" | sha256sum --check
uv pip install --python "$PYTHON" --target "$ROOT/orbbec-camera-1.1.0-shmfix" --no-deps \
  "$ARCHIVE" forge-common==1.0.1 forge-msgs==1.1.0 opencv-python-headless==4.11.0.86
"$PYTHON" "$HERE/scripts/patch_orbbec_shutdown.py" "$ROOT/orbbec-camera-1.1.0-shmfix"
