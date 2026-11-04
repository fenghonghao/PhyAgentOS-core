#!/usr/bin/env bash
# Launch a camera adapter release binary (not in the PAOS registry; Dora 0.4.1 builds).
#   run_camera.sh orbbec|realsense|usb --config <file>
# Binaries live in ${CAMERA_NODES_DIR:-~/.PhyAgentOS/camera-nodes}; fetch them with
# scripts/fetch_camera_nodes.sh (sha256-verified GitHub releases).
set -euo pipefail
KIND="${1:?usage: run_camera.sh orbbec|realsense|usb --config <file>}"; shift
ROOT="${CAMERA_NODES_DIR:-$HOME/.PhyAgentOS/camera-nodes}"
# The start hook saves the fixed exposure of this run (scripts/lock_exposure.py). The adapter reads
# it only from its config file, so put it into a copy of the config.
RUN_DIR="$HOME/.PhyAgentOS/run/piper-vla"
if [ "$KIND" = orbbec ] && [ "${1:-}" = --config ] && [ -s "$RUN_DIR/camera_exposure_us" ]; then
  CONFIG="$RUN_DIR/$(basename "$2")"
  sed -E "s/(exposure_us: *)[0-9]+/\1$(cat "$RUN_DIR/camera_exposure_us")/" "$2" > "$CONFIG"
  set -- --config "$CONFIG"
fi
case "$KIND" in
  orbbec)    exec "$ROOT/orbbec_camera-v1.1.0-ubuntu20.04-x86_64/orbbec_camera" "$@" ;;
  realsense) exec "$ROOT/realsense_camera-v1.1.0-ubuntu20.04-x86_64/realsense_camera" "$@" ;;
  usb)       exec "$ROOT/usb_camera-v1.0.7-ubuntu20.04-x86_64/usb_camera" run "$@" ;;
  *) echo "unknown camera kind: $KIND" >&2; exit 2 ;;
esac
