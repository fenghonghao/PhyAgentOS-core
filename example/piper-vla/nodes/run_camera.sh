#!/usr/bin/env bash
# Launch a camera adapter release binary (not in the PAOS registry; Dora 0.4.1 builds).
#   run_camera.sh orbbec|realsense|usb --config <file>
# Binaries live in ${CAMERA_NODES_DIR:-~/.PhyAgentOS/camera-nodes}; fetch them with
# scripts/fetch_camera_nodes.sh (sha256-verified GitHub releases).
set -euo pipefail
KIND="${1:?usage: run_camera.sh orbbec|realsense|usb --config <file>}"; shift
ROOT="${CAMERA_NODES_DIR:-$HOME/.PhyAgentOS/camera-nodes}"
case "$KIND" in
  orbbec)    exec "$ROOT/orbbec_camera-v1.1.0-ubuntu20.04-x86_64/orbbec_camera" "$@" ;;
  realsense) exec "$ROOT/realsense_camera-v1.1.0-ubuntu20.04-x86_64/realsense_camera" "$@" ;;
  usb)       exec "$ROOT/usb_camera-v1.0.7-ubuntu20.04-x86_64/usb_camera" run "$@" ;;
  *) echo "unknown camera kind: $KIND" >&2; exit 2 ;;
esac
