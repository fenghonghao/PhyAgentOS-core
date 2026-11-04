#!/usr/bin/env bash
# Download the Dora-0.4.1 camera adapter releases, verify sha256, extract. Needs `gh` (logged in).
set -euo pipefail
DEST="${CAMERA_NODES_DIR:-$HOME/.PhyAgentOS/camera-nodes}"; mkdir -p "$DEST"; cd "$DEST"
fetch() {  # tag repo asset
  local tag="$1" repo="$2" asset="$3"
  [ -f "$asset" ] && sha256sum -c "$asset.sha256" >/dev/null 2>&1 && { echo "ok   $asset"; } || {
    gh release download "$tag" -R "$repo" -p "$asset" -p "$asset.sha256" --clobber
    sha256sum -c "$asset.sha256"; }
  local d="${asset%.tar.gz}"; mkdir -p "$d"; tar -xzf "$asset" -C "$d"
}
fetch v1.1.0 PhyAgentOS/adapter-camera-orbbec    orbbec_camera-v1.1.0-ubuntu20.04-x86_64.tar.gz
fetch v1.1.0 PhyAgentOS/adapter-camera-realsense realsense_camera-v1.1.0-ubuntu20.04-x86_64.tar.gz
fetch v1.0.7 PhyAgentOS/adapter-camera-usb       usb_camera-v1.0.7-ubuntu20.04-x86_64.tar.gz
ls -d */
