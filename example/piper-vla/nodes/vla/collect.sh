#!/usr/bin/env bash
# Teleoperated data collection (outside the competition run; stop the paos skill first, the
# drivers own the CAN buses).
#
#   collect.sh real|sim [--dataset NAME] [--task "text"]...
#
# Opens the operator console (camera previews + status). Keys: see nodes/vla/src/piper_vla/ui.py
# (W/S A/D R/F move, Q/E yaw, T/G tilt, Space gripper, 1/2 arm, H home, Enter record/save,
# Backspace discard, Esc quit). Episodes append to ~/.PhyAgentOS/vla/datasets/<NAME>.
# Ctrl-C after Esc stops the drivers (they return to their reset pose).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TARGET="${1:?usage: collect.sh real|sim [--dataset NAME] [--task TEXT]...}"; shift
DATASET=""; TASKS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --task) TASKS+=("$2"); shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done

nodebin() {  # newest installed Forge node executable by name
  find "$HOME/.PhyAgentOS/forge_runtime/nodes/$1/versions" -type f -name "$1" -perm -u+x 2>/dev/null | sort | tail -1
}
case "$TARGET" in
  real) PROFILE="$ROOT/profiles/real-piper"; TEMPLATE="$HERE/collect/real-piper.yaml"
        PIPER_DRIVER="$(nodebin robots_agilex_piper)"; [ -n "$PIPER_DRIVER" ] || { echo "robots_agilex_piper not installed (paos skill install / forge-node install)"; exit 1; }
        MUJOCO_SIM="" ;;
  sim)  PROFILE="$ROOT/profiles/sim-piper"; TEMPLATE="$HERE/collect/sim-piper.yaml"
        MUJOCO_SIM="$(nodebin mujoco_sim)"; [ -n "$MUJOCO_SIM" ] || { echo "mujoco_sim not installed"; exit 1; }
        PIPER_DRIVER="" ;;
  *) echo "target must be real or sim"; exit 2 ;;
esac
command -v dora >/dev/null || { echo "dora CLI not on PATH"; exit 1; }
[ -n "${DISPLAY:-}" ] || { echo "no DISPLAY: the operator console needs a desktop session"; exit 1; }

# task texts go through one YAML double-quoted env value, joined by \n
TASKS_ESC=""
for t in "${TASKS[@]+"${TASKS[@]}"}"; do
  t="${t//\\/\\\\}"; t="${t//\"/\\\"}"
  t="${t//&/\\&}"; t="${t//|/\\|}"   # sed replacement specials
  TASKS_ESC="${TASKS_ESC:+$TASKS_ESC\\n}$t"
done
OUT="$PROFILE/collect.local.yaml"     # rendered next to the profile's device configs (node cwd)
sed -e "s|@PIPER_DRIVER@|$PIPER_DRIVER|g" -e "s|@MUJOCO_SIM@|$MUJOCO_SIM|g" \
    -e "s|@NODES@|$ROOT/nodes|g" -e "s|@PROFILE@|$PROFILE|g" -e "s|@SIMPROFILE@|$PROFILE|g" \
    -e "s|@HERE@|$HERE/collect|g" -e "s|@DISPLAY@|$DISPLAY|g" -e "s|@DATASET@|$DATASET|g" \
    -e "s|@TASKS@|$TASKS_ESC|g" "$TEMPLATE" > "$OUT"
echo "== collecting ($TARGET) dataset=${DATASET:-<config default>} -> ~/.PhyAgentOS/vla/datasets"
cd "$PROFILE"
exec dora run "$OUT"
