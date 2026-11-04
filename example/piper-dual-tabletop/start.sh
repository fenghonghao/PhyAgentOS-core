#!/usr/bin/env bash
# Prepare the skill_server venv (outside the bundle: archives must not contain venvs) and run
# pre-flight checks. Needs no sudo; prints what needs it.
# PAOS runs this as the bundle start hook on every `paos skill start` with PAOS_SKILL_PROFILE set;
# sim-* profiles skip the CAN/camera checks. Manually: ./start.sh [--sim]
# Dependencies are (re)installed only when the requirement files change, so a start does not
# depend on the network once the venv is built.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SKILL_SERVER_VENV:-$HERE/../.venvs/piper-dual-tabletop-skill-server}"
SIM=0
[ "${1:-}" = "--sim" ] && SIM=1
case "${PAOS_SKILL_PROFILE:-}" in sim-*) SIM=1 ;; esac
echo "== profile: ${PAOS_SKILL_PROFILE:-manual} (sim=$SIM)"
fail=0
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== skill_server venv: $VENV"
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q --python 3.12 "$VENV" || bad "uv venv failed"
fi
REQ="$HERE/nodes/skill_server"
WANT="$(cat "$REQ/requirements.lock" "$REQ/requirements-sam3.txt" "$REQ/requirements-hf.txt" "$REQ/requirements-orbbec.txt" | sha256sum | cut -c1-16)"
HAVE="$(cat "$VENV/.req-hash" 2>/dev/null || true)"
if [ "$WANT" = "$HAVE" ]; then
  ok "deps up to date (requirements hash $WANT)"
else
  uv pip install -q --python "$VENV/bin/python" -r "$REQ/requirements.lock" \
    && uv pip install -q --python "$VENV/bin/python" -r "$REQ/requirements-sam3.txt" \
    && uv pip install -q --python "$VENV/bin/python" -r "$REQ/requirements-hf.txt" \
    && uv pip install -q --python "$VENV/bin/python" --no-deps -r "$REQ/requirements-orbbec.txt" \
    && echo "$WANT" > "$VENV/.req-hash" && ok "deps installed (pinned + SAM3: torch cu128, transformers)" \
    || bad "dependency install failed (offline? pre-fill the uv cache)"
fi
SAM3_DIR="${SAM3_DIR:-$HOME/.PhyAgentOS/models/sam3}"
[ -s "$SAM3_DIR/model.safetensors" ] && [ -s "$SAM3_DIR/config.json" ] && [ -s "$SAM3_DIR/tokenizer.json" ] \
  && ok "SAM3 weights in $SAM3_DIR" || bad "SAM3 weights missing: run $HERE/nodes/skill_server/download_sam3.sh $SAM3_DIR"
"$VENV/bin/python" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
  && ok "torch CUDA available" || warn "torch CUDA not available: SAM3 would run on CPU (slow)"
PYTHONPATH="$HERE/nodes/skill_server/src" "$VENV/bin/python" -c \
  "import dora, forge_msgs, numpy, yaml, forge_tool, tabletop.dora_node" 2>/dev/null \
  && ok "node imports" || bad "node imports"

echo "== tools"
command -v dora >/dev/null && ok "$(dora --version | head -1)" || bad "dora CLI missing"
command -v "${PAOS_BIN:-paos}" >/dev/null && ok "paos found" || warn "paos not on PATH (set PAOS_BIN)"

if [ "$SIM" -eq 0 ]; then
  CAMERA_ROOT="${CAMERA_NODES_DIR:-$HOME/.PhyAgentOS/camera-nodes}"
  if [ ! -f "$CAMERA_ROOT/orbbec-camera-1.1.0-shmfix/cv2/__init__.py" ]; then
    SKILL_SERVER_VENV="$VENV" bash "$HERE/scripts/setup_guarded_camera.sh" \
      || bad "guarded camera dependency install failed"
  fi
  "$VENV/bin/python" "$HERE/scripts/patch_orbbec_shutdown.py" "$CAMERA_ROOT/orbbec-camera-1.1.0-shmfix" \
    || bad "camera shutdown patch failed"
  echo "== CAN"
  dpkg -s can-utils >/dev/null 2>&1 && ok "can-utils" || bad "can-utils missing: sudo apt install can-utils"
  # the CAN interface(s) named in the profile's driver config(s) (single booth arm: piper.yaml)
  for drv in "$HERE/profiles/${PAOS_SKILL_PROFILE:-real-piper}"/piper*.yaml; do
    [ -f "$drv" ] || continue
    ifc="$(awk '/^port:/{print $2; exit}' "$drv")"
    if ip link show "$ifc" >/dev/null 2>&1; then
      state="$(ip -details link show "$ifc" | grep -o 'state [A-Z]*' | head -1)"
      rate="$(ip -details link show "$ifc" | grep -o 'bitrate [0-9]*' | head -1)"
      [ "$state" = "state UP" ] && [ "$rate" = "bitrate 1000000" ] && ok "$ifc $state $rate" \
        || bad "$ifc $state $rate (activate: piper_sdk can_activate.sh $ifc 1000000 <usb-port>)"
    else
      bad "$ifc missing (name it with piper_sdk can_activate.sh by USB port)"
    fi
  done
  echo "== cameras"
  id -nG | grep -qw video && ok "user in video group" \
    || warn "this session is not in the video group (after usermod, log out and in again); cameras may lack access"
  echo "== camera exposure"
  # Fixed for this run, measured from the head camera's auto exposure. The scale brings the image
  # to the brightness chosen in the booth: auto gave about 1700 us, 2500 us was chosen (2026-11-04).
  RUN_DIR="$HOME/.PhyAgentOS/run/piper-dual-tabletop"; mkdir -p "$RUN_DIR"
  if out="$(cd "$RUN_DIR" && "$VENV/bin/python" "$HERE/scripts/lock_exposure.py" \
      "$HERE/profiles/${PAOS_SKILL_PROFILE:-real-piper}/camera_head.yaml" camera_exposure_us \
      --scale "${EXPOSURE_SCALE:-1.47}" 2>&1)"; then
    ok "$(printf '%s\n' "$out" | grep '^exposure:')"
  else
    bad "exposure not measured: $(printf '%s\n' "$out" | tail -1)"
  fi
fi

echo "== GPU"
command -v nvidia-smi >/dev/null && ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" \
  || warn "no NVIDIA GPU visible"

if [ "$SIM" -eq 0 ] && [ "$fail" -eq 0 ]; then
  echo "== live view"
  # Only a page to watch: when it does not start the skill still does.
  if out="$(SKILL_SERVER_VENV="$VENV" bash "$HERE/nodes/run_live_view.sh" 2>&1)"; then ok "$out"; else warn "$out"; fi
fi

[ "$fail" -eq 0 ] && { echo "PRE-FLIGHT OK"; exit 0; } || { echo "PRE-FLIGHT FAILED"; exit 1; }
