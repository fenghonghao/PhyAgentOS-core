#!/usr/bin/env bash
# Bundle start hook (PAOS runs it on every `paos skill start` with PAOS_SKILL_PROFILE set):
# builds the VLA venv outside the bundle (archives must not contain venvs), checks every policy
# listed in the profile's vla_policy.yaml (weights present, sha256), and on the real robot the
# CAN interface and camera access. Manually: ./start.sh [--sim]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${PAOS_SKILL_PROFILE:-real-piper}"
SIM=0
[ "${1:-}" = "--sim" ] && { SIM=1; PROFILE="${PAOS_SKILL_PROFILE:-sim-piper}"; }
case "$PROFILE" in sim-*) SIM=1 ;; esac
echo "== profile: $PROFILE (sim=$SIM)"
fail=0
ok()   { printf '  [ok]   %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }
warn() { printf '  [warn] %s\n' "$*"; }

echo "== VLA venv"
VLA_VENV="${VLA_VENV:-$HERE/../.venvs/piper-vla}"
VLA_VENV="$VLA_VENV" "$HERE/nodes/vla/setup_venv.sh" >/tmp/paos-vla-setup.log 2>&1 \
  && ok "VLA venv, tokenizer cache, ACT backbone ($VLA_VENV)" || bad "VLA setup failed: see /tmp/paos-vla-setup.log"
PYTHONPATH="$HERE/nodes/vla/src" "$VLA_VENV/bin/python" -c \
  "import dora, forge_msgs, forge_tool, piper_vla.tool_server, piper_vla.policy_node" 2>/dev/null \
  && ok "node imports" || bad "node imports"

echo "== policies (profiles/$PROFILE/vla_policy.yaml)"
"$VLA_VENV/bin/python" - "$HERE/profiles/$PROFILE/vla_policy.yaml" <<'PY' || bad "vla_policy.yaml policies"
import hashlib, sys, yaml
from pathlib import Path
pols = (yaml.safe_load(open(sys.argv[1])) or {}).get("policies") or []
if not pols:
    print("  [warn] no policies in vla_policy.yaml: vla.run has nothing to run")
bad = 0
for p in pols:
    w = Path(p["path"]).expanduser() / "model.safetensors"
    if not w.exists():
        print(f"  [FAIL] policy {p['name']}: {w} missing"); bad = 1; continue
    if p.get("sha256"):
        h = hashlib.sha256()
        with w.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != p["sha256"]:
            print(f"  [FAIL] policy {p['name']}: sha256 mismatch"); bad = 1; continue
    print(f"  [ok]   policy {p['name']} ({w.stat().st_size / 1e6:.0f} MB)")
sys.exit(bad)
PY

echo "== tools"
command -v dora >/dev/null && ok "$(dora --version | head -1)" || bad "dora CLI missing"

if [ "$SIM" -eq 0 ]; then
  echo "== CAN"
  dpkg -s can-utils >/dev/null 2>&1 && ok "can-utils" || bad "can-utils missing: sudo apt install can-utils"
  ifc="$(awk '/^port:/{print $2; exit}' "$HERE/profiles/$PROFILE/piper.yaml")"
  if ip link show "$ifc" >/dev/null 2>&1; then
    state="$(ip -details link show "$ifc" | grep -o 'state [A-Z]*' | head -1)"
    rate="$(ip -details link show "$ifc" | grep -o 'bitrate [0-9]*' | head -1)"
    [ "$state" = "state UP" ] && [ "$rate" = "bitrate 1000000" ] && ok "$ifc $state $rate" \
      || bad "$ifc $state $rate (activate: piper_sdk can_activate.sh $ifc 1000000 <usb-port>)"
  else
    bad "$ifc missing (name it with piper_sdk can_activate.sh by USB port)"
  fi
  echo "== cameras"
  id -nG | grep -qw video && ok "user in video group" \
    || warn "this session is not in the video group (after usermod, log out and in again); cameras may lack access"
  echo "== camera exposure"
  # Fixed for this run, measured from the head camera's auto exposure. The scale brings the image
  # to the brightness chosen in the booth: auto gave about 1700 us, 2500 us was chosen (2026-11-04).
  RUN_DIR="$HOME/.PhyAgentOS/run/piper-vla"; mkdir -p "$RUN_DIR"
  if out="$(cd "$RUN_DIR" && "$VLA_VENV/bin/python" "$HERE/scripts/lock_exposure.py" \
      "$HERE/profiles/$PROFILE/camera_head.yaml" camera_exposure_us \
      --scale "${EXPOSURE_SCALE:-1.47}" 2>&1)"; then
    ok "$(printf '%s\n' "$out" | grep '^exposure:')"
  else
    bad "exposure not measured: $(printf '%s\n' "$out" | tail -1)"
  fi
fi

echo "== GPU"
command -v nvidia-smi >/dev/null && ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)" \
  || warn "no NVIDIA GPU visible"

[ "$fail" -eq 0 ] && { echo "PRE-FLIGHT OK"; exit 0; } || { echo "PRE-FLIGHT FAILED"; exit 1; }
