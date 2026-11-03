#!/usr/bin/env bash
# Copy a trained checkpoint to the model store and print the vla_policy.yaml entry (with the
# weights' sha256, checked by the policy node at load) to paste into the profile before sealing.
#
#   export_model.sh <run_dir | pretrained_model dir> <policy-name> ["description"] ["task text"]
set -euo pipefail
SRC="${1:?run dir or pretrained_model dir}"
NAME="${2:?policy name}"
DESC="${3:-}"
TASK="${4:-}"
STORE="${VLA_MODEL_STORE:-$HOME/.PhyAgentOS/models/piper-vla}"
if [ -f "$SRC/model.safetensors" ]; then PM="$SRC"; else PM="$SRC/checkpoints/last/pretrained_model"; fi
[ -f "$PM/model.safetensors" ] || { echo "no model.safetensors under $PM"; exit 1; }
DST="$STORE/$NAME"
mkdir -p "$STORE"
rm -rf "$DST.tmp"
cp -rL "$PM" "$DST.tmp"
rm -rf "$DST"
mv "$DST.tmp" "$DST"
SHA="$(sha256sum "$DST/model.safetensors" | cut -d' ' -f1)"
TYPE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["type"])' "$DST/config.json")"
echo "== exported $TYPE policy to $DST"
cat <<EOF
# paste under 'policies:' in profiles/real-piper/vla_policy.yaml
  - name: $NAME
    path: $DST
    sha256: $SHA
    description: "$DESC"
    task: "$TASK"
EOF
