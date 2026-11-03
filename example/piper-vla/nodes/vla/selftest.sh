#!/usr/bin/env bash
# One-command check of a machine for the B plan (run first thing on the RTX 4090 workstation):
# synthetic 14-D dataset -> ACT 200 steps -> pi0.5 (from pi05_base) 20 steps -> load + infer
# each with the policy node's own loader, printing load time, GPU memory and per-step latency.
#   selftest.sh            ACT + pi0.5
#   selftest.sh act        ACT only (e.g. on the 8 GB laptop)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VLA_VENV:-$HOME/.PhyAgentOS/skills/.venvs/piper-vla}"
WHICH="${1:-all}"
WORK="${SELFTEST_DIR:-$HOME/.PhyAgentOS/vla/selftest}"
rm -rf "$WORK"; mkdir -p "$WORK/datasets" "$WORK/runs" "$WORK/models"
export PYTHONPATH="$HERE/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLA_DATA_ROOT="$WORK/datasets" VLA_OUT_ROOT="$WORK/runs" VLA_MODEL_STORE="$WORK/models"
PY="$VENV/bin/python"
echo "== synthetic dataset"
"$PY" -m piper_vla.selftest record "$WORK/datasets/selftest" 2>&1 | grep -v -E "Svt\[|Map:" | tail -4
echo "== ACT"
"$HERE/train.sh" act selftest --steps=200 --save_freq=200 --log_freq=100 2>&1 | tr '\r' '\n' \
  | grep -E "INFO.*step:|End of training|Error" | tail -4
"$HERE/export_model.sh" "$(ls -d "$WORK"/runs/selftest-act-*/ | tail -1)" st_act >/dev/null
"$PY" -m piper_vla.selftest infer "$WORK/models/st_act" 2>&1 | grep -E "cuda|loaded|step"
if [ "$WHICH" != act ]; then
  echo "== pi0.5 (bf16, expert only, from pi05_base)"
  "$HERE/train.sh" pi05 selftest --steps=20 --save_freq=20 --log_freq=10 --batch_size=4 2>&1 | tr '\r' '\n' \
    | grep -E "INFO.*step:|End of training|Error|Could not load|without loading" | tail -6
  "$HERE/export_model.sh" "$(ls -d "$WORK"/runs/selftest-pi05-*/ | tail -1)" st_pi05 >/dev/null
  "$PY" -m piper_vla.selftest infer "$WORK/models/st_pi05" 2>&1 | grep -E "cuda|loaded|step|Error"
fi
echo "== selftest done (work dir $WORK; delete it afterwards, the pi0.5 checkpoint is ~7 GB)"
