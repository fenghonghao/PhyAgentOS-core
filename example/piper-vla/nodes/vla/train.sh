#!/usr/bin/env bash
# Train an ACT or pi0.5 policy on a dataset recorded by collect.sh (LeRobot 0.6.1, same venv
# as inference so checkpoints load unchanged). Run on the RTX 4090 workstation.
#
#   train.sh act  <dataset> [--steps N] [--batch_size B] [extra lerobot-train args...]
#   train.sh pi05 <dataset> [...]
#
# Presets (override any with extra args; later args win):
#   act : chunk 50 (1.7 s at 30 fps), batch 16, 30k steps (~1 h on a 4090 at 320x240 x3)
#   pi05: from pi05_base, bfloat16, gradient checkpointing, action expert only (fits 24 GB),
#         batch 8, 3k steps
# Output: $VLA_OUT_ROOT/<dataset>-<policy>-<stamp>/checkpoints/last/pretrained_model
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${1:?usage: train.sh act|pi05 <dataset> [lerobot-train args]}"
DATASET="${2:?dataset name}"
shift 2
VENV="${VLA_VENV:-$HOME/.PhyAgentOS/skills/.venvs/piper-vla}"
DATA_ROOT="${VLA_DATA_ROOT:-$HOME/.PhyAgentOS/vla/datasets}"
OUT_ROOT="${VLA_OUT_ROOT:-$HOME/.PhyAgentOS/vla/runs}"
PI05_BASE="${PI05_BASE:-$HOME/.PhyAgentOS/models/pi05_base}"
[ -x "$VENV/bin/lerobot-train" ] || { echo "no lerobot-train in $VENV (run setup_venv.sh)"; exit 1; }
[ -f "$DATA_ROOT/$DATASET/meta/info.json" ] || { echo "no dataset at $DATA_ROOT/$DATASET"; exit 1; }
STAMP="$(date +%m%d-%H%M%S)"
OUT="$OUT_ROOT/$DATASET-$POLICY-$STAMP"
mkdir -p "$OUT_ROOT"
# the PaliGemma tokenizer and all weights come from local caches (see setup_venv.sh)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

COMMON=(--dataset.repo_id="local/$DATASET" --dataset.root="$DATA_ROOT/$DATASET"
        --output_dir="$OUT" --job_name="$DATASET-$POLICY" --policy.device=cuda
        --policy.push_to_hub=false --wandb.enable=false --log_freq=50 --num_workers=8)
case "$POLICY" in
  act)
    ARGS=(--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50
          --batch_size=16 --steps=30000 --save_freq=5000)
    ;;
  pi05)
    [ -f "$PI05_BASE/model.safetensors" ] || { echo "pi05_base missing at $PI05_BASE"; exit 1; }
    ARGS=(--policy.type=pi05 --policy.pretrained_path="$PI05_BASE"
          --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.train_expert_only=true
          --batch_size=8 --steps=3000 --save_freq=1000)
    ;;
  *) echo "policy must be act or pi05"; exit 1 ;;
esac
echo "== $POLICY on $DATASET -> $OUT"
"$VENV/bin/lerobot-train" "${COMMON[@]}" "${ARGS[@]}" "$@" 2>&1 | tee "$OUT_ROOT/$DATASET-$POLICY-$STAMP.log"
echo "== done: $OUT/checkpoints/last/pretrained_model"
echo "   export: $HERE/export_model.sh $OUT <policy-name>"
