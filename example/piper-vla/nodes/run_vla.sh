#!/usr/bin/env bash
# Launch a piper_vla Dora node (collect_node | policy_node) from the VLA venv.
#   run_vla.sh collect_node --config <file> [...]   |   run_vla.sh policy_node --config <file>
# The VLA venv (lerobot 0.6.1); start.sh builds it with vla/setup_venv.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD="${1:?usage: run_vla.sh collect_node|policy_node --config <file>}"; shift
VENV="${VLA_VENV:-${PAOS_SKILL_ROOT:-$HERE/..}/../.venvs/piper-vla}"
[ -x "$VENV/bin/python" ] || VENV="$HOME/.PhyAgentOS/skills/.venvs/piper-vla"
export PYTHONPATH="$HERE/vla/src${PYTHONPATH:+:$PYTHONPATH}"
# models and the PaliGemma tokenizer are read from local caches only (no network at run time)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
exec "$VENV/bin/python" -m "piper_vla.$MOD" "$@"
