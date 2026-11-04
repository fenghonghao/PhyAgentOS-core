#!/usr/bin/env bash
# Launch the skill_server Dora node from its venv (built by start.sh outside the bundle).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SKILL_SERVER_VENV:-${PAOS_SKILL_ROOT:-$HERE/..}/../.venvs/piper-dual-tabletop-skill-server}"
[ -x "$VENV/bin/python" ] || VENV="$HERE/skill_server/.venv"   # development fallback
export PYTHONPATH="$HERE/skill_server/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" -m tabletop.dora_node "$@"
