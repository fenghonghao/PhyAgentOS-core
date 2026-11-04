#!/usr/bin/env bash
# Competition entry: one scene = one call.
#   ./run_scene.sh <scene_id> "<task text from the referee>" [target_image]
# target_image: path of the task's target image, appended to the text so the model passes it
# to show(image=path). SCENE_LIMIT_S: the organizers' time limit (default 180).
# Writes the scene deadline (start + SCENE_LIMIT_S) for the skill_server (fresh namespace, time_left())
# and for PAOS (seconds left on every model request), records start/end, runs the agent once.
# There is deliberately no timeout here: the model ends the scene, or the referee stops it.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <scene_id> \"<task text>\" [target_image]" >&2; exit 2; }
SCENE="$1"; TASK="$2"
if [ $# -ge 3 ]; then
  IMG="$(readlink -f "$3")"; [ -f "$IMG" ] || { echo "no such image: $3" >&2; exit 2; }
  TASK="$TASK
目标图像（target image）：$IMG"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIMIT_S="${SCENE_LIMIT_S:-180}"
DEADLINE_FILE="${PAOS_SCENE_DEADLINE_FILE:-/tmp/paos_scene_deadline}"   # = robot.deadline_file in skill_server.yaml
LOG_DIR="${HOME}/.PhyAgentOS/logs/piper-dual-tabletop/scenes"
PAOS="${PAOS_BIN:-paos}"
mkdir -p "$LOG_DIR"

START="$(date +%s.%N)"
DEADLINE="$(awk -v s="$START" -v l="$LIMIT_S" 'BEGIN{printf "%.3f", s + l}')"
echo "$DEADLINE" > "$DEADLINE_FILE"
export PAOS_SCENE_DEADLINE="$DEADLINE"
STAMP="$(date +%Y%m%d-%H%M%S)"
META="$LOG_DIR/${STAMP}_${SCENE}.json"
SKILL_VERSION="$(awk -F'"' '/^version:/{print $2; exit}' "$HERE/skill.yaml")"
CORE_REV="$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)"
printf '{"scene":"%s","task":%s,"start":%s,"deadline":%s,"skill_version":"%s","core_rev":"%s"}\n' \
  "$SCENE" "$(printf '%s' "$TASK" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" \
  "$START" "$DEADLINE" "$SKILL_VERSION" "$CORE_REV" > "$META"
echo "[scene $SCENE] start $(date +%T)  deadline in ${LIMIT_S}s  log $META"

set +e
"$PAOS" agent -m "$TASK" -s "scene:${SCENE}:${STAMP}" --no-markdown 2>&1 | tee "$LOG_DIR/${STAMP}_${SCENE}_agent.log"
STATUS=${PIPESTATUS[0]}
set -e
END="$(date +%s.%N)"
python3 - "$META" "$END" "$STATUS" <<'PY'
import json, sys
p, end, status = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
m = json.loads(open(p).read()); m.update(end=end, elapsed_s=round(end - m["start"], 2), agent_exit=status)
open(p, "w").write(json.dumps(m, ensure_ascii=False) + "\n"); print(f"[scene {m['scene']}] elapsed {m['elapsed_s']} s, exit {status}")
PY
exit "$STATUS"
