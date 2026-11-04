#!/usr/bin/env bash
# Start the read-only live view page in the background; start.sh calls this on `paos skill start`.
# The page is its own process, not a Dora node: if it dies the skill keeps running. It stays up after
# `paos skill stop`, attaches to the next dataflow by itself, and is replaced at the next start.
# Prints the page's address, or why it did not start (exit 1).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${SKILL_SERVER_VENV:-$HERE/../../.venvs/piper-dual-tabletop-skill-server}"
DEPS="$(dirname "$VENV")/piper-dual-tabletop-live-view"
RUN_DIR="$HOME/.PhyAgentOS/run/piper-dual-tabletop"
LOG="$RUN_DIR/live_view.log"
PORT=8780
REQ="$HERE/live_view/requirements.txt"
[ -f "$HERE/live_view/server.py" ] || { echo "no page here: the bundle carries nodes/live_view"; exit 1; }
mkdir -p "$RUN_DIR"

WANT="$(sha256sum "$REQ" | cut -d' ' -f1)"
if [ "$(cat "$DEPS/.req-hash" 2>/dev/null || true)" != "$WANT" ]; then
  rm -rf "$DEPS"
  uv pip install -q --python "$VENV/bin/python" --target "$DEPS" --no-deps -r "$REQ" \
    || { echo "dependency install failed (offline? pre-fill the uv cache)"; exit 1; }
  echo "$WANT" > "$DEPS/.req-hash"
fi

holder() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u; }
for pid in $(holder); do
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'live_view/server.py' \
    || { echo "port $PORT is held by another program (pid $pid)"; exit 1; }
  kill "$pid"
done
for _ in $(seq 1 50); do [ -z "$(holder)" ] && break; sleep 0.2; done
[ -z "$(holder)" ] || { echo "the previous page did not stop (port $PORT)"; exit 1; }

PYTHONPATH="$DEPS" setsid nohup "$VENV/bin/python" "$HERE/live_view/server.py" --port "$PORT" \
  > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 50); do
  [ -n "$(holder)" ] && { echo "live view http://127.0.0.1:$PORT"; exit 0; }
  sleep 0.2
done
echo "live view did not start, see $LOG"
exit 1
