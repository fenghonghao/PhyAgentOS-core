#!/usr/bin/env bash
# Build the VLA venv and offline model caches (laptop and RTX 4090 workstation alike).
#
#   setup_venv.sh            venv + PaliGemma tokenizer cache + ResNet18 backbone weights
#   setup_venv.sh --pi05     also fetch pi05_base (14.5 GB, ModelScope mirror, sha256-checked)
#
# Everything lands outside the bundle:
#   venv       $VLA_VENV (default ~/.PhyAgentOS/skills/.venvs/piper-vla)
#   pi05_base  ~/.PhyAgentOS/models/pi05_base
#   tokenizer  ~/.PhyAgentOS/models/paligemma-3b-pt-224-tokenizer, linked into the HF hub cache
#              as google/paligemma-3b-pt-224 (LeRobot hardcodes that gated repo id)
#   resnet18   ~/.cache/torch/hub/checkpoints (ACT backbone init)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VLA_VENV:-$HOME/.PhyAgentOS/skills/.venvs/piper-vla}"
MODELS="$HOME/.PhyAgentOS/models"
TOK="$MODELS/paligemma-3b-pt-224-tokenizer"
PI05="$MODELS/pi05_base"
PI05_SHA=0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
MS="https://www.modelscope.cn/models"
fail=0
ok() { echo "  ok   $*"; }
bad() { echo "  FAIL $*"; fail=1; }
noproxy() { env -u https_proxy -u HTTPS_PROXY -u http_proxy -u HTTP_PROXY -u ALL_PROXY -u all_proxy "$@"; }

echo "== venv $VENV"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV" || bad "uv venv"
WANT="$(cat "$HERE/requirements-torch.txt" "$HERE/requirements-vla.txt" "$HERE/requirements-orbbec.txt" | sha256sum | cut -c1-16)"
if [ "$(cat "$VENV/.req-hash" 2>/dev/null)" = "$WANT" ]; then ok "deps up to date"
else
  uv pip install -q --python "$VENV/bin/python" -r "$HERE/requirements-torch.txt" \
    && uv pip install -q --python "$VENV/bin/python" -r "$HERE/requirements-vla.txt" \
    && uv pip install -q --python "$VENV/bin/python" --no-deps -r "$HERE/requirements-orbbec.txt" \
    && echo "$WANT" > "$VENV/.req-hash" && ok "deps installed" || bad "pip install"
fi
"$VENV/bin/python" -c "import lerobot, torch; print('  lerobot', lerobot.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || bad "import"

echo "== PaliGemma tokenizer"
mkdir -p "$TOK"
for f in tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json added_tokens.json config.json preprocessor_config.json generation_config.json; do
  [ -s "$TOK/$f" ] || noproxy curl -sSfL -o "$TOK/$f" "$MS/google/paligemma-3b-pt-224/resolve/master/$f" || bad "download $f"
done
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}/models--google--paligemma-3b-pt-224"
REV=local0000000000000000000000000000000000
mkdir -p "$HUB/refs" "$HUB/snapshots/$REV"
[ -s "$HUB/refs/main" ] || echo -n "$REV" > "$HUB/refs/main"
SNAP="$HUB/snapshots/$(cat "$HUB/refs/main")"
mkdir -p "$SNAP"
for f in "$TOK"/*; do [ -e "$SNAP/$(basename "$f")" ] || cp "$f" "$SNAP/"; done
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$VENV/bin/python" -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('google/paligemma-3b-pt-224'); print('  tokenizer', type(t).__name__, len(t))" \
  || bad "tokenizer does not load offline"

echo "== ResNet18 (ACT backbone)"
R18="$HOME/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
if [ -s "$R18" ]; then ok "$R18"
else mkdir -p "$(dirname "$R18")"; curl -sSfL -o "$R18" https://download.pytorch.org/models/resnet18-f37072fd.pth && ok "downloaded" || bad "resnet18"; fi

if [ "${1:-}" = "--pi05" ]; then
  echo "== pi05_base -> $PI05"
  mkdir -p "$PI05"
  for f in config.json policy_preprocessor.json policy_postprocessor.json; do
    [ -s "$PI05/$f" ] || noproxy curl -sSfL -o "$PI05/$f" "$MS/lerobot/pi05_base/resolve/master/$f" || bad "download $f"
  done
  if [ "$(stat -c %s "$PI05/model.safetensors" 2>/dev/null)" != 14467165872 ]; then
    for i in $(seq 1 40); do
      noproxy curl -sSfL -C - --speed-limit 300000 --speed-time 60 -o "$PI05/model.safetensors" \
        "$MS/lerobot/pi05_base/resolve/master/model.safetensors" && break
      echo "  retry $i"; sleep 3
    done
  fi
  echo "$PI05_SHA  $PI05/model.safetensors" | sha256sum -c --quiet && ok "pi05_base sha256" || bad "pi05_base sha256"
fi
[ $fail = 0 ] && echo "== VLA setup OK" || { echo "== VLA setup has failures"; exit 1; }
