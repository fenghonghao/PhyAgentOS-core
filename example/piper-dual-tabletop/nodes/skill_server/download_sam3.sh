#!/usr/bin/env bash
# SAM3 (transformers format) from the ModelScope mirror of facebook/sam3; resumable.
set -u
DEST="${1:-$HOME/.PhyAgentOS/models/sam3}"; mkdir -p "$DEST"; cd "$DEST"
for f in config.json processor_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt LICENSE model.safetensors; do
  curl -sSL -C - --retry 5 -o "$f" "https://www.modelscope.cn/models/facebook/sam3/resolve/master/$f" || echo "FAILED $f"
done
ls -la
