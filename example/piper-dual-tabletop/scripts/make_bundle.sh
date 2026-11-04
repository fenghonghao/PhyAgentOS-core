#!/usr/bin/env bash
# Package the paos skill into a flat, versioned tar.gz with archive-manifest.json
# (per-file size + sha256). Excludes venvs, caches, Dora run logs, locally rendered dataflows
# and the offline evaluation (dev/). Output: ${OUT_DIR:-<repo>/dist}/<name>-<version>.tar.gz
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${OUT_DIR:-$(cd "${SKILL_ROOT}/../.." && pwd)/dist}"
mkdir -p "$OUT_DIR"
read -r NAME VERSION < <(awk -F'"' '/^name:/{n=$0; sub(/^name: */,"",n)} /^version:/{v=$2} END{print n, v}' "$SKILL_ROOT/skill.yaml")
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
BUNDLE="${OUT_DIR}/${NAME}-${VERSION}.tar.gz"

python3 - "$SKILL_ROOT" "$STAGE" <<'PY'
import fnmatch, hashlib, json, shutil, sys
from pathlib import Path
src, stage = Path(sys.argv[1]), Path(sys.argv[2])
skip_dirs = {".venv", "__pycache__", ".pytest_cache", "out", "dev", ".git"}
skip_files = ["dataflow.local.yaml", "*.local.yaml", "*.pyc", "archive-manifest.json", ".gitignore"]
files = []
for p in sorted(src.rglob("*")):
    rel = p.relative_to(src)
    if any(part in skip_dirs for part in rel.parts) or not p.is_file():
        continue
    if any(fnmatch.fnmatch(p.name, pat) for pat in skip_files):
        continue
    if p.is_symlink():
        sys.exit(f"symlink not allowed in bundle: {rel}")
    dst = stage / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dst)
    data = dst.read_bytes()
    files.append({"path": rel.as_posix(), "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
(stage / "archive-manifest.json").write_text(json.dumps({"files": files}, separators=(",", ":")), encoding="utf-8")
print(f"    {len(files)} files")
PY
rm -f "$BUNDLE"
(cd "$STAGE" && find . -type f | sed 's|^\./||' | LC_ALL=C sort | tar -czf "$BUNDLE" -T -)
echo "bundle: $BUNDLE"
echo "size:   $(wc -c < "$BUNDLE") bytes"
echo "sha256: $(sha256sum "$BUNDLE" | cut -d' ' -f1)"
