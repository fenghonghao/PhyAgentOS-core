"""One-page digest of a scene run, for updating the SKILL.md playbooks after a public task.

Reads what run_scene.sh and skill_server already write:
  ~/.PhyAgentOS/logs/piper-dual-tabletop/scenes/<stamp>_<scene>.json   (task, start, end)
  ~/.PhyAgentOS/logs/piper-dual-tabletop/scenes/<stamp>_<scene>_agent.log
  ~/.PhyAgentOS/logs/piper-dual-tabletop/ledger.jsonl                 (every robot.run block)
  ~/.PhyAgentOS/logs/piper-dual-tabletop/look/                         (images sent by look())

Usage:
  python3 dev/scene_digest.py                 # the latest scene
  python3 dev/scene_digest.py --last 3        # the three latest scenes
  python3 dev/scene_digest.py --scene pub-2   # scenes whose id contains pub-2
  python3 dev/scene_digest.py --ledger ~/.PhyAgentOS/logs/piper-dual-tabletop/sim-ledger.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

LOGS = Path.home() / ".PhyAgentOS" / "logs" / "piper-dual-tabletop"
MOTION = {"move_to", "home", "set_gripper", "pick", "place", "drag"}


def scenes(scene_filter: str | None) -> list[dict]:
    out = []
    for p in sorted((LOGS / "scenes").glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except ValueError:
            continue
        if scene_filter and scene_filter not in str(m.get("scene")):
            continue
        m["_meta"] = p
        out.append(m)
    return out


def ledger(path: Path, t0: float, t1: float) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if t0 <= float(r.get("t", 0)) <= t1:
            recs.append(r)
    return recs


def short(v, n: int = 160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def failure(call: dict) -> str | None:
    """Domain outcomes that went wrong, in one phrase."""
    r = call.get("result")
    if "error" in call:
        return f"raised {call['error']}"
    if not isinstance(r, dict):
        return None
    name = call["call"]
    if name in ("move_to", "home") and r.get("reached") is False:
        return f"not reached: {r.get('reason')} {short(r.get('detail') or '', 80)}"
    if name in ("pick", "place", "drag") and r.get("ok") is False:
        return f"{r.get('reason')} {short(r.get('note') or '', 80)}"
    if name == "locate" and (not r.get("found") or not r.get("usable")):
        return f"found={r.get('found')} usable={r.get('usable')} note={short(r.get('note') or '', 100)}"
    return None


def digest(m: dict, ledger_path: Path, nxt: dict | None) -> str:
    t0 = float(m["start"]) - 1
    t1 = float(m.get("end") or (float(nxt["start"]) - 1 if nxt else time.time()))
    lines = [f"## scene {m['scene']}  ({time.strftime('%m-%d %H:%M:%S', time.localtime(float(m['start'])))})",
             f"task: {short(m.get('task', ''), 400)}",
             f"elapsed: {m.get('elapsed_s', '?')} s   agent exit: {m.get('agent_exit', '?')}   "
             f"skill {m.get('skill_version')} core {m.get('core_rev')}"]
    log = m["_meta"].with_name(m["_meta"].stem + "_agent.log")
    if log.exists():
        tail = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()][-4:]
        lines.append("agent log tail: " + " | ".join(short(t, 200) for t in tail))
    blocks = ledger(ledger_path, t0, t1)
    per = collections.defaultdict(lambda: [0, 0.0])
    lines.append(f"\n{len(blocks)} robot.run block(s)")
    for k, b in enumerate(blocks, 1):
        calls = b.get("ledger") or []
        top = [c for c in calls if not c.get("nested")]
        for c in top:
            per[c["call"]][0] += 1
            per[c["call"]][1] += float(c.get("dt_s") or 0)
        status = "ok" if b.get("ok") else ("FAULT" if b.get("fault") else "error")
        lines.append(f"\n### block {k}: {status}, {b.get('elapsed_s')} s, {len(calls)} primitive calls")
        lines.append("```python\n" + (b.get("code") or "").rstrip() + "\n```")
        if b.get("output"):
            lines.append("output:\n```\n" + b["output"].rstrip()[-2500:] + "\n```")
        if b.get("error"):
            lines.append("error: " + short(b["error"], 600))
        if b.get("images"):
            lines.append("images shown to the model: " + ", ".join(p.rsplit("/", 1)[-1] for p in b["images"]))
        for c in calls:
            if c["call"] == "look":
                q = (c.get("args") or {}).get("question", "")
                lines.append(f"- {c['call']} ({c.get('dt_s')} s): {short(q, 140)} -> {short(c.get('result'), 300)}")
        for c in calls:
            f = failure(c)
            if f:
                lines.append(f"- FAIL {c['call']} {short(c.get('args'), 120)}: {f}")
    if per:
        lines.append("\ntime by top-level primitive: " + ", ".join(
            f"{k} x{n} {t:.1f}s" for k, (n, t) in sorted(per.items(), key=lambda kv: -kv[1][1])))
    imgs = [p.name for p in sorted((LOGS / "look").glob("*.jpg")) if t0 <= p.stat().st_mtime <= t1 + 1]
    if imgs:
        lines.append(f"vision images ({LOGS / 'look'}): " + ", ".join(imgs[:20]) + (" ..." if len(imgs) > 20 else ""))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=1)
    ap.add_argument("--scene")
    ap.add_argument("--ledger", default=str(LOGS / "ledger.jsonl"))
    args = ap.parse_args()
    all_scenes = scenes(args.scene)
    if not all_scenes:
        print("no scenes recorded under", LOGS / "scenes")
        return
    chosen = all_scenes[-args.last:]
    for m in chosen:
        i = all_scenes.index(m)
        print(digest(m, Path(args.ledger).expanduser(), all_scenes[i + 1] if i + 1 < len(all_scenes) else None))
        print()


if __name__ == "__main__":
    main()
