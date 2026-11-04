"""Offline trial: DeepSeek (thinking on) writes robot.run programs against the fake world.

Measures per-request latency, reasoning/output tokens, rounds, estimated execution time and
task success. The round cap and HTTP timeout below belong to this harness only; the skill
itself has no stop logic.

Usage:
  nodes/skill_server/.venv/bin/python dev/llm_trial.py --runs 3 --tasks all --out dev/trials.jsonl
The API key is read from ~/.PhyAgentOS/config.json (providers.deepseek) and never printed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nodes" / "skill_server" / "src"))

from tabletop.executor import Executor  # noqa: E402
from tabletop.fake import Box, FakeWorld  # noqa: E402
from tabletop.primitives import Robot, RobotConfig  # noqa: E402

SCENE_S = 180.0
EST_S = {"move_to": 2.5, "home": 2.5, "set_gripper": 1.0, "locate": 1.2}  # locate: SAM3 ~0.55 s/frame x 2 cameras
HARNESS_MAX_ROUNDS = 30  # sanity cap only; the scene ends at 180 s like a referee stop


def api_section() -> str:
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- BEGIN API.*?-->(.*)<!-- END API -->", text, re.S)
    assert m, "API markers missing in SKILL.md"
    return m.group(1).strip()


SYSTEM = """You control a dual-arm AgileX PIPER tabletop robot (left arm at y=+0.30, right arm at y=-0.30,
both bases at x=0, table surface z=0, objects lie roughly at x=0.25..0.50).
You have exactly one tool, robot_run(code), which executes Python on the robot computer.
Write one block that completes the whole task when possible; if output shows a failure, write a
short repair block (earlier variables/functions persist). When the task is complete, reply with
a one-line summary and no tool call.

""" + api_section()

TOOL = {
    "type": "function",
    "function": {
        "name": "robot_run",
        "description": "Execute a Python block with the robot API. Returns printed output, error traceback and elapsed time.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source"}},
                       "required": ["code"]},
    },
}


# ----------------------------------------------------------------------------- tasks

def base_boxes():
    return [
        Box("red block", [0.38, 0.12], 0.0, [0.05, 0.03, 0.04], yaw_deg=30, aliases=["red cube", "red"]),
        Box("blue block", [0.42, -0.12], 0.0, [0.06, 0.04, 0.05], yaw_deg=-20, aliases=["blue cube", "blue"]),
        Box("green block", [0.30, -0.02], 0.0, [0.04, 0.04, 0.03], aliases=["green cube", "green"]),
    ]


def on_top(w, upper, lower, tol_xy=0.025):
    a, b = w.boxes[upper], w.boxes[lower]
    return abs(a.z_bottom - b.top) < 0.004 and np.linalg.norm(np.subtract(a.center_xy, b.center_xy)) < tol_xy


def free(w, name):
    return all(arm.held != name for arm in w.arm.values())


def in_tray(w, name):
    t, b = w.boxes["tray"], w.boxes[name]
    return free(w, name) and t.contains_xy(np.array([*b.center_xy, t.center[2]])) and b.z_bottom >= t.top - 0.004


TASKS = {
    "stack2": dict(
        text="把红色方块放到蓝色方块上面。",
        boxes=base_boxes,
        check=lambda w: on_top(w, "red block", "blue block") and free(w, "red block"),
    ),
    "stack3": dict(
        text="把红、绿、蓝三个方块叠成一摞：蓝色在最下面，绿色在中间，红色在最上面。",
        boxes=base_boxes,
        check=lambda w: on_top(w, "green block", "blue block") and on_top(w, "red block", "green block")
        and free(w, "red block") and free(w, "green block"),
    ),
    "handover": dict(
        text="左臂拿起黄色瓶子，交给右臂；右臂再把瓶子竖直放到蓝色方块旁边（离蓝色方块中心 15 厘米以内）的桌面上。",
        boxes=lambda: base_boxes() + [Box("yellow bottle", [0.30, 0.22], 0.0, [0.04, 0.04, 0.12], aliases=["bottle", "yellow bottle"])],
        check=lambda w: free(w, "yellow bottle") and w.boxes["yellow bottle"].z_bottom < 0.004
        and np.linalg.norm(np.subtract(w.boxes["yellow bottle"].center_xy, w.boxes["blue block"].center_xy)) < 0.15
        and any("right grasped yellow bottle" in line for line in w.log)
        and any("left grasped yellow bottle" in line for line in w.log),
    ),
    "tray_all": dict(
        text="把桌上所有方块都放进托盘里。",
        boxes=lambda: base_boxes() + [Box("tray", [0.48, 0.02], 0.0, [0.16, 0.14, 0.01], aliases=["托盘", "plate"])],
        check=lambda w: all(in_tray(w, n) for n in ("red block", "blue block", "green block")),
    ),
    "parallel": dict(
        text="两只手臂同时工作：左臂把红色方块、右臂把蓝色方块分别放进托盘里。",
        boxes=lambda: base_boxes() + [Box("tray", [0.48, 0.02], 0.0, [0.16, 0.14, 0.01], aliases=["托盘", "plate"])],
        check=lambda w: in_tray(w, "red block") and in_tray(w, "blue block")
        and any("left grasped red block" in line for line in w.log) and any("right grasped blue block" in line for line in w.log),
    ),
}


# ----------------------------------------------------------------------------- LLM

def load_provider():
    cfg = json.loads((Path.home() / ".PhyAgentOS" / "config.json").read_text())
    p = cfg["providers"]["deepseek"]
    return p["apiBase"].rstrip("/"), p["apiKey"], p.get("defaultModel") or "deepseek-flash"


def chat(base, key, model, messages, thinking=True):
    body = {"model": model, "messages": messages, "tools": [TOOL], "max_tokens": 8192}
    if thinking:
        body["thinking"] = {"type": "enabled"}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:  # harness-only HTTP timeout
        data = json.loads(resp.read())
    return data, time.time() - t0


def est_exec_s(ledger):
    return sum(EST_S.get(e["call"], 0.0) for e in ledger)


def run_trial(task_id, base, key, model, thinking=True):
    spec = TASKS[task_id]
    world = FakeWorld(spec["boxes"]())
    robot = Robot(world, world, RobotConfig(handover_xyz=[0.32, 0.0, 0.14], handover_table_xy=[0.32, 0.0]))
    ex = Executor(robot)
    used = 0.0
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"{spec['text']}\n[time_left_s={SCENE_S:.1f}]"}]
    rec = {"task": task_id, "thinking": thinking, "requests": [], "blocks": [], "error": None}
    try:
        for _ in range(HARNESS_MAX_ROUNDS):
            if used >= SCENE_S:
                rec["stopped_by_referee"] = True
                break
            data, dt = chat(base, key, model, messages, thinking)
            used += dt
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            rec["requests"].append({
                "latency_s": round(dt, 2),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("prompt_cache_hit_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            })
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("reasoning_content") is not None:
                assistant["reasoning_content"] = msg["reasoning_content"]
            calls = msg.get("tool_calls") or []
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)
            if not calls:
                rec["final"] = (msg.get("content") or "")[:300]
                break
            for call in calls:
                try:
                    code = json.loads(call["function"]["arguments"]).get("code", "")
                except json.JSONDecodeError as exc:
                    code, out = "", {"ok": False, "error": f"bad arguments: {exc}"}
                else:
                    out = ex.run(code)
                exec_s = est_exec_s(out.get("ledger", []))
                used += exec_s
                rec["blocks"].append({"lines": code.count("\n") + 1, "ok": out.get("ok"), "error": (out.get("error") or "")[-400:],
                                      "output": (out.get("output") or "")[-600:], "est_exec_s": round(exec_s, 1),
                                      "primitive_calls": len(out.get("ledger", []))})
                result = {k: out.get(k) for k in ("ok", "output", "error", "fault", "elapsed_s")}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False) + f"\n[time_left_s={SCENE_S - used:.1f}]"})
    except Exception as exc:  # network / API errors are recorded, not raised
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    rec["success"] = bool(spec["check"](world))
    rec["rounds"] = len(rec["requests"])
    rec["llm_s"] = round(sum(r["latency_s"] for r in rec["requests"]), 1)
    rec["est_exec_s"] = round(sum(b["est_exec_s"] for b in rec["blocks"]), 1)
    rec["est_total_s"] = round(rec["llm_s"] + rec["est_exec_s"], 1)
    rec["world_log_tail"] = world.log[-8:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "dev" / "trials.jsonl"))
    args = ap.parse_args()
    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    base, key, model = load_provider()
    jobs = [t for t in tasks for _ in range(args.runs)]
    out = Path(args.out)
    with cf.ThreadPoolExecutor(args.workers) as pool, out.open("a", encoding="utf-8") as fh:
        futs = {pool.submit(run_trial, t, base, key, model, not args.no_thinking): t for t in jobs}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            lat = [r["latency_s"] for r in rec["requests"]]
            print(f"{rec['task']:9s} success={rec['success']!s:5s} rounds={rec['rounds']} llm={rec['llm_s']}s "
                  f"exec~{rec['est_exec_s']}s total~{rec['est_total_s']}s lat={lat} err={rec['error']}", flush=True)


if __name__ == "__main__":
    main()
