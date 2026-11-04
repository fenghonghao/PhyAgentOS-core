"""Offline trial for the single-arm booth: DeepSeek writes robot.run programs from SKILL.md for
the three competition task types, against the fake world.

look / inspect / orient are oracles answering from the fake world's ground truth in the JSON
shapes SKILL.md asks for, so this measures the model's code, not the vision model. Object
phrases match by words (all words of the phrase must describe the object), like SAM3 would.

Usage:
  nodes/skill_server/.venv/bin/python dev/llm_trial_booth.py --runs 2 --tasks all
The API key is read from ~/.PhyAgentOS/config.json (providers.deepseek) and never printed.
The round cap and HTTP timeout belong to this harness only; the skill has no stop logic.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes" / "skill_server" / "src"))
sys.path.insert(0, str(ROOT / "dev"))

from llm_trial import TOOL, api_section, chat, load_provider  # noqa: E402
from tabletop import geometry as geo  # noqa: E402
from tabletop.backend import Detection  # noqa: E402
from tabletop.executor import Executor  # noqa: E402
from tabletop.fake import Box, FakeWorld  # noqa: E402
from tabletop.primitives import Robot, RobotConfig  # noqa: E402

SCENE_S = 180.0
EST_S = {"move_to": 2.5, "home": 2.5, "set_gripper": 1.0, "locate": 1.2, "locate_all": 0.8, "look": 1.5,
         "inspect": 2.0, "orient": 2.0}
HARNESS_MAX_ROUNDS = 12
STOP = {"a", "an", "the", "of", "on", "small", "big", "object", "objects", "piece", "toy"}

SYSTEM = """You control one AgileX PIPER arm in a tabletop booth through the PhyAgentOS skill below
(world frame = the arm's base: x forward, y left, z up; objects lie at about x = 0.18..0.50, y = -0.25..0.25).
You have exactly one tool, robot_run(code), which executes Python on the robot computer.
Write one block that completes the whole task when possible; if output shows a failure, write a
short repair block (earlier variables/functions persist). When the task is complete, reply with
a one-line summary and no tool call.

""" + api_section()


class Booth(FakeWorld):
    """FakeWorld whose segmentation matches phrases by words, like an open-vocabulary model."""

    def segment(self, frame, text):
        words = {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}
        words = {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words}
        ids, names = frame.meta["ids"], frame.meta["names"]
        dets = []
        for i, name in enumerate(names):
            b = self.boxes[name]
            desc = set(re.findall(r"[a-z]+", " ".join([name, *b.aliases]).lower()))
            if words and words <= desc:
                mask = ids == i
                if mask.sum() >= 15:
                    dets.append(Detection(mask=mask, score=0.9 - 0.01 * len(dets)))
        dets.sort(key=lambda d: -int(d.mask.sum()))
        return dets


def blk(name, xy, color, **meta):
    return Box(name, list(xy), 0.0, [0.04, 0.04, 0.04], yaw_deg=meta.pop("yaw", 0.0),
               aliases=["block", "cube", "building block", "wooden block", "toy block"], color=color,
               meta={"kind": "block", **meta})


def fruit(name, xy, color):
    return Box(name, list(xy), 0.0, [0.055, 0.055, 0.05], aliases=["fruit", "toy fruit"], color=color,
               meta={"kind": "fruit"})


def bowl(name, xy, color):
    return Box(name, list(xy), 0.0, [0.13, 0.13, 0.05], aliases=["bowl", "container", "box", "basket"], color=color,
               meta={"kind": "container"})


# ----------------------------------------------------------------------------- tasks

def inside(w, obj, cont):
    o, c = w.boxes[obj], w.boxes[cont]
    return c.contains_xy(np.array([*o.center_xy, c.center[2]])) and o.z_bottom >= c.top - 0.004


def unmoved(w, start, names):
    return all(np.allclose(w.boxes[n].center_xy, start[n], atol=0.002) for n in names)


def target_boxes():
    return [bowl("blue bowl", [0.38, -0.16], (40, 60, 200)), bowl("green basket", [0.40, 0.17], (40, 170, 70)),
            blk("red block", [0.27, 0.05], (210, 30, 30)), blk("yellow block", [0.30, -0.05], (230, 210, 40)),
            fruit("orange", [0.46, 0.0], (250, 140, 20))]


def sort_boxes():
    return [blk("red block", [0.27, 0.16], (210, 30, 30)), blk("blue block", [0.33, 0.19], (40, 60, 200)),
            blk("green block", [0.30, 0.10], (40, 170, 70)),
            fruit("apple", [0.29, -0.16], (200, 20, 40)), fruit("banana", [0.36, -0.19], (240, 220, 60)),
            fruit("orange", [0.33, -0.10], (250, 140, 20)),
            blk("yellow block", [0.38, -0.05], (230, 210, 40)),       # stray, next to the fruits
            fruit("pear", [0.45, 0.12], (180, 210, 60))]             # stray, away from the fruits


EQ = [("2", [0.26, 0.20], 90, 7), ("6", [0.40, 0.21], 0, 17), ("3", [0.44, 0.05], 180, 27), ("+", [0.24, -0.20], 270, 37),
      ("-", [0.30, -0.24], 0, 47), ("×", [0.41, -0.20], 90, 57), ("÷", [0.46, -0.10], 0, 67), ("=", [0.20, 0.08], 180, 77)]


def eq_boxes():
    return [blk(f"block {s}", xy, (235, 200, 120), symbol=s, top_deg=t, yaw=y) for s, xy, t, y in EQ]


def eq_check(w, start):
    moved = [n for n, b in w.boxes.items() if not np.allclose(b.center_xy, start[n], atol=0.002)]
    if len(moved) != 5:
        return False
    syms = {n: w.boxes[n].meta["symbol"] for n in moved}
    pos = {n: np.asarray(w.boxes[n].center_xy) for n in moved}
    order = sorted(moved, key=lambda n: pos[n][1])            # READ_RIGHT_DEG 90: left -> right = +y
    xs = [pos[n][0] for n in order]
    if max(xs) - min(xs) > 0.02:
        return False
    text = "".join(syms[n] for n in order)
    sides = text.split("=")
    if len(sides) != 2:
        return False
    vals = [calc(side) for side in sides]
    ok = None not in vals and abs(vals[0] - vals[1]) < 1e-9
    tops = [(w.boxes[n].yaw_deg + w.boxes[n].meta["top_deg"]) % 360 for n in order if syms[n].isdigit()]
    return ok and all(abs((t - 180 + 180) % 360 - 180) < 10 for t in tops)


def target_alt_boxes():
    return [Box("white plate", [0.42, 0.14], 0.0, [0.16, 0.16, 0.02], aliases=["plate", "dish", "container"],
                color=(245, 245, 240), meta={"kind": "container"}),
            bowl("red bowl", [0.30, -0.18], (200, 30, 30)),
            fruit("green apple", [0.25, 0.05], (120, 200, 60)), blk("blue block", [0.36, -0.02], (40, 60, 200)),
            blk("purple block", [0.47, -0.08], (130, 50, 160))]


def sort_alt_boxes():  # groups front / back instead of left / right, strays elsewhere
    return [blk("red block", [0.22, -0.06], (210, 30, 30)), blk("blue block", [0.22, 0.02], (40, 60, 200)),
            blk("green block", [0.26, 0.09], (40, 170, 70)), blk("white block", [0.25, -0.13], (240, 240, 240)),
            fruit("apple", [0.44, -0.08], (200, 20, 40)), fruit("lemon", [0.46, 0.02], (250, 240, 60)),
            fruit("peach", [0.30, 0.20], (250, 180, 140)),           # stray, at the side
            blk("yellow block", [0.42, 0.11], (230, 210, 40))]       # stray, next to the fruits


EQ_ALT = [("8", [0.44, -0.18], 270, 3), ("4", [0.25, 0.18], 0, 12), ("2", [0.40, 0.17], 90, -20), ("+", [0.22, -0.05], 0, 0),
          ("-", [0.47, 0.02], 90, 30), ("×", [0.30, -0.22], 0, 15), ("÷", [0.33, 0.24], 180, 8), ("=", [0.46, -0.06], 90, 40)]


def eq_alt_boxes():
    return [blk(f"block {s}", xy, (235, 200, 120), symbol=s, top_deg=t, yaw=y) for s, xy, t, y in EQ_ALT]


def sort_alt_ok(w, start):
    xy = {n: np.asarray(b.center_xy) for n, b in w.boxes.items()}
    blocks_c = np.mean([xy[n] for n in ("red block", "blue block", "green block", "white block")], axis=0)
    fruits_c = np.mean([xy[n] for n in ("apple", "lemon")], axis=0)
    for n, own, other in (("yellow block", blocks_c, fruits_c), ("peach", fruits_c, blocks_c)):
        if not (np.linalg.norm(xy[n] - own) < 0.15 and np.linalg.norm(xy[n] - own) + 0.05 < np.linalg.norm(xy[n] - other)):
            return False
    return unmoved(w, start, ["red block", "blue block", "green block", "white block", "apple", "lemon"])


def calc(expr: str) -> float | None:
    """Value of "7", "2+3", "6÷2" ... (one operator at most); None if malformed."""
    m = re.fullmatch(r"(\d+)(?:([-+×÷])(\d+))?", expr)
    if not m:
        return None
    a = float(m.group(1))
    if m.group(2) is None:
        return a
    b = float(m.group(3))
    return {"+": a + b, "-": a - b, "×": a * b, "÷": a / b if b else None}[m.group(2)]


TASKS = {
    "target": dict(text="把目标图像中的物品移动到对应的容器中。目标图像：/tmp/booth_target_1.png",
                   boxes=target_boxes, jobs=[{"object": "red block", "container": "blue bowl"}],
                   check=lambda w, s: inside(w, "red block", "blue bowl")
                   and unmoved(w, s, ["blue bowl", "green basket", "yellow block", "orange"])),
    "target2": dict(text="把目标图像中的物品移动到对应的容器中。目标图像：/tmp/booth_target_2.png",
                    boxes=target_boxes, jobs=[{"object": "red block", "container": "green basket"},
                                              {"object": "orange", "container": "blue bowl"}],
                    check=lambda w, s: inside(w, "red block", "green basket") and inside(w, "orange", "blue bowl")
                    and unmoved(w, s, ["blue bowl", "green basket", "yellow block"])),
    "sort": dict(text="将桌上积木和水果分类摆放好。", boxes=sort_boxes,
                 check=lambda w, s: sort_ok(w, s)),
    "target_alt": dict(text="把目标图像中的物品移动到对应的容器中。目标图像：/tmp/booth_target_3.png",
                       boxes=target_alt_boxes, jobs=[{"object": "green apple", "container": "white plate"},
                                                     {"object": "blue block", "container": "red bowl"}],
                       check=lambda w, s: inside(w, "green apple", "white plate") and inside(w, "blue block", "red bowl")
                       and unmoved(w, s, ["white plate", "red bowl", "purple block"])),
    "sort_alt": dict(text="将桌上积木和水果分类摆放好。", boxes=sort_alt_boxes, check=lambda w, s: sort_alt_ok(w, s)),
    "equation_alt": dict(text="桌上有3个数字块和加、减、乘、除、等号5个符号块。用全部3个数字块，选1个运算符和等号，摆成一条成立的等式，让站在桌子前方的裁判能正着读。",
                         boxes=eq_alt_boxes, check=lambda w, s: eq_check(w, s)),
    "equation": dict(text="桌上有3个数字块和加、减、乘、除、等号5个符号块。用全部3个数字块，选1个运算符和等号，摆成一条成立的等式，让站在桌子前方的裁判能正着读。",
                     boxes=eq_boxes, check=eq_check),
}


def sort_ok(w, start):
    xy = {n: np.asarray(b.center_xy) for n, b in w.boxes.items()}
    blocks_c = np.mean([xy[n] for n in ("red block", "blue block", "green block")], axis=0)
    fruits_c = np.mean([xy[n] for n in ("apple", "banana", "orange")], axis=0)
    for n, own, other in (("yellow block", blocks_c, fruits_c), ("pear", fruits_c, blocks_c)):
        if not (np.linalg.norm(xy[n] - own) < 0.15 and np.linalg.norm(xy[n] - own) + 0.05 < np.linalg.norm(xy[n] - other)):
            return False
    return unmoved(w, start, ["red block", "blue block", "green block", "apple", "banana", "orange"])


# ----------------------------------------------------------------------------- oracles

def oracles(world: FakeWorld, spec: dict):
    def near_box(m):
        c = np.asarray(getattr(m, "center_xyz", m)[:2])
        return min(world.boxes.values(), key=lambda b: np.linalg.norm(np.asarray(b.center_xy) - c))

    def tile_dir(m, b):
        rel = (b.yaw_deg + b.meta.get("top_deg", 0) - (m.long_axis_deg or 0)) % 360
        return ["right", "up", "left", "down"][int(round(rel / 90)) % 4]

    def look(question, camera=None, marks=None, image=None, json=False, zoom=False):
        q = question.lower()
        marks = list(marks or [])
        if '"object"' in q or "reference" in q or "target" in q or image is not None:
            ans = spec.get("jobs", [])
        elif marks:
            ans = {}
            for i, m in enumerate(marks, 1):
                b = near_box(m)
                if "symbol" in q:
                    ans[str(i)] = {"symbol": b.meta.get("symbol"), "top": "up"} if "top" in q else b.meta.get("symbol")
                elif "fruit" in q or "block" in q:
                    ans[str(i)] = b.meta.get("kind")
                else:
                    ans[str(i)] = b.name
        else:
            ans = {"objects": [f"{b.name} ({b.meta.get('kind')})" for b in world.boxes.values()]}
        return ans if json else globals()["json"].dumps(ans, ensure_ascii=False)

    def inspect(objects, question, json=False):
        q = question.lower()
        ans = {}
        for i, m in enumerate(list(objects), 1):
            b = near_box(m)
            if "top" in q and "symbol" in q:
                ans[str(i)] = {"symbol": b.meta.get("symbol"), "top": tile_dir(m, b)}
            elif "symbol" in q or "digit" in q or "print" in q:
                ans[str(i)] = b.meta.get("symbol")
            else:
                ans[str(i)] = b.name
        return ans if json else globals()["json"].dumps(ans, ensure_ascii=False)

    def orient(objects, labels=None):
        out = []
        for m in list(objects):
            b = near_box(m)
            out.append(round((b.yaw_deg + b.meta.get("top_deg", 0) + 180) % 360 - 180, 1))
        return out

    return {"look": look, "inspect": inspect, "orient": orient}


# ----------------------------------------------------------------------------- loop

def run_trial(task_id, base, key, model, thinking=True):
    spec = TASKS[task_id]
    world = Booth(spec["boxes"](), arms=("left",), resolution=(320, 240))
    start = {n: list(b.center_xy) for n, b in world.boxes.items()}
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    ex = Executor(robot)
    ex.ns.update(oracles(world, spec))
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
            rec["requests"].append({"latency_s": round(dt, 2), "completion_tokens": usage.get("completion_tokens")})
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
                exec_s = sum(EST_S.get(e["call"], 0.0) for e in out.get("ledger", []))
                used += exec_s
                rec["blocks"].append({"code": code[-2500:], "ok": out.get("ok"), "error": (out.get("error") or "")[-500:],
                                      "output": (out.get("output") or "")[-800:], "est_exec_s": round(exec_s, 1)})
                result = {k: out.get(k) for k in ("ok", "output", "error", "fault", "elapsed_s")}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(result, ensure_ascii=False) + f"\n[time_left_s={SCENE_S - used:.1f}]"})
    except Exception as exc:  # network / API errors are recorded, not raised
        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
    rec["success"] = bool(spec["check"](world, start))
    rec["rounds"] = len(rec["requests"])
    rec["llm_s"] = round(sum(r["latency_s"] for r in rec["requests"]), 1)
    rec["est_exec_s"] = round(sum(b["est_exec_s"] for b in rec["blocks"]), 1)
    rec["est_total_s"] = round(rec["llm_s"] + rec["est_exec_s"], 1)
    rec["world_log_tail"] = world.log[-10:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "dev" / "trials_booth.jsonl"))
    args = ap.parse_args()
    tasks = list(TASKS) if args.tasks == "all" else args.tasks.split(",")
    base, key, model = load_provider()
    jobs = [t for t in tasks for _ in range(args.runs)]
    with cf.ThreadPoolExecutor(args.workers) as pool, Path(args.out).open("a", encoding="utf-8") as fh:
        futs = {pool.submit(run_trial, t, base, key, model, not args.no_thinking): t for t in jobs}
        for fut in cf.as_completed(futs):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            lat = [r["latency_s"] for r in rec["requests"]]
            errs = [b["error"].splitlines()[-1] for b in rec["blocks"] if b.get("error")]
            print(f"{rec['task']:9s} success={rec['success']!s:5s} rounds={rec['rounds']} llm={rec['llm_s']}s "
                  f"exec~{rec['est_exec_s']}s total~{rec['est_total_s']}s lat={lat} err={rec['error']} block_errs={errs[:2]}",
                  flush=True)


if __name__ == "__main__":
    main()
