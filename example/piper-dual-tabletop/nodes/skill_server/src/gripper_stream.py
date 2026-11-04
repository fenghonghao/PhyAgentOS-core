"""Dora node: the gripper's Action server, with a target that moves.

The firmware's gripper command is a position and an effort, with no speed, so a slow close has to
be a target that moves a little on every 20 ms tick, followed by the firmware's position loop. A
goal with ``max_velocity`` does that; without it the target goes to the goal at once. Between goals
the target keeps being sent, so the fingers keep pressing on what they closed on.

``GripperStream`` is the deterministic core (no Dora, time is passed in); ``run`` is the Dora loop.
This module is outside the ``tabletop`` package so that the node loads only what it needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml
from forge_msgs import GripperCommandGoal, GripperCommandResult, JointCommand, JointState

logger = logging.getLogger("gripper_stream")


@dataclass(frozen=True)
class StreamConfig:
    joint_name: str
    min_position: float
    max_position: float
    feedback_timeout_ms: float           # no joint state for this long: the driver has a fault
    goal_tolerance: float                # this close to the goal counts as reached
    stall_velocity_threshold: float      # moving slower than this (m/s) counts as stopped
    stall_timeout_ms: float              # stopped for this long: the goal ends, on what the fingers closed on


@dataclass
class Out:
    """What one input makes the node send: a target, and a result when a goal ends."""

    command: float | None = None
    goal_id: str | None = None
    result: GripperCommandResult | None = None
    status: str | None = None
    trace: dict[str, Any] | None = None


@dataclass
class _Goal:
    goal_id: str
    position: float
    speed: float | None
    target: float
    started_ns: int
    ticked_ns: int
    moved_ns: int
    samples: list[list[float]] = field(default_factory=list)


class GripperStream:
    def __init__(self, cfg: StreamConfig) -> None:
        self.cfg = cfg
        self._position: float | None = None
        self._seen_ns: int | None = None
        self._goal: _Goal | None = None
        self._hold: float | None = None

    def on_state(self, position: float, now_ns: int) -> None:
        goal = self._goal
        if goal is not None and now_ns > self._seen_ns:
            speed = abs(position - self._position) / ((now_ns - self._seen_ns) / 1e9)
            if speed > self.cfg.stall_velocity_threshold:
                goal.moved_ns = now_ns
        self._position, self._seen_ns = position, now_ns

    def on_goal(self, goal_id: str, goal: GripperCommandGoal, now_ns: int) -> Out | None:
        """None when accepted; a refusal is a result right away."""
        if self._goal is not None:
            return self._refuse(goal_id, "BUSY", "another gripper goal is running")
        if goal.max_effort is not None:
            return self._refuse(goal_id, "UNSUPPORTED_EFFORT", "this node commands positions only")
        if not self.cfg.min_position <= goal.position <= self.cfg.max_position:
            return self._refuse(goal_id, "POSITION_LIMIT_VIOLATION", f"{goal.position} is outside the stroke")
        if not self._fresh(now_ns):
            return self._refuse(goal_id, "NO_FRESH_ROBOT_STATE", "no recent joint state from the driver")
        target = goal.position if goal.max_velocity is None else self._position
        self._goal = _Goal(goal_id, goal.position, goal.max_velocity, target, now_ns, now_ns, now_ns)
        return None

    def on_cancel(self, goal_id: str, now_ns: int) -> Out | None:
        goal = self._goal
        if goal is None or goal.goal_id != goal_id:
            return None
        return self._finish(goal, "CANCELED", "gripper goal canceled", "canceled", now_ns, hold=self._position)

    def tick(self, now_ns: int) -> Out | None:
        goal = self._goal
        if goal is None:
            return None if self._hold is None else Out(command=self._hold)
        if not self._fresh(now_ns):
            return self._finish(goal, "FEEDBACK_STALE", "joint state stopped coming", "aborted", now_ns, hold=self._position)
        if abs(goal.position - self._position) <= self.cfg.goal_tolerance:
            return self._finish(goal, "SUCCESS", "gripper goal reached", "succeeded", now_ns, hold=goal.position,
                                reached_goal=True)
        if self._seen_ns - goal.moved_ns >= self.cfg.stall_timeout_ms * 1e6:  # by the states seen: silence is no stall
            return self._finish(goal, "SUCCESS", "fingers stopped before the goal", "succeeded", now_ns,
                                hold=goal.position, stalled=True)
        if goal.speed is not None:
            step = goal.speed * (now_ns - goal.ticked_ns) / 1e9
            goal.target += max(-step, min(step, goal.position - goal.target))
        goal.ticked_ns = now_ns
        goal.samples.append([round((now_ns - goal.started_ns) / 1e9, 3), round(goal.target, 6), round(self._position, 6)])
        return Out(command=goal.target)

    def _fresh(self, now_ns: int) -> bool:
        return self._seen_ns is not None and now_ns - self._seen_ns <= self.cfg.feedback_timeout_ms * 1e6

    def _refuse(self, goal_id: str, code: str, message: str) -> Out:
        result = GripperCommandResult(error_code=code, message=message, elapsed_ns=0, position=self._position,
                                      stalled=False, reached_goal=False)
        return Out(goal_id=goal_id, result=result, status="aborted")

    def _finish(self, goal: _Goal, code: str, message: str, status: str, now_ns: int, hold: float,
                stalled: bool = False, reached_goal: bool = False) -> Out:
        self._goal, self._hold = None, hold
        result = GripperCommandResult(error_code=code, message=message, elapsed_ns=now_ns - goal.started_ns,
                                      position=self._position, stalled=stalled, reached_goal=reached_goal)
        trace = {"goal": goal.position, "max_velocity": goal.speed, "error_code": code, "stalled": stalled,
                 "reached_goal": reached_goal, "final": self._position, "elapsed_s": result.elapsed_ns / 1e9,
                 "samples": goal.samples}
        return Out(command=hold, goal_id=goal.goal_id, result=result, status=status, trace=trace)


def _goal_id(event: dict[str, Any]) -> str:
    metadata = event["metadata"]
    return str(metadata.get("parameters", metadata)["goal_id"])


def _handle(core: GripperStream, event: dict[str, Any], now_ns: int) -> Out | None:
    input_id = event.get("id")
    if input_id == "tick":
        return core.tick(now_ns)
    if input_id == "joint_state":
        state = JointState.from_arrow(event["value"])
        core.on_state(float(state.position[list(state.name).index(core.cfg.joint_name)]), now_ns)
        return None
    if input_id == "gripper_goal":
        return core.on_goal(_goal_id(event), GripperCommandGoal.from_arrow(event["value"]), now_ns)
    if input_id == "gripper_cancel":
        return core.on_cancel(_goal_id(event), now_ns)
    return None


def _emit(node: Any, joint_name: str, out: Out, trace_path: Path | None) -> None:
    if out.command is not None:
        command = JointCommand(name=[joint_name], mode="position", position=[out.command])
        node.send_output("joint_command", command.to_arrow())
    if out.result is not None:
        node.send_output("gripper_result", out.result.to_arrow(),
                         metadata={"goal_id": out.goal_id, "goal_status": out.status})
    if out.trace is not None and trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": time.time(), **out.trace}) + "\n")


def run(node: Any, core: GripperStream, trace_path: Path | None = None,
        clock: Callable[[], int] = time.monotonic_ns) -> None:
    for event in node:
        kind = event.get("type")
        if kind == "STOP":
            return
        if kind == "ERROR":
            logger.error("dora error: %s", event.get("error"))
        if kind != "INPUT":
            continue
        try:
            out = _handle(core, event, clock())
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("ignored invalid input %s: %s", event.get("id"), exc)
            continue
        if out is not None:
            _emit(node, core.cfg.joint_name, out, trace_path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gripper node: streams a moving target at the tick rate")
    ap.add_argument("--config", required=True)
    ap.add_argument("--trace", help="append every goal's target and measured opening to this JSONL file")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = StreamConfig(**yaml.safe_load(Path(args.config).read_text()))
    from dora import Node

    run(Node(), GripperStream(cfg), Path(args.trace).expanduser() if args.trace else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
