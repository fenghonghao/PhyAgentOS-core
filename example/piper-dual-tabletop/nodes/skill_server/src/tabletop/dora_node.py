"""Dora node: Forge Action provider ``robot.skill_server/run`` + Dora Action client for the arm.

Threads
-------
* Main thread: the Dora event loop. Renews the endpoint lease, dispatches ToolMessages,
  sends motion/gripper goals queued on the ``Bus`` and resolves them when results arrive.
  It never runs model code, so the lease keeps renewing during long blocks.
* Worker thread (one per ``robot.run``): executes the model's code through ``Executor``.

No time limit is applied anywhere in this node. A block ends when its code ends, when it
raises, or on an explicit Action cancel.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from forge_tool import (
    TOOL_ENDPOINT_PROTOCOL,
    ToolAccepted,
    ToolContext,
    ToolControlResponse,
    ToolEndpointDescriptor,
    ToolEndpointError,
    ToolEndpointHandler,
    ToolError,
    ToolEvent,
    ToolEventEmitter,
    ToolExecutionKey,
    ToolExecutionStatus,
    ToolOperationDescriptor,
    ToolRequest,
    ToolResult,
    ToolResultResponse,
)
from forge_tool._tool_message import ToolMessage
from forge_tool.dora import DoraToolEndpointBinding, tool_envelope_to_message, tool_message_to_envelope

from .dora_backend import Bus, DoraArmBackend, MotionConfig, load_arm_configs
from .endpoint_lease import EndpointLease
from .executor import Executor
from .primitives import Robot, RobotConfig

logger = logging.getLogger("skill_server")

ENDPOINT_ID = "robot.skill_server"
OPERATION = "run"
DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=ENDPOINT_ID,
    operations=(ToolOperationDescriptor(name=OPERATION, semantics="action", cancellable=True,
                                        status_supported=True, max_concurrency=1),),
)


def _err(code: str, message: str, *, retryable: bool = False) -> ToolEndpointError:
    return ToolEndpointError(ToolError(code=code, message=message, retryable=retryable))


@dataclass
class _Execution:
    key: ToolExecutionKey
    arguments_json: str
    emitter: ToolEventEmitter
    phase: str = "accepted"
    result: ToolResult | None = None


class RunEndpoint:
    """Action SPI for ``run``. ``start`` returns at once; the block runs in a worker thread."""

    def __init__(self, executor: Executor, done: queue.Queue, ledger_path: Path | None,
                 scene_marker: "SceneMarker", result_cache_size: int = 64) -> None:
        self.executor = executor
        self.done = done
        self.ledger_path = ledger_path
        self.scene = scene_marker
        self._executions: OrderedDict[ToolExecutionKey, _Execution] = OrderedDict()
        self._active: ToolExecutionKey | None = None
        self._accepting = True
        self._cache = result_cache_size

    async def start(self, request: ToolRequest, context: ToolContext, events: ToolEventEmitter) -> ToolAccepted:
        if context.operation != OPERATION:
            raise _err("FORGE_PROTOCOL_UNKNOWN_OPERATION", f"unknown operation {context.operation!r}")
        if not self._accepting:
            raise _err("FORGE_ENDPOINT_STOPPING", "skill_server is shutting down", retryable=True)
        key = context.execution_key
        arguments_json = json.dumps(request.arguments, sort_keys=True, separators=(",", ":"))
        existing = self._executions.get(key)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                raise _err("FORGE_EXECUTION_CONFLICT", "execution key reused with different arguments")
            return ToolAccepted(details={"idempotent": True})
        if self._active is not None:
            raise _err("BUSY", "another robot.run block is still executing", retryable=True)
        code = request.arguments.get("code") if isinstance(request.arguments, dict) else None
        if not isinstance(code, str) or not code.strip():
            raise _err("INVALID_ARGUMENTS", "arguments.code must be a non-empty string")
        if self.scene.changed():
            self.executor.reset()  # a new scene started: fresh namespace
        execution = _Execution(key, arguments_json, events, phase="running")
        self._executions[key] = execution
        self._active = key

        def work() -> None:
            try:
                out = self.executor.run(code)
            except BaseException as exc:  # executor already catches model errors; this is a node bug
                out = {"ok": False, "output": "", "error": f"internal error: {exc!r}", "fault": True,
                       "cancelled": False, "elapsed_s": 0.0, "ledger": []}
            self.done.put((key, code, out))

        threading.Thread(target=work, name="robot-run", daemon=True).start()
        logger.info("run accepted invocation=%s attempt=%s", key.invocation_id, key.attempt_id)
        return ToolAccepted(details={"idempotent": False})

    async def cancel(self, key: ToolExecutionKey, reason: str | None = None) -> ToolControlResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolControlResponse(command="cancel", status="rejected",
                                       error=ToolError(code="FORGE_EXECUTION_NOT_FOUND", message="execution key is unknown"))
        if execution.result is not None:
            return ToolControlResponse(command="cancel", status="terminal")
        if execution.phase != "stopping":
            self.executor.cancel()
            execution.phase = "stopping"
        return ToolControlResponse(command="cancel", status="accepted", details={"reason": reason})

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        execution = self._executions.get(key)
        if execution is None:
            return ToolExecutionStatus(phase="unknown", error=ToolError(code="FORGE_EXECUTION_NOT_FOUND",
                                                                        message="execution key is unknown or expired"))
        err = execution.result.error if execution.result is not None and execution.phase == "failed" else None
        return ToolExecutionStatus(phase=execution.phase, error=err, details={"operation": OPERATION})  # type: ignore[arg-type]

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolResultResponse(status="not_found")
        if execution.result is None:
            return ToolResultResponse(status="pending")
        return ToolResultResponse(status="available", result=execution.result)

    async def finish(self, key: ToolExecutionKey, code: str, out: dict[str, Any]) -> None:
        execution = self._executions.get(key)
        if execution is None or execution.result is not None:
            return
        self._write_ledger(key, code, out)
        outputs = {k: out.get(k) for k in ("ok", "output", "error", "elapsed_s")}
        outputs["primitive_calls"] = len(out.get("ledger", []))
        outputs["images"] = out.get("images") or []  # image files of show(): PAOS hands them to the model
        if out.get("cancelled"):
            execution.phase, event = "cancelled", "cancelled"
            execution.result = ToolResult(status="cancelled", outputs=outputs)
        elif out.get("fault"):
            execution.phase, event = "failed", "executor_failed"
            execution.result = ToolResult(status="failed", outputs=outputs,
                                          error=ToolError(code="ROBOT_FAULT", message=(out.get("error") or "")[-500:]))
        else:  # model-code errors are domain results: the model reads them and repairs
            execution.phase, event = "completed", "executor_completed"
            execution.result = ToolResult(status="succeeded", outputs=outputs)
        if self._active == key:
            self._active = None
        try:
            await execution.emitter.emit(ToolEvent(type=event, data={"ok": bool(out.get("ok"))}))
            logger.info("run finished invocation=%s phase=%s event=%s elapsed=%.1fs",
                        key.invocation_id, execution.phase, event, out.get("elapsed_s") or 0.0)
        except Exception:
            logger.exception("failed to emit terminal event")
        self._executions.move_to_end(key)
        while sum(e.result is not None for e in self._executions.values()) > self._cache:
            old = next(k for k, e in self._executions.items() if e.result is not None)
            self._executions.pop(old)

    def begin_shutdown(self) -> None:
        self._accepting = False
        if self._active is not None:
            self.executor.cancel()

    def _write_ledger(self, key: ToolExecutionKey, code: str, out: dict[str, Any]) -> None:
        if self.ledger_path is None:
            return
        rec = {"t": time.time(), "invocation_id": key.invocation_id, "attempt_id": key.attempt_id,
               "code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code": code,
               **{k: out.get(k) for k in ("ok", "output", "error", "fault", "cancelled", "elapsed_s", "ledger", "images")}}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


class SceneMarker:
    """Detects a new scene by the deadline file that run_scene.sh rewrites."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._last: str | None = self._read()

    def _read(self) -> str | None:
        if self.path is None or not self.path.exists():
            return None
        return self.path.read_text().strip()

    def changed(self) -> bool:
        now = self._read()
        if now != self._last:
            self._last = now
            return True
        return False


def _params(event: dict[str, Any]) -> dict[str, Any]:
    md = event.get("metadata")
    if not isinstance(md, dict):
        return {}
    p = md.get("parameters")
    return p if isinstance(p, dict) else md


class Runner:
    def __init__(self, node: Any, cfg: dict[str, Any], base_dir: Path) -> None:
        from forge_msgs import GripperCommandResult, JointState, MoveJointsResult, MovePoseResult

        self._msgs = {"move_pose_result": MovePoseResult, "move_joints_result": MoveJointsResult,
                      "gripper_result": GripperCommandResult}
        self._JointState = JointState
        self.node = node
        self.bus = Bus()
        arms = load_arm_configs(cfg.get("arms", {}), base_dir)
        self.arm_names = list(arms)
        self.backend = DoraArmBackend(arms, MotionConfig(**cfg.get("motion", {})), self.bus)
        robot_cfg = RobotConfig(**cfg.get("robot", {}))
        self.perception = self._make_perception(cfg, robot_cfg)
        self.robot = Robot(self.backend, self.perception, robot_cfg, vision=self._make_vision(cfg))
        look_dir = (cfg.get("vision") or {}).get("save_dir")
        self.robot.look_dir = Path(look_dir).expanduser() if look_dir else None
        if cfg.get("show_dir"):
            self.robot.show_dir = Path(cfg["show_dir"]).expanduser()
        if cfg.get("grip_snapshot_dir"):
            self.robot.grip_snapshot_dir = Path(cfg["grip_snapshot_dir"]).expanduser()
        self.executor = Executor(self.robot)
        self.done: queue.Queue = queue.Queue()
        ledger = cfg.get("ledger_path")
        self.endpoint = RunEndpoint(self.executor, self.done, Path(ledger).expanduser() if ledger else None,
                                    SceneMarker(robot_cfg.deadline_file))
        self.iid = str(uuid.uuid4())
        self.handler = ToolEndpointHandler(DESCRIPTOR, endpoint_instance_id=self.iid, operations={OPERATION: self.endpoint})
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(DESCRIPTOR, self.iid)
        self.aio = asyncio.Runner()
        self._registered = False

    @staticmethod
    def _make_vision(cfg: dict[str, Any]):
        """look(): OpenAI-compatible vision model (``vision:`` in the YAML; absent = disabled)."""
        vcfg = dict(cfg.get("vision") or {})
        if not vcfg or vcfg.pop("enabled", True) is False:
            return None
        vcfg.pop("save_dir", None)
        from .vision import VisionClient, VisionConfig

        return VisionClient(VisionConfig(**vcfg))

    def _make_perception(self, cfg: dict[str, Any], robot_cfg: RobotConfig):
        cams = cfg.get("cameras") or []
        if not cams:
            return None
        from forge_msgs import Image

        from .perception import CameraConfig, ColorSegmenter, DoraPerception, Sam3Segmenter

        self._Image = Image
        seg_cfg = dict(cfg.get("segmenter") or {})
        kind = seg_cfg.pop("kind", "sam3")
        if kind == "sam3":
            seg = Sam3Segmenter(**seg_cfg)  # loads in the background
        elif kind == "color":
            seg = ColorSegmenter(**seg_cfg)
        else:
            raise ValueError(f"unknown segmenter {kind!r}")
        self.camera_names = [c["name"] for c in cams]
        return DoraPerception([CameraConfig(**c) for c in cams], seg, self.backend, head=robot_cfg.head_camera)

    def _handle_camera(self, input_id: str, value: Any) -> bool:
        if self.perception is None:
            return False
        name, _, kind = input_id.rpartition("_")
        if name not in getattr(self, "camera_names", []) or kind not in ("color", "depth"):
            return False
        img = self._Image.from_arrow(value).to_numpy()
        if kind == "color":
            self.perception.on_color(name, img)
        else:
            self.perception.on_depth(name, img)
        return True

    async def _send_batch(self, batch: pa.RecordBatch) -> None:
        self.node.send_output("tool_out", batch)

    def _send_envelope(self, envelope) -> None:
        self.node.send_output("tool_out", tool_envelope_to_message(envelope).to_arrow())

    def _drive_lease(self) -> None:
        env = self.lease.poll(time.monotonic())
        if env is not None:
            if env.message_type != "endpoint.register" or self._registered:
                logger.info("lease sends %s", env.message_type)
            self._registered = True
            self._send_envelope(env)

    def _flush(self) -> None:
        for output_id, payload, md in self.bus.drain():
            if md.get("cancel"):
                self.node.send_output(output_id, pa.array([True], type=pa.bool_()), metadata={"goal_id": md["goal_id"]})
            else:
                self.node.send_output(output_id, payload, metadata=md)
        while True:
            try:
                key, code, out = self.done.get_nowait()
            except queue.Empty:
                break
            self.aio.run(self.endpoint.finish(key, code, out))

    def _handle(self, input_id: str, event: dict[str, Any]) -> None:
        if input_id == "tool_in":
            message = ToolMessage.from_arrow(event["value"])
            env = tool_message_to_envelope(message)
            if env.message_type == "endpoint.registry.response":
                self.lease.acknowledge(env, time.monotonic())
                return
            if env.message_type not in ("tool.status.request",):
                logger.info("tool_in %s invocation=%s", env.message_type, env.invocation_id)
            for batch in self.aio.run(self.binding.dispatch_input(event["value"])):
                self.node.send_output("tool_out", batch)
            return
        if self._handle_camera(input_id, event["value"]):
            return
        arm, _, kind = input_id.partition("_")
        if arm not in self.arm_names:
            return
        if kind == "state":
            js = self._JointState.from_arrow(event["value"])
            self.backend.on_joint_state(list(js.name), list(js.position))
        elif kind in self._msgs:
            p = _params(event)
            self.bus.resolve(str(p.get("goal_id", "")), str(p.get("goal_status", "")), self._msgs[kind].from_arrow(event["value"]))

    def run(self) -> int:
        self._drive_lease()
        for event in self.node:
            etype = event.get("type")
            if etype == "STOP":
                self.endpoint.begin_shutdown()
                env = self.lease.stop()
                if env is not None:
                    self._send_envelope(env)
                self._flush()
                self.aio.close()
                return 0
            if etype == "ERROR":
                logger.error("dora error: %s", event.get("error"))
                continue
            if etype != "INPUT":
                continue
            self._drive_lease()
            try:
                self._handle(str(event.get("id")), event)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("ignored invalid input %s: %s", event.get("id"), exc)
            self._flush()
            self._drive_lease()
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="piper-dual-tabletop skill_server Dora node")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # model code (worker thread) can be CPU-heavy (IK); switch threads often so the Dora loop
    # keeps renewing the endpoint lease and forwarding results promptly
    import sys
    sys.setswitchinterval(0.001)
    path = Path(args.config).resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    from dora import Node

    return Runner(Node(), cfg, path.parent).run()


if __name__ == "__main__":
    raise SystemExit(main())
