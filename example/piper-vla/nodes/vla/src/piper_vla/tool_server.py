"""Dora node: the Forge tools of the piper-vla skill (Gateway endpoint ``vla.server``).

Tools (``gateway.yaml`` maps them onto these operations):

* ``vla.policies`` (query)  trained policies: name, state (loading / ready / error), description, task
* ``vla.state``    (query)  joints and gripper opening per arm, the current run and homing
* ``vla.home``     (action) move to the demonstrations' start pose with the gripper open
* ``vla.run``      (action) let policy ``policy`` drive the arm for ``seconds``, then hold still

The node talks to ``vla_policy`` (piper_vla.policy_node) with ``policy_cmd`` JSON and reads its
``policy_status``. Nothing here stops on time except a run's own ``seconds``, which the model
chooses; an Action cancel stops a run or a homing at once (the driver then holds the last
target). A policy node that stops publishing its status is reported as a failed Action.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

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

from .endpoint_lease import EndpointLease

logger = logging.getLogger("piper_vla.tool_server")

ENDPOINT_ID = "vla.server"
DESCRIPTOR = ToolEndpointDescriptor(
    protocol_version=TOOL_ENDPOINT_PROTOCOL,
    endpoint_id=ENDPOINT_ID,
    operations=(
        ToolOperationDescriptor(name="policies", semantics="query"),
        ToolOperationDescriptor(name="state", semantics="query"),
        ToolOperationDescriptor(name="home", semantics="action", cancellable=True, status_supported=True,
                                max_concurrency=1),
        ToolOperationDescriptor(name="run", semantics="action", cancellable=True, status_supported=True,
                                max_concurrency=1),
    ),
)
STATUS_STALE_S = 3.0


class PolicyNodeDown(RuntimeError):
    pass


def _err(code: str, message: str, *, retryable: bool = False) -> ToolEndpointError:
    return ToolEndpointError(ToolError(code=code, message=message, retryable=retryable))


class PolicyLink:
    """Latest ``policy_status`` plus the ``policy_cmd`` outbox (sent by the Dora loop)."""

    def __init__(self) -> None:
        self.status: dict[str, Any] = {}
        self.status_t = 0.0
        self.cv = threading.Condition()
        self.outbox: queue.Queue = queue.Queue()

    def on_status(self, raw: str) -> None:
        st = json.loads(raw)
        with self.cv:
            self.status, self.status_t = st, time.monotonic()
            self.cv.notify_all()

    def send(self, cmd: dict[str, Any]) -> None:
        self.outbox.put(cmd)

    def alive(self) -> bool:
        return self.status_t > 0 and time.monotonic() - self.status_t < STATUS_STALE_S

    def snapshot(self) -> dict[str, Any]:
        if not self.alive():
            raise PolicyNodeDown("vla_policy is not publishing its status")
        with self.cv:
            return json.loads(json.dumps(self.status))

    def wait_for(self, pred: Callable[[dict[str, Any]], bool], cancelled: Callable[[], bool]) -> dict[str, Any]:
        """Block until ``pred(status)``; no time limit, only cancel or a dead policy node end it."""
        with self.cv:
            while not pred(self.status):
                if cancelled():
                    raise InterruptedError
                if not self.alive():
                    raise PolicyNodeDown("vla_policy stopped publishing its status")
                self.cv.wait(0.1)
            return self.status


# ---------------------------------------------------------------------- operations

def op_policies(link: PolicyLink, args: dict[str, Any]) -> dict[str, Any]:
    return {"policies": link.snapshot().get("policies", [])}


def op_state(link: PolicyLink, args: dict[str, Any]) -> dict[str, Any]:
    st = link.snapshot()
    joints, grip = {}, {}
    for arm, values in (st.get("state") or {}).items():
        joints[arm], grip[arm] = [round(v, 4) for v in values[:-1]], round(values[-1], 4)
    return {"joints": joints, "gripper_width_m": grip, "run": st.get("run"), "home": st.get("home")}


def op_home(link: PolicyLink, args: dict[str, Any], cancelled: Callable[[], bool]) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    t0 = time.monotonic()
    link.send({"op": "home", "run_id": run_id})
    try:
        st = link.wait_for(lambda s: (s.get("home") or {}).get("run_id") == run_id and (s.get("home") or {}).get("done"),
                           cancelled)
    except InterruptedError:
        link.send({"op": "stop", "run_id": run_id})
        raise
    err = (st.get("home") or {}).get("error") or None
    return {"reached": not err, "error": err, "elapsed_s": round(time.monotonic() - t0, 2)}


def op_run(link: PolicyLink, args: dict[str, Any], cancelled: Callable[[], bool]) -> dict[str, Any]:
    name, seconds, task = args["policy"], float(args["seconds"]), args.get("task") or None
    pol = {p["name"]: p for p in link.snapshot().get("policies", [])}
    if name not in pol:
        return {"policy": name, "steps": 0, "error": f"unknown policy {name!r}; available: {sorted(pol)}"}
    if pol[name]["state"] == "loading":   # loading always ends (ready or error): wait for it
        st = link.wait_for(lambda s: any(p["name"] == name and p["state"] != "loading" for p in s.get("policies", [])),
                           cancelled)
        pol = {p["name"]: p for p in st.get("policies", [])}
    if pol[name]["state"] != "ready":
        return {"policy": name, "steps": 0, "error": f"policy {name} is {pol[name]['state']}: {pol[name].get('error', '')}"}
    run_id = str(uuid.uuid4())
    link.send({"op": "start", "run_id": run_id, "policy": name, "task": task})
    record: dict[str, Any] = {}
    try:
        st = link.wait_for(lambda s: (s.get("run") or {}).get("run_id") == run_id, cancelled)
        record = dict(st["run"])
        end = time.monotonic() + seconds
        while time.monotonic() < end and not cancelled():   # the model's chosen duration, nothing else
            with link.cv:
                link.cv.wait(min(0.1, max(0.0, end - time.monotonic())))
                run = link.status.get("run") or {}
            if run.get("run_id") == run_id:
                record = dict(run)
                if record.get("error"):
                    break
            elif not link.alive():
                raise PolicyNodeDown("vla_policy stopped publishing its status during the run")
    except InterruptedError:
        pass
    finally:
        link.send({"op": "stop", "run_id": run_id})
        # confirmed stopped before returning, so a following vla.home cannot race the policy
        link.wait_for(lambda s: (s.get("run") or {}).get("run_id") != run_id, lambda: False)
    steps = int(record.get("steps") or 0)
    st = link.status
    return {"policy": name, "task": record.get("task") or task or "", "seconds": seconds, "steps": steps,
            "skipped": int(record.get("skipped") or 0), "rate_hz": round(steps / seconds, 1) if seconds else 0.0,
            "infer_ms": float(record.get("infer_ms_mean") or 0.0), "error": record.get("error") or None,
            "cancelled": bool(cancelled()),
            "gripper_width_m": {a: round(v[-1], 4) for a, v in (st.get("state") or {}).items()}}


# ---------------------------------------------------------------------- Forge SPI

class QueryOp:
    def __init__(self, link: PolicyLink, fn: Callable[[PolicyLink, dict[str, Any]], dict[str, Any]]) -> None:
        self.link, self.fn = link, fn

    async def query(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        try:
            return ToolResult(status="succeeded", outputs=self.fn(self.link, dict(request.arguments or {})))
        except PolicyNodeDown as exc:
            return ToolResult(status="failed", error=ToolError(code="POLICY_NODE_DOWN", message=str(exc)))


@dataclass
class _Execution:
    key: ToolExecutionKey
    arguments_json: str
    emitter: ToolEventEmitter
    cancel: threading.Event
    phase: str = "running"
    result: ToolResult | None = None


class ActionOp:
    """Action SPI: ``start`` returns at once; the work runs in a thread, ``finish`` publishes it."""

    def __init__(self, name: str, link: PolicyLink, fn, validate, done: queue.Queue, cache: int = 32) -> None:
        self.name, self.link, self.fn, self.validate, self.done = name, link, fn, validate, done
        self._executions: OrderedDict[ToolExecutionKey, _Execution] = OrderedDict()
        self._cache = cache

    async def start(self, request: ToolRequest, context: ToolContext, events: ToolEventEmitter) -> ToolAccepted:
        args = dict(request.arguments or {})
        key = context.execution_key
        arguments_json = json.dumps(args, sort_keys=True, separators=(",", ":"))
        existing = self._executions.get(key)
        if existing is not None:
            if existing.arguments_json != arguments_json:
                raise _err("FORGE_EXECUTION_CONFLICT", "execution key reused with different arguments")
            return ToolAccepted(details={"idempotent": True})
        problem = self.validate(args)
        if problem:
            raise _err("INVALID_ARGUMENTS", problem)
        if not self.link.alive():
            raise _err("POLICY_NODE_DOWN", "vla_policy is not publishing its status", retryable=True)
        execution = _Execution(key, arguments_json, events, threading.Event())
        self._executions[key] = execution

        def work() -> None:
            try:
                out, status, error = self.fn(self.link, args, execution.cancel.is_set), "succeeded", None
                if execution.cancel.is_set():
                    status = "cancelled"
            except PolicyNodeDown as exc:
                out, status, error = {}, "failed", ToolError(code="POLICY_NODE_DOWN", message=str(exc))
            except InterruptedError:
                out, status, error = {}, "cancelled", None
            except Exception as exc:  # a bug in this node: report it, keep serving
                logger.exception("%s failed", self.name)
                out, status, error = {}, "failed", ToolError(code="INTERNAL", message=repr(exc)[:400])
            self.done.put((self, key, out, status, error))

        threading.Thread(target=work, name=f"vla-{self.name}", daemon=True).start()
        return ToolAccepted(details={"idempotent": False})

    async def cancel(self, key: ToolExecutionKey, reason: str | None = None) -> ToolControlResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolControlResponse(command="cancel", status="rejected",
                                       error=ToolError(code="FORGE_EXECUTION_NOT_FOUND", message="execution key is unknown"))
        if execution.result is not None:
            return ToolControlResponse(command="cancel", status="terminal")
        execution.cancel.set()
        execution.phase = "stopping"
        return ToolControlResponse(command="cancel", status="accepted", details={"reason": reason})

    async def status(self, key: ToolExecutionKey) -> ToolExecutionStatus:
        execution = self._executions.get(key)
        if execution is None:
            return ToolExecutionStatus(phase="unknown", error=ToolError(code="FORGE_EXECUTION_NOT_FOUND",
                                                                        message="execution key is unknown or expired"))
        err = execution.result.error if execution.result is not None and execution.phase == "failed" else None
        return ToolExecutionStatus(phase=execution.phase, error=err, details={"operation": self.name})  # type: ignore[arg-type]

    async def result(self, key: ToolExecutionKey) -> ToolResultResponse:
        execution = self._executions.get(key)
        if execution is None:
            return ToolResultResponse(status="not_found")
        if execution.result is None:
            return ToolResultResponse(status="pending")
        return ToolResultResponse(status="available", result=execution.result)

    async def finish(self, key: ToolExecutionKey, out: dict[str, Any], status: str, error: ToolError | None) -> None:
        execution = self._executions.get(key)
        if execution is None or execution.result is not None:
            return
        execution.result = ToolResult(status=status, outputs=out, error=error)  # type: ignore[arg-type]
        execution.phase = {"succeeded": "completed", "cancelled": "cancelled"}.get(status, "failed")
        event = {"succeeded": "executor_completed", "cancelled": "cancelled"}.get(status, "executor_failed")
        try:
            await execution.emitter.emit(ToolEvent(type=event, data={"operation": self.name}))
        except Exception:
            logger.exception("failed to emit terminal event")
        logger.info("%s finished invocation=%s status=%s", self.name, key.invocation_id, status)
        self._executions.move_to_end(key)
        while sum(e.result is not None for e in self._executions.values()) > self._cache:
            old = next(k for k, e in self._executions.items() if e.result is not None)
            self._executions.pop(old)


def _validate_run(args: dict[str, Any]) -> str | None:
    if not isinstance(args.get("policy"), str) or not args["policy"]:
        return "arguments.policy must be a policy name (see vla.policies)"
    s = args.get("seconds")
    if not isinstance(s, (int, float)) or isinstance(s, bool) or s <= 0:
        return "arguments.seconds must be a positive number"
    if args.get("task") is not None and not isinstance(args["task"], str):
        return "arguments.task must be a string"
    return None


# ---------------------------------------------------------------------- Dora loop

class Server:
    def __init__(self, node: Any) -> None:
        self.node = node
        self.link = PolicyLink()
        self.done: queue.Queue = queue.Queue()
        ops = {
            "policies": QueryOp(self.link, op_policies),
            "state": QueryOp(self.link, op_state),
            "home": ActionOp("home", self.link, op_home, lambda a: None, self.done),
            "run": ActionOp("run", self.link, op_run, _validate_run, self.done),
        }
        self.iid = str(uuid.uuid4())
        self.handler = ToolEndpointHandler(DESCRIPTOR, endpoint_instance_id=self.iid, operations=ops)
        self.binding = DoraToolEndpointBinding(self.handler, event_sink=self._send_batch)
        self.lease = EndpointLease(DESCRIPTOR, self.iid)
        self.aio = asyncio.Runner()
        self._registered = False

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
        while True:
            try:
                cmd = self.link.outbox.get_nowait()
            except queue.Empty:
                break
            self.node.send_output("policy_cmd", pa.array([json.dumps(cmd)]))
        while True:
            try:
                op, key, out, status, error = self.done.get_nowait()
            except queue.Empty:
                break
            self.aio.run(op.finish(key, out, status, error))

    def _handle(self, input_id: str, value: Any) -> None:
        if input_id == "tool_in":
            env = tool_message_to_envelope(ToolMessage.from_arrow(value))
            if env.message_type == "endpoint.registry.response":
                self.lease.acknowledge(env, time.monotonic())
                return
            for batch in self.aio.run(self.binding.dispatch_input(value)):
                self.node.send_output("tool_out", batch)
        elif input_id == "policy_status":
            self.link.on_status(value[0].as_py())

    def run(self) -> int:
        self._drive_lease()
        for event in self.node:
            etype = event.get("type")
            if etype == "STOP":
                env = self.lease.stop()
                if env is not None:
                    self._send_envelope(env)
                self.aio.close()
                return 0
            if etype != "INPUT":
                continue
            self._drive_lease()
            try:
                self._handle(str(event.get("id")), event["value"])
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("ignored invalid input %s: %s", event.get("id"), exc)
            self._flush()
            self._drive_lease()
        return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="piper-vla Forge tool server (Dora node)")
    ap.add_argument("--config", help="vla_policy.yaml (read for logging only)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.config:
        raw = yaml.safe_load(open(args.config)) or {}
        logger.info("policies configured: %s", [p.get("name") for p in raw.get("policies") or []])
    from dora import Node

    return Server(Node()).run()


if __name__ == "__main__":
    raise SystemExit(main())
