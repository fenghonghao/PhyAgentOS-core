"""piper-vla tools against a fake policy node, and the policy node's homing."""

import json
import threading
import time

import numpy as np
import pytest

from piper_vla import policy_node as pn
from piper_vla.spec import load_config
from piper_vla.tool_server import (DESCRIPTOR, PolicyLink, PolicyNodeDown, Server, _validate_run, op_home, op_run,
                                   op_state)

READY = [{"name": "cup", "state": "ready", "error": "", "description": "cup into bowl, 12 s", "task": "put the cup in the bowl"}]


class FakePolicyNode:
    """Answers policy_cmd like piper_vla.policy_node: run / stop / home, status every 50 ms."""

    def __init__(self, link: PolicyLink, policies, home_s: float = 0.2, alive: bool = True):
        self.link, self.policies, self.home_s = link, policies, home_s
        self.run, self.home, self.log = None, None, []
        self.state = {"left": [0.1, 0.5, -0.2, 0.0, 0.3, 0.0, 0.07]}
        self.alive = alive
        threading.Thread(target=self._loop, daemon=True).start()

    def publish(self):
        self.link.on_status(json.dumps({"policies": self.policies, "run": self.run, "home": self.home,
                                        "state": self.state, "last_command": {}}))

    def _loop(self):
        while True:
            try:
                cmd = self.link.outbox.get(timeout=0.05)
                self.log.append(cmd)
                if cmd["op"] == "start":
                    self.run = {"run_id": cmd["run_id"], "policy": cmd["policy"], "task": cmd.get("task") or "",
                                "steps": 0, "skipped": 0, "busy": 0, "infer_ms_mean": 4.0, "infer_ms_max": 6.0, "error": ""}
                elif cmd["op"] == "stop":
                    self.run = None
                    if self.home and self.home["run_id"] == cmd.get("run_id"):
                        self.home = None
                elif cmd["op"] == "home":
                    self.home = {"run_id": cmd["run_id"], "done": False, "error": "", "t0": time.monotonic()}
            except Exception:
                pass
            if self.run is not None:
                self.run["steps"] += 1
            if self.home is not None and not self.home["done"] and time.monotonic() - self.home["t0"] > self.home_s:
                self.home["done"] = True
            if self.alive:
                self.publish()


def link_with(policies=READY, **kw):
    link = PolicyLink()
    node = FakePolicyNode(link, policies, **kw)
    time.sleep(0.1)
    return link, node


def test_run_drives_for_the_chosen_seconds_then_stops():
    link, node = link_with()
    t0 = time.monotonic()
    out = op_run(link, {"policy": "cup", "seconds": 0.4, "task": "put the cup in the bowl"}, lambda: False)
    assert 0.35 < time.monotonic() - t0 < 2.0
    assert out["steps"] > 0 and out["error"] is None and not out["cancelled"]
    assert out["gripper_width_m"] == {"left": 0.07}
    assert [c["op"] for c in node.log] == ["start", "stop"] and node.run is None


def test_unknown_or_failed_policy_does_not_move():
    link, node = link_with([{**READY[0], "state": "error", "error": "OOM"}])
    assert "unknown policy" in op_run(link, {"policy": "nope", "seconds": 1}, lambda: False)["error"]
    out = op_run(link, {"policy": "cup", "seconds": 1}, lambda: False)
    assert out["steps"] == 0 and "OOM" in out["error"] and node.log == []


def test_loading_policy_is_waited_for():
    link, node = link_with([{**READY[0], "state": "loading"}])
    threading.Timer(0.3, lambda: setattr(node, "policies", READY)).start()
    out = op_run(link, {"policy": "cup", "seconds": 0.2}, lambda: False)
    assert out["error"] is None and out["steps"] > 0


def test_cancel_stops_a_run_at_once():
    link, node = link_with()
    t0 = time.monotonic()
    out = op_run(link, {"policy": "cup", "seconds": 30}, lambda: time.monotonic() - t0 > 0.2)
    assert time.monotonic() - t0 < 2 and out["cancelled"] and node.run is None


def test_dead_policy_node_is_reported():
    link = PolicyLink()
    with pytest.raises(PolicyNodeDown):
        op_state(link, {})


def test_home_and_state():
    link, node = link_with()
    out = op_home(link, {}, lambda: False)
    assert out["reached"] and out["error"] is None
    st = op_state(link, {})
    assert st["joints"]["left"] == [0.1, 0.5, -0.2, 0.0, 0.3, 0.0] and st["gripper_width_m"]["left"] == 0.07


def test_home_cancel_sends_stop():
    link, node = link_with(home_s=30)
    t0 = time.monotonic()
    with pytest.raises(InterruptedError):
        op_home(link, {}, lambda: time.monotonic() - t0 > 0.2)
    time.sleep(0.15)
    assert [c["op"] for c in node.log] == ["home", "stop"] and node.home is None


def test_run_arguments_are_checked():
    assert _validate_run({"policy": "cup", "seconds": 2}) is None
    assert "seconds" in _validate_run({"policy": "cup", "seconds": 0})
    assert "policy" in _validate_run({"seconds": 2})
    assert "seconds" in _validate_run({"policy": "cup", "seconds": True})


def test_server_wires_all_four_operations():
    class Node:
        def send_output(self, *a, **k):
            pass

    s = Server(Node())
    assert [o.name for o in DESCRIPTOR.operations] == ["policies", "state", "home", "run"]
    s.aio.close()


# ---------------------------------------------------------------- policy node homing

class RecNode:
    def __init__(self):
        self.out = []

    def send_output(self, output_id, value, metadata=None):
        self.out.append((output_id, value))


def test_policy_node_homes_smoothly_and_stop_cancels(monkeypatch):
    raw, spec = load_config(str(__import__("pathlib").Path(__file__).resolve().parents[3] / "profiles/sim-piper/vla_policy.yaml"))
    raw = {**raw, "policies": []}
    node = RecNode()
    runner = pn.PolicyRunner(node, raw, spec)
    clock = [100.0]
    monkeypatch.setattr(pn.time, "monotonic", lambda: clock[0])
    arm = spec.arms[0]
    runner.state[arm.name] = (clock[0], {**{j: 0.4 for j in arm.joint_names}, arm.gripper_joint: 0.0})
    runner.on_cmd({"op": "home", "run_id": "h1"})
    dur = runner.homing["duration"]
    target = np.array([*arm.home_joints, arm.home_gripper_m])
    sent = []
    for k in range(int(dur * 30) + 3):
        clock[0] = 100.0 + k / 30.0
        runner.state[arm.name] = (clock[0], runner.state[arm.name][1])
        n0 = len(node.out)
        runner.tick()
        sent += [v for o, v in node.out[n0:] if o == arm.command_output]
    final = np.asarray(runner.last_cmd)
    assert np.allclose(final, target, atol=1e-6) and runner.status()["home"]["done"]
    assert len(sent) >= int(dur * 30)              # one command per tick along the path
    runner.on_cmd({"op": "home", "run_id": "h2"})
    runner.on_cmd({"op": "stop", "run_id": "h2"})
    assert runner.homing is None
