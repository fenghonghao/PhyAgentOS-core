"""The gripper node: its core on a simulated finger at 50 Hz, its Dora loop on a fake node."""

import json
import os
from pathlib import Path

import pytest
import yaml
from forge_msgs import GripperCommandGoal, GripperCommandResult, JointCommand, JointState

from gripper_stream import GripperStream, StreamConfig, run
from tabletop.dora_backend import gripper_fault

ROOT = Path(__file__).resolve().parents[3]
TICK_NS = 20_000_000
STALL_S = 0.25


def config():
    return StreamConfig(joint_name="gripper", min_position=0.0, max_position=0.105, feedback_timeout_ms=300,
                        goal_tolerance=0.001, stall_velocity_threshold=0.0005, stall_timeout_ms=250)


class Rig:
    """The core as the dataflow runs it: a joint state and a tick every 20 ms, on a finger that follows
    the last command at full speed and stops on an object."""

    def __init__(self, width=0.05, object_width=0.0, finger_speed=0.5):
        self.core = GripperStream(config())
        self.now = 0
        self.width = width
        self.object_width = object_width
        self.finger_speed = finger_speed
        self.command = None
        self.sent = []
        self.moved = []
        self.core.on_state(width, self.now)

    def goal(self, position, speed=None, goal_id="g1", effort=None):
        goal = GripperCommandGoal(position=position, max_velocity=speed, max_effort=effort)
        return self.core.on_goal(goal_id, goal, self.now)

    def step(self, state=True):
        self.now += TICK_NS
        if self.command is not None:
            room = self.finger_speed * TICK_NS / 1e9
            width = self.width + max(-room, min(room, self.command - self.width))
            if width < self.width:
                width = max(width, min(self.width, self.object_width))
            self.moved.append(abs(width - self.width))
            self.width = width
        if state:
            self.core.on_state(self.width, self.now)
        out = self.core.tick(self.now)
        if out is not None and out.command is not None:
            self.command = out.command
            self.sent.append(out.command)
        return out

    def until_result(self, limit_s=10.0, state=True):
        start = self.now
        while (self.now - start) / 1e9 < limit_s:
            out = self.step(state)
            if out is not None and out.result is not None:
                return out, (self.now - start) / 1e9
        raise AssertionError(f"no result in {limit_s} s; fingers at {self.width}")


# ---------------------------------------------------------------- the target

def test_a_goal_with_a_speed_moves_the_target_that_fast():
    rig = Rig(width=0.05)
    assert rig.goal(0.0, speed=0.02) is None
    for _ in range(25):                                              # half a second
        rig.step()
    steps = [a - b for a, b in zip(rig.sent, rig.sent[1:])]
    assert all(s == pytest.approx(0.02 * 0.02) for s in steps), steps
    assert rig.sent[-1] == pytest.approx(0.05 - 0.02 * 0.5)


def test_a_goal_without_a_speed_commands_the_goal_at_once():
    rig = Rig(width=0.05)
    rig.goal(0.03)
    rig.step()
    assert rig.sent == [0.03]


def test_a_slow_target_starts_where_the_fingers_are():
    rig = Rig(width=0.05)
    rig.goal(0.0, speed=0.02)
    rig.step()
    assert rig.sent[0] == pytest.approx(0.05, abs=0.001)


def test_nothing_is_sent_before_the_first_goal():
    rig = Rig()
    assert rig.step() is None and rig.sent == []


def test_the_target_keeps_being_sent_between_goals():
    rig = Rig(width=0.05)
    rig.goal(0.03)
    rig.until_result()
    rig.sent.clear()
    for _ in range(5):
        rig.step()
    assert rig.sent == [0.03] * 5


# ---------------------------------------------------------------- how a goal ends

def test_a_slow_close_on_an_object_ends_soon_after_the_fingers_stop():
    rig = Rig(width=0.045, object_width=0.03)
    rig.goal(0.0, speed=0.02)
    out, elapsed = rig.until_result()
    assert out.goal_id == "g1" and out.status == "succeeded"
    assert out.result.error_code == "SUCCESS" and out.result.stalled and not out.result.reached_goal
    assert out.result.position == pytest.approx(0.03)
    assert elapsed == pytest.approx(0.015 / 0.02 + STALL_S, abs=0.1)
    assert max(rig.moved) <= 0.02 * 0.02 + 1e-9                      # the fingers never went faster than asked


def test_after_closing_on_an_object_the_fingers_keep_pressing():
    rig = Rig(width=0.045, object_width=0.03)
    rig.goal(0.0, speed=0.02)
    out, _ = rig.until_result()
    assert out.command == 0.0
    rig.sent.clear()
    rig.step()
    assert rig.sent == [0.0] and rig.width == pytest.approx(0.03)


def test_a_slow_close_on_nothing_reaches_the_goal():
    rig = Rig(width=0.02)
    rig.goal(0.0, speed=0.02)
    out, elapsed = rig.until_result()
    assert out.result.error_code == "SUCCESS" and out.result.reached_goal and not out.result.stalled
    assert elapsed == pytest.approx(0.02 / 0.02, abs=0.15)


def test_fingers_that_cannot_start_are_a_stall():
    rig = Rig(width=0.03, object_width=0.03)
    rig.goal(0.0, speed=0.02)
    out, elapsed = rig.until_result()
    assert out.result.stalled and elapsed == pytest.approx(STALL_S, abs=0.05)


def test_a_full_speed_close_on_an_object_also_ends_on_the_stall():
    rig = Rig(width=0.08, object_width=0.03)
    rig.goal(0.0)
    out, elapsed = rig.until_result()
    assert out.result.stalled and out.result.position == pytest.approx(0.03)
    assert elapsed == pytest.approx(0.05 / 0.5 + STALL_S, abs=0.1)


def test_a_reading_that_only_jitters_counts_as_stopped():
    rig = Rig(width=0.03, object_width=0.03)
    rig.goal(0.0, speed=0.02)
    jitter = 0.000002                                                # 0.1 mm/s, under the threshold
    for i in range(30):
        rig.now += TICK_NS
        rig.core.on_state(0.03 + jitter * (i % 2), rig.now)
        out = rig.core.tick(rig.now)
        if out.result is not None:
            break
    assert out.result is not None and out.result.stalled


def test_a_stalled_close_is_not_a_fault_for_the_skill_server():
    rig = Rig(width=0.045, object_width=0.03)
    rig.goal(0.0, speed=0.02)
    out, _ = rig.until_result()
    assert gripper_fault(out.status, out.result.error_code, out.result.message) is None


def test_fingers_moving_slowly_with_the_target_are_not_stalled():
    rig = Rig(width=0.10)
    rig.goal(0.0, speed=0.02)
    out, elapsed = rig.until_result()
    assert out.result.reached_goal and elapsed == pytest.approx(0.10 / 0.02, abs=0.2)


# ---------------------------------------------------------------- refusals and faults

def test_a_second_goal_is_refused_while_one_runs():
    rig = Rig(width=0.05)
    rig.goal(0.0, speed=0.02)
    rig.step()
    refused = rig.goal(0.04, goal_id="g2")
    assert refused.goal_id == "g2" and refused.status == "aborted" and refused.result.error_code == "BUSY"
    rig.step()
    assert rig.sent[-1] < 0.05 and rig.sent[-1] != 0.04              # the first one carries on


def test_a_goal_outside_the_stroke_is_refused():
    rig = Rig()
    assert rig.goal(0.2).result.error_code == "POSITION_LIMIT_VIOLATION"


def test_a_goal_with_an_effort_is_refused():
    rig = Rig()
    assert rig.goal(0.0, effort=1.0).result.error_code == "UNSUPPORTED_EFFORT"


def test_a_goal_is_refused_with_no_state_yet():
    core = GripperStream(config())
    out = core.on_goal("g1", GripperCommandGoal(position=0.0), 0)
    assert out.status == "aborted" and out.result.error_code == "NO_FRESH_ROBOT_STATE"


def test_a_goal_is_refused_with_an_old_state():
    rig = Rig()
    rig.now += 400_000_000
    assert rig.goal(0.0).result.error_code == "NO_FRESH_ROBOT_STATE"


def test_state_that_stops_coming_aborts_the_goal_and_holds_where_the_fingers_were():
    rig = Rig(width=0.05)
    rig.goal(0.0, speed=0.02)
    out, elapsed = rig.until_result(state=False)
    assert out.status == "aborted" and out.result.error_code == "FEEDBACK_STALE"
    assert gripper_fault(out.status, out.result.error_code, out.result.message) is not None
    assert elapsed == pytest.approx(0.3, abs=0.05)
    assert out.command == pytest.approx(0.05)


def test_cancel_stops_the_target_where_the_fingers_are():
    rig = Rig(width=0.05)
    rig.goal(0.0, speed=0.02)
    for _ in range(10):
        rig.step()
    out = rig.core.on_cancel("g1", rig.now)
    assert out.status == "canceled" and out.result.error_code == "CANCELED"
    assert out.command == pytest.approx(rig.width)
    assert gripper_fault(out.status, out.result.error_code, out.result.message) is None


def test_cancel_of_another_goal_changes_nothing():
    rig = Rig(width=0.05)
    rig.goal(0.0, speed=0.02)
    assert rig.core.on_cancel("other", rig.now) is None
    assert rig.step().result is None


# ---------------------------------------------------------------- the Dora loop

class FakeNode:
    def __init__(self, events):
        self.events = events
        self.sent = []

    def __iter__(self):
        return iter(self.events)

    def send_output(self, output_id, value, metadata=None):
        self.sent.append((output_id, value, metadata))


def state_event(width):
    return {"type": "INPUT", "id": "joint_state", "metadata": {},
            "value": JointState(name=["joint1", "gripper"], position=[0.0, width]).to_arrow()}


def goal_event(position, speed=None, metadata=None):
    return {"type": "INPUT", "id": "gripper_goal", "metadata": {"goal_id": "g1"} if metadata is None else metadata,
            "value": GripperCommandGoal(position=position, max_velocity=speed).to_arrow()}


TICK = {"type": "INPUT", "id": "tick", "metadata": {}, "value": None}


def clock():
    now = iter(range(TICK_NS, 10_000 * TICK_NS, TICK_NS))
    return lambda: next(now)


def test_the_loop_sends_joint_commands_for_the_gripper_only():
    node = FakeNode([state_event(0.05), goal_event(0.03), TICK])
    run(node, GripperStream(config()), clock=clock())
    assert [out for out, _, _ in node.sent] == ["joint_command"]
    command = JointCommand.from_arrow(node.sent[0][1])
    assert command.name == ["gripper"] and command.mode == "position" and command.position == [0.03]


def test_the_loop_reports_the_result_as_the_skill_server_reads_it():
    node = FakeNode([state_event(0.03), goal_event(0.03), TICK])
    run(node, GripperStream(config()), clock=clock())
    (_, value, metadata), = [item for item in node.sent if item[0] == "gripper_result"]
    assert metadata == {"goal_id": "g1", "goal_status": "succeeded"}
    result = GripperCommandResult.from_arrow(value)
    assert result.error_code == "SUCCESS" and result.reached_goal and result.position == pytest.approx(0.03)


def test_the_loop_finds_the_goal_id_under_parameters():
    node = FakeNode([state_event(0.03), goal_event(0.03, metadata={"parameters": {"goal_id": "g7"}}), TICK])
    run(node, GripperStream(config()), clock=clock())
    assert [md for out, _, md in node.sent if out == "gripper_result"] == [{"goal_id": "g7", "goal_status": "succeeded"}]


def test_the_loop_cancels_by_goal_id():
    cancel = {"type": "INPUT", "id": "gripper_cancel", "metadata": {"goal_id": "g1"}, "value": None}
    node = FakeNode([state_event(0.05), goal_event(0.0, speed=0.02), TICK, cancel])
    run(node, GripperStream(config()), clock=clock())
    assert [md for out, _, md in node.sent if out == "gripper_result"] == [{"goal_id": "g1", "goal_status": "canceled"}]


def test_the_loop_survives_a_bad_message_and_ends_on_stop():
    bad = {"type": "INPUT", "id": "gripper_goal", "metadata": {"goal_id": "g1"}, "value": "not arrow"}
    stop = {"type": "STOP"}
    node = FakeNode([state_event(0.05), bad, goal_event(0.03), TICK, stop, TICK])
    run(node, GripperStream(config()), clock=clock())
    assert len(node.sent) == 1


def test_the_loop_writes_a_trace_of_each_goal(tmp_path):
    trace = tmp_path / "logs" / "trace.jsonl"
    events = [state_event(0.032), goal_event(0.03, speed=0.02), TICK, state_event(0.0305), TICK]
    run(FakeNode(events), GripperStream(config()), trace_path=trace, clock=clock())
    line = json.loads(trace.read_text())
    assert line["goal"] == 0.03 and line["max_velocity"] == 0.02 and line["error_code"] == "SUCCESS"
    assert line["reached_goal"] and not line["stalled"] and line["final"] == pytest.approx(0.0305)
    assert line["samples"] and all(len(sample) == 3 for sample in line["samples"])       # seconds, target, measured


# ---------------------------------------------------------------- the profiles run this node

@pytest.mark.parametrize("profile,node_id,driver", [("real-piper", "gripper", "piper"), ("sim-piper", "gripper_left", "mujoco")])
def test_the_profile_runs_this_node_at_50_hz(profile, node_id, driver):
    flow = yaml.safe_load((ROOT / "profiles" / profile / "dataflow.yaml").read_text())
    nodes = {n["id"]: n for n in flow["nodes"]}
    gripper = nodes[node_id]
    assert gripper["path"] == "${PAOS_SKILL_ROOT}/nodes/run_gripper.sh"
    assert gripper["inputs"]["tick"] == "dora/timer/millis/20"
    assert nodes[driver]["inputs"]["action/gripper"] == f"{node_id}/joint_command"
    assert set(gripper["outputs"]) == {"joint_command", "gripper_result"}


@pytest.mark.parametrize("profile", ["real-piper", "sim-piper"])
def test_the_profile_config_loads(profile):
    data = yaml.safe_load((ROOT / "profiles" / profile / "gripper_controller.yaml").read_text())
    assert StreamConfig(**data).stall_timeout_ms == 250


def test_the_launcher_is_executable():
    assert os.access(ROOT / "nodes" / "run_gripper.sh", os.X_OK)


@pytest.mark.parametrize("profile", ["real-piper", "sim-piper"])
def test_the_manifest_no_longer_needs_the_old_controller(profile):
    manifest = yaml.safe_load((ROOT / "skill.yaml").read_text())
    assert "gripper_action_controller" not in manifest["profiles"][profile]["required_binaries"]
    assert "gripper_action_controller" not in manifest["artifacts"]["nodes"]
