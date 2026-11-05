"""DoraArmBackend on the real PIPER URDF. A stand-in answers for the Dora side: no dataflow runs."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("forge_msgs")
from forge_msgs import GripperCommandGoal, MoveJointsGoal  # noqa: E402

from tabletop import kinematics as kin  # noqa: E402
from tabletop.backend import Unreachable  # noqa: E402
from tabletop.dora_backend import ArmConfig, Bus, DoraArmBackend, MotionConfig  # noqa: E402
from tabletop.primitives import Robot, RobotConfig  # noqa: E402
from tabletop.types import Sight  # noqa: E402

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "robots" / "piper"
URDF = str(ASSETS / "piper_with_gripper_real.urdf")
HOME = [0.0, 0.5, -0.02, 0.0, 0.0, 0.0]
JOINTS = [f"joint{i}" for i in range(1, 7)]
FLOOR = 0.0155


class Stack(Bus):
    """Answers at once, like a motion stack that reaches every goal. ``script``: error codes to
    answer first. ``object_width``: where closing fingers stop."""

    def __init__(self):
        super().__init__()
        self.backend = None
        self.goals = []
        self.history = []
        self.q = list(HOME)
        self.width = 0.10
        self.object_width = 0.0
        self.script = []

    def request(self, output_id, payload):
        self.goals.append(output_id)
        if self.script:
            code = self.script.pop(0)
            return "aborted", SimpleNamespace(error_code=code, message="scripted", position=self.width)
        if output_id.endswith("move_joints_goal"):
            self.q = list(MoveJointsGoal.from_arrow(payload).positions)
            self.history.append((list(self.q), self.width))
            self.publish()
            return "succeeded", SimpleNamespace(error_code="SUCCESS", message="")
        assert output_id.endswith("gripper_goal"), output_id
        self.grip_goal = GripperCommandGoal.from_arrow(payload)
        target = self.grip_goal.position
        self.width = max(target, self.object_width) if target < self.width else target
        self.publish()
        return "succeeded", SimpleNamespace(error_code="SUCCESS", message="", position=self.width)

    def publish(self):
        self.backend.on_joint_state([*JOINTS, "gripper"], [*self.q, self.width])


@pytest.fixture
def stack():
    s = Stack()
    s.backend = DoraArmBackend({"left": ArmConfig(name="left", urdf=URDF, home_joints=HOME, max_opening_m=0.10)},
                               MotionConfig(), s)
    s.publish()
    return s


@pytest.fixture
def b(stack):
    return stack.backend


@pytest.fixture
def robot(stack):
    return Robot(stack.backend, None, RobotConfig(tcp_floor_z_m=FLOOR, finger_half_depth_m=0.01))


def sight(x, y, top=0.08, width=0.075, yaw=90.0, grasp_below=0.03, text="thing"):
    z = max(top - grasp_below, FLOOR)
    return Sight(text=text, score=0.9, camera="head", pixel=[0, 0], center_xyz=[x, y, top / 2], top_z=top,
                 size=[width, width, top], long_axis_deg=0.0, grasp_yaw_deg=yaw, color=[200, 100, 100],
                 grasp_xyz=[x, y, z], open_width_m=min(width + 0.03, 0.10))


def lowest_finger_point(b, q, width, half_depth=0.01):
    T = b._fk_world(np.asarray(q))
    R = T[:3, :3]
    tip = T[2, 3] + 0.0125 * R[2, 2]
    return float(tip - abs(R[2, 1]) * width / 2 - abs(R[2, 0]) * half_depth)


def test_more_than_one_arm_is_refused():
    arms = {n: ArmConfig(name=n, urdf=URDF, home_joints=HOME) for n in ("left", "right")}
    with pytest.raises(ValueError, match="one arm"):
        DoraArmBackend(arms, MotionConfig(), Stack())


def test_a_plan_moves_nothing_until_it_is_executed(b, stack):
    b.plan([0.30, 0.05, 0.10], 100.0, 15.0, "joint")
    assert stack.goals == []


@pytest.mark.parametrize("xyz,yaw,tilt", [([0.30, 0.05, 0.10], 100.0, 15.0), ([0.25, -0.10, 0.06], 40.0, 0.0),
                                          ([0.45, 0.20, 0.15], 170.0, 30.0)])
def test_the_arm_ends_at_the_commanded_pose(b, xyz, yaw, tilt):
    assert b.execute(b.plan(xyz, yaw, tilt, "joint")).reached
    got, got_yaw, got_tilt = b.tcp_pose()
    assert got == pytest.approx(xyz, abs=1e-3)
    assert abs((got_yaw - yaw + 180) % 360 - 180) < 0.5 and got_tilt == pytest.approx(tilt, abs=0.5)


def test_a_line_stays_on_the_line_with_the_same_orientation(b):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    start, end = np.array([0.30, 0.0, 0.10]), np.array([0.34, 0.03, 0.03])
    plan = b.plan(end.tolist(), 90.0, 0.0, "line")
    assert len(plan.steps) >= 4
    d = (end - start) / np.linalg.norm(end - start)
    for q in plan.steps:
        T = b._fk_world(np.asarray(q))
        p = T[:3, 3] - start
        assert np.linalg.norm(p - (p @ d) * d) < 1e-3
        assert np.allclose(T[:3, 2], [0, 0, -1], atol=5e-3) and np.allclose(T[:3, 1], [0, 1, 0], atol=5e-3)
    assert b._fk_world(np.asarray(plan.steps[-1]))[:3, 3] == pytest.approx(end, abs=1e-3)


def test_a_turn_on_the_spot_is_a_line_in_small_steps(b):
    b.execute(b.plan([0.30, 0.0, 0.05], 90.0, 0.0, "joint"))
    plan = b.plan([0.30, 0.0, 0.05], 30.0, 0.0, "line")
    assert len(plan.steps) >= 7                                       # 60 degrees, about 8 per step
    steps = np.array([HOME, *plan.steps])[1:]
    assert np.abs(np.diff(steps, axis=0)).max() < math.radians(10)


def test_a_line_that_leaves_the_reach_is_refused_whole(b, stack):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    stack.goals.clear()
    with pytest.raises(Unreachable):
        b.plan([0.75, 0.0, 0.10], 90.0, 0.0, "line")
    assert stack.goals == []


def test_a_joint_move_out_of_reach_is_refused(b):
    with pytest.raises(Unreachable, match="no joint solution"):
        b.plan([0.85, 0.0, 0.10], 90.0, 0.0, "joint")


def test_auto_is_a_line_when_short_and_a_joint_move_when_long(b):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    assert len(b.plan([0.30, 0.0, 0.05], 90.0, 0.0, "auto").steps) > 1
    assert len(b.plan([0.20, 0.25, 0.05], 90.0, 0.0, "auto").steps) == 1


def test_a_plan_can_continue_from_the_end_of_another(b, stack):
    above = b.plan([0.40, 0.10, 0.12], 110.0, 15.0, "joint")
    down = b.plan([0.40, 0.10, 0.05], 110.0, 15.0, "line", start=above)
    assert stack.goals == []
    assert np.abs(np.asarray(down.steps[0]) - np.asarray(above.end_joints)).max() < math.radians(15)
    assert down.end_xyz == pytest.approx([0.40, 0.10, 0.05])


def test_a_joint_swing_that_dips_below_the_floor_is_refused(b):
    start = b.plan([0.20, 0.0, 0.14], 90.0, 15.0, "joint")
    b.execute(start)
    q0, q1 = np.asarray(start.end_joints), np.asarray(HOME)
    target = [float(v) for v in b._fk_world(q1)[:3, 3]]
    low = min(b._fk_world(q0 + (q1 - q0) * f)[2, 3] for f in np.linspace(0, 1, 21))
    ends = min(b._fk_world(q0)[2, 3], b._fk_world(q1)[2, 3])
    assert low < ends - 0.003, "this pair no longer dips: pick another"
    with pytest.raises(Unreachable, match="below"):
        b.plan(target, 90.0, 67.5, "joint", floor_z=low + 0.002)
    assert b.plan(target, 90.0, 67.5, "joint", floor_z=low - 0.002).steps


def test_execute_sends_one_joint_goal_per_step(b, stack):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    stack.goals.clear()
    plan = b.plan([0.30, 0.0, 0.03], 90.0, 0.0, "line")
    assert b.execute(plan).reached
    assert stack.goals == ["left_move_joints_goal"] * len(plan.steps)
    assert stack.q == pytest.approx(list(plan.steps[-1]))


def test_execute_stops_between_steps_when_cancelled(b, stack):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    stack.goals.clear()
    out = b.execute(b.plan([0.30, 0.0, 0.03], 90.0, 0.0, "line"), cancelled=lambda: len(stack.goals) >= 2)
    assert not out.reached and out.reason == "cancelled" and len(stack.goals) == 2


def test_a_busy_controller_is_asked_again(b, stack):
    stack.script = ["BUSY"]
    out = b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    assert out.reached and len(stack.goals) == 2


def test_a_failed_trajectory_is_a_fault(b, stack):
    stack.script = ["TRAJECTORY_EXECUTION_FAILED"]
    out = b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    assert not out.reached and "TRAJECTORY_EXECUTION_FAILED" in out.fault


def test_home_goes_to_the_home_joints(b, stack):
    b.execute(b.plan([0.30, 0.0, 0.10], 90.0, 0.0, "joint"))
    assert b.execute(b.plan_home()).reached
    assert stack.q == pytest.approx(HOME)


def test_grip_reports_where_the_fingers_stopped(b, stack):
    stack.object_width = 0.05
    out = b.grip(0.0)
    assert out.width_m == pytest.approx(0.05) and out.fault is None
    assert b.gripper_width() == pytest.approx(0.05)


def test_grip_asks_for_the_speed_it_is_given(b, stack):
    b.grip(0.0, 0.02)
    assert stack.grip_goal.max_velocity == 0.02 and stack.grip_goal.max_effort is None
    b.grip(0.06)
    assert stack.grip_goal.max_velocity is None


def test_a_gripper_hardware_fault_is_reported(b, stack):
    stack.script = ["HARDWARE_FAULT"]
    assert "HARDWARE_FAULT" in b.grip(0.0).fault


def test_stop_cancels_the_goals_in_flight(b, stack):
    stack._pending["g1"] = None
    b.stop()
    sent = stack.drain()
    assert {out for out, _, md in sent} == {"left_move_joints_cancel", "left_gripper_cancel"}
    assert all(md == {"goal_id": "g1", "cancel": True} for _, _, md in sent)


def test_the_real_urdf_stops_joint_six_where_the_manual_does(b):
    assert b.chain.limits[5] == pytest.approx((-math.radians(100), math.radians(100)), abs=1e-3)


# ---------------------------------------------------------------- the primitives on the real arm's kinematics

@pytest.mark.parametrize("r,heading", [(0.20, 0), (0.30, 0), (0.40, 30), (0.45, -40), (0.50, 60), (0.55, 10)])
def test_a_fruit_can_be_picked_anywhere_it_can_lie(robot, stack, r, heading):
    x, y = r * math.cos(math.radians(heading)), r * math.sin(math.radians(heading))
    stack.object_width = 0.075
    p = robot.pick(sight(x, y))
    assert p.ok, p
    assert all(lowest_finger_point(robot.arms_backend, q, w) > 0.003 - 1e-3 for q, w in stack.history)


@pytest.mark.parametrize("r,heading,tilt", [(0.20, 0, 0), (0.30, 20, 0), (0.35, -30, 0), (0.45, 0, 0), (0.50, 45, 15)])
def test_a_flat_piece_keeps_the_fingertips_above_the_table(robot, stack, r, heading, tilt):
    x, y = r * math.cos(math.radians(heading)), r * math.sin(math.radians(heading))
    tangent = heading + 90.0
    stack.object_width = 0.03
    p = robot.pick(sight(x, y, top=0.01, width=0.03, yaw=tangent, grasp_below=0.005))
    assert p.ok and p.tilt_deg == tilt, p
    for q, w in stack.history:
        assert lowest_finger_point(robot.arms_backend, q, w) >= 0.003 - 1e-3


def test_a_picked_fruit_goes_into_a_box_at_a_reachable_spot(robot, stack):
    stack.object_width = 0.075
    assert robot.pick(sight(0.30, -0.10)).ok
    box = sight(0.40, 0.25, top=0.15, width=0.20, text="box")
    p = robot.place(box)
    assert p.ok, p
    assert p.release_xyz == pytest.approx([0.40, 0.25, 0.15 + 0.03 + 0.05], abs=2e-3)
    assert stack.width == pytest.approx(0.075 + 0.015)


def test_the_box_as_measured_on_the_real_table_is_within_reach(robot, stack):
    """Opening at about (0.43, 0.476), rim 0.135 to 0.16 m (other session, 2026-11-04)."""
    stack.object_width = 0.075
    assert robot.pick(sight(0.30, 0.10)).ok
    p = robot.place(sight(0.43, 0.476, top=0.16, width=0.20, text="box"))
    assert p.ok, p


def test_a_piece_can_be_dragged_across_the_table(robot, stack):
    stack.object_width = 0.03
    s = sight(0.25, -0.15, top=0.01, width=0.03, yaw=60.0, grasp_below=0.005)
    d = robot.drag(s, [0.40, -0.05])
    assert d.ok, d
    assert d.final_xyz[:2] == pytest.approx([0.40, -0.05], abs=1e-3)


def test_a_drag_to_where_it_must_lean_stays_at_one_height_above_the_table(robot, stack):
    stack.object_width = 0.03
    d = robot.drag(sight(0.30, 0.0, top=0.01, width=0.03, yaw=90.0, grasp_below=0.005), [0.50, 0.0])
    assert d.ok, d
    b = robot.arms_backend
    heights = [round(float(b._fk_world(np.asarray(q))[2, 3]), 3) for q, w in stack.history if w < 0.04]
    assert len(heights) > 5 and max(heights) - min(heights) <= 0.001
    assert min(heights) > FLOOR


def test_a_drag_the_wrist_cannot_follow_is_refused_before_anything_moves(robot, stack):
    stack.object_width = 0.03
    s = sight(0.30, -0.30, top=0.01, width=0.03, yaw=0.0, grasp_below=0.005)   # 135 deg off the tangent at the far end
    width = stack.width
    d = robot.drag(s, [0.30, 0.30], yaw_deg=0.0)
    assert not d.ok and d.reason == "unreachable"
    assert stack.history == [] and stack.width == width


# ---------------------------------------------------------------- crossing between two heights
# Scene simple-cherry-strawberry-2, 2026-11-04: a pick right after a place was one joint move from above
# the box to a low hover, and it swept the gripper through the box and turned it over.

BOX_CENTRE = np.array([0.3428, 0.445])
BOX_SIZE = (0.2365, 0.1375)
BOX_AXIS_DEG = -23.25
BOX_TOP = 0.173
OPENING = dict(top=0.1706, width=0.13, text="box opening")


def over_the_box(xy) -> bool:
    a = math.radians(BOX_AXIS_DEG)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    d = np.asarray(xy) - BOX_CENTRE
    return abs(d @ u) <= BOX_SIZE[0] / 2 and abs(d @ v) <= BOX_SIZE[1] / 2


def gripper_points(b, flange, q):
    """The gripper as a box of points, from the flange to the fingertips."""
    T = b._fk_world(q)
    R = T[:3, :3]
    tips = T[:3, 3] + R[:, 2] * 0.0125
    base = (b.T_world_base @ flange.fk(q))[:3, 3]
    for along in np.linspace(0.0, 1.0, 7):
        middle = base + (tips - base) * along
        for across in (-0.05, 0.0, 0.05):
            for side in (-0.04, 0.0, 0.04):
                yield middle + R[:, 1] * across + R[:, 0] * side


def lowest_over_the_box(b, joints) -> float:
    flange = kin.Chain(URDF, b.c.reference_frame, "link6")
    low = math.inf
    for q0, q1 in zip(joints, joints[1:]):
        for f in np.linspace(0.0, 1.0, 21):
            q = np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * f
            for p in gripper_points(b, flange, q):
                if over_the_box(p[:2]):
                    low = min(low, float(p[2]))
    return low


def carried_into_the_box(robot, stack):
    stack.object_width = 0.027
    s = sight(0.4434, 0.0938, top=0.015, width=0.027, yaw=199.05, grasp_below=0.0)
    assert robot.pick(s, yaw_deg=199.05, width_m=0.042).ok
    assert robot.place(sight(0.333, 0.4504, **OPENING)).ok
    stack.history.clear()
    stack.object_width = 0.0
    return list(stack.q)


@pytest.mark.parametrize("x,y,yaw", [(0.4337, 0.1216, -1.28), (0.25, 0.20, 90.0), (0.25, -0.20, 45.0)])
def test_a_pick_right_after_a_place_does_not_sweep_through_the_box(robot, stack, x, y, yaw):
    start = carried_into_the_box(robot, stack)
    r = robot.pick(sight(x, y, top=0.015, width=0.027, yaw=yaw, grasp_below=0.0), yaw_deg=yaw, width_m=0.042)
    assert r.reason == "empty", r
    assert lowest_over_the_box(robot.arms_backend, [start, *[q for q, _ in stack.history]]) >= BOX_TOP


def test_a_drag_right_after_a_place_does_not_sweep_through_the_box(robot, stack):
    start = carried_into_the_box(robot, stack)
    stack.object_width = 0.03
    d = robot.drag(sight(0.30, 0.0, top=0.01, width=0.03, yaw=90.0, grasp_below=0.005), [0.35, -0.05])
    assert d.ok, d
    assert lowest_over_the_box(robot.arms_backend, [start, *[q for q, _ in stack.history]]) >= BOX_TOP


def test_a_big_drop_crosses_at_the_start_height_then_comes_down_above_the_hover(robot, stack, b):
    b.execute(b.plan([0.30, 0.25, 0.25], 90.0, 45.0, "joint"))
    stack.history.clear()
    stack.object_width = 0.03
    assert robot.pick(sight(0.35, -0.15, top=0.01, width=0.03, yaw=60.0, grasp_below=0.005)).ok
    first = b._fk_world(np.asarray(stack.history[0][0]))[:3, 3]
    assert first[2] == pytest.approx(0.25, abs=1e-3)
    assert first[:2] == pytest.approx([0.35, -0.15], abs=0.05)                  # above the hover, not on the way


def test_a_pick_from_home_goes_straight_to_a_hover_a_little_lower(robot, stack, b):
    stack.object_width = 0.03
    assert robot.pick(sight(0.30, 0.0, top=0.01, width=0.03, yaw=90.0, grasp_below=0.005)).ok
    first = b._fk_world(np.asarray(stack.history[0][0]))[:3, 3]
    assert first == pytest.approx([0.30, 0.0, FLOOR + 0.08], abs=2e-3)          # no extra leg: this costs time


def test_a_pick_of_something_tall_from_low_rises_on_the_spot_first(robot, stack, b):
    b.execute(b.plan([0.30, 0.0, 0.05], 90.0, 0.0, "joint"))
    stack.history.clear()
    stack.object_width = 0.05
    assert robot.pick(sight(0.35, 0.20, top=0.20, width=0.05, yaw=90.0)).ok
    first = b._fk_world(np.asarray(stack.history[0][0]))[:3, 3]
    assert first[:2] == pytest.approx([0.30, 0.0], abs=2e-3)
    assert first[2] > 0.15


# ---------------------------------------------------------------- the way round of Playbook C on the real arm

def manual_put():
    """The stack, the robot, the ``pick`` results and ``put`` as the manual writes it."""
    text = (Path(__file__).resolve().parents[3] / "SKILL.md").read_text(encoding="utf-8")
    start = text.index("def put(s, slot, turn):")
    s = Stack()
    s.backend = DoraArmBackend({"left": ArmConfig(name="left", urdf=URDF, home_joints=HOME, max_opening_m=0.10)},
                               MotionConfig(), s)
    s.publish()
    robot = Robot(s.backend, None, RobotConfig(tcp_floor_z_m=FLOOR, finger_half_depth_m=0.01))
    picks = []

    def pick(*args, **kwargs):
        picks.append(robot.pick(*args, **kwargs))
        return picks[-1]

    scope = {"math": math, "pick": pick, "place": robot.place, "TCP_FLOOR_Z": FLOOR}
    exec(text[start:text.index("\nAWAY = ", start)], scope)  # noqa: S102
    return s, robot, picks, scope["put"]


def flat_piece(x, y, length, width, grasp_yaw):
    piece = sight(x, y, top=0.008, width=width, yaw=grasp_yaw, grasp_below=0.005)
    piece.size = [length, width, 0.008]
    return piece


def test_a_flat_piece_is_grasped_straight_down_after_the_piece_before_it():
    # A real scene: with the closing line chosen for the turn, this "=" was reached leaning 55 degrees
    # and the fingers closed above it.
    s, _, picks, put = manual_put()
    s.object_width = 0.015
    assert put(flat_piece(0.3995, 0.1982, 0.036, 0.015, -16.84), np.array([0.29, 0.36]), 20.0).ok
    s.object_width = 0.0237
    assert put(flat_piece(0.3576, 0.0756, 0.0353, 0.0237, 29.22), np.array([0.29, 0.24]), -35.0).ok
    assert picks[-1].tilt_deg == 0.0, picks[-1]


def test_a_turn_the_wrist_can_make_is_made():
    s, _, picks, put = manual_put()
    s.object_width = 0.03
    p = put(flat_piece(0.35, 0.25, 0.03, 0.03, math.degrees(math.atan2(0.25, 0.35)) + 60), np.array([0.28, 0.20]), 30.0)
    assert p.ok, p
    assert abs((p.yaw_deg - picks[0].yaw_deg - 30.0 + 180.0) % 360.0 - 180.0) < 1.0


# where the pieces lay in a real scene, the farthest 0.59 m from the base, and two more
@pytest.mark.parametrize("x,y", [(.387, .272), (.319, .311), (.347, .387), (.279, .131), (.300, .205), (.342, .112),
                                 (.448, .388), (.400, .170), (.25, .05), (.45, .35)])
def test_the_wrist_camera_gets_over_a_piece_leaning_wherever_it_lies(x, y):
    import yaml

    from tabletop import geometry as geo
    from tabletop.perception import CameraConfig

    text = (Path(__file__).resolve().parents[3] / "SKILL.md").read_text(encoding="utf-8")
    start = text.index("def over(s):")
    s = Stack()
    s.backend = DoraArmBackend({"left": ArmConfig(name="left", urdf=URDF, home_joints=HOME, max_opening_m=0.10)},
                               MotionConfig(), s)
    s.publish()
    robot = Robot(s.backend, None, RobotConfig(tcp_floor_z_m=FLOOR, finger_half_depth_m=0.01))
    scope = {"math": math, "move_to": robot.move_to}
    exec(text[start:text.index("\ndef unread():", start)], scope)  # noqa: S102
    r = scope["over"](SimpleNamespace(center_xyz=[x, y, 0.008]))
    assert r.reached, r

    profile = Path(__file__).resolve().parents[3] / "profiles/real-piper/skill_server.yaml"
    wrist = next(c for c in yaml.safe_load(profile.read_text(encoding="utf-8"))["cameras"] if c["name"] != "head")
    camera = CameraConfig(**{k: wrist[k] for k in ("name", "K", "mount", "T_matrix", "T_inverse", "wrist_frame", "xyz", "rpy_deg") if k in wrist})
    T_world_camera = s.backend._fk_world(np.array(robot.state().joints)) @ camera.T()
    uv, depth = geo.project(np.array(wrist["K"], float), T_world_camera, np.array([[x, y, 0.008]]))
    assert abs(uv[0][0] - 320) < 40 and abs(uv[0][1] - 240) < 40, uv     # in the middle of the image
    assert 0.15 < depth[0] < 0.25, depth


@pytest.mark.parametrize("turn", [120, 150, 172, 180, -120, -150, -180])
@pytest.mark.parametrize("at,slot", [((0.35, 0.25), (0.28, 0.20)), ((0.30, 0.05), (0.28, 0.32)), ((0.40, 0.38), (0.34, 0.14)),
                                     ((0.251, 0.365), (0.26, 0.30))])   # the last: where a real scene left the piece in the hand
def test_a_big_turn_goes_in_one_grasp_and_the_piece_ends_in_its_slot(at, slot, turn):
    s, robot, picks, put = manual_put()
    s.object_width = 0.03
    piece = sight(at[0], at[1], top=0.01, width=0.03, yaw=math.degrees(math.atan2(at[1], at[0])) + 60, grasp_below=0.005)
    p = put(piece, np.array(slot), float(turn))
    assert p.ok, p
    assert p.release_xyz[:2] == pytest.approx(list(slot), abs=2e-3)
    assert not robot.state().holding
    assert len(picks) == 1                                           # a turn out of reach stays undone
