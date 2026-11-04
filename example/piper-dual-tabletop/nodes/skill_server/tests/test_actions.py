"""The arm primitives: move_to, set_gripper, home, state, pick, place, drag."""

import dataclasses
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tabletop.backend import MotionOutcome
from tabletop.executor import Executor
from tabletop.fake import Box, FakeWorld
from tabletop.primitives import Robot, RobotConfig
from tabletop.types import Cancelled, RobotFault

FLOOR = 0.0155


def scene(**kw):
    boxes = [
        Box("red block", [0.30, 0.12], 0.0, [0.04, 0.04, 0.04], aliases=["block"]),
        Box("peach", [0.40, -0.05], 0.0, [0.08, 0.08, 0.08], aliases=["fruit"]),
        Box("flat star", [0.28, -0.15], 0.0, [0.05, 0.04, 0.02]),
        Box("cardboard box", [0.30, 0.32], 0.0, [0.20, 0.14, 0.15]),
    ]
    kw.setdefault("max_opening_m", 0.10)
    return FakeWorld(boxes, **kw)


@pytest.fixture
def world():
    return scene()


@pytest.fixture
def robot(world):
    return Robot(world, world, RobotConfig())


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def hold(robot, name="red block"):
    r = robot.pick(robot.locate(name)[0])
    assert r.ok, r
    return r


# ---------------------------------------------------------------- what the model can call

def test_the_model_sees_the_agreed_names(robot):
    ns = Executor(robot).ns
    for name in ("locate", "show", "look", "capture", "move_to", "set_gripper", "home", "state", "pick", "place",
                 "drag", "time_left", "TABLE_Z", "TCP_FLOOR_Z", "MAX_OPENING_M", "READ_UP_DEG", "READ_RIGHT_DEG"):
        assert name in ns, name
    for name in ("wait", "place_at", "place_in", "handover", "TOPDOWN_MAX_Z"):
        assert name not in ns, name


def test_pick_and_place_run_from_model_code(robot, world):
    out = Executor(robot).run("s = locate('red block')[0]\nr = pick(s)\np = place([0.20, 0.15])\nprint(r.ok, p.ok)")
    assert out["ok"], out
    assert out["output"] == "True True\n"


# ---------------------------------------------------------------- move_to

def test_move_to_goes_there_and_reports_where_it_ended(robot):
    m = robot.move_to([0.30, 0.0, 0.10], yaw_deg=20, tilt_deg=0)
    assert m.reached and m.reason is None and m.err_m == pytest.approx(0.0, abs=1e-6)
    assert m.final_xyz == pytest.approx([0.30, 0.0, 0.10])
    assert m.yaw_deg == pytest.approx(20) and m.tilt_deg == pytest.approx(0)


def test_move_to_none_keeps_the_current_angles(robot):
    robot.move_to([0.30, 0.0, 0.20], yaw_deg=20, tilt_deg=30)
    m = robot.move_to([0.32, 0.0, 0.20])
    assert m.yaw_deg == pytest.approx(20) and m.tilt_deg == pytest.approx(30)


def test_move_to_tries_half_a_turn_with_an_empty_gripper(robot, world):
    world.bad_yaw = [(-10, 10)]
    m = robot.move_to([0.30, 0.1, 0.10], yaw_deg=0, tilt_deg=0)
    assert m.reached and abs(m.yaw_deg) == pytest.approx(180)


def test_move_to_never_turns_a_held_object_on_its_own(robot, world):
    hold(robot)
    world.bad_yaw = [(-10, 10)]
    world.moves.clear()
    m = robot.move_to([0.30, 0.1, 0.10], yaw_deg=0, tilt_deg=0)
    assert not m.reached and m.reason == "ik_failed" and world.moves == []


def test_move_to_out_of_reach_moves_nothing_and_says_why(robot, world):
    before = robot.state().tcp_xyz
    m = robot.move_to([0.90, 0.0, 0.10], yaw_deg=0, tilt_deg=0)
    assert not m.reached and m.reason == "ik_failed" and m.detail
    assert world.moves == [] and m.final_xyz == pytest.approx(before)
    assert m.err_m > 0.1


def test_move_to_refuses_the_table_and_moves_nothing(world):
    robot = Robot(world, world, RobotConfig(tcp_floor_z_m=FLOOR))
    m = robot.move_to([0.30, 0.0, 0.010], yaw_deg=0, tilt_deg=0)
    assert not m.reached and m.reason == "below_floor" and world.moves == []
    assert robot.move_to([0.30, 0.0, FLOOR], yaw_deg=0, tilt_deg=0).reached


def test_the_floor_rises_when_a_leaning_finger_reaches_lower(world):
    robot = Robot(world, world, RobotConfig(tcp_floor_z_m=FLOOR, fingertip_m=0.0125, finger_half_depth_m=0.01))
    assert robot.move_to([0.30, 0.0, FLOOR], yaw_deg=90, tilt_deg=25).reason == "below_floor"
    assert robot.move_to([0.30, 0.0, FLOOR + 0.004], yaw_deg=90, tilt_deg=25).reached


def test_the_floor_rises_more_when_the_closing_line_leans_with_open_fingers(world):
    robot = Robot(world, world, RobotConfig(tcp_floor_z_m=FLOOR, fingertip_m=0.0125, finger_half_depth_m=0.01))
    robot.set_gripper(0.08)
    assert robot.move_to([0.30, 0.0, 0.025], yaw_deg=90, tilt_deg=25).reached      # closing across the lean
    assert robot.move_to([0.30, 0.0, 0.025], yaw_deg=0, tilt_deg=25).reason == "below_floor"  # along it


def test_move_to_refuses_outside_the_workspace(world):
    robot = Robot(world, world, RobotConfig(workspace={"x": [0.1, 0.5], "y": [-0.3, 0.3], "z_max": 0.3}))
    m = robot.move_to([0.55, 0.0, 0.10], yaw_deg=0, tilt_deg=0)
    assert not m.reached and m.reason == "outside_workspace" and "x=0.550" in m.detail and world.moves == []
    assert robot.move_to([0.30, 0.0, 0.10], yaw_deg=0, tilt_deg=0).reached


@pytest.mark.parametrize("args,kwargs", [(([0.3, 0.0],), {}), (("left", [0.3, 0.0, 0.1]), {}),
                                         (([0.3, 0.0, 0.1],), {"path": "fly"})])
def test_move_to_misuse_raises(robot, args, kwargs):
    with pytest.raises((ValueError, TypeError)):
        robot.move_to(*args, **kwargs)


# ---------------------------------------------------------------- set_gripper, home, state

def test_set_gripper_opens_to_a_width(robot):
    g = robot.set_gripper(0.06)
    assert g.width_m == pytest.approx(0.06) and not g.holding


def test_set_gripper_stays_within_the_stroke(robot):
    assert robot.set_gripper(0.5).width_m == pytest.approx(0.10)


def test_set_gripper_takes_a_width_not_a_word(robot):
    with pytest.raises(ValueError):
        robot.set_gripper("open")


def test_closing_on_nothing_is_not_holding(robot):
    g = robot.set_gripper(0)
    assert not g.holding and g.width_m < 0.004


def test_holding_means_the_fingers_stopped_wider_than_asked(robot, world):
    s = robot.locate("red block")[0]
    robot.set_gripper(0.08)
    robot.move_to(s.grasp_xyz, yaw_deg=0, tilt_deg=0)
    g = robot.set_gripper(0)
    assert g.holding and g.width_m == pytest.approx(0.04, abs=0.003) and robot.state().holding
    assert not robot.set_gripper(0.08).holding and not robot.state().holding


def test_home_returns_and_leaves_the_gripper_alone(robot, world):
    start = robot.state().tcp_xyz
    robot.set_gripper(0.05)
    robot.move_to([0.30, 0.0, 0.10], yaw_deg=0, tilt_deg=0)
    m = robot.home()
    assert m.reached and robot.state().tcp_xyz == pytest.approx(start)
    assert robot.state().gripper_width_m == pytest.approx(0.05)


def test_state_has_only_what_a_single_arm_needs(robot):
    assert set(robot.state().to_dict()) == {"tcp_xyz", "yaw_deg", "tilt_deg", "joints", "gripper_width_m", "holding"}


# ---------------------------------------------------------------- pick

def test_pick_opens_goes_above_down_closes_and_lifts(robot, world):
    s = robot.locate("red block")[0]
    r = robot.pick(s)
    assert r.ok and r.reason is None and r.width_m == pytest.approx(0.04, abs=0.003)
    assert world.arm["left"].held == "red block" and robot.state().holding
    assert [m["path"] for m in world.moves] == ["joint", "line", "line"]
    hover, down, up = (np.array(m["xyz"]) for m in world.moves)
    assert down == pytest.approx(s.grasp_xyz) and up == pytest.approx(hover)
    assert np.linalg.norm(hover - down) == pytest.approx(0.08)
    assert world.boxes["red block"].z_bottom > 0.05                # it came up with the gripper
    assert world.log.index("closed on red block") > world.log.index("fingers to %.3f" % s.open_width_m)


def test_pick_takes_a_sight_and_never_looks_for_itself(robot):
    with pytest.raises(ValueError, match="locate"):
        robot.pick("red block")


def test_the_hover_leans_only_if_it_must_and_the_grasp_stays_straight_if_it_can():
    w = scene(topdown_max_z=0.095)                                 # a hover 8 cm above is too high to be straight
    robot = Robot(w, w, RobotConfig())
    r = robot.pick(robot.locate("red block")[0])
    assert r.ok and r.tilt_deg == 0
    hover, down, up = w.moves
    assert (hover["tilt"], down["tilt"], up["tilt"]) == (25, 0, 25)
    h, d = np.array(hover["xyz"]), np.array(down["xyz"])
    t = math.radians(25)
    u = d[:2] / np.linalg.norm(d[:2])
    assert h[:2] == pytest.approx(d[:2] - u * math.sin(t) * 0.08)   # back along the leaning fingers
    assert h[2] == pytest.approx(d[2] + math.cos(t) * 0.08)


def test_the_grasp_leans_when_the_point_itself_is_out_of_reach_straight_down():
    w = scene(topdown_max_z=0.03)
    robot = Robot(w, w, RobotConfig())
    r = robot.pick(robot.locate("red block")[0], z=0.035)
    assert r.ok and r.tilt_deg == 25 and w.moves[1]["tilt"] == 25


def test_pick_stays_straight_down_when_it_can(robot):
    assert hold(robot).tilt_deg == 0


def test_pick_z_and_yaw_override_the_measurement(robot, world):
    s = robot.locate("red block")[0]
    r = robot.pick(s, z=0.03, yaw_deg=30)
    assert r.ok and r.yaw_deg == pytest.approx(30)
    assert world.moves[1]["xyz"][2] == pytest.approx(0.03) and world.moves[0]["yaw"] == pytest.approx(30)


def test_pick_width_sets_how_far_the_fingers_open(robot, world):
    s = robot.locate("red block")[0]
    assert robot.pick(s, width_m=0.055).ok
    assert "fingers to 0.055" in world.log


def test_pick_width_defaults_to_the_object_plus_a_margin(robot, world):
    s = robot.locate("red block")[0]
    assert robot.pick(s).ok
    assert "fingers to %.3f" % s.open_width_m in world.log


def test_a_narrow_opening_lowers_the_floor_of_a_leaning_grasp(world):
    robot = Robot(world, world, RobotConfig(tcp_floor_z_m=FLOOR, fingertip_m=0.0125, finger_half_depth_m=0.0,
                                            approach_tilts_deg=[25.0]))
    s = robot.locate("red block")[0]
    assert robot.pick(s, yaw_deg=0, width_m=0.05).ok
    narrow = world.moves[1]["xyz"][2]
    robot.place(s.center_xyz[:2])
    world.moves.clear()
    s = robot.locate("red block")[0]
    assert robot.pick(s, yaw_deg=0, width_m=0.09).ok
    assert world.moves[1]["xyz"][2] > narrow + 0.005                # the lower finger reaches further down


def test_pick_keeps_the_fingertips_off_the_table(world):
    robot = Robot(world, world, RobotConfig(tcp_floor_z_m=FLOOR))
    r = robot.pick(robot.locate("flat star")[0], z=0.001)
    assert r.ok and world.moves[1]["xyz"][2] == pytest.approx(FLOOR)


def test_pick_tries_half_a_turn_when_the_wrist_cannot_do_it(robot, world):
    world.bad_yaw = [(25, 35)]
    r = robot.pick(robot.locate("red block")[0], yaw_deg=30)
    assert r.ok and r.yaw_deg == pytest.approx(-150)


def test_pick_on_its_own_starts_from_the_way_round_nearer_the_wrists_middle(robot):
    s = dataclasses.replace(robot.locate("red block")[0], grasp_yaw_deg=-80.0)
    r = robot.pick(s)
    heading = math.degrees(math.atan2(s.center_xyz[1], s.center_xyz[0]))
    assert r.ok and abs(wrap(r.yaw_deg - heading - 90)) <= 90
    assert abs(wrap(r.yaw_deg - s.grasp_yaw_deg)) in (pytest.approx(0, abs=1e-6), pytest.approx(180, abs=1e-6))


def test_pick_out_of_reach_moves_nothing_not_even_the_fingers(world):
    w = scene(reach_m=0.2)
    robot = Robot(w, w, RobotConfig())
    width = robot.state().gripper_width_m
    r = robot.pick(robot.locate("red block")[0])
    assert not r.ok and r.reason == "unreachable" and r.detail
    assert w.moves == [] and robot.state().gripper_width_m == width


def test_pick_reports_an_empty_grasp_once_and_lifts_back(robot, world):
    s = robot.locate("red block")[0]
    world.boxes["red block"].center_xy = [0.10, 0.30]              # it is no longer where it was seen
    r = robot.pick(s)
    assert not r.ok and r.reason == "empty" and not robot.state().holding
    assert [m["path"] for m in world.moves] == ["joint", "line", "line"]        # one attempt, ending above


def test_pick_needs_a_height_or_a_given_z():
    w = scene(depth=False)
    robot = Robot(w, w, RobotConfig())
    s = robot.locate("red block")[0]
    r = robot.pick(s)
    assert not r.ok and r.reason == "height_unknown" and w.moves == []
    assert robot.pick(s, z=0.02).ok


def test_pick_refuses_what_cannot_fit_between_the_fingers(world):
    world.boxes["melon"] = Box("melon", [0.30, -0.30], 0.0, [0.14, 0.13, 0.08])
    robot = Robot(world, world, RobotConfig())
    r = robot.pick(robot.locate("melon")[0])
    assert not r.ok and r.reason == "too_wide" and world.moves == []


def test_pick_refuses_while_already_holding(robot, world):
    hold(robot)
    world.moves.clear()
    r = robot.pick(robot.locate("peach")[0])
    assert not r.ok and r.reason == "already_holding" and world.moves == []


# ---------------------------------------------------------------- place

def test_place_sets_the_object_down_and_opens_only_a_little(robot, world):
    s = robot.locate("red block")[0]
    held = robot.pick(s).width_m
    world.moves.clear()
    p = robot.place([0.20, 0.15])
    assert p.ok and p.reason is None and not robot.state().holding
    assert p.release_xyz == pytest.approx([0.20, 0.15, 0.01 + s.grasp_xyz[2]])  # bottom 1 cm above the table
    b = world.boxes["red block"]
    assert b.center_xy == pytest.approx([0.20, 0.15], abs=0.006) and b.z_bottom == pytest.approx(0.0)
    assert robot.state().gripper_width_m == pytest.approx(held + 0.015)
    assert [m["path"] for m in world.moves][-2:] == ["line", "line"]
    assert world.moves[-1]["xyz"] == pytest.approx(world.moves[-3]["xyz"])       # back up the way it came


def test_place_into_a_container_releases_above_its_rim(robot, world):
    box = robot.locate("cardboard box")[0]
    s = robot.locate("peach")[0]
    assert robot.pick(s).ok
    p = robot.place(box)
    assert p.ok and p.release_xyz[:2] == pytest.approx(box.center_xyz[:2])
    assert p.release_xyz[2] == pytest.approx(box.top_z + 0.03 + s.grasp_xyz[2])  # bottom 3 cm above the rim
    assert world.boxes["cardboard box"].contains_xy(world.boxes["peach"].center)


def test_place_bottom_z_says_how_high_the_object_is_let_go(robot, world):
    s = robot.locate("red block")[0]
    robot.pick(s)
    p = robot.place([0.20, 0.15], bottom_z=0.05)
    assert p.release_xyz[2] == pytest.approx(0.05 + s.grasp_xyz[2])


def test_place_works_after_a_grasp_made_by_hand(robot, world):
    s = robot.locate("red block")[0]
    robot.set_gripper(0.08)
    robot.move_to(s.grasp_xyz, yaw_deg=0, tilt_deg=0)
    robot.set_gripper(0)
    robot.move_to([s.grasp_xyz[0], s.grasp_xyz[1], 0.09], path="line")
    p = robot.place([0.20, 0.15])
    assert p.ok and p.release_xyz[2] == pytest.approx(0.01 + s.grasp_xyz[2])


def test_place_without_something_held_moves_nothing(robot, world):
    p = robot.place([0.20, 0.15])
    assert not p.ok and p.reason == "not_holding" and world.moves == []


def test_place_yaw_turns_the_object(robot, world):
    robot.pick(robot.locate("red block")[0])
    before = world.boxes["red block"].yaw_deg
    p = robot.place([0.20, 0.15], yaw_deg=robot.state().yaw_deg + 40)
    assert p.ok and abs(wrap(world.boxes["red block"].yaw_deg - before - 40)) < 1e-6


def test_place_at_a_yaw_the_wrist_cannot_do_keeps_holding_and_moves_nothing(robot, world):
    robot.pick(robot.locate("red block")[0])
    world.moves.clear()
    world.bad_yaw = [(40, 60)]
    p = robot.place([0.20, 0.15], yaw_deg=50)
    assert not p.ok and p.reason == "unreachable" and p.detail
    assert world.moves == [] and robot.state().holding


def test_place_keeps_the_heading_when_the_wrist_allows_it(robot, world):
    robot.pick(robot.locate("red block")[0], yaw_deg=100)
    assert robot.place([0.25, 0.05]).yaw_deg == pytest.approx(100)


def test_place_turns_only_as_far_as_the_wrist_needs(robot, world):
    robot.pick(robot.locate("flat star")[0], yaw_deg=0)              # about 62 deg from the middle of the wrist there
    x, y = -0.05, 0.30                                                # heading 99.5 deg: 0 would be 170 off
    p = robot.place([x, y])
    heading = math.degrees(math.atan2(y, x))
    assert p.ok and abs(wrap(p.yaw_deg - heading - 90)) == pytest.approx(90)


def test_place_rises_on_the_spot_before_travelling_to_something_higher(robot, world):
    box = robot.locate("cardboard box")[0]
    robot.pick(robot.locate("peach")[0])
    start = list(world.moves[-1]["xyz"])
    world.moves.clear()
    assert robot.place(box).ok
    rise, travel = world.moves[0], world.moves[1]
    assert rise["xyz"][:2] == pytest.approx(start[:2]) and rise["xyz"][2] == pytest.approx(travel["xyz"][2])
    assert travel["xyz"][2] > start[2]


def test_place_does_not_rise_first_when_it_is_already_high_enough(robot, world):
    robot.pick(robot.locate("red block")[0])
    world.moves.clear()
    assert robot.place([0.20, 0.15]).ok
    assert [m["path"] for m in world.moves] == ["joint", "line", "line"]


# ---------------------------------------------------------------- drag

def test_drag_slides_the_object_along_the_table_without_lifting(robot, world):
    s = robot.locate("flat star")[0]
    r = robot.drag(s, [0.25, -0.05])
    assert r.ok and r.reason is None and r.width_m == pytest.approx(0.05, abs=0.02)
    b = world.boxes["flat star"]
    assert b.center_xy == pytest.approx([0.25, -0.05], abs=0.006) and b.z_bottom == pytest.approx(0.0)
    assert r.final_xyz[:2] == pytest.approx([0.25, -0.05])
    assert not robot.state().holding and world.arm["left"].held is None
    assert [m["path"] for m in world.moves] == ["joint", "line", "line", "line"]
    hover, down, slide, up = (m["xyz"] for m in world.moves)
    assert slide[2] == pytest.approx(down[2]) and up[2] > slide[2]
    closed, opened = world.log.index("closed on flat star"), world.log.index("released flat star")
    assert sum("move" in line for line in world.log[closed:opened]) == 1          # only the slide between them


def test_drag_leans_only_where_it_must_and_stays_level_with_the_table():
    w = scene(topdown_max_z=0.07)                                  # a hover 8 cm above is too high to be straight
    robot = Robot(w, w, RobotConfig())
    r = robot.drag(robot.locate("flat star")[0], [0.25, -0.05])
    assert r.ok and r.tilt_deg == 0
    hover, down, slide, up = w.moves
    assert down["tilt"] == 0 and slide["tilt"] == 0 and hover["tilt"] > 0 and up["tilt"] > 0
    assert slide["xyz"][2] == pytest.approx(down["xyz"][2])


def test_drag_defaults_to_the_height_pick_would_use(robot, world):
    s = robot.locate("peach")[0]
    assert robot.drag(s, [0.30, -0.10]).ok
    assert world.moves[1]["xyz"][2] == pytest.approx(s.grasp_xyz[2])


def test_drag_width_sets_how_far_the_fingers_open(robot, world):
    assert robot.drag(robot.locate("flat star")[0], [0.25, -0.05], width_m=0.065).ok
    assert "fingers to 0.065" in world.log


def test_drag_of_nothing_reports_empty_and_does_not_slide(robot, world):
    s = robot.locate("flat star")[0]
    world.boxes["flat star"].center_xy = [0.10, 0.30]
    r = robot.drag(s, [0.25, -0.05])
    assert not r.ok and r.reason == "empty" and not robot.state().holding
    assert all(np.linalg.norm(np.array(m["xyz"][:2]) - s.center_xyz[:2]) < 0.09 for m in world.moves)


def test_drag_to_somewhere_out_of_reach_moves_nothing(robot, world):
    width = robot.state().gripper_width_m
    r = robot.drag(robot.locate("flat star")[0], [0.90, 0.0])
    assert not r.ok and r.reason == "unreachable" and world.moves == []
    assert robot.state().gripper_width_m == width


def test_drag_refuses_misuse(robot, world):
    with pytest.raises(ValueError, match="locate"):
        robot.drag("flat star", [0.25, -0.05])
    hold(robot)
    world.moves.clear()
    assert robot.drag(robot.locate("flat star")[0], [0.25, -0.05]).reason == "already_holding"
    assert world.moves == []


# ---------------------------------------------------------------- faults, cancel, ledger

def test_a_driver_fault_raises(robot, world):
    world.execute = lambda plan, cancelled=None: MotionOutcome(False, None, "trajectory", fault="CAN disconnected")
    with pytest.raises(RobotFault, match="CAN disconnected"):
        robot.move_to([0.30, 0.0, 0.10], yaw_deg=0, tilt_deg=0)
    with pytest.raises(RobotFault, match="CAN disconnected"):
        robot.pick(robot.locate("red block")[0])


def test_cancel_stops_between_the_steps_of_a_pick(robot, world):
    s = robot.locate("red block")[0]
    world.move_delay_s = 0.05
    threading.Timer(0.02, robot.cancel_event.set).start()
    with pytest.raises(Cancelled):
        robot.pick(s)
    assert len(world.moves) < 3


def test_the_ledger_shows_the_pick_and_its_steps(robot):
    s = robot.locate("red block")[0]
    robot.ledger.drain()
    robot.pick(s)
    entries = robot.ledger.drain()
    assert [e["call"] for e in entries if not e["nested"]] == ["pick"]
    assert {"set_gripper", "motion"} <= {e["call"] for e in entries if e["nested"]}


class _Cameras:
    """Just what the snapshots use; a camera without a frame yet is skipped."""

    def __init__(self):
        self.frames = {"head": np.full((48, 64, 3), 200, np.uint8), "wrist": None}

    def cameras(self):
        return list(self.frames)

    def latest_rgb(self, camera):
        return self.frames[camera]


def _wait_for(paths, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not all(p.exists() and p.stat().st_size for p in paths):
        assert time.monotonic() < deadline, "snapshot not written"
        time.sleep(0.01)


def test_grip_snapshots_are_logged_and_never_reach_the_model(robot, tmp_path):
    robot.perception = _Cameras()
    robot.grip_snapshot_dir = tmp_path / "grip"
    robot.ledger.drain()
    robot.take_shown()

    grip = robot.set_gripper(0.05)

    entry = robot.ledger.drain()[-1]
    paths = [Path(p) for p in entry["args"]["before_images"]]
    assert [p.name.split("_")[-1] for p in paths] == ["head.jpg"]
    assert "w050mm" in paths[0].name
    _wait_for(paths)
    assert robot.take_shown() == []           # what the model is handed after a block
    assert "before_images" not in str(grip)   # what its code gets back


def test_no_grip_snapshots_unless_configured(robot):
    robot.perception = _Cameras()
    robot.ledger.drain()
    robot.set_gripper(0.05)
    assert "before_images" not in robot.ledger.drain()[-1]["args"]
