import time

import numpy as np


def check_box(s, box, tol_xy=0.006, tol_h=0.006, tol_deg=4.0):
    assert np.allclose(s.center_xyz[:2], box.center_xy, atol=tol_xy), (s.center_xyz, box.center_xy)
    assert abs(s.top_z - box.top) < tol_h, (s.top_z, box.top)
    assert abs(s.size[0] - max(box.size[:2])) < 0.012 and abs(s.size[1] - min(box.size[:2])) < 0.012, s.size
    long_true = box.yaw_deg if box.size[0] >= box.size[1] else box.yaw_deg + 90
    diff = (s.long_axis_deg - long_true + 90) % 180 - 90
    assert abs(diff) < tol_deg, (s.long_axis_deg, long_true)


def test_locate_with_depth(robot, world):
    s = robot.locate("red block")[0]
    check_box(s, world.boxes["red block"])
    assert abs(s.grasp_xyz[2] - (world.boxes["red block"].top - 0.02)) < 0.006


def test_time_left_env(robot, monkeypatch):
    assert robot.time_left() is None
    monkeypatch.setenv("PAOS_SCENE_DEADLINE", str(time.time() + 100))
    assert 98 < robot.time_left() <= 100


def test_ledger_records_calls(robot):
    robot.locate("red block")
    entries = robot.ledger.drain()
    assert entries[0]["call"] == "locate" and "result" in entries[0]


