import pytest

from tabletop.fake import Box, FakeWorld
from tabletop.executor import Executor
from tabletop.primitives import Robot, RobotConfig


def make_scene(depth=True, **kw):
    boxes = [
        Box("red block", [0.38, 0.10], 0.0, [0.05, 0.03, 0.04], yaw_deg=30, aliases=["red cube"]),
        Box("blue block", [0.42, -0.12], 0.0, [0.06, 0.04, 0.05], yaw_deg=-20, aliases=["blue cube"]),
        Box("green block", [0.30, -0.02], 0.0, [0.04, 0.04, 0.03]),
        Box("yellow bottle", [0.30, 0.22], 0.0, [0.04, 0.04, 0.12], aliases=["bottle"]),
    ]
    return FakeWorld(boxes, depth=depth, **kw)


@pytest.fixture(autouse=True)
def no_real_piece_templates(monkeypatch):
    """The fake world's boxes are not the real pieces: nothing is read unless a test gives templates."""
    monkeypatch.setattr("tabletop.primitives.load_templates", lambda: [])


@pytest.fixture
def world():
    return make_scene()


@pytest.fixture
def robot(world):
    return Robot(world, world, RobotConfig())


@pytest.fixture
def executor(robot):
    return Executor(robot)
