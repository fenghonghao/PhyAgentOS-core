"""The head camera streams 1080p. Objects are found on a window of it reduced to 640x480, which is what
the camera's own 640x480 mode shows; everything the model sees is on the full image."""

import io
from pathlib import Path

import numpy as np
from PIL import Image

from tabletop import geometry as geo
from tabletop.backend import Detection, Frame
from tabletop.fake import Box, FakeWorld
from tabletop.perception import reduced
from tabletop.primitives import Robot, RobotConfig

WINDOW = [320, 60, 1280, 960, 2]
K_1080 = np.array([[1033.43, 0, 960.853], [0, 1033.183, 545.645], [0, 0, 1.0]])
LOOKING_DOWN = np.array([[1.0, 0, 0, 0.3], [0, -1, 0, 0.1], [0, 0, -1, 0.75], [0, 0, 0, 1]])
RED = (210, 30, 30)


def frame_1080(depth=True):
    rgb = np.full((1080, 1920, 3), 200, np.uint8)
    d = np.full((1080, 1920), 0.75, np.float32) if depth else None
    return Frame(camera="head", K=K_1080.copy(), T_wc=LOOKING_DOWN.copy(), rgb=rgb, depth=d, meta={"t": 1.0})


def test_reduced_is_the_window_at_half_the_size():
    frame = frame_1080()
    small = reduced(frame, WINDOW)
    assert small.rgb.shape == (480, 640, 3) and small.rgb.dtype == np.uint8
    assert small.depth.shape == (480, 640)
    assert small.camera == "head" and small.meta["t"] == 1.0


def test_reduced_keeps_the_full_frame_and_a_frame_that_is_not_reduced_has_none():
    frame = frame_1080()
    assert reduced(frame, WINDOW).full is frame
    assert frame.full is None


def test_reduced_intrinsics_match_the_cameras_own_640x480_mode():
    small = reduced(frame_1080(), WINDOW)
    assert np.allclose(small.K, [[516.715, 0, 320.18], [0, 516.59, 242.57], [0, 0, 1]], atol=0.01)


def test_a_world_point_lands_on_the_same_thing_in_both_images():
    point = np.array([[0.36, 0.16, 0.0]])
    frame = frame_1080()
    uv, _ = geo.project(frame.K, frame.T_wc, point)
    u, v = np.rint(uv[0]).astype(int)
    frame.rgb[v - 6:v + 7, u - 6:u + 7] = RED
    small = reduced(frame, WINDOW)
    rows, cols = np.nonzero(small.rgb[..., 1] < 100)
    uv_small, _ = geo.project(small.K, small.T_wc, point)
    assert abs(cols.mean() - uv_small[0, 0]) < 0.6 and abs(rows.mean() - uv_small[0, 1]) < 0.6


def test_reduced_without_depth_stays_without_depth():
    assert reduced(frame_1080(depth=False), WINDOW).depth is None


def test_full_pixel_is_the_same_place_on_the_full_image():
    frame = frame_1080()
    frame.rgb[100:120, 500:530] = RED
    small = reduced(frame, WINDOW)
    rows, cols = np.nonzero(small.rgb[..., 1] < 100)
    row, col = small.full_pixel(int(rows.mean()), int(cols.mean()))
    assert frame.rgb[row, col, 1] < 100
    assert frame_1080().full_pixel(7, 9) == (7, 9)


class World1080(FakeWorld):
    """The fake world streaming 1080p, reduced as the real head camera is."""

    def __init__(self, boxes):
        super().__init__(boxes, arms=("left",), resolution=(1920, 1080))

    def capture(self, camera):
        return reduced(super().capture(camera), WINDOW)

    def segment(self, frame, text):
        x0, y0, w, h, f = WINDOW
        return [Detection(mask=d.mask[y0:y0 + h:f, x0:x0 + w:f], score=d.score) for d in super().segment(frame.full, text)]


def robot_1080():
    world = World1080([Box("red block", [0.33, 0.12], 0.0, [0.06, 0.04, 0.04], color=RED)])
    return Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))


def test_locate_gives_the_same_position_as_on_the_full_image():
    full = FakeWorld([Box("red block", [0.33, 0.12], 0.0, [0.06, 0.04, 0.04], color=RED)], arms=("left",), resolution=(1920, 1080))
    expected = Robot(full, full, RobotConfig(wrist_cameras={"left": "wrist_left"})).locate("red block")[0]
    sight = robot_1080().locate("red block")[0]
    assert np.allclose(sight.center_xyz, expected.center_xyz, atol=0.004)
    assert abs(sight.top_z - expected.top_z) < 0.003


def test_sight_pixel_and_capture_share_the_full_image():
    robot = robot_1080()
    sight = robot.locate("red block")[0]
    f = robot.capture("head")
    assert f.rgb.shape == (1080, 1920, 3) and f.world.shape == (1080, 1920, 3)
    row, col = sight.pixel
    assert tuple(f.rgb[row, col]) == RED
    assert np.allclose(f.world[row, col, :2], sight.center_xyz[:2], atol=0.02)


def test_show_is_the_full_view_at_the_size_sent_to_a_model(tmp_path):
    robot = robot_1080()
    robot.show_dir = Path(tmp_path)
    robot.show("head")
    with Image.open(robot.take_shown()[0]) as image:
        assert image.size == (1024, 576)


def test_marks_are_drawn_on_the_full_view_at_the_right_place(tmp_path):
    robot = robot_1080()
    robot.show_dir = Path(tmp_path)
    sight = robot.locate("red block")[0]
    robot.show("head")
    robot.show("head", marks=[sight])
    plain, marked = [np.asarray(Image.open(io.BytesIO(Path(p).read_bytes())).convert("RGB")).astype(int) for p in robot.take_shown()]
    rows, cols = np.nonzero(np.abs(marked - plain).sum(axis=2) > 90)
    row, col = [v * 1024 / 1920 for v in sight.pixel]
    ring = np.hypot(rows - row, cols - col) < 40  # the number sits outside the ring, further away
    assert (rows[ring] < row).any() and (rows[ring] > row).any() and (cols[ring] < col).any() and (cols[ring] > col).any()
