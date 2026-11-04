"""locate / look / capture / show as model code sees them (single-arm booth, fake world)."""

import asyncio
import base64
import io
import json
import urllib.error

import numpy as np
import pytest
from PIL import Image

from tabletop import geometry as geo
from tabletop.backend import Detection
from tabletop.executor import Executor
from tabletop.fake import Box, FakeWorld
from tabletop.primitives import Robot, RobotConfig
from tabletop.vision import VisionClient, VisionConfig, VisionError

RED = (210, 30, 30)
TABLE = (205, 205, 200)  # the fake world's table colour


def booth(**kw):
    boxes = [
        Box("blue bowl", [0.35, -0.15], 0.0, [0.12, 0.12, 0.05], aliases=["bowl"], color=(40, 60, 200)),
        Box("red block", [0.30, 0.12], 0.0, [0.04, 0.04, 0.04], aliases=["block"], color=RED),
        Box("green block", [0.42, 0.05], 0.0, [0.04, 0.04, 0.04], aliases=["block"], color=(30, 180, 60)),
        Box("yellow block", [0.25, -0.02], 0.0, [0.04, 0.04, 0.04], aliases=["block"], color=(230, 210, 40)),
    ]
    return FakeWorld(boxes, arms=("left",), **kw)


def make_robot(world, tmp_path=None, **kw):
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}), **kw)
    if tmp_path is not None:
        robot.show_dir = tmp_path / "show"
    return robot


@pytest.fixture
def world():
    return booth()


@pytest.fixture
def robot(world, tmp_path):
    return make_robot(world, tmp_path)


@pytest.fixture
def ex(robot):
    return Executor(robot)


def spy_captures(world):
    seen = []
    real = world.capture

    def capture(camera):
        seen.append(camera)
        return real(camera)

    world.capture = capture
    return seen


def decode(path):
    with Image.open(path) as im:
        assert im.format == "JPEG"
        return np.asarray(im.convert("RGB"))


def run(ex, code):
    out = ex.run(code)
    assert out["ok"], out
    return out


# ---------------------------------------------------------------- locate


def test_locate_returns_every_instance_best_score_first(robot):
    items = robot.locate("block")
    assert len(items) == 3
    assert [s.text for s in items] == ["block"] * 3
    scores = [s.score for s in items]
    assert scores == sorted(scores, reverse=True)
    # the fake world scores red > green > yellow but lists them by image area
    assert np.allclose(items[0].center_xyz[:2], [0.30, 0.12], atol=0.01)
    assert np.allclose(items[2].center_xyz[:2], [0.25, -0.02], atol=0.01)


def test_locate_counts_an_object_once_when_a_mask_lies_inside_a_better_one(robot, world):
    """SAM3 gave the real peach twice: with its leaf and without."""
    real = world.segment

    def segment(frame, text):
        dets = real(frame, text)
        smaller = dets[0].mask.copy()
        smaller[np.nonzero(dets[0].mask)[0].max()] = False        # the same object minus a row
        return dets + [Detection(mask=smaller, score=0.2), Detection(mask=dets[1].mask.copy(), score=0.95)]

    world.segment = segment
    items = robot.locate("block")
    assert len(items) == 3
    assert [round(s.score, 2) for s in items][0] == 0.95            # the better of the two is kept


def test_locate_keeps_the_whole_when_a_part_of_it_scores_higher(robot, world):
    """On 2 of 10 real frames a box's floor outscored the whole inside, and top_z read 3 cm."""
    real = world.segment

    def segment(frame, text):
        whole = real(frame, text)[0].mask
        part = whole.copy()
        rows = np.nonzero(whole.any(axis=1))[0]
        part[rows[:len(rows) // 2]] = False
        return [Detection(mask=part, score=0.9), Detection(mask=whole, score=0.5)]

    world.segment = segment
    items = robot.locate("bowl")
    assert [s.score for s in items] == [0.5]
    assert items[0].top_z == pytest.approx(0.05, abs=0.004)


def test_locate_ignores_a_mask_that_merges_several_objects(robot, world):
    real = world.segment

    def segment(frame, text):
        dets = real(frame, text)
        union = dets[0].mask | dets[1].mask
        return dets + [Detection(mask=union, score=0.1)]

    world.segment = segment
    assert len(robot.locate("block")) == 3


def test_locate_returns_an_empty_list_when_nothing_matches(robot):
    assert robot.locate("purple cube") == []


def test_locate_never_moves_the_arm(robot, world):
    tcp = list(world.arm["left"].tcp)
    robot.locate("block")
    robot.locate("purple cube")
    assert world.arm["left"].tcp == tcp and world.log == []


def test_locate_reads_one_image_of_the_requested_camera(robot, world):
    seen = spy_captures(world)
    robot.locate("red block")
    assert seen == ["head"]
    seen.clear()
    robot.locate("red block", camera="wrist")
    assert seen == ["wrist_left"]


def test_locate_from_the_wrist_above_an_object(robot, world):
    robot.move_to([0.30, 0.12, 0.10], yaw_deg=0, tilt_deg=0)
    s = robot.locate("red block", camera="wrist")[0]
    assert s.camera == "wrist_left"
    assert np.allclose(s.center_xyz[:2], [0.30, 0.12], atol=0.006)
    assert s.top_z == pytest.approx(0.04, abs=0.003)


@pytest.mark.parametrize("bad", ["", "   ", ["red block", "red cube"], None])
def test_locate_wants_one_noun_phrase(robot, bad):
    with pytest.raises(ValueError):
        robot.locate(bad)


def test_locate_rejects_an_unknown_camera(robot):
    with pytest.raises(ValueError, match="side"):
        robot.locate("red block", camera="side")


def test_locate_measures_a_rotated_flat_object_from_one_camera():
    w = FakeWorld([Box("tile", [0.30, 0.10], 0.0, [0.08, 0.04, 0.02], yaw_deg=30)], arms=("left",), max_opening_m=0.10)
    s = make_robot(w).locate("tile")[0]
    assert s.camera == "head" and s.note is None
    assert np.allclose(s.center_xyz, [0.30, 0.10, 0.01], atol=0.008)
    assert s.top_z == pytest.approx(0.02, abs=0.004)
    assert np.allclose(s.size, [0.08, 0.04, 0.02], atol=0.012)
    assert s.long_axis_deg == pytest.approx(30, abs=5)
    assert s.grasp_yaw_deg == pytest.approx(-60, abs=5)   # closes across the short side
    assert np.allclose(s.grasp_xyz, [0.30, 0.10, 0.01], atol=0.008)
    assert s.open_width_m == pytest.approx(0.07, abs=0.012)


def _no_depth_stream():
    return FakeWorld([Box("tile", [0.30, 0.10], 0.0, [0.06, 0.04, 0.01])], arms=("left",), depth=False)


def _depth_all_invalid():
    w = FakeWorld([Box("tile", [0.30, 0.10], 0.0, [0.06, 0.04, 0.01])], arms=("left",))
    real = w.capture

    def capture(camera):
        frame = real(camera)
        frame.depth = np.zeros_like(frame.depth)      # a black, glossy object gives no depth
        return frame

    w.capture = capture
    return w


@pytest.mark.parametrize("make_world", [_no_depth_stream, _depth_all_invalid])
def test_locate_without_depth_gives_the_table_position_and_says_height_is_unknown(make_world):
    s = make_robot(make_world()).locate("tile")[0]
    assert np.allclose(s.center_xyz[:2], [0.30, 0.10], atol=0.015)
    assert s.top_z == 0.0 and s.size[2] == 0.0
    assert s.grasp_xyz is None and s.grasp_yaw_deg is not None
    assert "height unknown" in s.note


def test_locate_takes_a_thin_objects_outline_from_its_whole_mask():
    """A real 3.6 x 1.3 cm piece, a few mm above the table, measured 1.7 x 0.5 cm: only the middle
    of it gave depth points above the table."""
    w = FakeWorld([Box("thin", [0.30, 0.10], 0.0, [0.036, 0.013, 0.007], yaw_deg=30)], arms=("left",), resolution=(640, 480))
    real = w.capture

    def capture(camera):
        frame = real(camera)
        mask = frame.meta["ids"] == 0
        rows, cols = np.nonzero(mask)
        middle = np.zeros_like(mask)
        middle[int(rows.mean()) - 1:int(rows.mean()) + 2, int(cols.mean()) - 2:int(cols.mean()) + 3] = True
        frame.depth = np.where(mask & ~middle, 0.0, frame.depth).astype(np.float32)
        return frame

    w.capture = capture
    s = make_robot(w).locate("thin")[0]
    assert np.allclose(s.size[:2], [0.036, 0.013], atol=0.006)
    assert np.allclose(s.center_xyz[:2], [0.30, 0.10], atol=0.004)
    assert s.long_axis_deg == pytest.approx(30, abs=8)
    assert s.top_z == pytest.approx(0.007, abs=0.002) and s.grasp_xyz is not None
    assert s.note is None


@pytest.mark.parametrize("xy", [[0.26, 0.20], [0.44, 0.05], [0.24, -0.20], [0.46, -0.10]])
def test_locate_does_not_count_the_side_a_camera_sees_into_a_thin_objects_centre(xy):
    """The near side is in the mask too and would pull the centre toward the camera by 5 mm."""
    w = FakeWorld([Box("thin", xy, 0.0, [0.04, 0.03, 0.012])], arms=("left",), resolution=(640, 480))
    s = make_robot(w).locate("thin")[0]
    assert np.allclose(s.center_xyz[:2], xy, atol=0.002), s.center_xyz


def test_locate_says_when_an_object_is_cut_off_by_the_image_edge():
    w = FakeWorld([Box("plank", [0.40, 0.75], 0.0, [0.10, 0.30, 0.03])], arms=("left",))
    s = make_robot(w).locate("plank")[0]
    assert "cut off" in s.note


def test_locate_says_when_an_object_is_too_wide_for_the_gripper():
    w = FakeWorld([Box("melon", [0.30, 0.0], 0.0, [0.09, 0.09, 0.08])], arms=("left",), max_opening_m=0.10)
    s = make_robot(w).locate("melon")[0]
    assert "too_wide" in s.note
    assert s.open_width_m == pytest.approx(0.10)


def test_sight_pixel_is_row_and_col_of_the_object_in_that_camera(robot, world):
    s = robot.locate("yellow block")[0]
    f = world.capture("head")
    uv, _ = geo.project(f.K, f.T_wc, np.array([[0.25, -0.02, 0.04]]))     # top face centre
    col, row = uv[0]
    assert abs(col - row) > 15                                              # a swap would show
    assert abs(s.pixel[0] - row) < 10 and abs(s.pixel[1] - col) < 10


def test_sight_color_is_the_typical_colour_of_the_object(robot):
    assert robot.locate("red block")[0].color == list(RED)
    assert robot.locate("blue bowl")[0].color == [40, 60, 200]


def test_printing_a_sight_shows_its_color(ex):
    assert "color=[210, 30, 30]" in run(ex, "print(locate('red block')[0])")["output"]


def test_sight_points_are_the_world_points_of_that_object(robot):
    s = robot.locate("red block")[0]
    assert s.points.ndim == 2 and s.points.shape[1] == 3 and len(s.points) > 20
    assert np.all(np.abs(s.points[:, 0] - 0.30) < 0.03) and np.all(np.abs(s.points[:, 1] - 0.12) < 0.03)
    assert s.points[:, 2].max() == pytest.approx(0.04, abs=0.002)


def test_printing_a_sight_stays_short_and_leaves_out_the_points(ex):
    out = run(ex, "s = locate('red block')[0]\nprint(s)")
    assert "center_xyz" in out["output"] and "points" not in out["output"]
    assert len(out["output"]) < 400
    assert len(json.dumps(out["ledger"])) < 1000


def test_sights_can_be_compared_in_model_code(ex):
    out = run(ex, "a = locate('block')\nprint(a[0] in a, a[1] in a[:1], a.index(a[2]))")
    assert out["output"].split() == ["True", "False", "2"]


# ---------------------------------------------------------------- capture


def test_capture_gives_world_coordinates_for_every_colour_pixel(robot, world):
    s = robot.locate("red block")[0]
    f = robot.capture()
    assert f.camera == "head" and f.rgb.shape == (240, 320, 3) and f.world.shape == (240, 320, 3)
    r, c = s.pixel
    assert tuple(int(v) for v in f.rgb[r, c]) == RED                        # same pixel grid
    assert np.allclose(f.world[r, c, :2], [0.30, 0.12], atol=0.03)
    assert 0.0 <= f.world[r, c, 2] <= 0.042                                 # on the block, not the table plane
    table = (f.rgb == np.array(TABLE, np.uint8)).all(axis=2)
    assert table.sum() > 1000 and np.abs(f.world[table, 2]).max() < 0.001


def test_capture_without_a_depth_stream_has_all_nan_coordinates():
    f = make_robot(booth(depth=False)).capture()
    assert f.rgb.shape == (240, 320, 3)
    assert np.isnan(f.world).all()


def test_capture_from_model_code_prints_short_and_allows_array_math(ex):
    out = run(ex, "f = capture('head')\nprint(f)\n"
                  "red = (f.rgb[..., 0] > 180) & (f.rgb[..., 1] < 60) & (f.world[..., 2] > TABLE_Z + 0.004)\n"
                  "m = f.world[red].mean(axis=0)\nprint(float(m[0]), float(m[1]))")
    first, second = out["output"].splitlines()
    assert len(first) < 100
    x, y = (float(v) for v in second.split())
    assert x == pytest.approx(0.30, abs=0.02) and y == pytest.approx(0.12, abs=0.02)
    assert len(json.dumps(out["ledger"])) < 1000


# ---------------------------------------------------------------- show


def test_show_hands_the_camera_image_to_the_block_result(ex):
    out = run(ex, "print(show('head'))")
    assert out["output"].strip() == "None"
    assert len(out["images"]) == 1
    assert decode(out["images"][0]).shape == (240, 320, 3)


def test_show_from_the_wrist_camera(ex, world):
    seen = spy_captures(world)
    out = run(ex, "show('wrist')")
    assert seen == ["wrist_left"] and len(out["images"]) == 1


def test_show_takes_the_picture_when_called_not_when_the_block_ends(ex, robot, world):
    r, c = robot.locate("red block")[0].pixel

    def shift():
        world.boxes["red block"].center_xy = [0.20, -0.25]

    ex.ns["shift"] = shift
    out = run(ex, "show('head')\nshift()\nshow('head')")
    first, second = (decode(p) for p in out["images"])
    assert first[r, c, 0] > 150 and first[r, c, 1] < 90                     # the red block is there
    assert second[r, c, 1] > 170                                            # gone: bright table


def test_each_block_returns_only_its_own_images(ex):
    assert len(run(ex, "show('head')")["images"]) == 1
    assert run(ex, "print(1)")["images"] == []


def test_a_block_that_raises_still_returns_its_images(ex):
    out = ex.run("show('head')\nraise ValueError('boom')")
    assert not out["ok"] and "boom" in out["error"]
    assert len(out["images"]) == 1


def test_show_draws_numbered_marks_on_the_camera_image(ex):
    plain = decode(run(ex, "show('head')")["images"][0]).astype(int)
    marked = decode(run(ex, "show('head', marks=locate('block'))")["images"][0]).astype(int)
    assert (np.abs(plain - marked).sum(axis=2) > 60).sum() > 100


def test_show_a_reference_image_file_downscaled_without_touching_a_camera(ex, world, tmp_path):
    path = tmp_path / "target.png"
    Image.fromarray(np.full((1500, 2000, 3), 200, np.uint8)).save(path)
    seen = spy_captures(world)
    out = run(ex, f"show(image={str(path)!r})")
    assert seen == []
    assert decode(out["images"][0]).shape == (768, 1024, 3)


def test_show_an_array_from_capture(ex):
    out = run(ex, "f = capture('head')\nshow(image=f.rgb[100:160, 50:130])")
    assert decode(out["images"][0]).shape == (60, 80, 3)


def test_show_refuses_image_files_outside_the_allowed_folders(ex):
    out = ex.run("show(image='/etc/hosts.png')")
    assert not out["ok"] and "ValueError" in out["error"] and out["images"] == []


def test_show_marks_need_a_camera_image(ex):
    out = ex.run("show(image=np.zeros((4, 4, 3)), marks=[[0.3, 0.0, 0.0]])")
    assert not out["ok"] and "ValueError" in out["error"]


def test_node_result_carries_the_image_paths(robot):
    from forge_tool import ToolExecutionKey
    from tabletop.dora_node import RunEndpoint, SceneMarker, _Execution

    class Emitter:
        async def emit(self, event):
            pass

    import queue

    key = ToolExecutionKey(invocation_id="i", attempt_id="a")
    endpoint = RunEndpoint(Executor(robot), queue.Queue(), None, SceneMarker(None))
    endpoint._executions[key] = _Execution(key, "{}", Emitter(), phase="running")
    out = {"ok": True, "output": "", "error": None, "elapsed_s": 0.1, "ledger": [], "images": ["/x/a.jpg"]}
    asyncio.run(endpoint.finish(key, "show()", out))
    result = endpoint._executions[key].result
    assert result.status == "succeeded" and result.outputs["images"] == ["/x/a.jpg"]


# ---------------------------------------------------------------- look


class FakeVLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, url, body, headers):
        self.calls.append(body)
        return {"choices": [{"message": {"content": self.answer}}]}


def vision(post, tmp_path):
    return VisionClient(VisionConfig(base_url="http://vlm.test", model="m", api_key_env="TEST_VLM_KEY",
                                     paos_config=str(tmp_path / "none.json")), post=post)


def sent(body):
    parts = body["messages"][0]["content"]
    images = [Image.open(io.BytesIO(base64.b64decode(p["image_url"]["url"].split(",", 1)[1])))
              for p in parts if p["type"] == "image_url"]
    return images, next(p["text"] for p in parts if p["type"] == "text")


def test_look_about_an_array_sends_only_that_image(world, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "k")
    vlm = FakeVLM("a star")
    robot = make_robot(world, vision=vision(vlm, tmp_path))
    seen = spy_captures(world)
    assert robot.look("What shape is this?", camera=None, image=np.zeros((30, 40, 3), np.uint8)) == "a star"
    images, text = sent(vlm.calls[0])
    assert seen == [] and [im.size for im in images] == [(40, 30)]
    assert "camera" not in text


def test_look_needs_an_image_and_marks_need_a_camera(world, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "k")
    vlm = FakeVLM("x")
    robot = make_robot(world, vision=vision(vlm, tmp_path))
    with pytest.raises(ValueError):
        robot.look("q", camera=None)
    with pytest.raises(ValueError):
        robot.look("q", camera=None, image=np.zeros((4, 4, 3), np.uint8), marks=[[0.3, 0.0, 0.0]])
    assert vlm.calls == []


def test_vision_failure_surfaces_at_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "k")
    calls = []

    def busy(url, body, headers):
        calls.append(1)
        raise urllib.error.HTTPError("u", 503, "Service is too busy", {}, None)

    with pytest.raises(VisionError, match="503"):
        vision(busy, tmp_path).ask("q", [])
    assert len(calls) == 1


def test_empty_vision_answer_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "k")
    with pytest.raises(VisionError, match="empty answer"):
        vision(FakeVLM(""), tmp_path).ask("q", [])
