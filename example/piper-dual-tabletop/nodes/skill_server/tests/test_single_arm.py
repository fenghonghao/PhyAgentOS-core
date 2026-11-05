"""look() in the booth: marks, given images, its failures."""

import base64
import io
import json

import numpy as np
import pytest

from tabletop.fake import Box, FakeWorld
from tabletop.primitives import Robot, RobotConfig
from tabletop.vision import VisionClient, VisionConfig, VisionError, draw_marks, parse_json


def booth(**kw):
    boxes = [
        Box("blue bowl", [0.35, -0.15], 0.0, [0.12, 0.12, 0.05], aliases=["bowl", "container"], color=(40, 60, 200)),
        Box("red block", [0.30, 0.12], 0.0, [0.04, 0.04, 0.04], aliases=["block", "cube"], color=(210, 30, 30)),
        Box("green block", [0.42, 0.05], 0.0, [0.04, 0.04, 0.04], aliases=["block", "cube"], color=(30, 180, 60)),
        Box("yellow block", [0.25, -0.02], 0.0, [0.04, 0.04, 0.04], aliases=["block", "cube"], color=(230, 210, 40)),
        Box("apple", [0.45, -0.02], 0.0, [0.06, 0.06, 0.06], aliases=["fruit"], color=(200, 20, 40)),
    ]
    return FakeWorld(boxes, arms=("left",), **kw)


@pytest.fixture
def world():
    return booth()


@pytest.fixture
def robot(world):
    return Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))


# ---------------------------------------------------------------- look()

class FakeVLM:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(self, url, body, headers):
        self.calls.append((url, body, headers))
        return {"choices": [{"message": {"content": self.answer}}]}


def vision(post, tmp_path):
    cfg = VisionConfig(base_url="http://vlm.test/v1", model="test-model", api_key_env="TEST_VLM_KEY",
                       paos_config=str(tmp_path / "none.json"))
    return VisionClient(cfg, post=post)


def images_of(body):
    parts = body["messages"][0]["content"]
    return [p for p in parts if p["type"] == "image_url"], [p["text"] for p in parts if p["type"] == "text"][0]


def test_look_sends_marked_head_image_and_parses_json(world, tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_VLM_KEY", "k-test")
    vlm = FakeVLM('```json\n{"1": "block", "2": "fruit"}\n```')
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}), vision=vision(vlm, tmp_path))
    items = robot.locate("red block") + robot.locate("apple")
    ans = robot.look("Is each marked object a block or a fruit? JSON", marks=items, json=True)
    assert ans == {"1": "block", "2": "fruit"}
    url, body, headers = vlm.calls[0]
    assert url == "http://vlm.test/v1/chat/completions" and body["model"] == "test-model"
    assert headers["Authorization"] == "Bearer k-test"
    imgs, text = images_of(body)
    assert len(imgs) == 1 and "Numbered colored rings (1 to 2)" in text and "head camera" in text


def test_look_with_given_image_sends_it_first(world, tmp_path, monkeypatch):
    from PIL import Image

    monkeypatch.setenv("TEST_VLM_KEY", "k-test")
    target = tmp_path / "target.png"
    Image.fromarray(np.full((40, 60, 3), 200, np.uint8)).save(target)
    vlm = FakeVLM("red block")
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}), vision=vision(vlm, tmp_path))
    assert robot.look("Which object is shown in image 1?", image=str(target)) == "red block"
    imgs, text = images_of(vlm.calls[0][1])
    assert len(imgs) == 2 and text.startswith("Image 1 is an image given with this question")
    first = Image.open(io.BytesIO(base64.b64decode(imgs[0]["image_url"]["url"].split(",", 1)[1])))
    assert first.size == (60, 40)
    with pytest.raises(ValueError):
        robot.look("x", image="/etc/passwd")


def test_look_without_config_or_key_fails_clearly(world, tmp_path, monkeypatch):
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    with pytest.raises(RuntimeError, match="not configured"):
        robot.look("what is on the table?")
    monkeypatch.delenv("TEST_VLM_KEY", raising=False)
    robot.vision = vision(FakeVLM("x"), tmp_path)
    with pytest.raises(VisionError, match="no API key"):
        robot.look("what is on the table?")


def test_draw_marks_changes_pixels_near_the_object(world):
    f = world.capture("head")
    s = Robot(world, world, RobotConfig()).locate("apple")[0]
    img, drawn = draw_marks(f, [s, [5.0, 5.0, 0.0]])
    assert drawn == [1]                                      # the far point is outside the view
    assert (img != f.rgb).any(axis=2).sum() > 50


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Answer:\n[1, 2]\nthanks') == [1, 2]
    with pytest.raises(ValueError):
        parse_json("no json here")
    assert json.dumps(parse_json('```json\n{"x": "÷"}\n```'), ensure_ascii=False) == '{"x": "÷"}'


