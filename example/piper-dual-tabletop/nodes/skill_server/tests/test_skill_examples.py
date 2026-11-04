"""The reference blocks of the three SKILL.md task playbooks, run verbatim in single-arm fake booths.

The fake world renders plain boxes, so what the main model would read from the images of
``show`` comes from the ground truth here: only the one constant that holds that reading is
replaced in the block. These tests check the blocks and the primitives they call, not the model.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from tabletop import geometry as geo
from tabletop.executor import Executor
from tabletop.fake import Box, FakeWorld
from tabletop.primitives import Robot, RobotConfig

_SKILL = (Path(__file__).resolve().parents[3] / "SKILL.md").read_text(encoding="utf-8")
PIECE = "colored number or math symbol piece"


def blocks(letter: str) -> list[str]:
    section = re.search(rf"### Playbook {letter}:(.*?)\n### ", _SKILL, re.S)
    assert section, f"Playbook {letter} missing in SKILL.md"
    return re.findall(r"```python\n(.*?)```", section.group(1), re.S)


def decided(code: str, name: str, value) -> str:
    """The block with the constant the model fills in after looking at the images."""
    out, n = re.subn(rf"^{name} = .*$", f"{name} = {value!r}", code, count=1, flags=re.M)
    assert n == 1, name
    return out


def setup(boxes, tmp_path):
    world = FakeWorld(boxes, arms=("left",), resolution=(640, 480))  # as the real cameras
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    robot.show_dir = tmp_path / "show"
    return world, Executor(robot)


def run(ex, code):
    out = ex.run(code)
    assert out["ok"], (out.get("error"), out.get("output"))
    return out


def nearest(world, sight) -> Box:
    c = np.asarray(sight.center_xyz[:2])
    return min(world.boxes.values(), key=lambda b: np.linalg.norm(np.asarray(b.center_xy) - c))


def fruit(name, xy):
    return Box(name, list(xy), 0.0, [0.05, 0.05, 0.05], aliases=["fruit"], meta={"kind": "fruit"})


def flat_block(name, xy):
    return Box(name, list(xy), 0.0, [0.04, 0.04, 0.012], aliases=["flat colored block"], meta={"kind": "block"})


def test_playbook_a_moves_the_chosen_fruit_into_the_box(tmp_path):
    boxes = [
        Box("cardboard box", [0.36, -0.16], 0.0, [0.14, 0.12, 0.06], aliases=["empty space inside the box"]),
        fruit("peach", [0.30, 0.12]), fruit("mangosteen", [0.42, 0.05]), fruit("apple", [0.45, -0.02]),
    ]
    world, ex = setup(boxes, tmp_path)
    see, act, check = blocks("A")

    assert len(run(ex, see)["images"]) == 1                      # the numbered head image: the task is words
    assert not any("closed on" in line for line in world.log)    # looking touched nothing
    number = 1 + [nearest(world, s).name for s in ex.ns["objects"]].index("peach")
    before = {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items() if n != "peach"}
    out = run(ex, decided(act, "TARGETS", [number]))
    peach, box = world.boxes["peach"], world.boxes["cardboard box"]
    assert box.contains_xy(peach.center) and abs(peach.z_bottom - box.top) < 1e-6, out["output"]
    assert {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items() if n != "peach"} == before

    out = run(ex, check)
    assert "box moved by 0.0" in out["output"] and len(out["images"]) == 1


def test_playbook_a_puts_things_in_the_whole_box_not_in_a_word_for_its_inside():
    """On the real box that word's top_z jumped between 0.06, 0.13 and 0.17 m."""
    code = "".join(blocks("A"))
    assert "empty space" not in code and "place(box)" in code


def box_and_two_fruits():
    return [
        Box("cardboard box", [0.36, -0.16], 0.0, [0.14, 0.12, 0.06], aliases=["empty space inside the box"]),
        fruit("peach", [0.30, 0.12]), fruit("apple", [0.45, -0.02]),
    ]


def test_playbook_a_moves_two_targets_into_the_box(tmp_path):
    world, ex = setup(box_and_two_fruits(), tmp_path)
    see, act, _check = blocks("A")
    run(ex, see)
    numbers = [i for i, s in enumerate(ex.ns["objects"], 1)]
    out = run(ex, decided(act, "TARGETS", numbers))
    box = world.boxes["cardboard box"]
    for name in ("peach", "apple"):
        assert box.contains_xy(world.boxes[name].center), (name, out["output"])


def test_playbook_a_does_not_grasp_at_a_target_that_went_with_the_first(tmp_path):
    """Two cherries on one stem: the first pick took both, and the second closed on nothing."""
    world, ex = setup(box_and_two_fruits(), tmp_path)
    see, act, _check = blocks("A")
    run(ex, see)
    place = ex.ns["place"]

    def place_and_lose_the_apple(*args, **kw):
        out = place(*args, **kw)
        world.boxes.pop("apple", None)
        return out

    ex.ns["place"] = place_and_lose_the_apple
    numbers = [i for i, s in enumerate(ex.ns["objects"], 1)]
    numbers.sort(key=lambda i: nearest(world, ex.ns["objects"][i - 1]).name != "peach")   # the peach first
    out = run(ex, decided(act, "TARGETS", numbers))["output"]
    assert not any("closed on nothing" in line for line in world.log), out


def test_playbook_b_sorts_strays_into_their_groups(tmp_path):
    boxes = [
        flat_block("b1", [0.28, 0.16]), flat_block("b2", [0.34, 0.18]), flat_block("b3", [0.31, 0.10]),
        fruit("f1", [0.30, -0.16]), fruit("f2", [0.37, -0.18]), fruit("f3", [0.33, -0.10]),
        flat_block("stray block", [0.38, -0.05]),                 # sitting next to the fruits
        fruit("stray fruit", [0.44, 0.12]),                       # away from the fruits
    ]
    start = {b.name: list(b.center_xy) for b in boxes}
    world, ex = setup(boxes, tmp_path)
    see, act, check = blocks("B")

    assert len(run(ex, see)["images"]) == 1
    assert len(ex.ns["items"]) == 8
    kinds = {i: nearest(world, s).meta["kind"] for i, s in enumerate(ex.ns["items"], 1)}
    out = run(ex, decided(act, "KINDS", kinds))

    xy = {n: np.asarray(b.center_xy) for n, b in world.boxes.items()}
    blocks_c = np.mean([xy[n] for n in ("b1", "b2", "b3")], axis=0)
    fruits_c = np.mean([xy[n] for n in ("f1", "f2", "f3")], axis=0)
    for n, own, other in (("stray block", blocks_c, fruits_c), ("stray fruit", fruits_c, blocks_c)):
        d_own, d_other = np.linalg.norm(xy[n] - own), np.linalg.norm(xy[n] - other)
        assert d_own < 0.15 and d_own + 0.05 < d_other, (n, xy[n], out["output"])
    for n in ("b1", "b2", "b3", "f1", "f2", "f3"):                # grouped objects stay where they were
        assert np.allclose(xy[n], start[n])
    names = list(world.boxes)
    for i, a in enumerate(names):                                 # nothing placed on top of another
        for b in names[i + 1:]:
            assert np.linalg.norm(xy[a] - xy[b]) > 0.045, (a, b)
    assert len(run(ex, check)["images"]) == 1


def image_direction(world: FakeWorld, box: Box, camera: str = "head") -> str:
    """Where the top of a piece points in an image, as the model reads it."""
    th = math.radians(box.yaw_deg + box.meta["top_deg"])
    f = world.capture(camera)
    uv, _ = geo.project(f.K, f.T_wc, np.vstack([box.center, box.center + 0.03 * np.array([math.cos(th), math.sin(th), 0.0])]))
    du, dv = uv[1] - uv[0]
    return ("right" if du > 0 else "left") if abs(du) > abs(dv) else ("down" if dv > 0 else "up")


ONE_DIGIT = ([("2", [0.26, 0.20], 90), ("6", [0.40, 0.21], 0), ("3", [0.44, 0.05], 180)],
             (["2", "×", "3", "=", "6"], ["3", "×", "2", "=", "6"], ["6", "÷", "2", "=", "3"], ["6", "÷", "3", "=", "2"]))
TWO_DIGIT = ([("18", [0.26, 0.20], 90), ("7", [0.40, 0.21], 0), ("11", [0.44, 0.05], 180)],
             (["18", "-", "7", "=", "11"], ["18", "-", "11", "=", "7"], ["7", "+", "11", "=", "18"], ["11", "+", "7", "=", "18"]))


def equation_pieces(numbers) -> list[Box]:
    spec = numbers + [("+", [0.24, -0.20], 270), ("-", [0.30, -0.24], 0), ("×", [0.41, -0.20], 90),
                      ("÷", [0.46, -0.10], 0), ("=", [0.20, 0.08], 180)]
    return [Box(f"piece {s}", list(xy), 0.0, [0.04, 0.03, 0.012], aliases=[PIECE], meta={"symbol": s, "top_deg": top})
            for s, xy, top in spec]


def assert_row_reads(world, out, equations):
    m = re.search(r"equation: (.+)", out)
    assert m and m.group(1).split() in equations, out
    used = [f"piece {s}" for s in m.group(1).split()]
    right = np.array([0.0, 1.0])                                  # READ_RIGHT_DEG default: +y
    pos = [np.asarray(world.boxes[n].center_xy) for n in used]
    along = [float(p @ right) for p in pos]
    assert all(b - a > 0.05 for a, b in zip(along, along[1:])), (along, out)  # left to right, spaced
    across = [float(p[0]) for p in pos]
    assert max(across) - min(across) < 0.01                       # one straight row
    for n in used:                                                # every top points to READ_UP_DEG
        b = world.boxes[n]
        top = (b.yaw_deg + b.meta["top_deg"]) % 360
        assert abs((top - 180.0 + 180) % 360 - 180) < 6.0, (n, top, out)


@pytest.mark.parametrize("numbers, equations", [ONE_DIGIT, TWO_DIGIT])   # two digits as in the organizers' example
def test_playbook_c_equation_row_reads_correctly(tmp_path, numbers, equations):
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, tiles, _above, act, check = blocks("C")

    assert len(run(ex, see)["images"]) == 1
    moves = sum("move" in line for line in world.log)
    assert len(run(ex, tiles)["images"]) == 8                     # one tile per unread piece
    assert sum("move" in line for line in world.log) == moves     # reading moves nothing
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    out = run(ex, decided(act, "READ", read))["output"]
    assert_row_reads(world, out, equations)
    assert len(run(ex, check)["images"]) == 1


def test_playbook_c_reading_from_above_touches_nothing_and_sets_the_turn(tmp_path):
    numbers, equations = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, tiles, above, act, _check = blocks("C")
    run(ex, see)
    run(ex, tiles)
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    number = next(i for i, (symbol, _) in read.items() if symbol == "6")
    six = world.boxes["piece 6"]
    run(ex, "move_to([0.30, 0.0, 0.20], yaw_deg=90)")             # the wrist image turns with the gripper
    before = {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()}

    out = run(ex, decided(above, "LOOK", [number]))

    assert len(out["images"]) == 1                                # the wrist image
    assert world.moves[-1]["xyz"][:2] == pytest.approx(six.center_xy[:2], abs=0.003)   # as locate measured it
    assert world.moves[-1]["tilt"] == pytest.approx(0.0)
    assert not any("closed on" in line for line in world.log)
    assert {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()} == before

    from_above = image_direction(world, six, "wrist_left")
    assert from_above != read[number][1]                          # else this test cannot tell the two images apart
    read[number] = ("6", from_above)
    assert_row_reads(world, run(ex, decided(act, "READ", read))["output"], equations)


def test_playbook_c_at_1080p_tiles_hold_whole_pieces_and_the_row_reads(tmp_path):
    from .test_two_resolutions import World1080

    numbers, equations = ONE_DIGIT
    world = World1080(equation_pieces(numbers))
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    robot.show_dir = tmp_path / "show"
    ex = Executor(robot)
    see, tiles, _above, act, _check = blocks("C")
    run(ex, see)
    run(ex, tiles)

    for s in ex.ns["pieces"]:
        t = ex.ns["tile"](s)
        own = np.array(s.color)
        h, w = t.shape[:2]
        assert np.abs(t[h // 2, w // 2].astype(int) - own).sum() < 30       # the piece is in the middle
        rim = np.concatenate([t[0], t[-1], t[:, 0], t[:, -1]])
        assert np.abs(rim.astype(int) - own).sum(axis=1).min() > 30, s      # and stops before the edge

    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    assert_row_reads(world, run(ex, decided(act, "READ", read))["output"], equations)
