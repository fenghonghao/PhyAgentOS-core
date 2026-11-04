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
from PIL import Image

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


FREE_PLACE = [0.33, 0.05]                                         # clear of equation_pieces


def in_free_place(act: str, read: dict) -> str:
    return decided(decided(act, "READ", read), "CENTRE", FREE_PLACE)


def equation_pieces(numbers) -> list[Box]:
    spec = numbers + [("+", [0.24, -0.20], 270), ("-", [0.30, -0.24], 0), ("×", [0.41, -0.20], 90),
                      ("÷", [0.46, -0.10], 0), ("=", [0.20, 0.08], 180)]
    return [Box(f"piece {s}", list(xy), 0.0, [0.04, 0.03, 0.012], aliases=[PIECE], meta={"symbol": s, "top_deg": top})
            for s, xy, top in spec]


def assert_row_reads(world, out, equations):
    m = re.search(r"equation: (.+)", out)
    assert m and m.group(1).split() in equations, out
    used = [f"piece {s}" for s in m.group(1).split()]
    right = np.array([0.0, -1.0])                                 # READ_RIGHT_DEG default: -y
    pos = [np.asarray(world.boxes[n].center_xy) for n in used]
    along = [float(p @ right) for p in pos]
    assert all(b - a > 0.05 for a, b in zip(along, along[1:])), (along, out)  # left to right, spaced
    across = [float(p[0]) for p in pos]
    assert max(across) - min(across) < 0.01                       # one straight row
    for n in used:                                                # every top points to READ_UP_DEG
        b = world.boxes[n]
        top = (b.yaw_deg + b.meta["top_deg"]) % 360
        assert abs((top + 180) % 360 - 180) < 6.0, (n, top, out)


@pytest.mark.parametrize("numbers, equations", [ONE_DIGIT, TWO_DIGIT])   # two digits as in the organizers' example
def test_playbook_c_equation_row_reads_correctly(tmp_path, numbers, equations):
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, act, check = blocks("C")

    before = {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()}
    assert len(run(ex, see)["images"]) == 1                       # the head image: no chart, no tiles
    assert {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()} == before   # reading touches nothing
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    out = run(ex, in_free_place(act, read))["output"]
    assert_row_reads(world, out, equations)
    assert len(run(ex, check)["images"]) == 1


def test_playbook_c_reading_from_above_touches_nothing_and_sets_the_turn(tmp_path):
    numbers, equations = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, above, act, _check = blocks("C")
    run(ex, see)
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    number = next(i for i, (symbol, _) in read.items() if symbol == "6")
    six = world.boxes["piece 6"]
    before = {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()}

    asked = []
    ex.ns["look"] = lambda question, **kw: asked.append((question, kw)) or "6, its top points up"
    out = run(ex, decided(above, "LOOK", {number: ["6", "8"]}))

    assert len(asked) == 1 and "6, 8" in asked[0][0] and asked[0][1] == {"camera": "wrist"}, asked
    assert "look: 6, its top points up" in out["output"]
    heading = math.atan2(six.center_xy[1], six.center_xy[0])
    short_of_it = [six.center_xy[0] - 0.04 * math.cos(heading), six.center_xy[1] - 0.04 * math.sin(heading)]
    assert world.moves[-1]["xyz"][:2] == pytest.approx(short_of_it, abs=0.003)
    assert world.moves[-1]["tilt"] == pytest.approx(25.0)
    assert not any("closed on" in line for line in world.log)
    assert {n: (list(b.center_xy), b.z_bottom) for n, b in world.boxes.items()} == before

    from_above = image_direction(world, six, "wrist_left")
    assert from_above != read[number][1]                          # else this test cannot tell the two images apart
    read[number] = ("6", from_above)
    assert_row_reads(world, run(ex, in_free_place(act, read))["output"], equations)


def test_playbook_c_at_1080p_the_row_reads(tmp_path):
    from .test_two_resolutions import World1080

    numbers, equations = ONE_DIGIT
    world = World1080(equation_pieces(numbers))
    robot = Robot(world, world, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    robot.show_dir = tmp_path / "show"
    ex = Executor(robot)
    see, _above, act, _check = blocks("C")
    run(ex, see)

    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    assert_row_reads(world, run(ex, in_free_place(act, read))["output"], equations)


def test_playbook_c_does_not_move_anything_when_an_unused_piece_is_in_the_way(tmp_path):
    numbers, _ = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, act, _check = blocks("C")
    run(ex, see)
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    moves = len(world.moves)
    out = run(ex, decided(decided(act, "READ", read), "CENTRE", [0.30, -0.24]))["output"]   # on the unused "-"
    assert "in the way: [" in out and "in the way: []" not in out, out
    assert len(world.moves) == moves


def test_playbook_c_turns_a_piece_whose_top_is_unknown_not_at_all(tmp_path):
    numbers, equations = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, act, _check = blocks("C")
    run(ex, see)
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    code = in_free_place(act, read).replace("s.top_every_deg = 360.0", "s.top_every_deg = 360.0\n    s.top_deg = None", 1)
    out = run(ex, code)["output"]
    assert re.search(r"equation: \S", out) and out.count("True") >= 5, out


def test_the_chart_of_the_pieces_can_be_shown(tmp_path):
    world, ex = setup([], tmp_path)
    out = run(ex, 'show(image=PIECE_CHART)')
    assert len(out["images"]) == 1
    assert Image.open(out["images"][0]).size[0] > 500


ALL_PIECES = [str(n) for n in range(1, 21)] + ["+", "-", "×", "÷", "="]


def test_each_piece_has_its_own_picture_to_show(tmp_path):
    world, ex = setup([], tmp_path)
    out = run(ex, 'print(sorted(PIECE_IMAGES))\nshow(image=PIECE_IMAGES["6"])\nshow(image=PIECE_IMAGES["÷"])')
    assert len(out["images"]) == 2
    assert all(Image.open(p).size[0] >= 200 for p in out["images"])
    assert sorted(ex.ns["PIECE_IMAGES"]) == sorted(ALL_PIECES)
    assert all(Path(p).is_file() for p in ex.ns["PIECE_IMAGES"].values())


def wrist_reads(locate, symbol, top_deg):
    """`locate` where only the wrist camera reads the pieces, the way the head camera did not."""

    def reading(text, camera="head"):
        found = locate(text, camera=camera)
        if camera == "wrist":
            for w in found:
                w.symbol, w.top_deg, w.top_every_deg = symbol, top_deg, 360.0
        return found

    return reading


def test_playbook_c_block_1_reads_unread_pieces_from_close_up_without_look(tmp_path):
    numbers, _ = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, above, _act, _check = blocks("C")
    asked = []
    ex.ns["locate"] = wrist_reads(ex.ns["locate"], "7", 30.0)
    ex.ns["look"] = lambda *a, **k: asked.append(a) or "?"
    out = run(ex, see)["output"]
    assert "still unread: []" in out and not asked, out
    assert all(s.symbol == "7" and s.top_deg == 30.0 for s in ex.ns["pieces"])

    moves = len(world.moves)
    run(ex, above)                                                # nothing filled in: nothing to ask
    assert len(world.moves) == moves and not asked


def test_playbook_c_a_piece_without_its_top_stays_unread_from_close_up(tmp_path):
    numbers, _ = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, _act, _check = blocks("C")
    ex.ns["locate"] = wrist_reads(ex.ns["locate"], "7", None)
    out = run(ex, see)["output"]
    assert "still unread: [1, 2, 3, 4, 5, 6, 7, 8]" in out, out
    assert "from close up: 7 None" in out, out                    # what it did see is there for the model


def test_playbook_c_blocks_copied_unchanged_keep_what_locate_read(tmp_path):
    numbers, equations = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, act, _check = blocks("C")
    locate = ex.ns["locate"]

    def reads_all(text, camera="head"):
        found = locate(text, camera=camera)
        for s in found:
            b = nearest(world, s)
            s.symbol, s.top_deg, s.top_every_deg = b.meta["symbol"], (b.yaw_deg + b.meta["top_deg"]) % 360, 360.0
        return found

    ex.ns["locate"] = reads_all
    moves = len(world.moves)
    out = run(ex, see)
    assert "still unread: []" in out["output"] and len(out["images"]) == 1, out
    assert len(world.moves) <= moves + 1                          # at most home
    assert_row_reads(world, run(ex, decided(act, "CENTRE", FREE_PLACE))["output"], equations)


def test_playbook_c_moves_a_piece_in_the_way_to_its_free_place_first(tmp_path):
    numbers, equations = ONE_DIGIT
    world, ex = setup(equation_pieces(numbers), tmp_path)
    see, _above, act, _check = blocks("C")
    run(ex, see)
    read = {i: (nearest(world, s).meta["symbol"], image_direction(world, nearest(world, s)))
            for i, s in enumerate(ex.ns["pieces"], 1)}
    code = decided(decided(decided(act, "READ", read), "CENTRE", [0.30, -0.12]), "AWAY", [[0.33, 0.12]])   # on the unused "-"
    out = run(ex, code)["output"]
    assert "in the way: [" in out and "in the way: []" not in out, out
    assert world.boxes["piece -"].center_xy[:2] == pytest.approx([0.33, 0.12], abs=0.01), out
    assert_row_reads(world, out, equations)
