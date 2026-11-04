"""Reading pieces by outline and colour against templates. The shapes are hand-built on the same
grid the real masks are put on; the colours are the ones measured on the real pieces."""

import math

import numpy as np
import pytest

from tabletop import symbols as sym
from tabletop.executor import Executor
from tabletop.fake import Box, FakeWorld
from tabletop.primitives import Robot, RobotConfig
from tabletop.symbols import make_template, read_symbol

GREEN, ORANGE, PINK = [127, 158, 120], [186, 139, 102], [164, 98, 117]
PURPLE, BLUE, YELLOW = [91, 81, 109], [117, 147, 176], [196, 199, 107]
N = sym.GRID_CELLS
AXIS = (np.arange(N) + 0.5) * sym.CELL_M - sym.HALF_M
X, Y = np.meshgrid(AXIS, AXIS)                       # grid[row, col]: row = y, col = x


def shape(inside, turn_deg=0.0):
    """Grid of the points where inside(x, y) holds, with the shape turned by turn_deg."""
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    return inside(c * X + s * Y, -s * X + c * Y)


def bar(length, width):
    return lambda x, y: (np.abs(x) <= length / 2) & (np.abs(y) <= width / 2)


def cross(x, y):
    return bar(0.034, 0.010)(x, y) | bar(0.010, 0.034)(x, y)


def divide(x, y):
    return bar(0.034, 0.010)(x, y) | bar(0.010, 0.024)(x, y)


def nine(x, y):
    """An upright 9: a ring at the top, a tail on the right going down."""
    ring = (np.hypot(x, y - 0.006) <= 0.012) & (np.hypot(x, y - 0.006) >= 0.004)
    return ring | ((x >= 0.004) & (x <= 0.012) & (y >= -0.017) & (y <= 0.006))


def seven(x, y):
    return ((y >= 0.010) & (y <= 0.017) & (np.abs(x) <= 0.012)) | (np.abs(x - y * 0.45 + 0.002) <= 0.004) & (np.abs(y) <= 0.017)


def off(angle, target, every=360.0):
    return abs((angle - target + every / 2) % every - every / 2)


LIBRARY = [
    make_template("+", GREEN, shape(cross), 0.0, every_deg=90.0),
    make_template("×", ORANGE, shape(cross), 45.0, every_deg=90.0),
    make_template("÷", PINK, shape(divide), 90.0),
    make_template("-", BLUE, shape(bar(0.036, 0.013)), 90.0),
    make_template("=", PURPLE, shape(bar(0.035, 0.023)), 90.0),
    make_template("9", PURPLE, shape(nine), 90.0),
    make_template("6", PINK, shape(nine, 180.0), 270.0),     # the same outline as the 9, upside down
    make_template("7", BLUE, shape(seven), 90.0),
]


@pytest.mark.parametrize("symbol, color, inside, top0, every", [
    ("+", GREEN, cross, 0.0, 90.0),
    ("×", ORANGE, cross, 45.0, 90.0),
    ("÷", PINK, divide, 90.0, 180.0),
    ("-", BLUE, bar(0.036, 0.013), 90.0, 180.0),
    ("=", PURPLE, bar(0.035, 0.023), 90.0, 180.0),
    ("9", PURPLE, nine, 90.0, 360.0),
    ("7", BLUE, seven, 90.0, 360.0),
])
@pytest.mark.parametrize("turn", [0, 25, 73, 140, 200, 311])
def test_any_piece_is_read_at_any_turn_with_its_top_direction(symbol, color, inside, top0, every, turn):
    r = read_symbol(color, shape(inside, turn), LIBRARY)
    assert r is not None and r.symbol == symbol
    assert r.every_deg == every
    assert off(r.top_deg, top0 + turn, every) < 5


def test_colour_tells_plus_from_times_on_the_same_outline():
    assert read_symbol(GREEN, shape(cross), LIBRARY).symbol == "+"
    assert read_symbol(ORANGE, shape(cross), LIBRARY).symbol == "×"


def test_colour_tells_six_from_nine_whichever_way_up():
    for turn in (0, 180):
        assert read_symbol(PURPLE, shape(nine, turn), LIBRARY).symbol == "9"
        assert read_symbol(PINK, shape(nine, turn), LIBRARY).symbol == "6"


def test_outline_tells_equals_from_nine_in_the_same_colour():
    assert read_symbol(PURPLE, shape(bar(0.035, 0.023), 40), LIBRARY).symbol == "="
    assert read_symbol(PURPLE, shape(nine, 40), LIBRARY).symbol == "9"


def test_a_shifted_mask_is_still_read():
    grid = np.roll(shape(cross, 20), (2, -3), axis=(0, 1))
    assert read_symbol(GREEN, grid, LIBRARY).symbol == "+"


@pytest.mark.parametrize("name, color, grid", [
    ("no template in that colour", YELLOW, shape(cross)),
    ("a disc is no green template", GREEN, shape(lambda x, y: np.hypot(x, y) <= 0.017)),
    ("a bar cut short by a hand", BLUE, shape(bar(0.019, 0.009))),
    ("a piece twice the size of any template", GREEN, shape(lambda x, y: bar(0.06, 0.02)(x, y) | bar(0.02, 0.06)(x, y))),
    ("next to nothing", GREEN, shape(bar(0.004, 0.004))),
])
def test_what_matches_no_template_gets_no_symbol(name, color, grid):
    assert read_symbol(color, grid, LIBRARY) is None, name


def test_template_round_trip_keeps_what_reading_needs():
    t = make_template("9", PURPLE, shape(nine), 90.0)
    again = sym.Template.from_json(t.to_json())
    assert (again.symbol, again.every_deg) == ("9", 360.0)
    r = read_symbol(PURPLE, shape(nine, 30), [again])
    assert r.symbol == "9" and off(r.top_deg, 120.0) < 5


def test_near_symmetric_outline_gets_half_turn_ambiguity_by_itself():
    assert make_template("0", GREEN, shape(lambda x, y: np.hypot(x / 1.4, y) <= 0.012), 90.0).every_deg == 180.0
    assert make_template("9", PURPLE, shape(nine), 90.0).every_deg == 360.0


def test_the_grid_of_a_real_mask_is_centred_and_on_the_table():
    f, K = 500.0, np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    T = np.diag([1.0, -1.0, -1.0, 1.0])
    T[:3, 3] = [0.30, 0.10, 0.80]                  # straight down from 0.8 m: 1 px = 1.6 mm at z = 0
    mask = np.zeros((480, 640), bool)
    mask[240 - 5:240 + 5, 320 - 10:320 + 10] = True  # 32 x 16 mm on the table
    grid = sym.table_grid(K, T, mask, 0.0)
    assert grid.sum() * sym.CELL_M ** 2 == pytest.approx(0.032 * 0.016, rel=0.25)
    rows, cols = np.nonzero(grid)
    assert abs(rows.mean() - (N - 1) / 2) <= 1 and abs(cols.mean() - (N - 1) / 2) <= 1
    assert np.ptp(cols) > np.ptp(rows)             # long along world x


def scene(library):
    w = FakeWorld([Box("bar", [0.30, 0.10], 0.0, [0.036, 0.013, 0.01], yaw_deg=30, color=tuple(BLUE), aliases=["piece"]),
                   Box("apple", [0.30, -0.10], 0.0, [0.06, 0.06, 0.06], color=(200, 20, 40), aliases=["piece"])],
                  arms=("left",), resolution=(640, 480))
    robot = Robot(w, w, RobotConfig(wrist_cameras={"left": "wrist_left"}))
    robot.symbol_templates = library
    return w, robot


def seen_bar_template():
    """Made the way the real templates are: from the outline the camera sees, sides of the piece included."""
    w = FakeWorld([Box("bar", [0.30, 0.10], 0.0, [0.036, 0.013, 0.01], color=tuple(BLUE), aliases=["piece"])],
                  arms=("left",), resolution=(640, 480))
    frame = Robot(w, w, RobotConfig(wrist_cameras={"left": "wrist_left"}))._frame("head")
    mask = w.segment(frame, "piece")[0].mask
    return make_template("-", BLUE, sym.table_grid(frame.K, frame.T_wc, mask, 0.01), 90.0)


def test_locate_fills_in_the_symbol_only_for_what_matches_a_template():
    _, robot = scene([seen_bar_template() if t.symbol == "-" else t for t in LIBRARY])
    by_symbol = {s.symbol: s for s in robot.locate("piece")}
    assert set(by_symbol) == {"-", None}
    minus = by_symbol["-"]
    assert np.allclose(minus.center_xyz[:2], [0.30, 0.10], atol=0.006)
    assert off(minus.top_deg, 120.0, 180.0) < 10 and minus.top_every_deg == 180.0
    assert by_symbol[None].top_deg is None and by_symbol[None].top_every_deg is None


def test_model_code_can_set_or_correct_a_symbol():
    _, robot = scene(LIBRARY)
    ex = Executor(robot)
    out = ex.run("p = locate('piece')\nunknown = [s for s in p if s.symbol is None][0]\n"
                 "unknown.symbol = '18'\nunknown.top_deg = 90.0\nunknown.top_every_deg = 360.0\nprint(unknown)")
    assert out["ok"], out
    assert "symbol='18'" in out["output"] and "top_deg=90.0" in out["output"]
