"""Reading the number and operator pieces of the equation task by outline and colour.

A piece's mask is put on a world grid on the table, so its outline has its true size and shape
whatever the camera's angle. The outline is turned against saved templates of the real pieces
(built with dev/label_pieces.py); colour limits which templates are compared, and tells apart
pieces with the same outline (+ and ×, 6 and 9). What matches no template stays unread, and so does
what fits two of them about equally: same-colour digits can be that close, and the model can look.
"""

from __future__ import annotations

import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry as geo

CELL_M = 0.001             # finer than a head camera pixel; thin pieces (-, 7) read wrong at 2 mm
HALF_M = 0.036
GRID_CELLS = int(round(2 * HALF_M / CELL_M))
MIN_CELLS = 120
MAX_PIECE_M = 0.06         # larger objects are not compared at all
MAX_HUE_DIFF_DEG = 18.0    # colours of different pieces differ by 30 degrees or more
MIN_OVERLAP = 0.80         # measured over 3 arrangements: blocks and background reach 0.79 on a digit
MIN_LEAD = 0.06            # over the next symbol. Measured: misreads lead by 0.04 at most (8 as 6, 9 as =)
TURN_STEP_DEG = 5.0
HALF_TURN_SAME = 0.85      # overlap of an outline with itself turned half a turn: top is ambiguous
TEMPLATE_FILE = Path(__file__).with_name("piece_templates.json")


def hue_of(color) -> float:
    return colorsys.rgb_to_hsv(*(float(c) / 255.0 for c in list(color)[:3]))[0] * 360.0


def _hue_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


_AXIS = (np.arange(GRID_CELLS) + 0.5) * CELL_M - HALF_M
_X, _Y = np.meshgrid(_AXIS, _AXIS)


def _turned(grid: np.ndarray, turn_deg: float) -> np.ndarray:
    """The grid's outline turned about the grid's middle (each new cell looks up where it came from)."""
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    col = np.rint((c * _X + s * _Y + HALF_M) / CELL_M - 0.5).astype(int)
    row = np.rint((-s * _X + c * _Y + HALF_M) / CELL_M - 0.5).astype(int)
    inside = (col >= 0) & (col < GRID_CELLS) & (row >= 0) & (row < GRID_CELLS)
    out = np.zeros_like(grid)
    out[inside] = grid[row[inside], col[inside]]
    return out


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    return int((a & b).sum()) / max(int((a | b).sum()), 1)


def _fit(a: np.ndarray, b: np.ndarray) -> float:
    """Best overlap with one outline moved by up to a cell (their middles are only known to a cell)."""
    return max(_overlap(np.roll(a, (dr, dc), axis=(0, 1)), b) for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _centred(grid: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(grid)
    shift = (int(round((GRID_CELLS - 1) / 2 - rows.mean())), int(round((GRID_CELLS - 1) / 2 - cols.mean())))
    return np.roll(grid, shift, axis=(0, 1))


def long_axis_deg(grid: np.ndarray) -> float:
    """Direction of the long side of the outline's smallest rectangle, degrees."""
    rows, cols = np.nonzero(grid)
    return float(geo.min_area_rect(np.column_stack([cols, rows]).astype(float) * CELL_M).long_axis_deg)


@dataclass
class Template:
    symbol: str
    hue: float
    top_deg: float        # heading of the symbol's top, with the outline as stored
    every_deg: float      # the outline looks the same turned by this much
    grid: np.ndarray      # the outline, centred on its middle

    def to_json(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "hue": round(self.hue, 1), "top_deg": round(self.top_deg, 1),
                "every_deg": self.every_deg,
                "rows": ["".join("#" if v else "." for v in row) for row in self.grid]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Template":
        grid = np.array([[c == "#" for c in row] for row in d["rows"]], dtype=bool)
        return cls(d["symbol"], float(d["hue"]), float(d["top_deg"]), float(d["every_deg"]), grid)


def make_template(symbol: str, color, grid: np.ndarray, top_deg: float, every_deg: float | None = None) -> Template:
    """A template from a piece's grid. Unless given, ``every_deg`` is 180 for an outline that
    looks about the same turned half a turn, else 360."""
    centred = _centred(grid)
    if every_deg is None:
        every_deg = 180.0 if _fit(_turned(centred, 180.0), centred) >= HALF_TURN_SAME else 360.0
    return Template(symbol, hue_of(color), float(top_deg), float(every_deg), centred)


def load_templates(path: Path = TEMPLATE_FILE) -> list[Template]:
    if not path.exists():
        return []
    return [Template.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def table_grid(K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray, z: float, center=None) -> np.ndarray:
    """The mask on the plane at height z, on a world grid around ``center`` (default: the mask's
    middle): grid[row, col] with col along world x and row along world y."""
    if center is None:
        center = geo.mask_to_plane(K, T_wc, mask, z)[:, :2].mean(axis=0)
    axis = (np.arange(GRID_CELLS) + 0.5) * CELL_M - HALF_M
    gx, gy = np.meshgrid(center[0] + axis, center[1] + axis)
    uv, depth = geo.project(K, T_wc, np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)]))
    col, row = np.floor(uv[:, 0]).astype(int), np.floor(uv[:, 1]).astype(int)
    h, w = mask.shape[:2]
    seen = (depth > 0) & (col >= 0) & (col < w) & (row >= 0) & (row < h)
    grid = np.zeros(gx.size, dtype=bool)
    grid[seen] = mask[row[seen], col[seen]]
    return grid.reshape(GRID_CELLS, GRID_CELLS)


@dataclass
class Reading:
    symbol: str
    top_deg: float    # world heading of the symbol's top
    every_deg: float  # the piece looks the same turned by this much
    overlap: float


def read_symbol(color, grid: np.ndarray, templates: list[Template]) -> Reading | None:
    """The template of this colour that the outline fits best at some turn, if it fits well."""
    if int(grid.sum()) < MIN_CELLS:
        return None
    hue = hue_of(color)
    candidates = [t for t in templates if _hue_diff(t.hue, hue) <= MAX_HUE_DIFF_DEG]
    if not candidates:
        return None
    target = _centred(grid)
    fits = []
    for t in candidates:
        coarse = max(np.arange(0.0, 360.0, TURN_STEP_DEG), key=lambda a: _overlap(_turned(t.grid, float(a)), target))
        fine = np.arange(coarse - TURN_STEP_DEG, coarse + TURN_STEP_DEG + 0.5, 1.0)
        turn = max(fine, key=lambda a: _fit(_turned(t.grid, float(a)), target))
        fits.append((_fit(_turned(t.grid, float(turn)), target), float(turn), t))
    overlap, turn, t = max(fits, key=lambda f: f[0])
    rival = max((f[0] for f in fits if f[2].symbol != t.symbol), default=0.0)
    if overlap < MIN_OVERLAP or overlap - rival < MIN_LEAD:
        return None
    return Reading(t.symbol, t.top_deg + turn, t.every_deg, overlap)


def distinct(detections: list) -> list:
    """Best score first, each object once. SAM3 gives some objects several masks. A mask mostly
    inside a better one adds nothing (a fruit with and without its leaf, a piece of its stem). A
    whole replaces a better-scoring part of it (the inside of a box and its floor alone), and a
    mask that holds several found objects is dropped."""

    def inside(a, b) -> bool:
        return int((a.mask & b.mask).sum()) > 0.7 * int(a.mask.sum())

    kept: list = []
    for d in sorted(detections, key=lambda d: -d.score):
        parts = [k for k in kept if inside(k, d)]
        if len(parts) > 1 or any(inside(d, k) for k in kept):
            continue
        kept = [k for k in kept if not any(k is p for p in parts)] + [d]
    return sorted(kept, key=lambda d: -d.score)
