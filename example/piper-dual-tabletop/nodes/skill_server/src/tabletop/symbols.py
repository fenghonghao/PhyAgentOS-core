"""Reading the number and operator pieces of the equation task by outline and colour.

A piece's mask is put on a world grid on the table, so its outline has its true size and shape
whatever the camera's angle. The outline is turned against saved templates of the real pieces
(built with dev/label_pieces.py and dev/build_colour_templates.py); colour limits which templates
are compared, and tells apart pieces with the same outline (+ and ×, 6 and 9).

Same-colour digits can have nearly the same mask, since a mask fills their holes in. When the mask
is not sure, a second outline decides: the piece's colour picked out on the camera's full image,
which keeps the holes. What neither is sure of stays unread: the model can look. A symbol can come
without a direction, when both outlines look about the same upside down.
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
MIN_OVERLAP = 0.70         # lowest fit of a right read on moved pieces: 0.71. The scenes hold no other blocks
MIN_LEAD = 0.03            # over the next symbol. Measured: misreads lead by 0.023 at most (6 as 8)
MIN_HALF_LEAD = 0.04       # over its own half turn. Measured: wrong directions lead by 0.027 at most (5)
COLOUR_MIN_OVERLAP = 0.60  # the same for the colour outline
COLOUR_MIN_LEAD = 0.06     # measured: misreads lead by 0.05 at most (13 as 16)
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
    outline: str = "mask" # mask: the mask's outline. colour: the piece's colour on the full image

    def to_json(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "hue": round(self.hue, 1), "top_deg": round(self.top_deg, 1),
                "every_deg": self.every_deg, "outline": self.outline,
                "rows": ["".join("#" if v else "." for v in row) for row in self.grid]}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Template":
        grid = np.array([[c == "#" for c in row] for row in d["rows"]], dtype=bool)
        return cls(d["symbol"], float(d["hue"]), float(d["top_deg"]), float(d["every_deg"]), grid,
                   d.get("outline", "mask"))


def make_template(symbol: str, color, grid: np.ndarray, top_deg: float, every_deg: float | None = None,
                  outline: str = "mask") -> Template:
    """A template from a piece's grid. Unless given, ``every_deg`` is 180 for an outline that
    looks about the same turned half a turn, else 360."""
    centred = _centred(grid)
    if every_deg is None:
        every_deg = 180.0 if _fit(_turned(centred, 180.0), centred) >= HALF_TURN_SAME else 360.0
    return Template(symbol, hue_of(color), float(top_deg), float(every_deg), centred, outline)


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


def _lab(rgb) -> np.ndarray:
    """CIE Lab of sRGB colours, on the last axis."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    t = xyz / np.array([0.9505, 1.0, 1.089])
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def _split_level(values: np.ndarray) -> float:
    """The level that best splits the values in two groups (Otsu)."""
    counts, edges = np.histogram(values, bins=64)
    middles = (edges[:-1] + edges[1:]) / 2
    weight, mass = np.cumsum(counts).astype(float), np.cumsum(counts * middles)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mass[-1] * weight - mass * weight[-1]) ** 2 / (weight * (weight[-1] - weight))
    return float(edges[int(np.nanargmax(between[:-1])) + 1])


def colour_outline(rgb: np.ndarray, small_mask: np.ndarray, color, window) -> np.ndarray:
    """The piece's colour on the camera's full image, inside its mask and a little around it. A mask
    found on the reduced ``window`` (x0, y0, width, height, factor) fills holes in; this keeps them."""
    x0, y0, w, h, f = window
    rows, cols = np.nonzero(small_mask)
    r0, r1 = max(y0 + rows.min() * f - 8, 0), y0 + (rows.max() + 1) * f + 8
    c0, c1 = max(x0 + cols.min() * f - 8, 0), x0 + (cols.max() + 1) * f + 8
    whole = np.zeros(rgb.shape[:2], bool)
    whole[y0:y0 + h, x0:x0 + w] = small_mask.repeat(f, axis=0).repeat(f, axis=1)
    box = whole[r0:r1, c0:c1]
    grown = box.copy()
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if dr * dr + dc * dc <= 9:
                grown |= np.roll(box, (dr, dc), axis=(0, 1))
    far = np.linalg.norm(_lab(rgb[r0:r1, c0:c1, :3]) - _lab(color), axis=-1)
    out = np.zeros(rgb.shape[:2], bool)
    out[r0:r1, c0:c1] = grown & (far < _split_level(far[grown]))
    return out


@dataclass
class Reading:
    symbol: str
    top_deg: float | None  # world heading of the symbol's top; None: the outlines cannot tell
    every_deg: float       # the piece looks the same turned by this much
    overlap: float


@dataclass
class _Fit:
    overlap: float
    top_deg: float
    half_lead: float       # over the same outline turned half a turn
    every_deg: float

    def has_direction(self) -> bool:
        return self.every_deg != 360.0 or self.half_lead >= MIN_HALF_LEAD


def _fits(color, grid: np.ndarray, templates: list[Template]) -> dict[str, _Fit]:
    """The best fit of each symbol of this colour, at its best turn."""
    hue = hue_of(color)
    target = _centred(grid)
    fits: dict[str, _Fit] = {}
    for t in templates:
        if _hue_diff(t.hue, hue) > MAX_HUE_DIFF_DEG:
            continue
        coarse = max(np.arange(0.0, 360.0, TURN_STEP_DEG), key=lambda a: _overlap(_turned(t.grid, float(a)), target))
        fine = np.arange(coarse - TURN_STEP_DEG, coarse + TURN_STEP_DEG + 0.5, 1.0)
        turn = float(max(fine, key=lambda a: _fit(_turned(t.grid, float(a)), target)))
        overlap = _fit(_turned(t.grid, turn), target)
        if t.symbol not in fits or overlap > fits[t.symbol].overlap:
            half = overlap - _fit(_turned(t.grid, turn + 180.0), target)
            fits[t.symbol] = _Fit(overlap, t.top_deg + turn, half, t.every_deg)
    return fits


def _sure(fits: dict[str, _Fit], min_overlap: float, min_lead: float) -> str | None:
    ranked = sorted(fits, key=lambda name: -fits[name].overlap)
    if not ranked or fits[ranked[0]].overlap < min_overlap:
        return None
    if len(ranked) > 1 and fits[ranked[0]].overlap - fits[ranked[1]].overlap < min_lead:
        return None
    return ranked[0]


def read_symbol(color, grid: np.ndarray, templates: list[Template], colour_grid: np.ndarray | None = None) -> Reading | None:
    """The symbol this piece is, if sure: by its mask's outline, else by its colour outline if given."""
    if int(grid.sum()) < MIN_CELLS:
        return None
    mask_fits = _fits(color, grid, [t for t in templates if t.outline == "mask"])
    colour_fits: dict[str, _Fit] = {}
    if colour_grid is not None and int(colour_grid.sum()) >= MIN_CELLS:
        colour_fits = _fits(color, colour_grid, [t for t in templates if t.outline == "colour"])
    name = _sure(mask_fits, MIN_OVERLAP, MIN_LEAD)
    if name is None or not mask_fits[name].has_direction():
        name = _sure(colour_fits, COLOUR_MIN_OVERLAP, COLOUR_MIN_LEAD)
    if name is None:
        return None
    for fits in (mask_fits, colour_fits):
        if name in fits and fits[name].has_direction():
            fit = fits[name]
            return Reading(name, fit.top_deg, fit.every_deg, fit.overlap)
    fit = mask_fits.get(name) or colour_fits[name]
    return Reading(name, None, fit.every_deg, fit.overlap)


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
