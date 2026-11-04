"""A picture of every number and symbol piece, upright and named, for the model to compare with.

  cd example/piper-dual-tabletop
  ~/.PhyAgentOS/skills/.venvs/piper-dual-tabletop-skill-server/bin/python dev/build_piece_chart.py

Cuts each piece out of the labelled 1080p layouts in dev/piece_layouts, turned so that its top is up. Needs
the GPU for SAM3 for a moment, not the camera. Writes src/tabletop/piece_chart.jpg and piece_images/, which ship in the skill.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "dev"))
import build_colour_templates as bc  # noqa: E402

lp, sym, geo = bc.lp, bc.sym, bc.sym.geo
OUT = PKG / "nodes/skill_server/src/tabletop/piece_chart.jpg"
FOLDER = PKG / "nodes/skill_server/src/tabletop/piece_images"     # one picture per piece
FILE_NAMES = {"+": "plus", "-": "minus", "×": "times", "÷": "divide", "=": "equals"}
ON_ITS_BACK = {"1"}                                               # lay face down in the layouts: mirror it back
ONE_SIZE = 224
ROWS = [("orange", ["×", "4", "11"]), ("yellow", ["1", "5", "15"]), ("green", ["+", "2", "18"]),
        ("teal", ["10", "12", "20"]), ("blue", ["-", "3", "7", "19"]), ("purple", ["=", "9", "17"]),
        ("pink/red", ["÷", "6", "8", "13", "14", "16"])]
# The layout with the cleanest background, chosen by eye; others come from the first layout that has them.
# 4 has none without a neighbour in the corner: see dev/piece_chart_reshoot.md.
CLEANEST = {"-": "layout3", "6": "layout3", "10": "layout2", "16": "layout2", "18": "layout2", "17": "layout2",
            "=": "layout2", "+": "layout3", "2": "layout2", "7": "layout2", "÷": "layout2", "20": "layout3"}
CUT, TILE, LABEL, SIDE = 76, 112, 22, 84  # half a cut in image pixels, tile, label height, row name width


def upright(rgb: np.ndarray, K: np.ndarray, xy, top_deg: float) -> Image.Image:
    """The piece at xy, cut from the image and turned so that its top points up."""
    th = math.radians(top_deg)
    centre = np.array([xy[0], xy[1], lp.Z])
    uv, _ = geo.project(K, lp.T, np.vstack([centre, centre + 0.03 * np.array([math.cos(th), math.sin(th), 0.0])]))
    du, dv = uv[1] - uv[0]
    u, v = int(round(uv[0][0])), int(round(uv[0][1]))
    cut = Image.fromarray(rgb[v - CUT:v + CUT, u - CUT:u + CUT])
    # the top points at (du, dv) in the image; turn that to straight up
    turned = cut.rotate(math.degrees(math.atan2(du, -dv)), resample=Image.BICUBIC)
    half = 28
    return turned.crop((CUT - half, CUT - half, CUT + half, CUT + half)).resize((ONE_SIZE, ONE_SIZE), Image.BICUBIC)


def times_from_a_tile() -> Image.Image:
    """No layout has the ×. A tile of it from a real scene: 96 image pixels shown three times the size."""
    tile = Image.open(bc.LAYOUTS / "times_tile_hard1.jpg").convert("RGB").resize((96, 96), Image.BOX)
    th = math.radians(33.5)                                          # as locate read it there
    centre = np.array([0.342, 0.112, lp.Z])
    uv, _ = geo.project(lp.cam_K_full, lp.T, np.vstack([centre, centre + 0.03 * np.array([math.cos(th), math.sin(th), 0.0])]))
    du, dv = uv[1] - uv[0]
    turned = tile.rotate(math.degrees(math.atan2(du, -dv)), resample=Image.BICUBIC)
    return turned.crop((20, 20, 76, 76)).resize((ONE_SIZE, ONE_SIZE), Image.BICUBIC)


def main() -> None:
    layouts = yaml.safe_load((bc.LAYOUTS / "layouts.yaml").read_text(encoding="utf-8"))
    masks = {t.symbol: t for t in sym.load_templates() if t.outline == "mask"}
    tiles: dict[str, Image.Image] = {}
    for name, entries in layouts.items():
        z = np.load(bc.LAYOUTS / f"{name}.npz")
        rgb = z["frames"].mean(axis=0).round().astype(np.uint8)
        for p in bc.views_of(name, entries):
            symbol = str(p["symbol"])
            if symbol in tiles or CLEANEST.get(symbol, name) != name:
                continue
            first = p["views"][0]
            top = sym._fits(first["color"], first["mask"], [masks[symbol]])[symbol].top_deg
            top += 180.0 if p.get("upside_down") else 0.0
            tiles[symbol] = upright(rgb, z["K"], p["xy"], top)
    lp.cam_K_full = np.load(bc.LAYOUTS / "layout1.npz")["K"]
    tiles["×"] = times_from_a_tile()
    wanted = {s for _, row in ROWS for s in row}
    print("missing", sorted(wanted - set(tiles)), "extra", sorted(set(tiles) - wanted))
    for symbol in ON_ITS_BACK:
        tiles[symbol] = ImageOps.mirror(tiles[symbol])
    FOLDER.mkdir(exist_ok=True)
    for symbol, tile in tiles.items():
        tile.save(FOLDER / f"{FILE_NAMES.get(symbol, symbol)}.jpg", quality=92)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    sheet = Image.new("RGB", (SIDE + 6 * (TILE + 8), len(ROWS) * (TILE + LABEL + 6)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for r, (colour, row) in enumerate(ROWS):
        y = r * (TILE + LABEL + 6)
        draw.text((6, y + TILE // 2), colour, fill=(0, 0, 0), font=font)
        for c, symbol in enumerate(row):
            x = SIDE + c * (TILE + 8)
            if symbol in tiles:
                sheet.paste(tiles[symbol].resize((TILE, TILE), Image.BICUBIC), (x, y))
            text = "−" if symbol == "-" else symbol
            draw.text((x + TILE // 2 - 8 * len(text), y + TILE), text, fill=(0, 0, 0), font=font)
    OUT.write_bytes(b"")
    sheet.save(OUT, quality=92)
    print("wrote", OUT, sheet.size, OUT.stat().st_size)


if __name__ == "__main__":
    main()
