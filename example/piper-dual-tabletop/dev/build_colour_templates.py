"""Templates of the second round: each piece's colour outline, with its holes, from labelled layouts.

  cd example/piper-dual-tabletop
  ~/.PhyAgentOS/skills/.venvs/piper-dual-tabletop-skill-server/bin/python dev/build_colour_templates.py

Reads dev/piece_layouts/<name>.npz (frames and K at 1920x1080, head camera) and layouts.yaml: per layout,
each piece's symbol and world position, read by eye from the 1080p image, and upside_down where the mask
templates turned it the wrong way. Replaces the colour templates in src/tabletop/piece_templates.json and
keeps the mask templates, which give the top. Needs the GPU for SAM3 for a moment, not the camera.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "dev"))
import label_pieces as lp  # noqa: E402
from tabletop.backend import Frame  # noqa: E402
from tabletop.perception import reduced  # noqa: E402

sym = lp.sym
LAYOUTS = PKG / "dev/piece_layouts"
EVERY = {"+": 90.0, "×": 90.0, "-": 180.0, "=": 180.0, "÷": 180.0, "8": 180.0}


def views_of(name: str, entries: list[dict]) -> list[dict]:
    """Each labelled piece of a layout: per frame its colour, mask outline and colour outline."""
    z = np.load(LAYOUTS / f"{name}.npz")
    native, K = z["frames"][:4], z["K"]
    small = [reduced(Frame("head", K, lp.T, rgb), lp.cam.find_window) for rgb in native]
    found = lp.find_all(np.array([f.rgb for f in small]))
    pieces = [dict(entry, views=[]) for entry in entries]
    for k, dets in enumerate(found):
        for det in dets:
            c = lp.small_enough(det.mask)
            if c is None:
                continue
            piece = min(pieces, key=lambda p: np.linalg.norm(np.array(p["xy"]) - c))
            if np.linalg.norm(np.array(piece["xy"]) - c) >= 0.012:
                continue
            color = np.median(small[k].rgb[det.mask], axis=0)
            piece["views"].append(dict(
                color=color,
                mask=sym.table_grid(small[k].K, lp.T, det.mask, lp.Z, center=c),
                colour=sym.table_grid(K, lp.T, sym.colour_outline(native[k], det.mask, color, lp.cam.find_window), lp.Z, center=c)))
    return [p for p in pieces if p["views"]]


def templates_of(pieces: list[dict], mask_templates: list[sym.Template]) -> list[sym.Template]:
    """Colour templates of one layout. The top comes from the piece's mask template, labelled by hand."""
    by_name = {t.symbol: t for t in mask_templates}
    out = []
    for p in pieces:
        name = str(p["symbol"])
        first = p["views"][0]
        top = sym._fits(first["color"], first["mask"], [by_name[name]])[name].top_deg
        top += 180.0 if p.get("upside_down") else 0.0
        votes = sum(sym._centred(v["colour"]).astype(int) for v in p["views"]) >= len(p["views"]) / 2
        color = np.median(np.array([v["color"] for v in p["views"]]), axis=0)
        out.append(sym.make_template(name, color, votes, top, EVERY.get(name, 360.0), outline="colour"))
    return out


def main() -> None:
    layouts = yaml.safe_load((LAYOUTS / "layouts.yaml").read_text(encoding="utf-8"))
    current = sym.load_templates()
    masks = [t for t in current if t.outline == "mask"]
    colours = []
    for name, entries in layouts.items():
        pieces = views_of(name, entries)
        colours += templates_of(pieces, masks)
        print(name, len(pieces), "of", len(entries), "pieces found")
    sym.TEMPLATE_FILE.write_text(json.dumps([t.to_json() for t in masks + colours], ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(len(masks), "mask templates,", len(colours), "colour templates in", sym.TEMPLATE_FILE)


if __name__ == "__main__":
    main()
