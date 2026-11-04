#!/usr/bin/env python3
"""Generate the A4 ChArUco board for the hand-eye and head-camera calibration.

The page is 1-bit at 20 px/mm, so every square and marker bit lands on whole pixels.
Usage: make_board.py [output.pdf]    (needs opencv-python-headless and pillow)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SQUARES = (7, 5)
SQUARE_MM = 38
MARKER_MM = 30
DICTIONARY = "DICT_4X4_50"
PAGE_MM = (297, 210)  # A4 landscape
PX_PER_MM = 20
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def make_board() -> cv2.aruco.CharucoBoard:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY))
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM / 1000, MARKER_MM / 1000, dictionary)


def make_page() -> Image.Image:
    mm = PX_PER_MM
    board_px = (SQUARES[0] * SQUARE_MM * mm, SQUARES[1] * SQUARE_MM * mm)
    board = make_board().generateImage(board_px, marginSize=0, borderBits=1)
    page = np.full((PAGE_MM[1] * mm, PAGE_MM[0] * mm), 255, dtype=np.uint8)
    left = (page.shape[1] - board_px[0]) // 2
    top = (page.shape[0] - board_px[1]) // 2
    page[top:top + board_px[1], left:left + board_px[0]] = board

    image = Image.fromarray(page)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 3 * mm)
    label = (f"ChArUco {SQUARES[0]}x{SQUARES[1]}  square {SQUARE_MM} mm  marker {MARKER_MM} mm  {DICTIONARY}"
             "  |  print at 100% (actual size), no fit-to-page")
    draw.text((left, 4 * mm), label, fill=0, font=font)

    # 100 mm scale bar with 10 mm ticks in the bottom margin: check the printed scale with a ruler
    y = page.shape[0] - 5 * mm
    draw.rectangle((left, y, left + 100 * mm, y + mm // 4), fill=0)
    for i in range(11):
        x = left + i * 10 * mm
        draw.rectangle((x, y - 2 * mm, x + mm // 4, y), fill=0)
    draw.text((left + 103 * mm, y - 3 * mm), "this bar must measure 100 mm", fill=0, font=font)
    return image.convert("1", dither=Image.Dither.NONE)


def main() -> None:
    default = Path(__file__).with_name(f"charuco_a4_{SQUARES[0]}x{SQUARES[1]}_{SQUARE_MM}mm.pdf")
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    make_page().save(out, "PDF", resolution=25.4 * PX_PER_MM)
    print(out)


if __name__ == "__main__":
    main()
