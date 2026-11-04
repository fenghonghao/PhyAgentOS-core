"""``look()``: ask a vision-language model about a camera image.

Model-written code calls ``look(question, ...)``. This module takes the latest frame of a
camera (optionally with numbered marks drawn next to given objects, and optionally a second,
reference image such as the task's target image), sends it to an OpenAI-compatible
chat-completions endpoint (deepseek-flash by default, the same model PAOS uses) and returns
the answer text, or the parsed JSON when asked.

Credentials come from an environment variable or from the PAOS config file; the key is never
logged, printed or returned. No timeout is set on the request (Track 2 rule: nothing in the
skill stops on time). A failed request raises ``VisionError``: the model reads it and decides.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import geometry as geo
from .backend import Frame

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_DIRS = ("~", "/tmp", "/media", "/mnt")  # where model code may read an image file from
MAX_SIDE_PX = 1024                            # longest image side sent to a model
JPEG_QUALITY = 85
MARK_COLORS = [(230, 25, 75), (0, 130, 200), (255, 140, 0), (145, 30, 180), (0, 160, 80),
               (240, 50, 230), (128, 0, 0), (0, 0, 128)]


class VisionError(RuntimeError):
    """The vision model could not be reached or gave no usable answer."""


@dataclass
class VisionConfig:
    base_url: str | None = None          # OpenAI-compatible endpoint; default: PAOS provider apiBase
    model: str | None = None             # default: PAOS provider defaultModel
    api_key_env: str = "DEEPSEEK_API_KEY"
    paos_config: str = "~/.PhyAgentOS/config.json"
    paos_provider: str = "deepseek"
    temperature: float = 0.0
    # DeepSeek "thinking": one task photo took 1.0-1.4 s with it off, 4.7-5.7 s at the default
    # effort, 9-14 s at max (measured 11-04). None = do not send the field (other providers).
    thinking: str | None = "disabled"


def _resolve(cfg: VisionConfig) -> tuple[str, str, str]:
    """(base_url, model, api_key). Never logs the key."""
    prov: dict[str, Any] = {}
    p = Path(cfg.paos_config).expanduser()
    if p.exists():
        try:
            prov = (json.loads(p.read_text()).get("providers") or {}).get(cfg.paos_provider) or {}
        except (OSError, ValueError):
            prov = {}
    key = os.environ.get(cfg.api_key_env) or prov.get("apiKey") or ""
    base = cfg.base_url or prov.get("apiBase") or "https://api.deepseek.com"
    model = cfg.model or prov.get("defaultModel") or "deepseek-flash"
    if not key:
        raise VisionError(f"no API key: set {cfg.api_key_env} or providers.{cfg.paos_provider}.apiKey in {cfg.paos_config}")
    return base.rstrip("/"), model, key


def read_image(path: str) -> np.ndarray:
    """Load an image file model code named (e.g. the task's target image). Only image files
    under ``IMAGE_DIRS`` are read."""
    from PIL import Image

    p = Path(str(path)).expanduser().resolve()
    roots = [Path(d).expanduser().resolve() for d in IMAGE_DIRS]
    if p.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"image must be one of {sorted(IMAGE_SUFFIXES)}: {path}")
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError(f"image must be under {IMAGE_DIRS}: {path}")
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    with Image.open(p) as im:
        return np.asarray(im.convert("RGB"))


def encode_jpeg(rgb: np.ndarray, max_side: int = MAX_SIDE_PX, quality: int = JPEG_QUALITY) -> bytes:
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8))
    w, h = img.size
    s = max(w, h)
    if s > max_side:
        img = img.resize((round(w * max_side / s), round(h * max_side / s)), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(quality))
    return buf.getvalue()


def _mark_point(m: Any) -> tuple[np.ndarray, float]:
    """World point and radius (m) of a mark: a Sight-like object or [x, y, z]."""
    c = getattr(m, "center_xyz", None)
    if c is None and not hasattr(m, "text"):
        c = m
    if c is None:
        raise ValueError("a mark needs center_xyz (a found Sight) or an [x, y, z] point")
    p = np.asarray(c, dtype=float).reshape(-1)
    if p.size == 2:
        p = np.array([p[0], p[1], 0.0])
    top = getattr(m, "top_z", None)
    if top is not None:
        p = np.array([p[0], p[1], float(top)])
    size = getattr(m, "size", None)
    r = 0.5 * float(max(size[0], size[1])) if size else 0.02
    return p[:3], max(r, 0.01)


def draw_marks(frame: Frame, marks: list[Any]) -> tuple[np.ndarray, list[int]]:
    """Copy of the frame's RGB with a ring around each mark and its number (1-based) beside it.

    Returns the image and the numbers actually drawn (marks outside the view are skipped).
    """
    from PIL import Image, ImageDraw, ImageFont

    assert frame.rgb is not None
    img = Image.fromarray(np.ascontiguousarray(frame.rgb[..., :3]).astype(np.uint8))
    d = ImageDraw.Draw(img)
    h, w = frame.rgb.shape[:2]
    fsize = max(14, int(round(h / 26)))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", fsize)
    except OSError:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fsize)
        except OSError:
            font = ImageFont.load_default()
    drawn = []
    for i, m in enumerate(marks, 1):
        p, r = _mark_point(m)
        edge = p + np.array([r, 0.0, 0.0])
        uv, z = geo.project(frame.K, frame.T_wc, np.vstack([p, edge]))
        if z[0] <= 0 or not (0 <= uv[0, 0] < w and 0 <= uv[0, 1] < h):
            continue
        u, v = float(uv[0, 0]), float(uv[0, 1])
        rp = max(8.0, float(np.linalg.norm(uv[1] - uv[0])) * 1.3 + 4.0)
        col = MARK_COLORS[(i - 1) % len(MARK_COLORS)]
        d.ellipse([u - rp, v - rp, u + rp, v + rp], outline=col, width=max(2, fsize // 7))
        label = str(i)
        tw = d.textlength(label, font=font)
        # label at the ring's upper-left, outside it, so it never covers the object's print
        lx = min(max(0.0, u - rp - tw - 2), w - tw - 4)
        ly = min(max(0.0, v - rp - fsize), h - fsize - 4)
        d.rectangle([lx - 2, ly - 1, lx + tw + 2, ly + fsize + 2], fill=col)
        d.text((lx, ly), label, fill=(255, 255, 255), font=font)
        drawn.append(i)
    return np.asarray(img), drawn


def parse_json(text: str) -> Any:
    """First JSON object/array in ``text`` (code fences allowed)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = t.find(open_c), t.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except ValueError:
                continue
    raise ValueError(f"the answer is not JSON: {text[:300]!r}")


class VisionClient:
    """Chat-completions client with images (OpenAI format)."""

    def __init__(self, cfg: VisionConfig, post: Callable[[str, dict, dict], dict] | None = None) -> None:
        self.cfg = cfg
        self._post = post or self._urllib_post
        self._resolved: tuple[str, str, str] | None = None

    @staticmethod
    def _urllib_post(url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req) as resp:  # no timeout: see module docstring
            return json.loads(resp.read().decode())

    def ask(self, question: str, images: list[bytes]) -> str:
        if self._resolved is None:
            self._resolved = _resolve(self.cfg)
        base, model, key = self._resolved
        content: list[dict[str, Any]] = [{"type": "text", "text": question}]
        for jpg in images:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()}})
        body: dict[str, Any] = {"model": model, "temperature": self.cfg.temperature,
                                "messages": [{"role": "user", "content": content}]}
        if self.cfg.thinking:
            body["thinking"] = {"type": self.cfg.thinking}
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
        try:
            out = self._post(base + "/chat/completions", body, headers)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise VisionError(f"vision model request failed ({type(exc).__name__}: {exc})") from exc
        text = ((out.get("choices") or [{}])[0].get("message") or {}).get("content")
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        if not text or not str(text).strip():
            raise VisionError(f"empty answer from {model}")
        return str(text).strip()


def build_question(question: str, n_marks: int, has_reference: bool, camera: str | None) -> str:
    """The question, preceded by what the images are. ``camera`` None: only the given image."""
    if camera is None:
        return question.strip()
    parts = []
    if has_reference:
        parts.append("Image 1 is the reference image given with the task.")
        parts.append(f"Image 2 is the current view of the robot's {camera} camera.")
    else:
        parts.append(f"The image is the current view of the robot's {camera} camera.")
    if n_marks:
        parts.append(f"Numbered colored rings (1 to {n_marks}) mark objects in the camera image; "
                     "each number is printed in a small box just outside its ring. Refer to objects by these numbers.")
    parts.append(question.strip())
    return "\n".join(parts)
