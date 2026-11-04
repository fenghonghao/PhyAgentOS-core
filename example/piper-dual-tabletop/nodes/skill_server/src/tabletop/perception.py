"""Real-camera Perception for ``locate``: frame store, camera poses, pluggable segmenters.

Camera nodes (Orbbec/RealSense adapters) publish ``image/color`` (rgb8) and ``image/depth``
(32FC1, metres, 0 = invalid, aligned to color when the adapter's align option is on).
Intrinsics are not published by the adapters, so they come from this node's config.

Camera pose: a fixed camera has ``T_world_cam`` from calibration; a wrist camera is
``T_world_tcp(joints at capture) @ T_tcp_cam`` with ``T_tcp_cam`` from hand-eye calibration.

``capture`` waits for the next frame of a live stream so a wrist view is taken after the arm
stopped. A stream that has gone quiet is not waited for: the latest frame is returned with
``meta['stale'] = True`` (no time-based stop of the task, just no blocking on a dead camera).
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from . import kinematics as kin
from .backend import Detection, Frame, Perception


class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, text: str) -> list[Detection]:
        """All instances matching ``text``, best first."""


# PIPER URDF: tcp = link6 + 0.13503 m along link6 z (tcp_joint); hand-eye results are often
# given relative to the flange (link6)
LINK6_TO_TCP = kin.transform([0.0, 0.0, 0.13503], np.eye(3))


@dataclass
class CameraConfig:
    """Camera intrinsics and mounting. Pose = camera (optical frame) in the parent frame:
    world for ``mount: fixed``; the arm's ``wrist_frame`` (tcp or link6) for ``mount: wrist``.
    Give ``xyz``/``rpy_deg`` or a 4x4 ``T_matrix`` (``T_inverse: true`` if it maps the other way).
    """

    name: str
    K: list[list[float]]
    mount: str = "fixed"  # fixed | wrist
    arm: str | None = None  # wrist cameras: which arm
    xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    T_matrix: list[list[float]] | None = None
    T_inverse: bool = False
    wrist_frame: str = "tcp"  # tcp | link6 (flange)
    stale_after_s: float = 0.5
    depth_scale: float = 1.0  # calibration: true distance / reported depth
    find_window: list[int] | None = None  # find objects on this part of the image: x0, y0, width, height, factor

    def T(self) -> np.ndarray:
        if self.T_matrix is not None:
            M = np.asarray(self.T_matrix, dtype=float)
            if M.shape != (4, 4):
                raise ValueError(f"camera {self.name}: T_matrix must be 4x4")
            M = np.linalg.inv(M) if self.T_inverse else M
        else:
            M = kin.pose_matrix(self.xyz, [math.radians(a) for a in self.rpy_deg])
        if self.mount == "wrist" and self.wrist_frame == "link6":
            M = np.linalg.inv(LINK6_TO_TCP) @ M  # camera in the tcp frame
        elif self.mount == "wrist" and self.wrist_frame != "tcp":
            raise ValueError(f"camera {self.name}: wrist_frame must be tcp or link6")
        return M


def window_K(K: Any, window: Any) -> np.ndarray:
    """The intrinsics of a window of an image, reduced by a whole factor."""
    x0, y0, _, _, f = window
    out = np.asarray(K, dtype=float).copy()
    out[0, 0] /= f
    out[1, 1] /= f
    out[0, 2] = (out[0, 2] - x0 - (f - 1) / 2) / f
    out[1, 2] = (out[1, 2] - y0 - (f - 1) / 2) / f
    return out


def reduced(frame: Frame, window: Any) -> Frame:
    """A window of the frame's image, each block of factor x factor pixels made one. ``full`` keeps
    the frame itself. Sizes are whole multiples of the factor. Colours are averaged; depths are
    sampled, since a mean across an edge is a depth where nothing is."""
    x0, y0, w, h, f = window
    rgb = frame.rgb[y0:y0 + h, x0:x0 + w, :3].reshape(h // f, f, w // f, f, 3)
    depth = None if frame.depth is None else frame.depth[y0:y0 + h:f, x0:x0 + w:f]
    return Frame(camera=frame.camera, K=window_K(frame.K, window), T_wc=frame.T_wc,
                 rgb=rgb.mean(axis=(1, 3)).round().astype(np.uint8), depth=depth, meta=frame.meta,
                 full=frame, window=tuple(window))


class _Slot:
    def __init__(self) -> None:
        self.rgb: np.ndarray | None = None
        self.depth: np.ndarray | None = None
        self.t_rgb = 0.0
        self.t_depth = 0.0
        self.T_wc: np.ndarray | None = None  # pose when the color frame arrived (wrist cameras)


class DoraPerception(Perception):
    def __init__(self, cams: list[CameraConfig], segmenter: Segmenter, arm_backend: Any | None = None,
                 head: str = "head") -> None:
        self.cfg = {c.name: c for c in cams}
        self.segmenter = segmenter
        self.arms = arm_backend
        self.head = head
        self._slots = {c.name: _Slot() for c in cams}
        self._cond = threading.Condition()

    # ------------------------------------------------------------------ inputs (Dora main thread)

    def on_color(self, cam: str, rgb: np.ndarray) -> None:
        if cam not in self._slots:
            return
        c = self.cfg[cam]
        has_state = getattr(self.arms, "has_state", None)
        if c.mount == "wrist" and has_state is not None and not has_state(c.arm):
            # the wrist pose needs the arm's joint state, which this same (Dora main) thread
            # delivers: waiting for it here would deadlock the node, so drop the frame
            return
        T = self._pose(cam)
        with self._cond:
            s = self._slots[cam]
            s.rgb, s.t_rgb, s.T_wc = rgb, time.monotonic(), T
            self._cond.notify_all()

    def on_depth(self, cam: str, depth: np.ndarray) -> None:
        if cam not in self._slots:
            return
        scale = self.cfg[cam].depth_scale
        if scale != 1.0:
            depth = depth * scale
        with self._cond:
            s = self._slots[cam]
            s.depth, s.t_depth = depth, time.monotonic()

    def _pose(self, cam: str) -> np.ndarray:
        c = self.cfg[cam]
        if c.mount == "wrist":
            if self.arms is None or c.arm is None:
                raise RuntimeError(f"wrist camera {cam} needs an arm backend")
            return self.arms._T_world_tcp(c.arm) @ c.T()
        return c.T()

    # ------------------------------------------------------------------ Perception

    def cameras(self) -> list[str]:
        with self._cond:
            live = [n for n, s in self._slots.items() if s.rgb is not None]
        return sorted(live, key=lambda n: (n != self.head, n))

    def latest_rgb(self, camera: str) -> np.ndarray | None:
        """The newest colour image as it is, without waiting for a new frame (debug snapshots)."""
        with self._cond:
            return self._slots[camera].rgb

    def capture(self, camera: str) -> Frame:
        c = self.cfg[camera]
        t_call = time.monotonic()
        with self._cond:
            s = self._slots[camera]
            if s.rgb is None:
                raise RuntimeError(f"camera {camera} has produced no frame")
            stale = t_call - s.t_rgb > c.stale_after_s
            if not stale:
                self._cond.wait_for(lambda: s.t_rgb > t_call)
            rgb, depth, T, t_rgb = s.rgb, s.depth, s.T_wc, s.t_rgb
            depth_age = t_call - s.t_depth if s.depth is not None else None
        if depth is not None and depth.shape[:2] != rgb.shape[:2]:
            depth = None  # not aligned to color: do not mix pixel grids
        frame = Frame(camera=camera, K=np.asarray(c.K, dtype=float), T_wc=T, rgb=rgb, depth=depth,
                      meta={"stale": stale, "age_s": max(0.0, t_call - t_rgb), "depth_age_s": depth_age, "t": t_rgb})
        return reduced(frame, c.find_window) if c.find_window else frame

    def segment(self, frame: Frame, text: str) -> list[Detection]:
        if frame.rgb is None:
            return []
        if isinstance(self.segmenter, Sam3Segmenter):
            return self.segmenter.segment(frame.rgb, text, key=(frame.camera, frame.meta.get("t")))
        return self.segmenter.segment(frame.rgb, text)


# --------------------------------------------------------------------------- color segmenter

# hue ranges in degrees [0, 360), saturation/value minimums in [0, 1]
_COLORS = {
    "red": [(345, 360), (0, 12)], "红": [(345, 360), (0, 12)],
    "orange": [(12, 40)], "橙": [(12, 40)],
    "yellow": [(40, 70)], "黄": [(40, 70)],
    "green": [(75, 165)], "绿": [(75, 165)],
    "blue": [(190, 255)], "蓝": [(190, 255)],
    "purple": [(260, 320)], "紫": [(260, 320)],
}


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-6
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = (60 * ((g - b)[rm] / d[rm])) % 360
    h[gm] = 60 * ((b - r)[gm] / d[gm]) + 120
    h[bm] = 60 * ((r - g)[bm] / d[bm]) + 240
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    return h, s, mx


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected components (two-pass union-find on run starts; fine at 640x480)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    nxt = 1
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        xs = np.flatnonzero(row)
        starts = xs[np.r_[True, np.diff(xs) > 1]]
        ends = xs[np.r_[np.diff(xs) > 1, True]] + 1
        for x0, x1 in zip(starts, ends):
            above = labels[y - 1, x0:x1] if y > 0 else np.empty(0, dtype=np.int32)
            ids = {find(int(v)) for v in np.unique(above) if v}
            if ids:
                root = min(ids)
                for other in ids:
                    parent[other] = root
            else:
                root = nxt
                parent.append(nxt)
                nxt += 1
            labels[y, x0:x1] = root
    roots = np.array([find(i) for i in range(nxt)], dtype=np.int32)
    labels = roots[labels]
    uniq = np.unique(labels[labels > 0])
    remap = np.zeros(nxt, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[labels], len(uniq)


class ColorSegmenter:
    """Debug/fallback segmenter: picks the color word out of the text (红/red ...)."""

    def __init__(self, min_sat: float = 0.35, min_val: float = 0.25, min_area_px: int = 150) -> None:
        self.min_sat, self.min_val, self.min_area = min_sat, min_val, min_area_px

    def segment(self, rgb: np.ndarray, text: str) -> list[Detection]:
        t = text.lower()
        ranges = next((r for word, r in _COLORS.items() if word in t), None)
        if ranges is None:
            return []
        h, s, v = rgb_to_hsv(rgb)
        m = np.zeros(h.shape, dtype=bool)
        for lo, hi in ranges:
            m |= (h >= lo) & (h < hi)
        m &= (s >= self.min_sat) & (v >= self.min_val)
        labels, n = label_components(m)
        dets = []
        for i in range(1, n + 1):
            comp = labels == i
            area = int(comp.sum())
            if area >= self.min_area:
                dets.append(Detection(mask=comp, score=float(min(0.9, 0.3 + area / 20000.0))))
        dets.sort(key=lambda d: -int(d.mask.sum()))
        return dets


# --------------------------------------------------------------------------- SAM3 segmenter


class Sam3Segmenter:
    """Text-prompted instance segmentation with SAM3 (Hugging Face transformers format).

    Loads in a background thread at node start so the Dora loop and endpoint lease are never
    blocked; the first ``segment`` call waits for the load to finish. Text features are cached
    per prompt, since one ``locate`` asks several cameras for the same text.
    """

    def __init__(self, model_dir: str, device: str = "cuda", dtype: str = "float16",
                 threshold: float = 0.3, mask_threshold: float = 0.5) -> None:
        import os

        self.model_dir = os.path.expanduser(model_dir)
        self.device, self.dtype_name = device, dtype
        self.threshold, self.mask_threshold = threshold, mask_threshold
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._text_cache: dict[str, Any] = {}
        # vision features of recent frames: the backbone is ~0.55 s/frame on an RTX 5060 laptop,
        # the text decoder ~0.05 s, so retries with other phrasings reuse the frame's features
        self._vision_cache: "OrderedDict[Any, Any]" = OrderedDict()
        threading.Thread(target=self._load, name="sam3-load", daemon=True).start()

    def _load(self) -> None:
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor

            dev = self.device if (self.device != "cuda" or torch.cuda.is_available()) else "cpu"
            dtype = getattr(torch, self.dtype_name) if dev != "cpu" else torch.float32
            self._torch, self._dev, self._dt = torch, dev, dtype
            self.processor = Sam3Processor.from_pretrained(self.model_dir)
            self.model = Sam3Model.from_pretrained(self.model_dir, dtype=dtype,
                                                   attn_implementation="sdpa").to(dev).eval()
        except BaseException as exc:  # surfaced on the first segment() call
            self._error = exc
        finally:
            self._ready.set()

    def ready(self) -> bool:
        return self._ready.is_set() and self._error is None

    def _text_embeds(self, text: str):
        if text not in self._text_cache:
            ti = self.processor(text=text, return_tensors="pt").to(self._dev)
            self._text_cache[text] = self.model.get_text_features(**ti)  # forward() takes the output object
        return self._text_cache[text]

    def _vision(self, rgb: np.ndarray, key: Any):
        if key is not None and key in self._vision_cache:
            self._vision_cache.move_to_end(key)
            return self._vision_cache[key]
        from PIL import Image as PILImage

        img = self.processor(images=PILImage.fromarray(rgb), return_tensors="pt")
        vis = self.model.get_vision_features(pixel_values=img.pixel_values.to(self._dev, dtype=self._dt))
        if key is not None:
            self._vision_cache[key] = vis
            while len(self._vision_cache) > 6:
                self._vision_cache.popitem(last=False)
        return vis

    def segment(self, rgb: np.ndarray, text: str, key: Any = None) -> list[Detection]:
        """``key`` identifies the frame (camera, capture time) so its features can be reused."""
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"SAM3 failed to load from {self.model_dir}: {self._error!r}")
        torch = self._torch
        with self._lock, torch.inference_mode():
            out = self.model(vision_embeds=self._vision(rgb, key), text_embeds=self._text_embeds(text))
            res = self.processor.post_process_instance_segmentation(
                out, threshold=self.threshold, mask_threshold=self.mask_threshold,
                target_sizes=[tuple(rgb.shape[:2])])[0]
            masks = res["masks"].cpu().numpy().astype(bool)
            scores = res["scores"].float().cpu().numpy()
        dets = [Detection(mask=m, score=float(s)) for m, s in zip(masks, scores) if m.any()]
        dets.sort(key=lambda d: -d.score)
        return dets
