"""Primitives exposed to model-written code.

Perception is four functions: ``locate`` (measure), ``show`` (the model sees an image when the
block ends), ``look`` (a one-off vision question inside a block), ``capture`` (raw arrays). The arm primitives
are in ``actions``.

Rules:
* Only what a program cannot compose from other primitives is a primitive.
* Perception never moves the arm. What a call does is what it says: no retries, no other camera.
* Domain outcomes are return values; hardware faults raise ``RobotFault``.
* No timeouts and no time-based stopping anywhere. ``time_left()`` is read-only information.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import geometry as geo
from . import vision as vis
from .actions import ArmActions
from .backend import ArmBackend, Detection, Frame, Perception
from .symbols import MAX_PIECE_M, distinct, load_templates, read_symbol, table_grid
from .types import Cancelled, Capture, RobotFault, Sight

THIN_M = 0.02  # an object lower than this is measured by its mask, with the height from depth


@dataclass
class RobotConfig:
    table_z: float = 0.0
    head_camera: str = "head"
    wrist_cameras: dict[str, str] = field(default_factory=lambda: {"left": "wrist_left"})
    # lowest allowed TCP z (world) with the gripper straight down: where the closed fingertips touch
    # the table, plus a margin. Grasp heights are raised to it and lower targets are refused. None = no check.
    tcp_floor_z_m: float | None = None
    fingertip_m: float = 0.0125         # fingertips below the TCP
    finger_half_depth_m: float = 0.0    # half a finger's width across the closing line: a lean lowers its edge
    holding_min_width_m: float = 0.004
    grasp_depth_max_m: float = 0.03
    open_margin_m: float = 0.03
    deadline_env: str = "PAOS_SCENE_DEADLINE"
    deadline_file: str | None = None
    # TCP targets outside this box are refused (reason "outside_workspace"): the booth's side
    # panels and the camera pole. {x: [min, max], y: [min, max], z_max: m}; None = no check.
    # MEASURE ON SITE.
    workspace: dict[str, Any] | None = None
    # pick / place / drag
    approach_m: float = 0.08      # come in along the fingers from this far back
    approach_tilts_deg: list[float] = field(default_factory=lambda: [0.0, 15.0, 25.0, 40.0, 55.0, 70.0])  # smallest that reaches
    place_gap_m: float = 0.01     # place on the table: object bottom this far above it when released
    drop_gap_m: float = 0.03      # place in a container: object bottom this far above its rim
    release_open_m: float = 0.015  # let go by opening this much, so the fingers do not push neighbours
    # how the judges read an arrangement (equation task): heading (deg, world) the top of a
    # printed symbol must point to, and the heading of the reading direction (left -> right).
    # Default: judges stand in front of the table, facing the arm. CONFIRM ON SITE.
    read_up_deg: float = 180.0
    read_right_deg: float = 90.0


class _Ledger:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []

    def add(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._entries.append(entry)

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            out, self._entries = self._entries, []
        return out


def _summary(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_summary(v) for v in value]
    return value


class Robot(ArmActions):
    """Holds backends, per-scene state and the primitive implementations."""

    def __init__(self, arms: ArmBackend, perception: Perception | None, config: RobotConfig | None = None,
                 vision: vis.VisionClient | None = None) -> None:
        self.arms_backend = arms
        self.perception = perception
        self.cfg = config or RobotConfig()
        self.vision = vision
        self.look_dir: Path | None = None  # where look() saves the images it sent (debugging)
        self.show_dir = Path(tempfile.gettempdir()) / "piper_tabletop_show"  # image files of show()
        self.symbol_templates = load_templates()  # outlines of the equation task's pieces
        self._shown: list[str] = []
        self._shown_count = 0
        self.ledger = _Ledger()
        self.cancel_event = threading.Event()
        self._depth = threading.local()

    # ------------------------------------------------------------------ plumbing

    def namespace(self) -> dict[str, Any]:
        """Names visible to model code (in addition to numpy/math added by the executor)."""
        floor = self.cfg.tcp_floor_z_m
        return {
            "locate": self.locate,
            "show": self.show,
            "look": self.look,
            "capture": self.capture,
            "pick": self.pick,
            "place": self.place,
            "drag": self.drag,
            "move_to": self.move_to,
            "set_gripper": self.set_gripper,
            "home": self.home,
            "state": self.state,
            "time_left": self.time_left,
            "RobotFault": RobotFault,
            "TABLE_Z": self.cfg.table_z,
            "TCP_FLOOR_Z": self.cfg.table_z if floor is None else floor,
            "MAX_OPENING_M": self.arms_backend.max_opening_m,
            "READ_UP_DEG": self.cfg.read_up_deg,
            "READ_RIGHT_DEG": self.cfg.read_right_deg,
        }

    def _call(self, name: str, args: dict[str, Any], fn: Callable[[], Any]) -> Any:
        if self.cancel_event.is_set():
            raise Cancelled("robot.run cancelled")
        depth = getattr(self._depth, "n", 0)
        self._depth.n = depth + 1
        t0 = time.time()
        try:
            result = fn()
            self.ledger.add({"call": name, "args": args, "t_start": t0, "dt_s": round(time.time() - t0, 3),
                             "nested": depth > 0, "result": _summary(result)})
            return result
        except Exception as exc:
            self.ledger.add({"call": name, "args": args, "t_start": t0, "dt_s": round(time.time() - t0, 3),
                             "nested": depth > 0, "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            self._depth.n = depth

    # ------------------------------------------------------------------ time

    def time_left(self) -> float | None:
        """Seconds left before the scene's time limit; None when no deadline is set. Read-only."""
        return self._call("time_left", {}, self._time_left)

    def _time_left(self) -> float | None:
        raw = os.environ.get(self.cfg.deadline_env)
        if not raw and self.cfg.deadline_file:
            p = Path(self.cfg.deadline_file)
            raw = p.read_text().strip() if p.exists() else None
        if not raw:
            return None
        return round(float(raw) - time.time(), 1)

    # ------------------------------------------------------------------ perception

    def locate(self, text: str, camera: str = "head") -> list[Sight]:
        """Every instance of ``text`` (one English noun phrase) in one image of ``camera``, best
        score first, each measured from the depth under its mask; [] when nothing matches."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be one English noun phrase")
        return self._call("locate", {"text": text, "camera": camera}, lambda: self._locate(text, camera))

    def _locate(self, text: str, camera: str) -> list[Sight]:
        frame = self._frame(camera)
        detections = distinct(self.perception.segment(frame, text))
        sights = [s for s in (self._measure(text, frame, d) for d in detections) if s is not None]
        return sights

    def _measure(self, text: str, frame: Frame, detection: Detection) -> Sight | None:
        table_z = self.cfg.table_z
        notes = []
        if geo.touches_border(detection.mask):
            notes.append("cut off at the image edge")
        points = np.zeros((0, 3))
        if frame.depth is not None:
            points = geo.backproject_depth(frame.K, frame.T_wc, frame.depth, detection.mask)
        fp = geo.footprint_from_points(points, table_z)
        if fp is None:  # no depth under the mask (black or glossy surface), or nothing above the table
            points = geo.mask_to_plane(frame.K, frame.T_wc, detection.mask, table_z)
            fp = geo.footprint_from_plane_points(points, table_z)
            notes.append("no usable depth: position on the table plane, height unknown")
        if fp is None:
            return None
        if 0 < fp.size[2] < THIN_M:  # few of its depth points clear the table: the mask has the outline
            outline = geo.footprint_from_plane_points(
                geo.mask_to_plane(frame.K, frame.T_wc, detection.mask, float(fp.top_z)), float(fp.top_z))
            if outline is not None:
                # the side facing the camera is in the mask too and lands beyond the near edge
                cam = frame.T_wc[:3, 3]
                center = np.asarray(outline.center_xyz[:2], dtype=float)
                center += 0.5 * fp.size[2] * (center - cam[:2]) / (cam[2] - fp.top_z)
                fp.center_xyz[:2] = [float(v) for v in center]
                fp.size[:2] = outline.size[:2]
                fp.long_axis_deg = outline.long_axis_deg
        grasp = None
        if fp.size[2] > 0:
            grasp = geo.grasp_geometry(fp.center_xyz, fp.top_z, fp.size, fp.long_axis_deg,
                                       self.arms_backend.max_opening_m, self.cfg.grasp_depth_max_m,
                                       self.cfg.open_margin_m, tcp_floor_z=self.cfg.tcp_floor_z_m)
            if grasp.too_wide:
                notes.append("too_wide for the gripper")
        u, v = geo.mask_centroid(detection.mask)
        color = [int(c) for c in np.median(frame.rgb[detection.mask][:, :3], axis=0)]
        reading = None
        if self.symbol_templates and max(fp.size[:2]) <= MAX_PIECE_M:
            grid = table_grid(frame.K, frame.T_wc, detection.mask, float(fp.top_z))
            reading = read_symbol(color, grid, self.symbol_templates)
        return Sight(text=text, score=float(detection.score), camera=frame.camera,
                     pixel=[int(p) for p in frame.full_pixel(int(v), int(u))],
                     center_xyz=[float(x) for x in fp.center_xyz], top_z=float(fp.top_z),
                     size=[float(x) for x in fp.size], long_axis_deg=float(fp.long_axis_deg),
                     grasp_yaw_deg=grasp.grasp_yaw_deg if grasp else geo.norm90(fp.long_axis_deg + 90.0),
                     color=color,
                     grasp_xyz=grasp.grasp_xyz if grasp else None, open_width_m=grasp.open_width_m if grasp else None,
                     note="; ".join(notes) or None, points=points,
                     symbol=reading.symbol if reading else None, top_deg=reading.top_deg if reading else None,
                     top_every_deg=reading.every_deg if reading else None)

    def capture(self, camera: str = "head") -> Capture:
        """The current frame of ``camera`` as arrays for your own calculations. Nothing goes to
        any model: only what the code prints comes back."""
        return self._call("capture", {"camera": camera}, lambda: self._capture(camera))

    def _capture(self, camera: str) -> Capture:
        frame = self._frame(camera).own
        if frame.rgb is None:
            raise RuntimeError(f"camera {frame.camera} has no colour image")
        world = (geo.world_map(frame.K, frame.T_wc, frame.depth) if frame.depth is not None
                 else np.full((*frame.rgb.shape[:2], 3), np.nan))
        return Capture(camera=frame.camera, rgb=frame.rgb, world=world)

    def show(self, camera: str = "head", marks=None, image=None) -> None:
        """Attach an image to this block's result: you see it when the block ends. The picture is
        taken now, so several calls in one block show the moments they were made, in call order.
        ``marks``: Sights / [x, y, z] drawn on a camera image as numbered rings 1..N.
        ``image``: a file path or an array to show instead of a camera image."""
        marks = list(marks) if marks is not None else []
        args = {"camera": camera, "marks": len(marks), "image": image if isinstance(image, str) else image is not None}
        self._call("show", args, lambda: self._show(camera, marks, image, args))

    def _show(self, camera: str, marks: list[Any], image: Any, args: dict[str, Any]) -> None:
        if image is not None:
            if marks:
                raise ValueError("marks are drawn on a camera image: do not pass image= with them")
            rgb, tag = self._image_rgb(image), "image"
        else:
            frame = self._frame(camera)
            rgb, _ = self._marked_rgb(frame, marks)
            tag = frame.camera
        self.show_dir.mkdir(parents=True, exist_ok=True)
        self._shown_count += 1
        path = self.show_dir / f"{time.strftime('%H%M%S')}_{self._shown_count:04d}_{tag}.jpg"
        path.write_bytes(vis.encode_jpeg(rgb))
        self._shown.append(str(path))
        args["path"] = str(path)

    def take_shown(self) -> list[str]:
        """Image files of the ``show`` calls since the last time (the executor's block result)."""
        shown, self._shown = self._shown, []
        return shown

    def look(self, question: str, camera: str | None = "head", marks=None, image=None, json: bool = False):
        """Ask the vision model one question from inside a block (it has no other context) and get
        its answer as text, or as parsed JSON with ``json=True``. Sends ``image`` (a file path or
        an array) if given, then the current image of ``camera`` (None: no camera image).
        ``marks``: Sights / [x, y, z] drawn on the camera image as numbered rings 1..N."""
        marks = list(marks) if marks is not None else []
        args = {"question": question, "camera": camera, "marks": len(marks),
                "image": image if isinstance(image, str) else image is not None, "json": json}
        return self._call("look", args, lambda: self._look(question, camera, marks, image, bool(json)))

    def _look(self, question: str, camera: str | None, marks: list[Any], image: Any, as_json: bool):
        if self.vision is None:
            raise RuntimeError("look() is not configured in this profile (vision: in skill_server.yaml)")
        if camera is None and image is None:
            raise ValueError("look() needs a camera or an image")
        if camera is None and marks:
            raise ValueError("marks are drawn on a camera image: give a camera")
        images = [] if image is None else [vis.encode_jpeg(self._image_rgb(image))]
        name, missing = None, []
        if camera is not None:
            frame = self._frame(camera)
            rgb, missing = self._marked_rgb(frame, marks)
            images.append(vis.encode_jpeg(rgb))
            name = "head" if frame.camera == self.cfg.head_camera else frame.camera
        q = vis.build_question(question, len(marks), image is not None, name)
        if missing:
            q += f"\n(Marks {missing} are outside this image; skip them.)"
        self._save_look(images, name or "image")
        text = self.vision.ask(q, images)
        return vis.parse_json(text) if as_json else text

    def _frame(self, camera: str) -> Frame:
        if self.perception is None:
            raise RuntimeError("no cameras configured")
        return self.perception.capture(self._camera_name(camera))

    @staticmethod
    def _marked_rgb(frame: Frame, marks: list[Any]) -> tuple[np.ndarray, list[int]]:
        """The camera's whole image with ``marks`` drawn, and the numbers outside the view."""
        frame = frame.own
        if frame.rgb is None:
            raise RuntimeError(f"camera {frame.camera} has no colour image")
        if not marks:
            return frame.rgb, []
        rgb, drawn = vis.draw_marks(frame, marks)
        return rgb, [i for i in range(1, len(marks) + 1) if i not in drawn]

    @staticmethod
    def _image_rgb(image: Any) -> np.ndarray:
        return vis.read_image(image) if isinstance(image, str) else np.asarray(image)

    def _camera_name(self, camera: str | None) -> str:
        """None/'head' -> the fixed camera; 'wrist' -> the (first) arm's wrist camera."""
        assert self.perception is not None
        cams = self.perception.cameras()
        if camera in (None, "head"):
            name = self.cfg.head_camera
        elif camera == "wrist":
            name = next(iter(self.cfg.wrist_cameras.values()), None)
        else:
            name = camera
        if name not in cams:
            raise ValueError(f"camera {camera!r} is not available; cameras: {cams} (or 'head' / 'wrist')")
        return name

    def _save_look(self, images: list[bytes], tag: str) -> None:
        if self.look_dir is None:
            return
        try:
            self.look_dir.mkdir(parents=True, exist_ok=True)
            stem = f"{time.strftime('%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{tag}"
            for k, jpg in enumerate(images):
                (self.look_dir / f"{stem}_{k}.jpg").write_bytes(jpg)
        except OSError:
            pass

