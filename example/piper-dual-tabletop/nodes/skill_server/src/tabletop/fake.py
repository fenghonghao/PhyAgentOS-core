"""A kinematic one-arm tabletop world with ray-cast cameras, for tests and offline LLM trials.

Not physics: the arm teleports (optionally with a delay), a closed gripper picks up the box whose
footprint contains the TCP and fits the opening, an opened gripper drops the box onto whatever
is below it. Cameras render depth and an object-id image by ray casting oriented boxes.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .backend import ArmBackend, Detection, Frame, GripperOutcome, MotionOutcome, Perception, Plan, Unreachable


@dataclass
class Box:
    name: str
    center_xy: list[float]
    z_bottom: float
    size: list[float]  # [l (local x), w (local y), h]
    yaw_deg: float = 0.0
    aliases: list[str] = field(default_factory=list)
    color: tuple[int, int, int] | None = None  # rendered RGB (default: derived from the name)
    meta: dict = field(default_factory=dict)    # ground truth for test oracles (kind, symbol, ...)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], self.z_bottom + self.size[2] / 2])

    @property
    def top(self) -> float:
        return self.z_bottom + self.size[2]

    def rot(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def contains_xy(self, p: np.ndarray, margin: float = 0.0) -> bool:
        local = self.rot().T @ (np.asarray(p, dtype=float) - self.center)
        return abs(local[0]) <= self.size[0] / 2 + margin and abs(local[1]) <= self.size[1] / 2 + margin

    def extent_along(self, heading_deg: float) -> float:
        d = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg)), 0.0])
        local = self.rot().T @ d
        return abs(local[0]) * self.size[0] + abs(local[1]) * self.size[1]


@dataclass
class FakeArm:
    name: str
    base: list[float]
    home_xyz: list[float]
    tcp: list[float] = field(default_factory=list)
    yaw: float = 0.0
    tilt: float = 45.0  # home pose leans forward: a top-down pose is not reachable that high
    width: float = 0.07
    held: str | None = None
    held_offset: np.ndarray | None = None  # object centre minus TCP at grasp time

    def __post_init__(self) -> None:
        if not self.tcp:
            self.tcp = list(self.home_xyz)


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, np.array([1.0, 0, 0]))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


class FakeWorld(ArmBackend, Perception):
    """Both backends in one object, sharing the world state."""

    def __init__(self, boxes: list[Box] | None = None, *, reach_m: float = 0.62, max_opening_m: float = 0.07,
                 depth: bool = True, resolution: tuple[int, int] = (320, 240), move_delay_s: float = 0.0,
                 wrist_cam_offset: tuple[float, float, float] = (0.0, 0.0, 0.08), table_z: float = 0.0,
                 topdown_max_z: float | None = 0.14, arms: tuple[str, ...] = ("left",)) -> None:
        self.table_z = table_z
        self.boxes: dict[str, Box] = {b.name: b for b in (boxes or [])}
        self.a = FakeArm("left", [0.0, 0.0, 0.0], [0.15, 0.0, 0.25], width=max_opening_m)  # at the origin, like the booth
        self.arm = {"left": self.a}
        self.reach_m = reach_m
        # PIPER joint5 (+-70 deg) limits a vertical approach to low heights (URDF sampling: ~0.14 m)
        self.topdown_max_z = topdown_max_z
        self.max_opening_m = max_opening_m
        self.depth_enabled = depth
        self.resolution = resolution
        self.move_delay_s = move_delay_s
        self.wrist_cam_offset = np.asarray(wrist_cam_offset, dtype=float)
        self.bad_yaw: list[tuple[float, float]] = []  # headings of the closing line the wrist cannot take
        self.hidden_from: dict[str, set[str]] = {}  # camera -> names it cannot segment
        self.camera_enabled = {"head": True, "wrist_left": True}
        self.head_T = look_at([-0.30, 0.0, 0.80], [0.40, 0.0, 0.0])
        self._lock = threading.RLock()
        self.log: list[str] = []
        self.moves: list[dict] = []  # every executed motion: xyz, yaw, tilt, path

    # ------------------------------------------------------------------ arm backend

    def tcp_pose(self):
        return list(self.a.tcp), self.a.yaw, self.a.tilt

    def joints(self):
        return [0.0] * 6

    def gripper_width(self):
        return self.a.width

    def _unreachable(self, xyz, yaw: float, tilt: float) -> bool:
        if np.linalg.norm(np.asarray(xyz) - np.asarray(self.a.base)) > self.reach_m or xyz[2] < self.table_z + 0.005:
            return True
        if self.topdown_max_z is not None and abs(tilt) < 25 and xyz[2] > self.a.base[2] + self.topdown_max_z:
            return True
        y = ((yaw + 180) % 360) - 180
        return any(lo <= y <= hi for lo, hi in self.bad_yaw)

    def plan(self, xyz, yaw_deg, tilt_deg, path, start=None, floor_z=None):
        if self._unreachable(xyz, yaw_deg, tilt_deg):
            raise Unreachable(f"no joint solution for xyz={np.round(xyz, 3).tolist()} "
                              f"yaw={yaw_deg:.0f} tilt={tilt_deg:.0f}")
        end = [float(v) for v in xyz]
        return Plan(steps=[{"xyz": end, "yaw": float(yaw_deg), "tilt": float(tilt_deg), "path": path}], end_xyz=end)

    def plan_home(self, start=None):
        return Plan(steps=[{"xyz": list(self.a.home_xyz), "yaw": 0.0, "tilt": 45.0, "path": "home"}],
                    end_xyz=list(self.a.home_xyz))

    def execute(self, plan, cancelled=lambda: False):
        for step in plan.steps:
            if self.move_delay_s > 0:
                time.sleep(self.move_delay_s)
            if cancelled():
                return MotionOutcome(False, list(self.a.tcp), "cancelled")
            self._teleport(step)
        return MotionOutcome(True, list(self.a.tcp))

    def _teleport(self, step: dict) -> None:
        with self._lock:
            a = self.a
            dyaw = step["yaw"] - a.yaw
            a.tcp, a.yaw, a.tilt = list(step["xyz"]), step["yaw"], step["tilt"]
            if a.held:
                b = self.boxes[a.held]
                if abs(dyaw) > 1e-9:  # the held object turns with the gripper
                    c, s_ = math.cos(math.radians(dyaw)), math.sin(math.radians(dyaw))
                    a.held_offset = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]) @ a.held_offset
                    b.yaw_deg = (b.yaw_deg + dyaw + 180.0) % 360.0 - 180.0
                c = np.asarray(a.tcp) + a.held_offset
                b.center_xy = [float(c[0]), float(c[1])]
                b.z_bottom = float(c[2] - b.size[2] / 2)
            self.moves.append(dict(step))
            self.log.append(f"move {np.round(a.tcp, 3).tolist()} yaw={a.yaw:.0f} tilt={a.tilt:.0f} {step['path']}")

    def _support_z(self, xy: np.ndarray, exclude: str) -> float:
        z = self.table_z
        p = np.array([xy[0], xy[1], 0.0])
        for b in self.boxes.values():
            if b.name == exclude:
                continue
            q = p.copy()
            q[2] = b.center[2]
            if b.contains_xy(q):
                z = max(z, b.top)
        return z

    def grip(self, width_m):
        with self._lock:
            a = self.a
            if a.held:
                if width_m < a.width:           # squeezing what it already holds
                    return GripperOutcome(a.width)
                a.width = float(width_m)
                b = self.boxes[a.held]
                a.held, a.held_offset = None, None
                b.z_bottom = self._support_z(np.asarray(b.center_xy), b.name)
                self.log.append(f"released {b.name}")
                return GripperOutcome(a.width)
            if width_m < a.width - 1e-6:        # closing
                tcp = np.asarray(a.tcp)
                for b in self.boxes.values():
                    inside_z = b.z_bottom - 0.005 <= tcp[2] <= b.top + 0.002
                    if inside_z and b.contains_xy(tcp, margin=0.005):
                        span = b.extent_along(a.yaw)
                        if width_m <= span <= a.width + 1e-6:
                            a.held, a.held_offset, a.width = b.name, b.center - tcp, float(span)
                            self.log.append(f"closed on {b.name}")
                            return GripperOutcome(a.width)
            a.width = float(width_m)
            self.log.append("closed on nothing" if width_m < 0.004 else f"fingers to {width_m:.3f}")
            return GripperOutcome(a.width)

    # ------------------------------------------------------------------ perception

    def cameras(self):
        return [c for c in ("head", "wrist_left") if self.camera_enabled.get(c, True)]

    def _wrist_T(self) -> np.ndarray:
        a = self.a
        c, s = math.cos(math.radians(a.yaw)), math.sin(math.radians(a.yaw))
        dx, dy, dz = self.wrist_cam_offset
        eye = np.asarray(a.tcp) + np.array([c * dx - s * dy, s * dx + c * dy, dz])
        T = look_at(eye, eye + np.array([0, 0, -1.0]), up=(c, s, 0))
        return T

    def capture(self, camera):
        w, hgt = self.resolution
        if camera == "head":
            K, T = intrinsics(w, hgt, 70.0), self.head_T
        else:
            K, T = intrinsics(w, hgt, 75.0), self._wrist_T()
        with self._lock:
            depth, ids, names = self._render(K, T, w, hgt)
            palette = np.array([self._color(n) for n in names] + [(205, 205, 200)], dtype=np.uint8)
        rgb = palette[ids]  # ids == -1 picks the last entry: the table
        frame = Frame(camera=camera, K=K, T_wc=T, depth=depth if self.depth_enabled else None,
                      rgb=rgb, meta={"ids": ids, "names": names})
        return frame

    def _color(self, name: str) -> tuple[int, int, int]:
        b = self.boxes[name]
        if b.color is not None:
            return tuple(int(c) for c in b.color)
        h = sum(ord(ch) * (i + 1) for i, ch in enumerate(name))
        return (60 + h * 37 % 180, 60 + h * 53 % 180, 60 + h * 71 % 180)

    def _render(self, K, T, w, h):
        us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        uv = np.stack([us.ravel(), vs.ravel()], axis=1)
        d_cam = (np.linalg.inv(K) @ np.hstack([uv, np.ones((len(uv), 1))]).T).T
        d_cam /= np.linalg.norm(d_cam, axis=1, keepdims=True)
        dirs = (T[:3, :3] @ d_cam.T).T
        origin = T[:3, 3]
        best_t = np.full(len(uv), np.inf)
        ids = np.full(len(uv), -1, dtype=int)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_table = (self.table_z - origin[2]) / dirs[:, 2]
        t_table[~np.isfinite(t_table) | (t_table <= 0)] = np.inf
        best_t = np.minimum(best_t, t_table)
        names = list(self.boxes)
        for i, name in enumerate(names):
            b = self.boxes[name]
            R = b.rot()
            o = R.T @ (origin - b.center)
            d = dirs @ R
            e = np.array(b.size) / 2
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-e - o) / d
                t2 = (e - o) / d
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0) & (tmin < best_t)
            best_t[hit] = tmin[hit]
            ids[hit] = i
        z_axis = T[:3, 2]
        depth = (best_t * (dirs @ z_axis)).reshape(h, w).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth, ids.reshape(h, w), names

    def segment(self, frame, text):
        t = text.strip().lower()
        ids, names = frame.meta["ids"], frame.meta["names"]
        hidden = self.hidden_from.get(frame.camera, set())
        dets = []
        for i, name in enumerate(names):
            b = self.boxes.get(name)
            if b is None or name in hidden:
                continue
            labels = {name.lower(), *(x.lower() for x in b.aliases)}
            if t not in labels:
                continue
            mask = ids == i
            if mask.sum() >= 15:
                dets.append(Detection(mask=mask, score=0.9 - 0.01 * len(dets)))
        dets.sort(key=lambda d: -int(d.mask.sum()))
        return dets
