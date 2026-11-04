"""Interfaces the primitives depend on.

``ArmBackend`` talks to the arm (Dora/motion_server on the real robot, a kinematic fake in
tests). ``Perception`` captures camera frames and segments objects by text.
Neither interface has timeouts: executing a plan blocks until the motion ends.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .types import _Printable


class Unreachable(Exception):
    """A motion that cannot be planned. Nothing has moved."""


@dataclass
class Plan:
    """A checked motion that has not moved yet; a following plan can start from its end."""

    steps: list[Any]                      # backend-specific: one motion per entry
    end_xyz: list[float]
    end_joints: list[float] | None = None


@dataclass(repr=False)
class MotionOutcome(_Printable):
    """Raw outcome of a motion as reported by the backend."""

    reached: bool
    final_xyz: list[float] | None = None
    reason: str | None = None  # ik_failed | joint_limit | cancelled | controller code
    fault: str | None = None  # set when the driver reported a hardware fault
    detail: str | None = None  # raw controller message, for diagnosis


@dataclass
class GripperOutcome:
    width_m: float
    fault: str | None = None


class ArmBackend(abc.ABC):
    """One arm. Poses are TCP poses in the world frame; ``yaw_deg`` and ``tilt_deg`` as defined
    in ``kinematics``, for the gripper's position relative to ``base_xy``."""

    max_opening_m: float = 0.07
    base_xy: tuple[float, float] = (0.0, 0.0)

    @abc.abstractmethod
    def tcp_pose(self) -> tuple[list[float], float, float]:
        """Current (xyz, yaw_deg, tilt_deg)."""

    @abc.abstractmethod
    def joints(self) -> list[float]: ...

    @abc.abstractmethod
    def gripper_width(self) -> float: ...

    @abc.abstractmethod
    def plan(self, xyz: list[float], yaw_deg: float, tilt_deg: float, path: str, start: Plan | None = None,
             floor_z: float | None = None) -> Plan:
        """Check a motion from ``start``'s end (default: the current pose) without moving.
        ``path``: "line" is the straight line or nothing, "joint" the joint-space move, "auto" a
        short line or a long joint move. ``floor_z``: lowest TCP height a joint-space move may
        pass through. Raises :class:`Unreachable`."""

    @abc.abstractmethod
    def plan_home(self, start: Plan | None = None) -> Plan: ...

    @abc.abstractmethod
    def execute(self, plan: Plan, cancelled: Callable[[], bool] = lambda: False) -> MotionOutcome:
        """Run a plan; blocks until it ends. ``cancelled`` is asked between steps."""

    @abc.abstractmethod
    def grip(self, width_m: float) -> GripperOutcome:
        """Command an opening; returns once the fingers stop."""

    def stop(self) -> None:
        """Best-effort stop of the current motion (used only for explicit cancel)."""


@dataclass
class Frame:
    camera: str
    K: np.ndarray
    T_wc: np.ndarray
    rgb: np.ndarray | None = None
    depth: np.ndarray | None = None  # float32 metres along optical axis, 0 = invalid
    meta: dict = field(default_factory=dict)
    full: "Frame | None" = None  # the camera's own frame, when this one is a reduced window of it
    window: tuple[int, int, int, int, int] | None = None  # x0, y0, width, height, factor

    @property
    def own(self) -> "Frame":
        """The camera's own frame: what the model sees."""
        return self if self.full is None else self.full

    def full_pixel(self, row: int, col: int) -> tuple[int, int]:
        """The same place, as (row, col) on the camera's own image."""
        if self.window is None:
            return row, col
        x0, y0, _, _, f = self.window
        return y0 + row * f + f // 2, x0 + col * f + f // 2

    @property
    def shape(self) -> tuple[int, int]:
        if self.depth is not None:
            return self.depth.shape[:2]
        assert self.rgb is not None
        return self.rgb.shape[:2]


@dataclass
class Detection:
    mask: np.ndarray  # bool HxW
    score: float


class Perception(abc.ABC):
    @abc.abstractmethod
    def cameras(self) -> list[str]:
        """Camera names; the first is the fixed head camera."""

    @abc.abstractmethod
    def capture(self, camera: str) -> Frame:
        """Latest frame of a camera, with its current extrinsics (wrist cameras move)."""

    @abc.abstractmethod
    def segment(self, frame: Frame, text: str) -> list[Detection]:
        """All instances matching ``text``, best first."""
