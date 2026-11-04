"""Result types returned by primitives to model-written code.

All positions are in the world (table) frame, metres; angles are degrees.
Domain outcomes (not reached, not holding, not found) are return values.
Only hardware faults raise :class:`RobotFault`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


class RobotFault(RuntimeError):
    """Driver disconnected, emergency stop or collision stop. Aborts the code block."""


class Cancelled(RuntimeError):
    """The robot.run invocation was cancelled through the Action cancel operation."""


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return [_round(v) for v in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


class _Printable:
    def to_dict(self) -> dict[str, Any]:
        return _round(asdict(self))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        body = ", ".join(f"{k}={v!r}" for k, v in self.to_dict().items() if v not in (None, [], ""))
        return f"{type(self).__name__}({body})"


@dataclass(repr=False, eq=False)
class Sight(_Printable):
    """One instance found by ``locate``, measured from one image of one camera."""

    text: str
    score: float
    camera: str
    pixel: list[int]            # [row, col] of the object's centre in that camera's image
    center_xyz: list[float]     # middle of the footprint rectangle, half its height above the table
    top_z: float                # table height when the height is unknown (see ``note``)
    size: list[float]           # [length, width, height], length >= width; height 0 = unknown
    long_axis_deg: float        # direction of the long side, [-90, 90): says nothing about front and back
    grasp_yaw_deg: float        # yaw that closes a top-down gripper across the short side
    color: list[int]            # typical [r, g, b] of the object in that image
    grasp_xyz: list[float] | None = None    # None when the height is unknown
    open_width_m: float | None = None
    note: str | None = None     # only when the measurement is limited
    symbol: str | None = None   # operator recognised by the colour and shape rules; model code may set it
    top_deg: float | None = None        # world heading of the symbol's top
    top_every_deg: float | None = None  # the symbol looks the same turned by this much
    points: Any = None          # the object's world points, N x 3 (metres); never printed

    def to_dict(self) -> dict[str, Any]:
        return _round({f.name: getattr(self, f.name) for f in fields(self) if f.name != "points"})


@dataclass(repr=False, eq=False)
class Capture(_Printable):
    """One camera frame as arrays for model code; ``rgb`` and ``world`` share one pixel grid."""

    camera: str
    rgb: Any                    # H x W x 3, uint8
    world: Any                  # H x W x 3, world coordinates (metres), NaN where depth is invalid

    def to_dict(self) -> dict[str, Any]:
        return {"camera": self.camera, "shape": [int(v) for v in self.rgb.shape[:2]]}


@dataclass(repr=False)
class Move(_Printable):
    """Result of ``move_to`` / ``home``. Not reached by a refusal: nothing moved."""

    reached: bool
    final_xyz: list[float] | None = None
    err_m: float | None = None          # from final_xyz to the target
    yaw_deg: float | None = None
    tilt_deg: float | None = None
    reason: str | None = None           # ik_failed | below_floor | outside_workspace | a controller code
    detail: str | None = None


@dataclass(repr=False)
class Grip(_Printable):
    """Result of ``set_gripper``."""

    width_m: float
    holding: bool


@dataclass(repr=False)
class ArmState(_Printable):
    tcp_xyz: list[float]
    yaw_deg: float
    tilt_deg: float
    joints: list[float]
    gripper_width_m: float
    holding: bool


@dataclass(repr=False)
class Pick(_Printable):
    """Result of ``pick``."""

    ok: bool                            # holding something afterwards
    reason: str | None = None           # empty | unreachable | already_holding | height_unknown | too_wide
    detail: str | None = None
    width_m: float | None = None        # opening on the object
    tilt_deg: float | None = None
    yaw_deg: float | None = None


@dataclass(repr=False)
class Place(_Printable):
    """Result of ``place``."""

    ok: bool                            # released at the planned spot
    reason: str | None = None           # not_holding | unreachable
    detail: str | None = None
    release_xyz: list[float] | None = None  # TCP when the gripper opened
    tilt_deg: float | None = None
    yaw_deg: float | None = None


@dataclass(repr=False)
class Drag(_Printable):
    """Result of ``drag``."""

    ok: bool                            # slid to the end of the line and let go
    reason: str | None = None           # empty | unreachable | already_holding | height_unknown | too_wide
    detail: str | None = None
    width_m: float | None = None        # opening on the object
    final_xyz: list[float] | None = None  # TCP at the end of the line
    tilt_deg: float | None = None
    yaw_deg: float | None = None
