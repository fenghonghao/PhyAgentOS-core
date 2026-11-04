"""Arm primitives exposed to model-written code: move_to, set_gripper, home, state, pick, place, drag.

What a call does is what it says: no retries and no perception inside. ``pick``, ``place`` and
``drag`` plan their whole route before anything moves, so a refusal leaves the arm and the fingers
as they were. Domain outcomes are return values; hardware faults raise ``RobotFault``.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from . import kinematics as kin
from .backend import MotionOutcome, Plan, Unreachable
from .types import ArmState, Cancelled, Drag, Grip, Move, Pick, Place, RobotFault, Sight

PATHS = ("auto", "line", "joint")
RISE_FIRST_M = 0.02        # a move that ends this much higher rises on the spot first, to clear what is near
KEEP_YAW_DEG = 95.0        # a closing line this far from the tangent can stay (joint 6 stops at 100)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def _sight(target: Any) -> Sight:
    if not isinstance(target, Sight):
        raise ValueError(f"target must be a Sight from locate(...), got {target!r}")
    return target


class ArmActions:
    """Mixed into ``Robot``, which provides arms_backend, cfg, cancel_event and _call."""

    _holding = False
    _above_bottom = 0.0        # TCP height above the held object's bottom, from when it was grasped

    # ------------------------------------------------------------------ state, move_to, home

    def state(self) -> ArmState:
        return self._call("state", {}, self._state)

    def _state(self) -> ArmState:
        xyz, yaw, tilt = self.arms_backend.tcp_pose()
        return ArmState(tcp_xyz=[float(v) for v in xyz], yaw_deg=float(yaw), tilt_deg=float(tilt),
                        joints=[float(v) for v in self.arms_backend.joints()],
                        gripper_width_m=float(self.arms_backend.gripper_width()), holding=self._holding)

    def move_to(self, xyz, yaw_deg: float | None = None, tilt_deg: float | None = None, path: str = "auto") -> Move:
        """Move the TCP to ``xyz`` (world, m). ``yaw_deg``: closing line seen from above.
        ``tilt_deg``: the fingers lean this far away from the base, 0 = straight down. None keeps
        the current value. Blocks until it stops; when not reachable nothing moves."""
        target = [float(v) for v in xyz]
        if len(target) != 3:
            raise ValueError("xyz must have 3 values")
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}")
        args = {"xyz": target, "yaw_deg": yaw_deg, "tilt_deg": tilt_deg, "path": path}
        return self._call("move_to", args, lambda: self._move(target, yaw_deg, tilt_deg, path))

    def _move(self, target: list[float], yaw_deg: float | None, tilt_deg: float | None, path: str) -> Move:
        cur, cur_yaw, cur_tilt = self.arms_backend.tcp_pose()
        yaw = float(cur_yaw if yaw_deg is None else yaw_deg)
        tilt = float(cur_tilt if tilt_deg is None else tilt_deg)

        def not_moved(reason: str, detail: str) -> Move:
            return Move(reached=False, final_xyz=[float(v) for v in cur], err_m=self._distance(cur, target),
                        yaw_deg=float(cur_yaw), tilt_deg=float(cur_tilt), reason=reason, detail=detail)

        refused = self._refusal(target, yaw, tilt, self.arms_backend.gripper_width())
        if refused:
            return not_moved(*refused)
        # half a turn is the same grasp while the gripper holds nothing
        detail = ""
        for candidate in [yaw] if self._holding else [yaw, yaw + 180.0]:
            try:
                plan = self.arms_backend.plan(target, candidate, tilt, path, floor_z=self.cfg.tcp_floor_z_m)
            except Unreachable as exc:
                detail = str(exc)
                continue
            return self._finish(plan, target)
        return not_moved("ik_failed", detail)

    def home(self) -> Move:
        """Back to the start pose, out of the head camera's way. The gripper keeps its opening."""
        return self._call("home", {}, lambda: self._finish(self.arms_backend.plan_home(), None))

    def _finish(self, plan: Plan, target: list[float] | None) -> Move:
        outcome = self._motion(plan)
        xyz, yaw, tilt = self.arms_backend.tcp_pose()
        return Move(reached=outcome.reached, final_xyz=[float(v) for v in xyz],
                    err_m=None if target is None else self._distance(xyz, target), yaw_deg=_wrap(float(yaw)),
                    tilt_deg=float(tilt), reason=None if outcome.reached else outcome.reason,
                    detail=None if outcome.reached else outcome.detail)

    @staticmethod
    def _distance(a, b) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    def _motion(self, plan: Plan) -> MotionOutcome:
        def run() -> MotionOutcome:
            outcome = self.arms_backend.execute(plan, self.cancel_event.is_set)
            if self.cancel_event.is_set():  # a cancelled goal fails at its controller: not a fault
                raise Cancelled("robot.run cancelled")
            if outcome.fault:
                raise RobotFault(outcome.fault)
            return outcome
        return self._call("motion", {"to": [round(float(v), 4) for v in plan.end_xyz]}, run)

    # ------------------------------------------------------------------ what a target may not be

    def _relative(self, x: float, y: float) -> tuple[float, float]:
        bx, by = self.arms_backend.base_xy
        return x - bx, y - by

    def _floor_z(self, x: float, y: float, yaw: float, tilt: float, width: float) -> float | None:
        """Lowest TCP height for this pose, with the lowest point of the open fingers clear of the
        table. ``tcp_floor_z_m`` was measured straight down: leaning, the tip comes up a little
        and an edge of a finger or the lower of the two fingers goes down."""
        floor = self.cfg.tcp_floor_z_m
        if floor is None:
            return None
        R = kin.yaw_tilt_to_matrix(yaw, tilt, self._relative(x, y))
        return float(floor - self.cfg.fingertip_m * (1.0 + R[2, 2])
                     + abs(R[2, 1]) * width / 2.0 + abs(R[2, 0]) * self.cfg.finger_half_depth_m)

    def _refusal(self, xyz, yaw: float, tilt: float, width: float) -> tuple[str, str] | None:
        floor = self._floor_z(float(xyz[0]), float(xyz[1]), yaw, tilt, width)
        if floor is not None and xyz[2] < floor - 1e-9:
            return "below_floor", f"z={xyz[2]:.3f} would bring the fingertips to the table: lowest here is {floor:.3f}"
        ws = self.cfg.workspace
        if ws:
            for i, axis in enumerate("xy"):
                lim = ws.get(axis)
                if lim is not None and not (float(lim[0]) <= xyz[i] <= float(lim[1])):
                    return "outside_workspace", f"{axis}={xyz[i]:.3f} outside [{float(lim[0]):.3f}, {float(lim[1]):.3f}]"
            z_max = ws.get("z_max")
            if z_max is not None and xyz[2] > float(z_max):
                return "outside_workspace", f"z={xyz[2]:.3f} above {float(z_max):.3f}"
        return None

    # ------------------------------------------------------------------ gripper

    def set_gripper(self, width_m: float) -> Grip:
        """Move the fingers to an opening (m), 0 = closed. ``holding``: they stopped wider than
        asked, so something is between them. Which object it is, this does not say."""
        target = max(0.0, min(float(width_m), float(self.arms_backend.max_opening_m)))
        return self._call("set_gripper", {"width_m": round(target, 4)}, lambda: self._grip(target))

    def _grip(self, target: float) -> Grip:
        out = self.arms_backend.grip(target)
        if self.cancel_event.is_set():
            raise Cancelled("robot.run cancelled")
        if out.fault:
            raise RobotFault(f"gripper: {out.fault}")
        holding = float(out.width_m) > target + self.cfg.holding_min_width_m
        if holding and not self._holding:  # what it closed on stood on the table
            self._above_bottom = float(self.arms_backend.tcp_pose()[0][2]) - self.cfg.table_z
        self._holding = holding
        return Grip(width_m=float(out.width_m), holding=holding)

    # ------------------------------------------------------------------ route planning

    def _wrist_offset(self, yaw: float, x: float, y: float) -> float:
        """Closing line relative to the tangent at (x, y): roughly joint 6's angle."""
        rx, ry = self._relative(x, y)
        return _wrap(yaw - math.degrees(math.atan2(ry, rx)) - 90.0)

    def _grasp_yaws(self, s: Sight, yaw_deg: float | None) -> list[float]:
        """Both ways round grasp the same. Left to itself, the one nearer the wrist's middle first."""
        yaw = float(s.grasp_yaw_deg if yaw_deg is None else yaw_deg)
        first, second = _wrap(yaw), _wrap(yaw + 180.0)
        x, y = s.center_xyz[:2]
        if yaw_deg is None and abs(self._wrist_offset(second, x, y)) < abs(self._wrist_offset(first, x, y)):
            return [second, first]
        return [first, second]

    def _keep_yaw(self, yaw: float, x: float, y: float) -> float:
        """The current closing line if the wrist can keep it at (x, y), else the nearest it can."""
        offset = self._wrist_offset(yaw, x, y)
        if abs(offset) <= KEEP_YAW_DEG:
            return _wrap(yaw)
        return _wrap(yaw - offset + math.copysign(90.0, offset))

    def _approach(self, x: float, y: float, tilt: float) -> np.ndarray:
        return kin.approach_axis(self._relative(x, y), tilt)

    def _plan_legs(self, legs: list[tuple[Any, float, float, str]]) -> list[Plan]:
        plans: list[Plan] = []
        start = None
        for xyz, yaw, tilt, path in legs:
            start = self.arms_backend.plan([float(v) for v in xyz], yaw, tilt, path, start=start,
                                           floor_z=self.cfg.tcp_floor_z_m)
            plans.append(start)
        return plans

    def _reaches(self, xyz, yaw: float, tilt: float) -> bool:
        try:
            self.arms_backend.plan([float(v) for v in xyz], yaw, tilt, "joint")
        except Unreachable:
            return False
        return True

    def _stops(self, x: float, y: float, z: float, yaw: float, width: float):
        """Poses the fingers can work from at (x, y), smallest lean first, as (point, tilt). The
        height is raised to what keeps the fingers off the table at that lean."""
        for tilt in map(float, self.cfg.approach_tilts_deg):
            floor = self._floor_z(x, y, yaw, tilt, width)
            point = np.array([x, y, z if floor is None else max(z, floor)])
            if self._refusal(point, yaw, tilt, width) is None and self._reaches(point, yaw, tilt):
                yield point, tilt

    def _hovers(self, point: np.ndarray, yaw: float, tilt: float, width: float):
        """Points ``approach_m`` back along the fingers, smallest lean not below the stop's first:
        a hover often has to lean where the stop below it does not."""
        for hover_tilt in map(float, self.cfg.approach_tilts_deg):
            if hover_tilt < tilt:
                continue
            hover = point - self._approach(point[0], point[1], hover_tilt) * self.cfg.approach_m
            if self._refusal(hover, yaw, hover_tilt, width) is None and self._reaches(hover, yaw, hover_tilt):
                yield hover, hover_tilt

    def _find_route(self, x: float, y: float, z: float, yaws: list[float], width: float,
                    legs: Callable[..., list]) -> tuple:
        """The first complete route over (x, y, z). Returns (yaw, tilt at the stop, plans)."""
        detail = f"no pose reaches ({x:.3f}, {y:.3f}, {z:.3f})"
        for yaw in yaws:
            for point, tilt in self._stops(x, y, z, yaw, width):
                for hover, hover_tilt in self._hovers(point, yaw, tilt, width):
                    try:
                        return yaw, tilt, self._plan_legs(legs(point, hover, yaw, tilt, hover_tilt))
                    except Unreachable as exc:
                        detail = str(exc)
        raise Unreachable(detail)

    # ------------------------------------------------------------------ pick

    def pick(self, target: Sight, z: float | None = None, yaw_deg: float | None = None,
             width_m: float | None = None) -> Pick:
        """Grasp a Sight and lift it: open, move above, straight down, close, straight back up.
        ``z``: TCP height of the grasp (default ``grasp_xyz``). ``yaw_deg``: closing line (default
        ``grasp_yaw_deg``). ``width_m``: how far to open (default the object + a margin). One
        attempt. When not reachable nothing moves."""
        s = _sight(target)
        return self._call("pick", {"text": s.text, "at": s.center_xyz, "z": z, "yaw_deg": yaw_deg, "width_m": width_m},
                          lambda: self._pick(s, z, yaw_deg, width_m))

    def _precheck(self, s: Sight, z: float | None) -> tuple[str, str | None] | None:
        if self._holding:
            return "already_holding", None
        if z is None and s.grasp_xyz is None:
            return "height_unknown", s.note
        if s.size[1] > self.arms_backend.max_opening_m:
            return "too_wide", f"{s.size[1]:.3f} m does not fit the {self.arms_backend.max_opening_m:.3f} m opening"
        return None

    def _open_width(self, s: Sight, width_m: float | None) -> float:
        most = float(self.arms_backend.max_opening_m)
        if width_m is not None:
            return max(0.0, min(float(width_m), most))
        if s.open_width_m is not None:
            return float(s.open_width_m)
        return min(float(s.size[1]) + self.cfg.open_margin_m, most)

    def _pick(self, s: Sight, z: float | None, yaw_deg: float | None, width_m: float | None) -> Pick:
        problem = self._precheck(s, z)
        if problem:
            return Pick(ok=False, reason=problem[0], detail=problem[1])
        grasp_z = float(z) if z is not None else float(s.grasp_xyz[2])
        width = self._open_width(s, width_m)
        x, y = float(s.center_xyz[0]), float(s.center_xyz[1])
        try:
            yaw, tilt, (above, down, up) = self._find_route(
                x, y, grasp_z, self._grasp_yaws(s, yaw_deg), width,
                lambda p, h, yw, t, ht: [(h, yw, ht, "joint"), (p, yw, t, "line"), (h, yw, ht, "line")])
        except Unreachable as exc:
            return Pick(ok=False, reason="unreachable", detail=str(exc))
        self.set_gripper(width)
        for plan in (above, down):
            m = self._motion(plan)
            if not m.reached:
                return Pick(ok=False, reason=m.reason, detail=m.detail, tilt_deg=tilt, yaw_deg=yaw)
        g = self.set_gripper(0.0)
        self._motion(up)
        return Pick(ok=g.holding, reason=None if g.holding else "empty", width_m=g.width_m, tilt_deg=tilt, yaw_deg=yaw)

    # ------------------------------------------------------------------ place

    def place(self, target, bottom_z: float | None = None, yaw_deg: float | None = None) -> Place:
        """Put the held object down: move above, straight down, open a little, straight back up.
        ``target``: [x, y] (the object's bottom ends 1 cm above the table) or a container's Sight
        (over its centre, the bottom 3 cm above its ``top_z``). ``bottom_z``: that height instead.
        ``yaw_deg``: turn the object to this closing line (default: keep it if the wrist can). When
        not reachable nothing moves and it stays held."""
        return self._call("place", {"target": getattr(target, "text", target), "bottom_z": bottom_z, "yaw_deg": yaw_deg},
                          lambda: self._place(target, bottom_z, yaw_deg))

    def _place(self, target, bottom_z: float | None, yaw_deg: float | None) -> Place:
        if not self._holding:
            return Place(ok=False, reason="not_holding")
        if isinstance(target, Sight):
            x, y = float(target.center_xyz[0]), float(target.center_xyz[1])
            bottom = float(target.top_z) + self.cfg.drop_gap_m if bottom_z is None else float(bottom_z)
        else:
            x, y = float(target[0]), float(target[1])
            bottom = self.cfg.table_z + self.cfg.place_gap_m if bottom_z is None else float(bottom_z)
        cur, cur_yaw, _ = self.arms_backend.tcp_pose()
        width = float(self.arms_backend.gripper_width())

        def legs(p, h, yw, t, ht):
            rise = [([cur[0], cur[1], h[2]], cur_yaw, ht, "joint")] if h[2] > cur[2] + RISE_FIRST_M else []
            return [*rise, (h, yw, ht, "joint"), (p, yw, t, "line"), (h, yw, ht, "line")]

        yaws = [_wrap(float(yaw_deg))] if yaw_deg is not None else [self._keep_yaw(float(cur_yaw), x, y)]
        try:
            yaw, tilt, plans = self._find_route(x, y, bottom + self._above_bottom, yaws, width, legs)
        except Unreachable as exc:
            return Place(ok=False, reason="unreachable", detail=str(exc))
        for plan in plans[:-1]:
            m = self._motion(plan)
            if not m.reached:
                return Place(ok=False, reason=m.reason, detail=m.detail, tilt_deg=tilt, yaw_deg=yaw)
        release = [float(v) for v in self.arms_backend.tcp_pose()[0]]
        self.set_gripper(min(width + self.cfg.release_open_m, float(self.arms_backend.max_opening_m)))
        self._motion(plans[-1])
        return Place(ok=True, release_xyz=release, tilt_deg=tilt, yaw_deg=yaw)

    # ------------------------------------------------------------------ drag

    def drag(self, target: Sight, to, z: float | None = None, yaw_deg: float | None = None,
             width_m: float | None = None) -> Drag:
        """Slide a Sight along the table to ``to`` = [x, y]: open, move above, straight down, close,
        straight line at that height, open a little, back up. ``z``, ``yaw_deg``, ``width_m``: as in
        ``pick``. Nothing senses contact: keep the line clear of other objects. When not reachable
        nothing moves."""
        s = _sight(target)
        goal = [float(to[0]), float(to[1])]
        return self._call("drag", {"text": s.text, "at": s.center_xyz, "to": goal, "z": z, "yaw_deg": yaw_deg,
                                   "width_m": width_m}, lambda: self._drag(s, goal, z, yaw_deg, width_m))

    def _drag_route(self, s: Sight, goal: list[float], z: float, width: float, yaw_deg: float | None) -> tuple:
        """Each end leans only as far as it must; the line keeps one height, the higher one needed.
        Returns (yaw, tilt at the start, plans): above, down, slide, up, and back up from the start."""
        x, y = float(s.center_xyz[0]), float(s.center_xyz[1])
        detail = f"no pose reaches ({x:.3f}, {y:.3f}) and ({goal[0]:.3f}, {goal[1]:.3f}) at {z:.3f}"
        for yaw in self._grasp_yaws(s, yaw_deg):
            for start, tilt in self._stops(x, y, z, yaw, width):
                for end, end_tilt in self._stops(goal[0], goal[1], z, yaw, float(s.size[1])):
                    start[2] = end[2] = max(start[2], end[2])
                    hover = next(self._hovers(start, yaw, tilt, width), None)
                    end_hover = next(self._hovers(end, yaw, end_tilt, width), None)
                    if hover is None or end_hover is None:
                        continue
                    try:
                        plans = self._plan_legs([(hover[0], yaw, hover[1], "joint"), (start, yaw, tilt, "line"),
                                                 (end, yaw, end_tilt, "line"), (end_hover[0], yaw, end_hover[1], "line")])
                        back = self.arms_backend.plan([float(v) for v in hover[0]], yaw, hover[1], "line", start=plans[1])
                    except Unreachable as exc:
                        detail = str(exc)
                        continue
                    return yaw, tilt, (*plans, back)
        raise Unreachable(detail)

    def _drag(self, s: Sight, goal: list[float], z: float | None, yaw_deg: float | None,
              width_m: float | None) -> Drag:
        problem = self._precheck(s, z)
        if problem:
            return Drag(ok=False, reason=problem[0], detail=problem[1])
        grasp_z = float(z) if z is not None else float(s.grasp_xyz[2])
        width = self._open_width(s, width_m)
        try:
            yaw, tilt, (above, down, slide, up, back) = self._drag_route(s, goal, grasp_z, width, yaw_deg)
        except Unreachable as exc:
            return Drag(ok=False, reason="unreachable", detail=str(exc))
        self.set_gripper(width)
        for plan in (above, down):
            m = self._motion(plan)
            if not m.reached:
                return Drag(ok=False, reason=m.reason, detail=m.detail, tilt_deg=tilt, yaw_deg=yaw)
        g = self.set_gripper(0.0)
        if not g.holding:
            self._motion(back)
            return Drag(ok=False, reason="empty", width_m=g.width_m, tilt_deg=tilt, yaw_deg=yaw)
        m = self._motion(slide)
        if not m.reached:
            return Drag(ok=False, reason=m.reason, detail=m.detail, width_m=g.width_m, tilt_deg=tilt, yaw_deg=yaw)
        final = [float(v) for v in self.arms_backend.tcp_pose()[0]]
        self.set_gripper(min(g.width_m + self.cfg.release_open_m, float(self.arms_backend.max_opening_m)))
        self._motion(up)
        return Drag(ok=True, width_m=g.width_m, final_xyz=final, tilt_deg=tilt, yaw_deg=yaw)
