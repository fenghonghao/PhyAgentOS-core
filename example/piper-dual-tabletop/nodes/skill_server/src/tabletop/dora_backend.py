"""ArmBackend over Dora Actions: one motion_server + gripper_action_controller.

Worker threads (model code) never touch the Dora node. They put outgoing messages on the
``Bus`` and block on a per-goal event; the Dora main loop drains the bus every tick and
resolves goals when ``*_result`` inputs arrive. Nothing here times out: a goal waits for its
result. Hardware-level failures (stale state, aborted trajectory) surface as faults.

All IK is solved here, before anything moves, and motion_server gets joint goals: it plans
joint-space trajectories only. A straight line is a series of small joint goals, each solved
from the last, so a line that cannot be followed is refused whole.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import kinematics as kin
from .backend import ArmBackend, GripperOutcome, MotionOutcome, Plan, Unreachable

FAULT_CODES = {"NO_FRESH_ROBOT_STATE", "TRAJECTORY_EXECUTION_FAILED", "CONTROLLER_UNAVAILABLE"}
RETRY_CODES = {"BUSY", "PLANNING_FAILED"}  # robot still settling: resend (count-limited, no timer)
IK_ITERATIONS = 120


class Bus:
    """Thread-safe bridge between worker threads and the Dora main loop."""

    def __init__(self) -> None:
        self.outbox: queue.Queue[tuple[str, Any, dict]] = queue.Queue()
        self._pending: dict[str, tuple[threading.Event, list]] = {}
        self._lock = threading.Lock()

    def request(self, output_id: str, payload: Any) -> tuple[str, Any]:
        """Send a goal and block until its result arrives (no timeout)."""
        gid = str(uuid.uuid4())
        ev, box = threading.Event(), []
        with self._lock:
            self._pending[gid] = (ev, box)
        self.outbox.put((output_id, payload, {"goal_id": gid}))
        ev.wait()
        return box[0]

    def send(self, output_id: str, payload: Any, metadata: dict | None = None) -> None:
        self.outbox.put((output_id, payload, metadata or {}))

    def resolve(self, goal_id: str, status: str, result: Any) -> bool:
        with self._lock:
            item = self._pending.pop(goal_id, None)
        if item is None:
            return False
        item[1].append((status, result))
        item[0].set()
        return True

    def active_goals(self) -> list[str]:
        with self._lock:
            return list(self._pending)

    def drain(self) -> list[tuple[str, Any, dict]]:
        out = []
        while True:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                return out


@dataclass
class ArmConfig:
    name: str
    urdf: str
    home_joints: list[float]
    base_xyz: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_rpy_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # alternatively the arm_base pose in the world as a 4x4 matrix (overrides xyz/rpy);
    # base_T_inverse: the matrix given is world-in-base (invert it)
    base_T: list[list[float]] | None = None
    base_T_inverse: bool = False
    group_name: str = "piper_arm"
    reference_frame: str = "arm_base"
    target_frame: str = "tcp"
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(1, 7)])
    gripper_joint: str = "gripper"
    max_opening_m: float = 0.07
    # extra IK seeds (joint vectors) tried after the current and home joints
    ik_seeds: list[list[float]] = field(default_factory=lambda: [
        [0.0, 1.6, -1.24, 0.0, 1.05, 0.0], [0.0, 1.0, -1.0, 0.0, 0.8, 0.0], [0.0, 2.0, -1.46, 0.0, 1.1, 0.0]])


@dataclass
class MotionConfig:
    velocity_scale: float = 0.5
    acceleration_scale: float = 0.5
    line_step_m: float = 0.02
    line_step_deg: float = 8.0          # a line that also turns: at most this much per step
    line_joint_jump_rad: float = 0.35   # more than this in one step is not following the line
    auto_line_max_m: float = 0.12
    settle_retries: int = 5
    settle_pause_s: float = 0.15


class DoraArmBackend(ArmBackend):
    def __init__(self, arms: dict[str, ArmConfig], motion: MotionConfig, bus: Bus) -> None:
        if len(arms) != 1:
            raise ValueError(f"expected one arm, got {sorted(arms)}")
        self.name, self.c = next(iter(arms.items()))
        self.motion = motion
        self.bus = bus
        self.chain = kin.Chain(self.c.urdf, self.c.reference_frame, self.c.target_frame)
        self.T_world_base = base_pose(self.c)
        self.T_base_world = np.linalg.inv(self.T_world_base)
        self.base_xy = (float(self.T_world_base[0, 3]), float(self.T_world_base[1, 3]))
        self.max_opening_m = self.c.max_opening_m
        self._state: dict[str, float] = {}
        self._state_lock = threading.Lock()
        self._state_seen = threading.Event()

    # ------------------------------------------------------------------ inputs from the main loop

    def on_joint_state(self, names: list[str], position: list[float]) -> None:
        with self._state_lock:
            self._state = dict(zip(names, position))
        self._state_seen.set()

    # ------------------------------------------------------------------ state

    def has_state(self, arm: str | None = None) -> bool:
        """Whether a joint state has arrived (never blocks)."""
        return self._state_seen.is_set()

    def joints(self) -> list[float]:
        self._state_seen.wait()
        with self._state_lock:
            return [float(self._state[j]) for j in self.c.joint_names]

    def gripper_width(self) -> float:
        self._state_seen.wait()
        with self._state_lock:
            return float(self._state.get(self.c.gripper_joint, 0.0))

    def _fk_world(self, q) -> np.ndarray:
        return self.T_world_base @ self.chain.fk(q)

    def _T_world_tcp(self, arm: str | None = None) -> np.ndarray:
        return self._fk_world(self.joints())

    def _relative(self, xyz) -> tuple[float, float]:
        return float(xyz[0]) - self.base_xy[0], float(xyz[1]) - self.base_xy[1]

    def tcp_pose(self):
        T = self._T_world_tcp()
        yaw, tilt = kin.matrix_to_yaw_tilt(T[:3, :3], self._relative(T[:3, 3]))
        return [float(v) for v in T[:3, 3]], yaw, tilt

    # ------------------------------------------------------------------ planning: nothing moves

    def _solve(self, xyz, R_world: np.ndarray, seeds) -> np.ndarray | None:
        T_base = self.T_base_world @ kin.transform(np.asarray(xyz, dtype=float), R_world)
        return kin.solve_ik(self.chain, T_base, seeds, iters=IK_ITERATIONS)

    def plan(self, xyz, yaw_deg, tilt_deg, path, start=None, floor_z=None):
        q0 = np.asarray(start.end_joints if start is not None else self.joints(), dtype=float)
        T0 = self._fk_world(q0)
        end = np.asarray(xyz, dtype=float)
        R1 = kin.yaw_tilt_to_matrix(yaw_deg, tilt_deg, self._relative(end))
        where = f"xyz={np.round(end, 3).tolist()} yaw={yaw_deg:.0f} tilt={tilt_deg:.0f}"
        dist = float(np.linalg.norm(end - T0[:3, 3]))
        if path == "line" or (path == "auto" and dist <= self.motion.auto_line_max_m):
            steps = self._line(q0, T0, end, R1, where)
        else:
            q = self._solve(end, R1, [q0, self.c.home_joints, *self.c.ik_seeds])
            if q is None:
                raise Unreachable(f"no joint solution for {where}")
            if floor_z is not None and not self._swing_clear(q0, q, float(floor_z)):
                raise Unreachable(f"the joint move to {where} passes below the table: go there higher, or use a line")
            steps = [q]
        return Plan(steps=[[float(v) for v in q] for q in steps], end_xyz=[float(v) for v in end],
                    end_joints=[float(v) for v in steps[-1]])

    def _line(self, q0: np.ndarray, T0: np.ndarray, end: np.ndarray, R1: np.ndarray, where: str) -> list[np.ndarray]:
        start, R0 = T0[:3, 3], T0[:3, :3]
        turn = math.degrees(float(np.linalg.norm(kin.rot_log(R1 @ R0.T))))
        m = self.motion
        n = max(1, math.ceil(float(np.linalg.norm(end - start)) / m.line_step_m), math.ceil(turn / m.line_step_deg))
        steps, q = [], q0
        for i in range(1, n + 1):
            f = i / n
            p = start + (end - start) * f
            next_q = self._solve(p, kin.slerp_matrix(R0, R1, f), [q])
            if next_q is None or float(np.abs(next_q - q).max()) > m.line_joint_jump_rad:
                raise Unreachable(f"no straight line to {where}: it breaks at {np.round(p, 3).tolist()}")
            steps.append(next_q)
            q = next_q
        return steps

    def _swing_clear(self, q0: np.ndarray, q1: np.ndarray, floor_z: float) -> bool:
        return all(self._fk_world(q0 + (q1 - q0) * f)[2, 3] >= floor_z for f in np.linspace(0.1, 0.9, 9))

    def plan_home(self, start=None):
        home = [float(v) for v in self.c.home_joints]
        return Plan(steps=[home], end_xyz=[float(v) for v in self._fk_world(home)[:3, 3]], end_joints=home)

    # ------------------------------------------------------------------ execution

    def _send_joints(self, q) -> tuple[bool, str, str]:
        from forge_msgs import MoveJointsGoal

        c, m = self.c, self.motion
        goal = MoveJointsGoal(group_name=c.group_name, joint_names=list(c.joint_names), positions=[float(v) for v in q],
                              velocity_scale=m.velocity_scale, acceleration_scale=m.acceleration_scale,
                              requested_duration_ns=None)
        status, code, msg = "", "UNKNOWN", ""
        for attempt in range(m.settle_retries + 1):
            status, result = self.bus.request(f"{self.name}_move_joints_goal", goal.to_arrow())
            code, msg = getattr(result, "error_code", "UNKNOWN"), getattr(result, "message", "")
            if code in RETRY_CODES and attempt < m.settle_retries:
                time.sleep(m.settle_pause_s)
                continue
            break
        return status == "succeeded" and code == "SUCCESS", code, msg

    def execute(self, plan: Plan, cancelled: Callable[[], bool] = lambda: False) -> MotionOutcome:
        ok, code, msg = True, "SUCCESS", ""
        for q in plan.steps:
            if cancelled():
                return MotionOutcome(False, self.tcp_pose()[0], "cancelled")
            ok, code, msg = self._send_joints(q)
            if not ok:
                break
        return _outcome(ok, code, msg, self.tcp_pose()[0])

    def grip(self, width_m: float) -> GripperOutcome:
        from forge_msgs import GripperCommandGoal

        goal = GripperCommandGoal(position=float(width_m), max_velocity=None, max_effort=None)
        status, result = self.bus.request(f"{self.name}_gripper_goal", goal.to_arrow())
        width = getattr(result, "position", None)
        fault = gripper_fault(status, getattr(result, "error_code", "UNKNOWN"), getattr(result, "message", ""))
        return GripperOutcome(float(self.gripper_width() if width is None else width), fault=fault)

    def stop(self) -> None:
        for gid in self.bus.active_goals():
            for kind in ("move_joints", "gripper"):
                self.bus.send(f"{self.name}_{kind}_cancel", None, {"goal_id": gid, "cancel": True})


def base_pose(c: ArmConfig) -> np.ndarray:
    if c.base_T is not None:
        T = np.asarray(c.base_T, dtype=float)
        if T.shape != (4, 4):
            raise ValueError(f"arm {c.name}: base_T must be 4x4")
        return np.linalg.inv(T) if c.base_T_inverse else T
    return kin.pose_matrix(c.base_xyz, [math.radians(a) for a in c.base_rpy_deg])


def gripper_fault(status: str, code: str, msg: str) -> str | None:
    """Fault text for a gripper result, or None. Reaching the target, stalling on an object and
    a cancelled goal (the cancel taking effect) are not hardware faults."""
    if code in ("SUCCESS", "STALLED", "CANCELED", "CANCELLED") or status in ("succeeded", "canceled", "cancelled"):
        return None
    return f"gripper {code}: {msg}"


def _outcome(ok: bool, code: str, msg: str, final) -> MotionOutcome:
    if ok:
        return MotionOutcome(True, final)
    detail = f"{code}: {msg}"[:300]
    if code in FAULT_CODES:
        return MotionOutcome(False, final, code.lower(), fault=detail, detail=detail)
    if code == "JOINT_LIMIT_VIOLATION":
        return MotionOutcome(False, final, "joint_limit", detail=detail)
    return MotionOutcome(False, final, code.lower(), detail=detail)


def load_arm_configs(raw: dict, base_dir: Path) -> dict[str, ArmConfig]:
    out = {}
    for name, c in (raw or {}).items():
        c = dict(c)
        urdf = Path(c.pop("urdf"))
        out[name] = ArmConfig(name=name, urdf=str(urdf if urdf.is_absolute() else (base_dir / urdf).resolve()), **c)
    return out
