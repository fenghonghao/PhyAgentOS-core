"""Cartesian teleoperation of one PIPER arm through our own IK (no leader arm needed).

The target is the TCP pose in the arm base frame. Each control tick integrates the operator's
velocity, solves IK seeded with the last command and limits the joint step, so the command
stream sent to the driver is smooth and never jumps. If IK has no solution the target stays
where it was (the arm stops at the workspace edge instead of flipping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import kinematics as kin

from .spec import ArmSpec


@dataclass
class Twist:
    """Operator command in the arm base frame (already scaled to physical units)."""
    vx: float = 0.0      # m/s
    vy: float = 0.0
    vz: float = 0.0
    yaw: float = 0.0     # rad/s about base z
    tilt: float = 0.0    # rad/s about the TCP closing axis (approach leans away from vertical)
    grip: float = 0.0    # m/s, + opens

    def is_zero(self) -> bool:
        return not any((self.vx, self.vy, self.vz, self.yaw, self.tilt, self.grip))


class TeleopArm:
    def __init__(self, spec: ArmSpec, max_joint_speed: float = 2.0, control_yaw_deg: float = 0.0) -> None:
        if not spec.urdf:
            raise ValueError(f"arm {spec.name}: teleop needs 'urdf'")
        self.spec = spec
        self.chain = kin.Chain(spec.urdf, "arm_base", "tcp")
        self.max_joint_speed = max_joint_speed
        c, s = math.cos(math.radians(control_yaw_deg)), math.sin(math.radians(control_yaw_deg))
        self._R_ctrl = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.q_meas: np.ndarray | None = None
        self.g_meas: float | None = None
        self.q_cmd: np.ndarray | None = None
        self.g_cmd: float | None = None
        self.T: np.ndarray | None = None
        self.last_status = "waiting for state"
        self._home_from: tuple[np.ndarray, float] | None = None
        self._home_t = 0.0
        self._home_dur = 0.0

    # ------------------------------------------------------------------ state

    def on_state(self, names: list[str], position: list[float]) -> None:
        pos = dict(zip(names, position))
        try:
            self.q_meas = np.array([pos[j] for j in self.spec.joint_names], dtype=float)
            self.g_meas = float(pos[self.spec.gripper_joint])
        except KeyError:
            return
        if self.q_cmd is None:
            self.sync()

    def sync(self) -> None:
        """Take the measured pose as the command (start-up, after pause)."""
        if self.q_meas is None:
            return
        self.q_cmd = self.q_meas.copy()
        self.g_cmd = float(self.g_meas or 0.0)
        self.T = self.chain.fk(self.q_cmd)
        self.last_status = "ok"

    @property
    def ready(self) -> bool:
        return self.q_cmd is not None

    def command_values(self) -> list[float]:
        assert self.q_cmd is not None and self.g_cmd is not None
        return [*map(float, self.q_cmd), float(self.g_cmd)]

    def tcp_xyz(self) -> list[float] | None:
        return None if self.T is None else [float(v) for v in self.T[:3, 3]]

    # ------------------------------------------------------------------ control

    def start_home(self, duration_s: float) -> None:
        if self.q_cmd is None:
            return
        self._home_from = (self.q_cmd.copy(), float(self.g_cmd or 0.0))
        self._home_t = 0.0
        self._home_dur = max(0.5, duration_s)

    @property
    def homing(self) -> bool:
        return self._home_from is not None

    def home_duration(self, joint_speed: float) -> float:
        if self.q_cmd is None:
            return 0.0
        d = float(np.max(np.abs(np.asarray(self.spec.home_joints) - self.q_cmd)))
        return d / max(joint_speed, 1e-3)

    def step(self, tw: Twist, dt: float) -> str:
        if self.q_cmd is None or self.T is None:
            return self.last_status
        if self._home_from is not None:
            self._home_t += dt
            f = min(1.0, self._home_t / self._home_dur)
            f = f * f * (3 - 2 * f)                       # smoothstep
            q0, g0 = self._home_from
            self.q_cmd = q0 + (np.asarray(self.spec.home_joints, dtype=float) - q0) * f
            self.g_cmd = g0 + (self.spec.home_gripper_m - g0) * f
            self.T = self.chain.fk(self.q_cmd)
            if f >= 1.0:
                self._home_from = None
            self.last_status = "homing"
            return self.last_status

        if tw.grip:
            self.g_cmd = float(np.clip((self.g_cmd or 0.0) + tw.grip * dt, 0.0, self.spec.max_opening_m))
        if tw.is_zero() or not any((tw.vx, tw.vy, tw.vz, tw.yaw, tw.tilt)):
            self.last_status = "ok"
            return self.last_status

        T_new = self.T.copy()
        T_new[:3, 3] += self._R_ctrl @ np.array([tw.vx, tw.vy, tw.vz]) * dt
        R = T_new[:3, :3]
        if tw.yaw:
            R = kin.axis_angle(np.array([0.0, 0.0, 1.0]), tw.yaw * dt) @ R
        if tw.tilt:
            R = R @ kin.axis_angle(np.array([0.0, 1.0, 0.0]), tw.tilt * dt)
        T_new[:3, :3] = R
        q = kin.solve_ik(self.chain, T_new, [self.q_cmd], iters=80)
        if q is None:
            self.last_status = "workspace limit"
            return self.last_status
        max_step = self.max_joint_speed * dt
        dq = q - self.q_cmd
        if float(np.max(np.abs(dq))) > max_step:
            self.q_cmd = self.q_cmd + np.clip(dq, -max_step, max_step)
            self.T = self.chain.fk(self.q_cmd)
            self.last_status = "joint speed limit"
        else:
            self.q_cmd = q
            self.T = T_new
            self.last_status = "ok"
        return self.last_status
