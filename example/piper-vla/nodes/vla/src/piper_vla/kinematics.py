"""URDF forward kinematics and the yaw/tilt <-> TCP orientation convention.

PIPER ``piper_with_gripper.urdf``: the ``tcp`` frame sits 0.135 m along link6 z, the fingers
(joint7/joint8) slide along link6 y. Hence for the TCP frame:
  z_tcp = approach axis (out of the fingers), y_tcp = closing direction, x_tcp = y × z.

Convention shared with primitives/skills:
  closing c = (cos yaw, sin yaw, 0)
  approach a = cos(tilt)·(0, 0, -1) + sin(tilt)·(-sin yaw, cos yaw, 0)
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def transform(xyz, R) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def pose_matrix(xyz, rpy) -> np.ndarray:
    return transform(np.asarray(xyz, dtype=float), rpy_matrix(*rpy))


@dataclass
class _Joint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float = -math.pi
    upper: float = math.pi


class Chain:
    """Serial chain from ``base`` to ``tip`` parsed from a URDF file."""

    def __init__(self, urdf_path: str, base: str = "arm_base", tip: str = "tcp") -> None:
        root = ET.parse(urdf_path).getroot()
        by_child: dict[str, tuple[str, _Joint]] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
            lim = j.find("limit")
            lower = float(lim.get("lower", -math.pi)) if lim is not None else -math.pi
            upper = float(lim.get("upper", math.pi)) if lim is not None else math.pi
            by_child[j.find("child").get("link")] = (j.find("parent").get("link"),
                                                     _Joint(j.get("name"), j.get("type"), pose_matrix(xyz, rpy), axis, lower, upper))
        joints: list[_Joint] = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"no chain from {base} to {tip} in {urdf_path}")
            parent, joint = by_child[link]
            joints.append(joint)
            link = parent
        self.joints = joints[::-1]
        self.joint_names = [j.name for j in self.joints if j.kind in ("revolute", "continuous", "prismatic")]
        self.limits = [(j.lower, j.upper) for j in self.joints if j.kind in ("revolute", "continuous", "prismatic")]

    def fk_jacobian(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Tip pose and the 6xN geometric Jacobian [linear; angular] in the base frame."""
        q = list(q)
        T = np.eye(4)
        axes, origins, kinds = [], [], []
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind in ("revolute", "continuous"):
                axes.append(T[:3, :3] @ (j.axis / np.linalg.norm(j.axis)))
                origins.append(T[:3, 3].copy())
                kinds.append("r")
                T = T @ transform([0, 0, 0], axis_angle(j.axis, q[i]))
                i += 1
            elif j.kind == "prismatic":
                axes.append(T[:3, :3] @ (j.axis / np.linalg.norm(j.axis)))
                origins.append(T[:3, 3].copy())
                kinds.append("p")
                T = T @ transform(j.axis * q[i], np.eye(3))
                i += 1
        p_tip = T[:3, 3]
        J = np.zeros((6, len(axes)))
        for k, (a, o, kind) in enumerate(zip(axes, origins, kinds)):
            if kind == "r":
                J[:3, k] = np.cross(a, p_tip - o)
                J[3:, k] = a
            else:
                J[:3, k] = a
        return T, J

    def fk(self, q) -> np.ndarray:
        q = list(q)
        if len(q) != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} joint values, got {len(q)}")
        T = np.eye(4)
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.kind in ("revolute", "continuous"):
                T = T @ transform([0, 0, 0], axis_angle(j.axis, q[i]))
                i += 1
            elif j.kind == "prismatic":
                T = T @ transform(j.axis * q[i], np.eye(3))
                i += 1
        return T


# --------------------------------------------------------------------------- orientation convention


def yaw_tilt_to_matrix(yaw_deg: float, tilt_deg: float) -> np.ndarray:
    y, t = math.radians(yaw_deg), math.radians(tilt_deg)
    c = np.array([math.cos(y), math.sin(y), 0.0])
    h = np.array([-math.sin(y), math.cos(y), 0.0])
    a = math.cos(t) * np.array([0.0, 0.0, -1.0]) + math.sin(t) * h
    x = np.cross(c, a)
    return np.column_stack([x, c, a])


def matrix_to_yaw_tilt(R: np.ndarray) -> tuple[float, float]:
    c, a = R[:, 1], R[:, 2]
    yaw = math.atan2(c[1], c[0])
    h = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    tilt = math.atan2(float(a @ h), float(-a[2]))
    return math.degrees(yaw), math.degrees(tilt)


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """(qx, qy, qz, qw)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw, qx, qy, qz = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw, qx, qy, qz = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return qx / n, qy / n, qz / n, qw / n


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def slerp_matrix(R0: np.ndarray, R1: np.ndarray, f: float) -> np.ndarray:
    """Spherical interpolation between two rotations (shortest path)."""
    q0 = np.array(matrix_to_quat(R0))
    q1 = np.array(matrix_to_quat(R1))
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - f) * th) * q0 + math.sin(f * th) * q1) / math.sin(th)
    return quat_to_matrix(*q)


PIPER_LIMITS = [(-2.6179, 2.6179), (0.0, 3.14), (-2.967, 0.0), (-1.745, 1.745), (-1.22, 1.22), (-2.09439, 2.09439)]


def rot_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if math.pi - th < 1e-6:
        # near pi: axis from the largest diagonal entry
        i = int(np.argmax(np.diag(R)))
        v = R[:, i] + np.eye(3)[i]
        return th * v / np.linalg.norm(v)
    return th / (2 * math.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def solve_ik(chain: "Chain", T_target: np.ndarray, seeds, limits=None, iters: int = 300,
             pos_tol: float = 5e-4, rot_tol: float = 5e-3) -> np.ndarray | None:
    """Damped least squares with joint limits, tried from several seeds; returns joints or None."""
    limits = limits or chain.limits
    lo = np.array([l for l, _ in limits])
    hi = np.array([h for _, h in limits])
    for seed in seeds:
        q = np.clip(np.asarray(seed, dtype=float), lo, hi)
        for _ in range(iters):
            T, J = chain.fk_jacobian(q)
            e = np.concatenate([T_target[:3, 3] - T[:3, 3], 0.3 * rot_log(T_target[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) / 0.3 < rot_tol:
                return q
            J = J.copy()
            J[3:, :] *= 0.3
            lam = 0.02
            step = J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
            q = np.clip(q + np.clip(step, -0.3, 0.3), lo, hi)
    return None
