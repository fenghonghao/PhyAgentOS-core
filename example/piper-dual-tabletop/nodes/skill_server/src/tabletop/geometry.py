"""Pure geometry used by ``locate``: camera rays, footprints, visual hull, grasp geometry.

Conventions
-----------
* World frame: table frame, z up, metres.
* Camera: pinhole intrinsics ``K`` (3x3); ``T_wc`` (4x4) maps camera coordinates to world.
  Camera optical frame is the usual one: x right, y down, z forward.
* Angles in degrees. ``long_axis_deg`` is the heading of the footprint's long side,
  measured from +x towards +y, normalized to [-90, 90).
* ``grasp_yaw_deg`` is the heading of the gripper *closing direction* on the table,
  i.e. across the short side (= long axis + 90), normalized to [-90, 90).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def norm90(angle_deg: float) -> float:
    """Normalize a line direction (period 180) to [-90, 90)."""
    a = (angle_deg + 90.0) % 180.0 - 90.0
    return float(a)


# --------------------------------------------------------------------------- camera


def pixel_rays(K: np.ndarray, T_wc: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ray origin (shared) and unit directions for pixel coordinates ``uv`` (N,2)."""
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    ones = np.ones((uv.shape[0], 1))
    d_cam = (np.linalg.inv(K) @ np.hstack([uv, ones]).T).T
    d_world = (T_wc[:3, :3] @ d_cam.T).T
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    return T_wc[:3, 3].copy(), d_world


def project(K: np.ndarray, T_wc: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N,3) to pixels. Returns (uv (N,2), depth along optical axis (N,))."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    T_cw = np.linalg.inv(T_wc)
    p_cam = (T_cw[:3, :3] @ pts.T).T + T_cw[:3, 3]
    z = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = (K @ p_cam.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, z


def backproject_depth(K: np.ndarray, T_wc: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points (N,3) for masked pixels with valid depth (metres along optical axis; 0 = invalid)."""
    vs, us = np.nonzero(mask & np.isfinite(depth) & (depth > 0))
    if us.size == 0:
        return np.zeros((0, 3))
    z = depth[vs, us].astype(float)
    uv1 = np.stack([us + 0.5, vs + 0.5, np.ones_like(z)], axis=1)
    p_cam = (np.linalg.inv(K) @ uv1.T).T * z[:, None]
    return (T_wc[:3, :3] @ p_cam.T).T + T_wc[:3, 3]


def world_map(K: np.ndarray, T_wc: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """World coordinates (H, W, 3) of every pixel; NaN where the depth is invalid (0 or NaN)."""
    h, w = depth.shape[:2]
    us, vs = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = np.asarray(depth, dtype=float)
    z = np.where(np.isfinite(z) & (z > 0), z, np.nan)
    rays = np.stack([us, vs, np.ones_like(us)], axis=-1) @ np.linalg.inv(K).T
    return (rays * z[..., None]) @ T_wc[:3, :3].T + T_wc[:3, 3]


def mask_to_plane(K: np.ndarray, T_wc: np.ndarray, mask: np.ndarray, plane_z: float) -> np.ndarray:
    """Intersect the rays of masked pixels with the horizontal plane z = plane_z."""
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return np.zeros((0, 3))
    origin, dirs = pixel_rays(K, T_wc, np.stack([us + 0.5, vs + 0.5], axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (plane_z - origin[2]) / dirs[:, 2]
    ok = np.isfinite(t) & (t > 0)
    return origin + dirs[ok] * t[ok, None]


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    vs, us = np.nonzero(mask)
    if us.size == 0:
        return None
    return np.array([us.mean() + 0.5, vs.mean() + 0.5])


# --------------------------------------------------------------------------- footprint


def convex_hull(xy: np.ndarray) -> np.ndarray:
    """Monotone-chain convex hull of 2D points, counter-clockwise, no repeated end point."""
    pts = np.unique(np.round(np.asarray(xy, dtype=float), 6), axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


@dataclass
class Rect:
    center: np.ndarray  # (2,)
    length: float  # long side
    width: float  # short side
    long_axis_deg: float  # [-90, 90)


def min_area_rect(xy: np.ndarray) -> Rect | None:
    """Minimum-area bounding rectangle of 2D points (rotating calipers over hull edges)."""
    hull = convex_hull(xy)
    if len(hull) == 0:
        return None
    if len(hull) == 1:
        return Rect(hull[0].copy(), 0.0, 0.0, 0.0)
    best: tuple[float, Rect] | None = None
    edges = np.roll(hull, -1, axis=0) - hull
    for ex, ey in edges:
        n = float(np.hypot(ex, ey))
        if n < 1e-12:
            continue
        ux = np.array([ex, ey]) / n
        uy = np.array([-ux[1], ux[0]])
        a = hull @ ux
        b = hull @ uy
        ext_a, ext_b = a.max() - a.min(), b.max() - b.min()
        area = ext_a * ext_b
        if best is None or area < best[0] - 1e-12:
            c = ux * (a.max() + a.min()) / 2 + uy * (b.max() + b.min()) / 2
            if ext_a >= ext_b:
                length, width, axis = ext_a, ext_b, ux
            else:
                length, width, axis = ext_b, ext_a, uy
            ang = norm90(float(np.degrees(np.arctan2(axis[1], axis[0]))))
            best = (area, Rect(c, float(length), float(width), ang))
    return best[1] if best else None


def trim_extremes(pts: np.ndarray, pct: float = 0.5) -> np.ndarray:
    """Drop the outer ``pct`` percent of points along each axis of the first-pass rectangle.

    Gentle on purpose: fused views sample the object unevenly, so a radial filter centred on
    the densest view would cut away parts only a sparser view sees.
    """
    if len(pts) < 20:
        return pts
    rect = min_area_rect(pts[:, :2])
    if rect is None:
        return pts
    ang = math.radians(rect.long_axis_deg)
    ux, uy = np.array([math.cos(ang), math.sin(ang)]), np.array([-math.sin(ang), math.cos(ang)])
    a, b = pts[:, :2] @ ux, pts[:, :2] @ uy
    lo_a, hi_a = np.percentile(a, [pct, 100 - pct])
    lo_b, hi_b = np.percentile(b, [pct, 100 - pct])
    keep = (a >= lo_a) & (a <= hi_a) & (b >= lo_b) & (b <= hi_b)
    return pts[keep]


def touches_border(mask: np.ndarray) -> bool:
    """True when a mask reaches the image edge (the object is cut off in that view)."""
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


@dataclass
class Footprint:
    center_xyz: list[float]
    top_z: float
    size: list[float]  # [length, width, height]
    long_axis_deg: float


def footprint_from_points(
    points: np.ndarray,
    table_z: float,
    min_above_table: float = 0.004,
    top_percentile: float = 95.0,
) -> Footprint | None:
    """Footprint rectangle and height of an object from its world points (depth or hull)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = pts[pts[:, 2] > table_z + min_above_table]
    if len(pts) < 3:
        return None
    pts = trim_extremes(pts)
    rect = min_area_rect(pts[:, :2])
    if rect is None:
        return None
    top = float(np.percentile(pts[:, 2], top_percentile))
    height = max(top - table_z, 0.0)
    center = [float(rect.center[0]), float(rect.center[1]), table_z + height / 2]
    return Footprint(center, top, [rect.length, rect.width, height], rect.long_axis_deg)


def footprint_from_plane_points(points: np.ndarray, table_z: float) -> Footprint | None:
    """Single-view, no-depth fallback: footprint of ray/plane hits; height unknown (0)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        return None
    rect = min_area_rect(pts[:, :2])
    if rect is None:
        return None
    return Footprint([float(rect.center[0]), float(rect.center[1]), table_z], table_z, [rect.length, rect.width, 0.0], rect.long_axis_deg)


# --------------------------------------------------------------------------- grasp


@dataclass
class GraspGeometry:
    grasp_xyz: list[float]
    open_width_m: float
    grasp_yaw_deg: float
    too_wide: bool


def grasp_geometry(
    center_xyz: list[float],
    top_z: float,
    size: list[float],
    long_axis_deg: float,
    max_opening_m: float,
    grasp_depth_max_m: float = 0.03,
    open_margin_m: float = 0.03,
    tcp_floor_z: float | None = None,
) -> GraspGeometry:
    """Top-down grasp across the short side.

    Depth below the top is min(height/2, grasp_depth_max_m), but the TCP stays at or above
    ``tcp_floor_z`` so the fingertips clear the table; opening is short side + margin,
    capped at the gripper's maximum; ``too_wide`` when short side + margin does not fit.
    """
    _, width, height = size
    depth = min(height / 2.0, grasp_depth_max_m) if height > 0 else 0.0
    want = width + open_margin_m
    too_wide = want > max_opening_m
    z = float(top_z - depth)
    if tcp_floor_z is not None:
        z = max(z, float(tcp_floor_z))
    return GraspGeometry(
        grasp_xyz=[float(center_xyz[0]), float(center_xyz[1]), z],
        open_width_m=float(min(want, max_opening_m)),
        grasp_yaw_deg=norm90(long_axis_deg + 90.0),
        too_wide=bool(too_wide),
    )
