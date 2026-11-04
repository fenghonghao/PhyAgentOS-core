import math

import numpy as np

from tabletop import geometry as geo


def rect_points(cx, cy, l, w, ang, n=400):
    rng = np.random.default_rng(0)
    pts = rng.uniform([-l / 2, -w / 2], [l / 2, w / 2], size=(n, 2))
    pts = np.vstack([pts, [[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]]])
    c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + [cx, cy]


def test_world_map_converts_depth_pixels_to_world_points():
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    depth = np.array([[2.0, 0.0], [np.nan, 2.0]], np.float32)
    w = geo.world_map(K, T, depth)
    assert w.shape == (2, 2, 3)
    assert np.allclose(w[0, 0], [0.99, 1.99, 5.0])         # pixel centre (0.5, 0.5), 2 m away
    assert np.allclose(w[1, 1], [1.01, 2.01, 5.0])
    assert np.isnan(w[0, 1]).all() and np.isnan(w[1, 0]).all()  # 0 and NaN depth are invalid


def test_world_map_applies_the_camera_rotation():
    K = np.array([[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]])
    T = np.diag([1.0, -1.0, -1.0, 1.0])                     # looking straight down from z = 1
    T[2, 3] = 1.0
    w = geo.world_map(K, T, np.ones((2, 2), np.float32))
    assert np.allclose(w[0, 1], [0.005, 0.005, 0.0])        # col 1, row 0: x = +0.5/100, y flips


def test_norm90():
    assert geo.norm90(100) == -80
    assert geo.norm90(-90) == -90
    assert geo.norm90(90) == -90
    assert geo.norm90(45) == 45


def test_min_area_rect_recovers_rotated_rectangle():
    r = geo.min_area_rect(rect_points(0.4, -0.1, 0.08, 0.03, 30))
    assert abs(r.length - 0.08) < 1e-6 and abs(r.width - 0.03) < 1e-6
    assert abs(r.long_axis_deg - 30) < 1e-4
    assert np.allclose(r.center, [0.4, -0.1], atol=1e-6)


def test_grasp_geometry_and_too_wide():
    g = geo.grasp_geometry([0.4, 0.0, 0.02], 0.04, [0.05, 0.03, 0.04], 30, max_opening_m=0.07)
    assert np.allclose(g.grasp_xyz, [0.4, 0.0, 0.02])  # depth = min(h/2, 3cm) = 2cm
    assert abs(g.open_width_m - 0.06) < 1e-9
    assert g.grasp_yaw_deg == -60 and not g.too_wide
    g2 = geo.grasp_geometry([0, 0, 0.05], 0.10, [0.08, 0.06, 0.10], 0, max_opening_m=0.07)
    assert g2.too_wide and g2.open_width_m == 0.07 and abs(g2.grasp_xyz[2] - 0.07) < 1e-9


def test_grasp_height_stays_above_the_fingertip_floor():
    # 2 cm tall tile on the table: half its height would put the TCP at 1 cm
    g = geo.grasp_geometry([0.3, 0.0, 0.01], 0.02, [0.04, 0.03, 0.02], 0, max_opening_m=0.10, tcp_floor_z=0.0155)
    assert abs(g.grasp_xyz[2] - 0.0155) < 1e-9
    # a taller object is not affected
    g2 = geo.grasp_geometry([0.3, 0.0, 0.03], 0.06, [0.04, 0.03, 0.06], 0, max_opening_m=0.10, tcp_floor_z=0.0155)
    assert abs(g2.grasp_xyz[2] - 0.03) < 1e-9


def test_project_backproject_roundtrip():
    from tabletop.fake import intrinsics, look_at
    K, T = intrinsics(320, 240, 70), look_at([-0.3, 0, 0.8], [0.4, 0, 0])
    p = np.array([[0.35, 0.05, 0.02]])
    uv, z = geo.project(K, T, p)
    origin, d = geo.pixel_rays(K, T, uv)
    t = (p[0] - origin) @ d[0]
    assert np.allclose(origin + d[0] * t, p[0], atol=1e-9)
