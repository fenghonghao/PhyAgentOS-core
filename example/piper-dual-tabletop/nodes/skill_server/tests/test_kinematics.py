from pathlib import Path

import numpy as np
import pytest

from tabletop import kinematics as kin

URDF = str(Path(__file__).resolve().parents[3] / "assets" / "robots" / "piper" / "piper_with_gripper.urdf")


@pytest.mark.parametrize("yaw,tilt,xy", [
    (0, 0, (0.3, 0.0)), (30, 0, (0.3, 0.1)), (-120, 0, (0.2, -0.2)), (45, 30, (0.4, 0.2)),
    (-60, 40, (0.3, -0.3)), (170, 15, (0.2, 0.3)), (90, 70, (0.45, 0.45)), (0, 25, (0.5, 0.0)),
])
def test_yaw_tilt_roundtrip(yaw, tilt, xy):
    R = kin.yaw_tilt_to_matrix(yaw, tilt, xy)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
    y, t = kin.matrix_to_yaw_tilt(R, xy)
    assert abs((y - yaw + 180) % 360 - 180) < 1e-6 and abs(t - tilt) < 1e-6
    q = kin.matrix_to_quat(R)
    assert np.allclose(kin.quat_to_matrix(*q), R, atol=1e-9)


def test_topdown_axes_do_not_depend_on_where_the_gripper_is():
    for xy in ((0.3, 0.0), (0.0, 0.4), (-0.2, 0.1)):
        R = kin.yaw_tilt_to_matrix(0, 0, xy)
        assert np.allclose(R[:, 2], [0, 0, -1]) and np.allclose(R[:, 1], [1, 0, 0])


@pytest.mark.parametrize("xy,outward", [((0.5, 0.0), (1, 0)), ((0.0, 0.3), (0, 1)), ((-0.2, 0.0), (-1, 0))])
def test_tilt_always_leans_the_fingers_away_from_the_base(xy, outward):
    a = kin.yaw_tilt_to_matrix(33, 30, xy)[:, 2]
    assert np.allclose(a, [0.5 * outward[0], 0.5 * outward[1], -np.cos(np.radians(30))])
    assert np.allclose(a, kin.approach_axis(xy, 30))


def test_tilt_is_independent_of_the_closing_line():
    leans = [kin.yaw_tilt_to_matrix(yaw, 25, (0.4, 0.1))[:, 2] for yaw in (0, 40, 90, 135, -70)]
    assert all(np.allclose(a, leans[0]) for a in leans)


def test_yaw_is_the_closing_line_seen_from_above_at_any_tilt():
    for yaw in (0, 40, 90, 135, -70):
        c = kin.yaw_tilt_to_matrix(yaw, 40, (0.4, 0.1))[:, 1]
        assert abs(np.degrees(np.arctan2(c[1], c[0])) - yaw) < 1e-9


def test_half_a_turn_is_the_same_grasp_at_any_tilt():
    R, flipped = (kin.yaw_tilt_to_matrix(yaw, 30, (0.4, 0.1)) for yaw in (20, 200))
    assert np.allclose(R[:, 2], flipped[:, 2])                  # the fingers point the same way
    assert np.allclose(R[:, 1], -flipped[:, 1])                 # and swap places on the same line


def test_piper_chain_fk():
    chain = kin.Chain(URDF)
    assert chain.joint_names == [f"joint{i}" for i in range(1, 7)]
    T = chain.fk([0, 0.0383, -0.0708, 0, 0.008, 0])  # sim initial pose
    # tcp in front of the base and above the table
    assert T[0, 3] > 0.05 and T[2, 3] > 0.1
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)


def test_analytic_jacobian_matches_finite_difference():
    chain = kin.Chain(URDF)
    q = np.array([0.2, 1.1, -1.0, 0.3, 0.7, -0.4])
    T, J = chain.fk_jacobian(q)
    assert np.allclose(T, chain.fk(q))
    for k in range(6):
        dq = np.zeros(6); dq[k] = 1e-6
        T2 = chain.fk(q + dq)
        assert np.allclose((T2[:3, 3] - T[:3, 3]) / 1e-6, J[:3, k], atol=1e-4)
        assert np.allclose(kin.rot_log(T2[:3, :3] @ T[:3, :3].T) / 1e-6, J[3:, k], atol=1e-4)


def test_ik_reaches_sampled_pose():
    chain = kin.Chain(URDF)
    q_true = np.array([-0.15, 1.6, -1.24, 0.07, 1.05, 0.85])
    T = chain.fk(q_true)
    import time
    t0 = time.time()
    q = kin.solve_ik(chain, T, [[0, 0.5, -0.02, 0, 0, 0], [0, 1.6, -1.24, 0, 1.05, 0]])
    assert q is not None and np.allclose(chain.fk(q)[:3, 3], T[:3, 3], atol=1e-3)
    assert time.time() - t0 < 1.0


def test_calibration_config_forms_agree():
    import math as _m
    from tabletop.dora_backend import ArmConfig, base_pose
    from tabletop.perception import CameraConfig, LINK6_TO_TCP
    T = kin.pose_matrix([0.1, 0.3, 0.0], [0.0, 0.0, _m.radians(30)])
    a = ArmConfig(name="left", urdf=URDF, home_joints=[0] * 6, base_xyz=[0.1, 0.3, 0.0], base_rpy_deg=[0, 0, 30])
    b = ArmConfig(name="left", urdf=URDF, home_joints=[0] * 6, base_T=T.tolist())
    c = ArmConfig(name="left", urdf=URDF, home_joints=[0] * 6, base_T=np.linalg.inv(T).tolist(), base_T_inverse=True)
    assert np.allclose(base_pose(a), T) and np.allclose(base_pose(b), T) and np.allclose(base_pose(c), T)
    cam_tcp = kin.pose_matrix([0.0, -0.05, -0.08], [0.1, 0.0, 0.2])
    w1 = CameraConfig(name="w", K=np.eye(3).tolist(), mount="wrist", arm="left", T_matrix=cam_tcp.tolist())
    w2 = CameraConfig(name="w", K=np.eye(3).tolist(), mount="wrist", arm="left", wrist_frame="link6",
                      T_matrix=(LINK6_TO_TCP @ cam_tcp).tolist())
    assert np.allclose(w1.T(), cam_tcp) and np.allclose(w2.T(), cam_tcp)
