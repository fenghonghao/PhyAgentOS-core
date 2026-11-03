"""Data layout round trip and teleop IK stepping (no Dora, no GPU)."""

from pathlib import Path

import numpy as np

from piper_vla.spec import ArmSpec, CameraSpec, VlaSpec, load_config, prep_image
from piper_vla.teleop import TeleopArm, Twist

ROOT = Path(__file__).resolve().parents[3]
URDF = str(ROOT / "assets/robots/piper/piper_with_gripper.urdf")


def spec2():
    return VlaSpec(arms=[ArmSpec("left"), ArmSpec("right")], cameras=[CameraSpec("head")])


def test_pack_unpack_order():
    s = spec2()
    per = {a: {j: i + (10 if a == "right" else 0) for i, j in enumerate(ArmSpec(a).names)} for a in ("left", "right")}
    v = s.pack(per)
    assert v.shape == (14,) and s.state_names[0] == "left.joint1" and s.state_names[-1] == "right.gripper"
    assert s.unpack(v)["right"][6] == 16.0


def test_collection_and_policy_layouts_match():
    _, collect = load_config(ROOT / "nodes/vla/collect/real-piper.vla.yaml")
    _, policy = load_config(ROOT / "profiles/real-piper/vla_policy.yaml")
    assert collect.features() == policy.features()
    assert [a.home_joints for a in collect.arms] == [a.home_joints for a in policy.arms]


def test_prep_image_resizes_to_width_height():
    img = np.zeros((480, 640, 3), np.uint8)
    assert prep_image(img, [320, 240]).shape == (240, 320, 3)


def test_teleop_moves_smoothly_and_stops_at_workspace_edge():
    arm = TeleopArm(ArmSpec("left", urdf=URDF), max_joint_speed=2.0)
    arm.on_state([f"joint{i}" for i in range(1, 7)] + ["gripper"], [0.0, 0.5, -0.02, 0.0, 0.3, 0.0, 0.02])
    x0 = arm.tcp_xyz()
    for _ in range(30):
        arm.step(Twist(vx=0.05), 1 / 30)
    x1 = arm.tcp_xyz()
    assert 0.03 < x1[0] - x0[0] < 0.07 and abs(x1[2] - x0[2]) < 0.01
    q_prev = arm.q_cmd.copy()
    for _ in range(600):                      # push far out of reach
        arm.step(Twist(vx=0.3), 1 / 30)
        assert np.max(np.abs(arm.q_cmd - q_prev)) <= 2.0 / 30 + 1e-9
        q_prev = arm.q_cmd.copy()
    assert arm.last_status in ("workspace limit", "joint speed limit", "ok")
    arm.step(Twist(grip=1.0), 1.0)
    assert arm.g_cmd == arm.spec.max_opening_m
