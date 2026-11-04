import threading
import time

import numpy as np

from tabletop.perception import CameraConfig, ColorSegmenter, DoraPerception, label_components, rgb_to_hsv
from tabletop.primitives import Robot, RobotConfig
from tests.conftest import make_scene

COLORS = {"red block": (220, 30, 30), "blue block": (30, 60, 220), "green block": (40, 200, 60),
          "yellow bottle": (230, 210, 40)}


def colored(frame):
    ids, names = frame.meta["ids"], frame.meta["names"]
    rgb = np.full(ids.shape + (3,), 200, np.uint8)  # grey table
    for i, n in enumerate(names):
        rgb[ids == i] = COLORS.get(n, (90, 90, 90))
    return rgb


def test_hsv_and_components():
    h, s, v = rgb_to_hsv(np.array([[[255, 0, 0], [0, 0, 255]]], np.uint8))
    assert abs(h[0, 0]) < 1e-3 and abs(h[0, 1] - 240) < 1e-3 and s.min() > 0.99
    m = np.zeros((6, 8), bool); m[1:3, 1:3] = True; m[4:6, 5:8] = True
    labels, n = label_components(m)
    assert n == 2 and labels[1, 1] != labels[4, 6]


def test_color_segmenter_finds_red_block():
    w = make_scene()
    f = w.capture("head")
    dets = ColorSegmenter(min_area_px=20).segment(colored(f), "红色方块")
    assert dets and (dets[0].mask & (f.meta["ids"] == f.meta["names"].index("red block"))).sum() > 0.9 * dets[0].mask.sum()


def test_dora_perception_capture_waits_for_fresh_frame_and_locates():
    w = make_scene()
    f = w.capture("head")
    cam = CameraConfig(name="head", K=f.K.tolist(), mount="fixed")
    T = f.T_wc
    p = DoraPerception([cam], ColorSegmenter(min_area_px=20))
    p.cfg["head"].T = lambda: T  # calibrated pose of the fake head camera
    rgb, depth = colored(f), f.depth
    stop = threading.Event()

    def stream():
        while not stop.is_set():
            p.on_depth("head", depth)
            p.on_color("head", rgb)
            time.sleep(0.03)

    th = threading.Thread(target=stream, daemon=True); th.start()
    try:
        time.sleep(0.1)
        t0 = time.monotonic()
        fr = p.capture("head")
        assert fr.meta["stale"] is False and fr.depth is not None
        r = Robot(w, p, RobotConfig())
        s = r.locate("red block")[0]
        assert np.allclose(s.center_xyz[:2], w.boxes["red block"].center_xy, atol=0.01)
    finally:
        stop.set(); th.join()
    # stream stopped: capture does not block, returns the last frame marked stale
    time.sleep(0.6)
    assert p.capture("head").meta["stale"] is True


def test_depth_scale_from_the_calibration_is_applied_to_captured_depth():
    K = [[500.0, 0.0, 2.0], [0.0, 500.0, 2.0], [0.0, 0.0, 1.0]]
    # stale_after_s=0: capture returns the frame it has instead of waiting for a newer one
    cam = CameraConfig(name="head", K=K, mount="fixed", depth_scale=1.01, stale_after_s=0.0)
    p = DoraPerception([cam], ColorSegmenter(min_area_px=20))
    depth = np.full((4, 4), 0.8, np.float32)
    depth[0, 0] = 0.0  # invalid stays invalid
    p.on_depth("head", depth)
    p.on_color("head", np.zeros((4, 4, 3), np.uint8))
    fr = p.capture("head")
    assert np.isclose(fr.depth[1, 1], 0.808) and fr.depth[0, 0] == 0.0
    assert depth[1, 1] == np.float32(0.8)  # the incoming buffer is not modified


def test_wrist_frame_before_first_joint_state_does_not_block_the_dora_loop():
    """on_color runs on the Dora main thread, which is also the only one that delivers joint
    state: waiting there for the wrist pose would deadlock the node at start-up."""
    from pathlib import Path

    from tabletop.dora_backend import Bus, DoraArmBackend, MotionConfig, load_arm_configs

    root = Path(__file__).resolve().parents[3]
    arms = load_arm_configs({"left": {"urdf": "assets/robots/piper/piper_with_gripper.urdf",
                                      "base_xyz": [0.0, 0.0, 0.0], "base_rpy_deg": [0.0, 0.0, 0.0],
                                      "home_joints": [0.0, 0.5, -0.02, 0.0, 0.0, 0.0], "max_opening_m": 0.07}}, root)
    backend = DoraArmBackend(arms, MotionConfig(), Bus())
    cam = CameraConfig(name="wrist", K=[[300.0, 0, 160], [0, 300.0, 120], [0, 0, 1]], mount="wrist", arm="left")
    per = DoraPerception([cam], ColorSegmenter(), backend, head="head")
    rgb = np.zeros((240, 320, 3), np.uint8)
    t = threading.Thread(target=per.on_color, args=("wrist", rgb), daemon=True)
    t.start()
    t.join(2.0)                                   # test-side guard only
    assert not t.is_alive(), "on_color blocked waiting for the first joint state"
    assert "wrist" not in per.cameras()           # no pose yet: the frame is not kept
    backend.on_joint_state([f"joint{i}" for i in range(1, 7)] + ["gripper"], [0.0] * 7)
    per.on_color("wrist", rgb)
    assert "wrist" in per.cameras()
