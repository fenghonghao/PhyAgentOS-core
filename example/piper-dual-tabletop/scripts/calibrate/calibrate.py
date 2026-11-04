#!/usr/bin/env python3
"""Calibrate the head and wrist cameras with the printed ChArUco sheet taped flat on the table.

  calibrate.py check RUN                        detect the sheet in both cameras (no motion)
  calibrate.py collect RUN --square-mm 36       walk the arm, photograph with both cameras
  calibrate.py solve RUN [--merge OTHER_RUN]    print and save the skill_server.yaml entries
  calibrate.py point RUN                        hover the TCP over corners the head camera located

collect options: --center X Y (rough sheet centre in the base frame, default: from the head pose in
skill_server.yaml) and --head-only (the sheet or the head camera moved; the wrist mount did not).

RUN is a directory (under ~/.PhyAgentOS/calibration unless absolute). Use the venv built from
requirements.txt. Measure a printed square with a ruler: printers scale pages. The arm moves on its
own at low speed, so keep people clear. Nothing here disables or resets the motors. Recalibrate the
head if the camera or its pole moved. To tell, compare the sheet's corners in the head image, not the
background: the booth panels and boxes do move. --merge combines runs with the same camera and sheet.

Method. The board pose comes from the J6 axis: with the gripper straight down and only J6 turning,
every board point seen at a fixed pixel moves on a circle around the point where the axis meets the
board. That needs neither the camera mount nor a tool model, and it is measured in the posture used
for grasping. Classic hand-eye over wide wrist poses is not used here: it scattered the board by
10 mm on this arm, because the reported pose is less exact away from J4 = 0.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares

from make_board import SQUARE_MM, SQUARES, make_board

SKILL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL / "nodes" / "skill_server" / "src"))
from tabletop import kinematics as kin  # noqa: E402

PROFILE = SKILL / "profiles" / "real-piper"
URDF = str(SKILL / "assets" / "robots" / "piper" / "piper_with_gripper_real.urdf")
ORBBEC = (Path(os.environ.get("CAMERA_NODES_DIR", Path.home() / ".PhyAgentOS" / "camera-nodes"))
          / "orbbec_camera-v1.1.0-ubuntu20.04-x86_64" / "orbbec_camera")

# Intrinsics at 640x480. Head: factory values. Wrist: fitted from 18 views of the board (the factory
# values are about 1% off); the first has no distortion, which is the model the skill uses, the
# second is only used to measure on the board. Re-fit if a camera is swapped.
K_HEAD = np.array([[516.72, 0.0, 320.18], [0.0, 516.59, 242.82], [0.0, 0.0, 1.0]])
K_WRIST = np.array([[524.50, 0.0, 316.00], [0.0, 523.30, 242.30], [0.0, 0.0, 1.0]])
K_WRIST_D = np.array([[520.3, 0.0, 315.7], [0.0, 519.6, 240.1], [0.0, 0.0, 1.0]])
D_WRIST_D = np.array([0.076, -0.1729, -0.0031, -0.0012, 0.0])

LINK6_TO_TCP_M = 0.13503
TCP_TOUCH_Z_M = 0.0125       # TCP height when the closed fingertips touch the table (measured)
TIP_LENGTH_M = LINK6_TO_TCP_M + TCP_TOUCH_Z_M
MIN_TIP_Z_M = 0.04           # lowest fingertip height while walking
MIN_FLANGE_Z_M = 0.11
MIN_Y_M = -0.05              # right side panel
MAX_TIP_RADIUS_M = 0.50
LIMITS_DEG = [(-20, 85), (5, 170), (-165, -5), (-60, 60), (-69.5, 69.5), (-60, 60)]  # narrow J1, J4, J6: cables
EFFORT_LIMIT_NM = [6.0, 20.0, 15.0, 5.0, 5.0, 5.0]
SPEED_PERCENT = 20
SPIN_OFFSETS_M = [(-0.046, -0.069), (0.054, -0.079), (-0.056, 0.031), (0.014, -0.009)]  # around the sheet centre
SPIN_ANGLES_DEG = [-55.0, -20.0, 20.0, 55.0]
SPIN_TCP_Z_M = (0.065, 0.055, 0.045)      # first reachable height is used
# tilted views like the skill's observe poses: distance back from the centre, TCP z, tilt, J6
TILT_VIEWS = [(0.13, 0.20, -30, 0), (0.13, 0.20, -30, 35), (0.13, 0.20, -30, -35),
              (0.17, 0.25, -45, 0), (0.17, 0.25, -45, 30), (0.17, 0.25, -45, -30)]
POINT_OFFSETS_M = [(-0.04, -0.06), (0.04, -0.04), (-0.05, 0.05), (0.0, 0.01)]

BOARD = make_board()
OBJ = BOARD.getChessboardCorners().astype(np.float64)  # nominal metres, before the print scale
MARKER_OBJ = BOARD.getObjPoints()
MARKER_IDS = BOARD.getIds().ravel().tolist()
BOARD_CENTER = np.array([SQUARES[0] * SQUARE_MM / 2000, SQUARES[1] * SQUARE_MM / 2000, 0.0])
_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR_FINE = cv2.aruco.CharucoDetector(BOARD, cv2.aruco.CharucoParameters(), _params)
DETECTOR = cv2.aruco.CharucoDetector(BOARD)
LINK6 = kin.Chain(URDF, "arm_base", "link6")
LIMITS_RAD = [(math.radians(a), math.radians(b)) for a, b in LIMITS_DEG]
SEEDS = [[0.6, 1.7, -1.1, 0.0, 1.0, 0.0], [0.6, 1.2, -1.0, 0.0, 0.8, 0.0], [0.8, 0.9, -0.7, 0.0, 0.5, 0.0]]


class Stop(Exception):
    pass


def run_dir(name: str) -> Path:
    p = Path(name) if os.path.isabs(name) else Path.home() / ".PhyAgentOS" / "calibration" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_T(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    T[:3, 3] = np.asarray(tvec, dtype=float).ravel()
    return T


def to_vec(T: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def snapshot(camera: str, out: Path, all_streams: bool = False) -> None:
    """One frame through the same adapter and config as the skill. Also writes <stem>_color.jpg and
    <stem>_depth.png (uint16 mm) with all_streams."""
    cmd = [str(ORBBEC), "snapshot", "--config", str(PROFILE / f"camera_{camera}.yaml"), "-o", str(out)]
    if all_streams:
        cmd.append("--all-streams")
    for attempt in range(4):
        try:
            subprocess.run(cmd, cwd=out.parent, check=True, capture_output=True, timeout=90)
            return
        except subprocess.SubprocessError as e:
            if attempt == 3:
                raise Stop(f"{camera} camera snapshot failed (is another process using it?): {e}") from e
            time.sleep(5)


def corners_of(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Interior corners: nominal board metres (N,3) and pixels (N,2), or empty arrays."""
    cc, ci, _, _ = DETECTOR.detectBoard(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    if ci is None:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return OBJ[ci.ravel()], cc.reshape(-1, 2).astype(np.float64)


def all_points_of(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Interior corners plus marker corners, for close views that see few squares."""
    cc, ci, mc, mi = DETECTOR_FINE.detectBoard(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
    obj, px = [], []
    if ci is not None and len(ci):
        obj.append(OBJ[ci.ravel()])
        px.append(cc.reshape(-1, 2))
    if mi is not None:
        for corners, i in zip(mc, mi.ravel()):
            obj.append(np.asarray(MARKER_OBJ[MARKER_IDS.index(int(i))], dtype=float))
            px.append(corners.reshape(-1, 2))
    if not obj:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return np.vstack(obj), np.vstack(px).astype(np.float64)


# ------------------------------------------------------------------------------------ arm


class Arm:
    def __init__(self) -> None:
        from piper_sdk import C_PiperInterface_V2

        self.p = C_PiperInterface_V2("can0")
        self.p.ConnectPort(piper_init=False)
        time.sleep(0.4)
        self.peak = [0.0] * 6
        self.gripper_mm = float(self.p.GetArmGripperMsgs().gripper_state.grippers_angle) / 1000
        self.speed = SPEED_PERCENT
        self.check()

    def check(self) -> None:
        s = self.p.GetArmStatus().arm_status
        if getattr(s.ctrl_mode, "value", s.ctrl_mode) != 1:
            raise Stop(f"the arm is not in CAN control: {s.ctrl_mode}")
        if getattr(s.arm_status, "value", s.arm_status) != 0 or s.err_code != 0:
            raise Stop(f"arm status {s.arm_status} err {s.err_code}")

    def joints_deg(self) -> list[float]:
        j = self.p.GetArmJointMsgs().joint_state
        return [getattr(j, f"joint_{i}") / 1000 for i in range(1, 7)]

    def firmware_pose(self) -> np.ndarray:
        """T_base_link6 as reported by the firmware (R = Rz * Ry * Rx)."""
        e = self.p.GetArmEndPoseMsgs().end_pose
        rx, ry, rz = (math.radians(v / 1000) for v in (e.RX_axis, e.RY_axis, e.RZ_axis))
        cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
        T = np.eye(4)
        T[:3, :3] = [[cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                     [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                     [-sy, sx * cy, cx * cy]]
        T[:3, 3] = [e.X_axis / 1e6, e.Y_axis / 1e6, e.Z_axis / 1e6]
        return T

    def _send(self, target: list[int]) -> None:
        self.p.MotionCtrl_2(0x01, 0x01, self.speed, 0x00)
        self.p.JointCtrl(*target)

    def goto(self, q_deg, tol: float = 0.25, limit_s: float = 40.0) -> None:
        target = [int(round(v * 1000)) for v in q_deg]
        t0 = time.monotonic()
        while True:
            self._send(target)
            time.sleep(0.02)
            self.check()
            h = self.p.GetArmHighSpdInfoMsgs()
            for i in range(6):
                eff = abs(getattr(h, f"motor_{i + 1}").effort / 1000)
                self.peak[i] = max(self.peak[i], eff)
                if eff > EFFORT_LIMIT_NM[i]:
                    raise Stop(f"joint {i + 1} effort {eff:.1f} N*m is over {EFFORT_LIMIT_NM[i]}")
            if max(abs(a - b) for a, b in zip(self.joints_deg(), q_deg)) < tol:
                break
            if time.monotonic() - t0 > limit_s:
                raise Stop("target not reached")
        t1 = time.monotonic()
        while time.monotonic() - t1 < 0.8:  # settle
            self._send(target)
            time.sleep(0.02)
            self.check()

    def gripper(self, opening_mm: float) -> None:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0:
            self.p.GripperCtrl(int(opening_mm * 1000), 1000, 0x01, 0)
            time.sleep(0.02)
        self.gripper_mm = opening_mm

    def assert_undisturbed(self, q_deg) -> None:
        """Only this tool may drive the arm: stop if the joints or the gripper changed on their own."""
        self.check()
        off = max(abs(a - b) for a, b in zip(self.joints_deg(), q_deg))
        if off > 0.5:
            raise Stop(f"joints are {off:.2f} deg off the last target: another process or a person moved the arm")
        width = self.p.GetArmGripperMsgs().gripper_state.grippers_angle / 1000
        if abs(width - self.gripper_mm) > 3.0:
            raise Stop(f"gripper reads {width:.1f} mm, expected {self.gripper_mm:.0f}: another process moved it")

    def close(self) -> None:
        print("q now", np.round(self.joints_deg(), 1).tolist(), "peak effort", np.round(self.peak, 1).tolist())
        self.p.DisconnectPort()


def pose_safe(q, min_tip_z: float = MIN_TIP_Z_M) -> bool:
    T = LINK6.fk(q)
    flange, tip = T[:3, 3], T[:3, 3] + T[:3, 2] * TIP_LENGTH_M
    return bool(tip[2] >= min_tip_z and flange[2] >= MIN_FLANGE_Z_M and min(flange[1], tip[1]) >= MIN_Y_M
                and np.hypot(tip[0], tip[1]) <= MAX_TIP_RADIUS_M)


def path_safe(q0, q1, min_tip_z: float = MIN_TIP_Z_M) -> bool:
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    return all(pose_safe(q0 + s * (q1 - q0), min_tip_z) for s in np.linspace(0.0, 1.0, 21))


def tcp_ik(xy, z: float, tilt_deg: float, seeds) -> np.ndarray | None:
    """Joints for the TCP at (x, y, z), J6 near 0; tilt < 0 leans the approach away from the base."""
    j1 = math.degrees(math.atan2(xy[1], xy[0]))
    R = kin.yaw_tilt_to_matrix(90.0 + j1, -float(tilt_deg), xy)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.array([xy[0], xy[1], z]) - R[:, 2] * LINK6_TO_TCP_M
    return kin.solve_ik(LINK6, T, [*seeds, *SEEDS], limits=LIMITS_RAD)


# ------------------------------------------------------------------------------------ commands


def board_scale(run: Path) -> float:
    return json.loads((run / "board.json").read_text())["square_mm"] / SQUARE_MM


def pose_from_profile(camera: str) -> np.ndarray | None:
    cfg = yaml.safe_load((PROFILE / "skill_server.yaml").read_text())
    for cam in cfg.get("cameras") or []:
        if cam.get("name") == camera and cam.get("T_matrix"):
            return np.array(cam["T_matrix"], dtype=float)
    return None


def sheet_center_from_head(run: Path, scale: float) -> np.ndarray:
    T_base_head = pose_from_profile("head")
    if T_base_head is None:
        raise Stop("no head pose in skill_server.yaml yet: pass --center X Y (rough sheet centre, metres)")
    path = run / "head_rough.jpg"
    snapshot("head", path)
    obj, px = corners_of(path)
    if len(obj) < 8:
        raise Stop("the head camera does not see the sheet")
    _, r, t = cv2.solvePnP(obj * scale, px, K_HEAD, None)
    return (T_base_head @ to_T(r, t) @ np.r_[BOARD_CENTER * scale, 1.0])[:3]


def cmd_check(args) -> None:
    run = run_dir(args.run)
    for camera, K in (("head", K_HEAD), ("wrist", K_WRIST)):
        path = run / f"check_{camera}.jpg"
        snapshot(camera, path)
        obj, px = corners_of(path)
        line = f"{camera}: corners {len(obj)}/{(SQUARES[0] - 1) * (SQUARES[1] - 1)}"
        if len(obj) >= 8:
            H = cv2.findHomography(obj[:, :2], px, 0)[0]
            flat = cv2.perspectiveTransform(obj[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2) - px
            line += f", flatness {np.sqrt(np.mean(np.sum(flat ** 2, axis=1))):.2f} px"
        print(line)


def plan_collect(center, q_start, head_only: bool) -> list[dict]:
    plan, seed = [], [q_start]
    for dx, dy in SPIN_OFFSETS_M:
        xy = (center[0] + dx, center[1] + dy)
        for z in SPIN_TCP_Z_M:
            q = tcp_ik(xy, z, 0.0, seed)
            if q is not None:
                break
        if q is None:
            print(f"spin place {np.round(xy, 3).tolist()} is out of reach, skipped")
            continue
        seed = [q]
        for a in SPIN_ANGLES_DEG:
            v = np.degrees(q).copy()
            v[5] = a
            plan.append({"kind": "spin", "place": [round(xy[0], 3), round(xy[1], 3)], "j6": a, "q": v})
    if len({tuple(p["place"]) for p in plan}) < 3:
        raise Stop("fewer than 3 spin places in reach: move the sheet closer to the arm")
    if not head_only:
        radial = np.asarray(center[:2]) / np.linalg.norm(center[:2])
        for back, z, tilt, j6 in TILT_VIEWS:
            q = tcp_ik(np.asarray(center[:2]) - radial * back, z, tilt, seed)
            if q is None:
                print(f"tilted view {tilt} at z {z} is out of reach, skipped")
                continue
            v = np.degrees(q).copy()
            v[5] = float(j6)
            plan.append({"kind": "tilt", "j6": float(j6), "q": v})
    walk = [np.asarray(q_start)] + [np.radians(p["q"]) for p in plan] + [np.zeros(6)]
    for a, b in zip(walk, walk[1:]):
        if not path_safe(a, b):
            raise Stop("a planned move leaves the safe region")
    return plan


def cmd_collect(args) -> None:
    run = run_dir(args.run)
    scale = args.square_mm / SQUARE_MM
    (run / "board.json").write_text(json.dumps({"square_mm": args.square_mm}))
    center = np.array(args.center) if args.center else sheet_center_from_head(run, scale)
    print("sheet centre (m):", np.round(center, 3).tolist())
    arm = Arm()
    log = []
    try:
        plan = plan_collect(center, np.radians(arm.joints_deg()), args.head_only)
        print(f"walking {len(plan)} views at {SPEED_PERCENT}% speed, keep clear")
        arm.gripper(100)
        for k, p in enumerate(plan):
            arm.goto(p["q"].tolist())
            q1, T1 = arm.joints_deg(), arm.firmware_pose()
            tilted = p["kind"] == "tilt"
            path = run / f"{p['kind']}_{k:02d}.jpg"
            snapshot("wrist", path, tilted)
            arm.assert_undisturbed(p["q"].tolist())
            image = path.with_name(path.stem + "_color.jpg") if tilted else path
            print(f"{k:2d} {p['kind']} J6 {p['j6']:+.0f}: {len(all_points_of(image)[0])} points", flush=True)
            log.append({"k": k, "kind": p["kind"], "place": p.get("place"), "j6": p["j6"], "q": q1,
                        "T_fw": T1.tolist(), "image": image.name})
            (run / "views.json").write_text(json.dumps(log))
        arm.goto([0.0] * 6)
        arm.gripper(0)
        time.sleep(1.0)
        for k in range(3):
            snapshot("head", run / f"head_{k}.jpg", True)
            print(f"head {k}: {len(corners_of(run / f'head_{k}_color.jpg')[0])} corners", flush=True)
        print(f"done: calibrate.py solve {args.run}")
    except Stop as e:
        print("STOP:", e)
    finally:
        arm.close()


def axis_on_board(run: Path, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, float]:
    """Where the J6 axis meets the sheet: nominal board metres, base metres at the table, circle fit rms."""
    grid = np.array([[u, v] for v in (60, 150, 240, 310) for u in (160, 320, 480)], float).reshape(-1, 1, 2)
    pix = cv2.undistortPoints(grid, K_WRIST_D, D_WRIST_D, P=K_WRIST_D)
    J = len(pix)
    best = None
    for sign in (1, -1):
        A, b = [], []
        for r in rows:
            obj, px = all_points_of(run / r["image"])
            und = cv2.undistortPoints(px.reshape(-1, 1, 2), K_WRIST_D, D_WRIST_D, P=K_WRIST_D).reshape(-1, 2)
            H, _ = cv2.findHomography(und, obj[:, :2], cv2.RANSAC, 1.5)
            P = cv2.perspectiveTransform(pix, H).reshape(-1, 2)
            t = sign * math.radians(r["q"][5])
            R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
            for j in range(J):
                a = np.zeros((2, 2 + 2 * J))
                a[:2, :2] = np.eye(2)
                a[:2, 2 + 2 * j:4 + 2 * j] = R
                A.append(a)
                b.append(P[j])
        A, b = np.vstack(A), np.concatenate(b)
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        rms = float(np.sqrt(np.mean((A @ x - b) ** 2) * 2))
        if best is None or rms < best[0]:
            best = (rms, x[:2].copy())
    T = np.mean([np.array(r["T_fw"]) for r in rows], axis=0)
    axis = T[:3, 2] / np.linalg.norm(T[:3, 2])
    return best[1], (T[:3, 3] + axis * -T[2, 3] / axis[2])[:2], best[0]


def fit_board(on_board: np.ndarray, in_base: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """base_xy = t + M @ (scale * board_xy). Seen from above the board axes are left-handed."""
    mb, ma = on_board.mean(0), in_base.mean(0)
    U, _, Vt = np.linalg.svd((in_base - ma).T @ ((on_board - mb) * [1, -1]))
    R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    M = R @ np.diag([1.0, -1.0])
    return M, ma - scale * M @ mb


def cmd_solve(args) -> None:
    run = run_dir(args.run)
    runs = [run] + [run_dir(m) for m in args.merge]
    scale = board_scale(run)
    views = []
    for r_ in runs:
        for v in json.loads((r_ / "views.json").read_text()):
            v["run"] = r_
            views.append(v)
    places: dict = {}
    for r in views:
        if r["kind"] == "spin":
            places.setdefault((r["run"].name, *r["place"]), []).append(r)
    on_board, in_base = [], []
    for place, rows in places.items():
        b, a, rms = axis_on_board(rows[0]["run"], rows)
        print(f"place {place}: circle fit {rms * 1000:.2f} mm")
        on_board.append(b)
        in_base.append(a)
    on_board, in_base = np.array(on_board), np.array(in_base)
    M, t2 = fit_board(on_board, in_base, scale)
    residual = np.linalg.norm(t2 + scale * on_board @ M.T - in_base, axis=1) * 1000
    print("board fit residual (mm):", np.round(residual, 2).tolist())
    loo = []
    for k in range(len(on_board)):
        keep = [i for i in range(len(on_board)) if i != k]
        Mk, tk = fit_board(on_board[keep], in_base[keep], scale)
        loo.append(float(np.linalg.norm(tk + scale * Mk @ on_board[k] - in_base[k]) * 1000))
    print("leave-one-out error (mm):", np.round(loo, 2).tolist())
    arm_scale = float(np.sum((in_base - in_base.mean(0)) * ((on_board - on_board.mean(0)) @ M.T))
                      / np.sum((on_board - on_board.mean(0)) ** 2))
    print(f"square from the arm motion alone: {arm_scale * SQUARE_MM:.2f} mm; measured with a ruler: {scale * SQUARE_MM:.2f} mm")
    B = np.eye(4)
    B[:2, :2] = M
    B[2, 2] = -1.0
    B[:2, 3] = t2  # the sheet lies on the table, z = table_z = 0

    # head camera: corners averaged over the images, then the depth against the board plane
    pixels: dict = {}
    frames = [p for r_ in runs for p in sorted(r_.glob("head_[0-9]_color.jpg"))]
    per_run: dict = {}
    for p in frames:
        cc, ci, _, _ = DETECTOR.detectBoard(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if ci is not None:
            for i, q in zip(ci.ravel(), cc.reshape(-1, 2)):
                pixels.setdefault(int(i), []).append(q)
                per_run.setdefault(p.parent.name, {}).setdefault(int(i), []).append(q)
    for name, corners in per_run.items():  # merged runs must share the head camera and the sheet
        shift = max(np.linalg.norm(np.mean(v, axis=0) - np.mean(pixels[i], axis=0)) for i, v in corners.items())
        if shift > 1.0:
            raise Stop(f"run {name}: head image of the sheet differs by {shift:.1f} px, the camera or the sheet moved")
    ids = sorted(pixels)
    px = np.array([np.mean(pixels[i], axis=0) for i in ids], dtype=np.float64)
    obj = OBJ[ids] * scale
    _, r, t = cv2.solvePnP(obj, px, K_HEAD, None)
    proj = cv2.projectPoints(obj, r, t, K_HEAD, None)[0].reshape(-1, 2)
    print(f"head: {len(frames)} images, {len(ids)} corners, reprojection {np.sqrt(np.mean(np.sum((proj - px) ** 2, axis=1))):.2f} px")
    T_cam_board = to_T(r, t)
    T_base_head = B @ np.linalg.inv(T_cam_board)
    fx, fy, cx, cy = K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2]
    vv, uu = np.mgrid[0:480, 0:640]
    normal = T_cam_board[:3, 2]
    plane_z = (normal @ T_cam_board[:3, 3]) / (((uu - cx) / fx) * normal[0] + ((vv - cy) / fy) * normal[1] + normal[2])
    mask = np.zeros((480, 640), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(px.astype(np.int32)), 255)
    ratios, depth = [], None
    for p in frames:
        d = cv2.imread(str(p).replace("_color.jpg", "_depth.png"), cv2.IMREAD_UNCHANGED).astype(float) / 1000
        if np.mean(d > 0) < 0.5:  # the first frames after opening the camera can be empty
            continue
        white = (cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 100) & (mask > 0) & (d > 0)
        ratios.append(float(np.median(d[white] / plane_z[white])))
        depth = d
    depth_scale = 1.0 / float(np.median(ratios))
    print(f"head depth / board plane {np.round(ratios, 4).tolist()} -> depth_scale {depth_scale:.4f}")
    err = []
    for i, q in zip(ids, px):
        u, v = int(round(q[0])), int(round(q[1]))
        w = depth[v - 3:v + 4, u - 3:u + 4]
        w = w[w > 0]
        if w.size:
            Z = float(np.median(w)) * depth_scale
            err.append((T_base_head @ np.array([(q[0] - cx) / fx * Z, (q[1] - cy) / fy * Z, Z, 1.0]))[:3]
                       - (B @ np.r_[OBJ[i] * scale, 1.0])[:3])
    err = np.array(err)
    print(f"runtime path (pixel + depth) on the corners: xy mean {np.linalg.norm(err[:, :2], axis=1).mean() * 1000:.1f} "
          f"max {np.linalg.norm(err[:, :2], axis=1).max() * 1000:.1f} mm, z std {err[:, 2].std() * 1000:.1f} max {np.abs(err[:, 2]).max() * 1000:.1f} mm")

    # wrist camera, with the board pose fixed
    X = None
    if any(r["kind"] == "tilt" for r in views):
        data = []
        for r in views:
            o, p = all_points_of(r["run"] / r["image"])
            data.append({"o": o * [scale, scale, 1.0], "p": p, "T": np.array(r["T_fw"]), "name": r["image"]})
        fw, fh, cwx, cwy = K_WRIST[0, 0], K_WRIST[1, 1], K_WRIST[0, 2], K_WRIST[1, 2]

        def resid(x, vs):
            Xm = to_T(x[:3], x[3:6])
            out = []
            for v in vs:
                T = np.linalg.inv(v["T"] @ Xm) @ B
                c = v["o"] @ T[:3, :3].T + T[:3, 3]
                out.append(np.column_stack([fw * c[:, 0] / c[:, 2] + cwx, fh * c[:, 1] / c[:, 2] + cwy]) - v["p"])
            return np.concatenate(out).ravel()

        A = math.radians(20.0)
        guess = np.array([[0.0, math.cos(A), math.sin(A), -0.07], [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -math.sin(A), math.cos(A), 0.04], [0.0, 0.0, 0.0, 1.0]])
        fit = least_squares(resid, to_vec(guess), args=(data,), x_scale="jac", loss="soft_l1", f_scale=2.0)
        X = to_T(fit.x[:3], fit.x[3:6])
        per = [float(np.sqrt(np.mean(resid(fit.x, [v]) ** 2) * 2)) for v in data]
        print(f"wrist: camera in link6 {np.round(X[:3, 3], 4).tolist()} m; reprojection per view {np.round(per, 1).tolist()} px")

    fmt = lambda T: "[" + ", ".join("[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in T) + "]"
    lines = ["cameras:", "  - name: head", f"    K: {fmt(K_HEAD)}", "    mount: fixed",
             f"    depth_scale: {depth_scale:.4f}", f"    T_matrix: {fmt(T_base_head)}"]
    if X is not None:
        t = X[:3, 3]
        lines += ["  - name: wrist", f"    K: {fmt(K_WRIST)}", "    mount: wrist", "    arm: left", "    wrist_frame: link6",
                  f"    T_matrix: {fmt(X)}",
                  f"# robot.wrist_cam_offset: {[round(float(t[1]), 4), round(float(t[0]), 4), round(float(LINK6_TO_TCP_M - t[2]), 4)]}"]
    text = "\n".join(lines) + "\n"
    (run / "result.yaml").write_text(text)
    # a --head-only run keeps the wrist calibration from the profile (point needs it)
    np.savez(run / "result.npz", T_base_board=B, T_base_head=T_base_head, depth_scale=depth_scale, scale=scale,
             X=pose_from_profile("wrist") if X is None else X, center=(B @ np.r_[BOARD_CENTER * scale, 1.0])[:3])
    print("\n" + text + f"saved in {run / 'result.yaml'}; paste into profiles/real-piper/skill_server.yaml")


def cmd_point(args) -> None:
    """Closed fingertips 10 mm above corners located through the head camera, as the skill would."""
    run = run_dir(args.run)
    res = np.load(run / "result.npz")
    B, T_base_head, depth_scale, scale = res["T_base_board"], res["T_base_head"], float(res["depth_scale"]), float(res["scale"])
    X, center = res["X"], res["center"]
    snapshot("head", run / "point_head.jpg", True)
    obj, px = corners_of(run / "point_head_color.jpg")
    depth = cv2.imread(str(run / "point_head_depth.png"), cv2.IMREAD_UNCHANGED).astype(float) / 1000 * depth_scale
    fx, fy, cx, cy = K_HEAD[0, 0], K_HEAD[1, 1], K_HEAD[0, 2], K_HEAD[1, 2]
    located = []
    for o, q in zip(obj, px):
        u, v = int(round(q[0])), int(round(q[1]))
        w = depth[v - 3:v + 4, u - 3:u + 4]
        w = w[w > 0]
        if w.size:
            Z = float(np.median(w))
            located.append((o, (T_base_head @ np.array([(q[0] - cx) / fx * Z, (q[1] - cy) / fy * Z, Z, 1.0]))[:3]))
    if not located:
        raise Stop("no head depth at the corners: run again")
    z_low = TCP_TOUCH_Z_M + 0.010
    arm = Arm()
    arm.speed = 15
    try:
        arm.gripper(0)
        for dx, dy in POINT_OFFSETS_M:
            target = center[:2] + [dx, dy]
            o, P = min(located, key=lambda item: float(np.hypot(*(item[1][:2] - target))))
            qs = None
            for z_hi in SPIN_TCP_Z_M + (0.035,):
                seed, trial = [np.radians(arm.joints_deg())], []
                for z in [*np.arange(z_hi, z_low, -0.008), z_low]:
                    q = tcp_ik(P[:2], float(z), 0.0, seed)
                    if q is None:
                        trial = None
                        break
                    trial.append(q)
                    seed = [q]
                if trial:
                    qs = trial
                    break
            if not qs:
                print(f"corner at {np.round(P[:2], 3).tolist()} is out of reach, skipped")
                continue
            if not path_safe(np.radians(arm.joints_deg()), qs[0], 0.02):
                raise Stop("the move to the next corner would pass too low")
            for q in qs:
                arm.goto(np.degrees(q).tolist(), tol=0.2)
            T = arm.firmware_pose()
            path = run / f"point_{int(round(o[0] * 1000)):03d}_{int(round(o[1] * 1000)):03d}.jpg"
            snapshot("wrist", path)
            arm.assert_undisturbed(np.degrees(qs[-1]).tolist())
            ob, pxw = all_points_of(path)
            H, _ = cv2.findHomography(pxw, ob[:, :2] * scale, cv2.RANSAC, 2.0)
            cam = np.linalg.inv(X) @ np.array([0.0, 0.0, T[2, 3] / abs(T[2, 2]), 1.0])
            axis_px = (K_WRIST @ cam[:3] / cam[2])[:2]
            under = cv2.perspectiveTransform(axis_px.reshape(1, 1, 2), H).ravel()
            miss = under - o[:2] * scale
            tip_z = T[2, 3] + T[2, 2] * TIP_LENGTH_M
            print(f"corner head says {np.round(P[:2] * 1000, 1).tolist()} mm: fingertip height {tip_z * 1000:.1f} mm, "
                  f"axis misses the corner by {np.linalg.norm(miss) * 1000:.1f} mm ({path.name})", flush=True)
            for q in reversed(qs):
                arm.goto(np.degrees(q).tolist(), tol=0.3)
        arm.goto([0.0] * 6)
    except Stop as e:
        print("STOP:", e)
    finally:
        arm.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check")
    c.add_argument("run")
    c.set_defaults(fn=cmd_check)
    c = sub.add_parser("collect")
    c.add_argument("run")
    c.add_argument("--square-mm", type=float, required=True, help="printed square, measured with a ruler")
    c.add_argument("--center", nargs=2, type=float, metavar=("X", "Y"), help="rough sheet centre in the base frame, m")
    c.add_argument("--head-only", action="store_true", help="no tilted views: keeps the wrist calibration")
    c.set_defaults(fn=cmd_collect)
    c = sub.add_parser("solve")
    c.add_argument("run")
    c.add_argument("--merge", nargs="*", default=[], help="other runs with the same head camera and sheet position")
    c.set_defaults(fn=cmd_solve)
    c = sub.add_parser("point")
    c.add_argument("run")
    c.set_defaults(fn=cmd_point)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    try:
        main()
    except Stop as e:
        sys.exit(f"STOP: {e}")
