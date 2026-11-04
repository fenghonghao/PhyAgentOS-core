#!/usr/bin/env python3
"""Turn an eye-to-hand calibration (head camera pose in each arm's base frame) into the
skill_server.yaml entries, with the world frame = the LEFT arm's arm_base frame.

Input YAML (4x4 row-major matrices, metres):
  head_in_left_base:  [[...], [...], [...], [...]]   # camera optical frame expressed in left arm_base
  head_in_right_base: [[...], [...], [...], [...]]   # camera optical frame expressed in right arm_base
  # if your matrices map base -> camera instead (base in camera frame), set: inverse: true
Usage:
  scripts/calib_compose.py calib.yaml   # prints the YAML snippet and a sanity report
"""
import math
import sys

import numpy as np
import yaml


def rpy_deg(R):
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return [round(math.degrees(a), 2) for a in (roll, pitch, yaw)]


def main(path):
    c = yaml.safe_load(open(path))
    L = np.asarray(c["head_in_left_base"], dtype=float)
    Rr = np.asarray(c["head_in_right_base"], dtype=float)
    if c.get("inverse"):
        L, Rr = np.linalg.inv(L), np.linalg.inv(Rr)
    right_in_left = L @ np.linalg.inv(Rr)  # right arm_base expressed in the left arm_base (world)
    fmt = lambda M: "[" + ", ".join("[" + ", ".join(f"{v:.6f}" for v in row) + "]" for row in M) + "]"
    print("# --- paste into profiles/real-piper/skill_server.yaml (world = the arm base) ---")
    print("arms:\n  left:\n    base_T: " + fmt(np.eye(4)) + "\n  right:\n    base_T: " + fmt(right_in_left))
    print("cameras:\n  - {name: head, mount: fixed, K: <from calibration>, T_matrix: " + fmt(L) + "}")
    d = right_in_left[:3, 3]
    print("\n# sanity: right base at", np.round(d, 4).tolist(), "m, distance", round(float(np.linalg.norm(d)), 4),
          "m, rpy_deg", rpy_deg(right_in_left[:3, :3]))
    print("# both bases on one table -> z offset and roll/pitch should be ~0; camera height above left base:",
          round(float(L[2, 3]), 3), "m")


if __name__ == "__main__":
    main(sys.argv[1])
