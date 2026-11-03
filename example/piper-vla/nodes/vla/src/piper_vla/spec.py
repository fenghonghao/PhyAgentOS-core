"""Shared data layout for collection, training and inference.

Collection writes LeRobot datasets with exactly this layout and inference rebuilds the same
observation from live Dora inputs, so the two must use these helpers and nothing else:

* ``observation.state`` / ``action``: per arm (config order) ``joint1..joint6, gripper``,
  joints in radians, gripper as total finger opening in metres (driver convention).
  ``action`` is the commanded joint target, not the next measured state.
* ``observation.images.<camera>``: RGB uint8, resized to the camera's ``size`` with
  ``prep_image`` (bilinear), height x width x 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ArmSpec:
    name: str
    joint_names: list[str] = field(default_factory=lambda: [f"joint{i}" for i in range(1, 7)])
    gripper_joint: str = "gripper"
    home_joints: list[float] = field(default_factory=lambda: [0.0, 0.5, -0.02, 0.0, 0.3, 0.0])
    home_gripper_m: float = 0.0
    max_opening_m: float = 0.07
    # URDF used by the teleop IK (relative paths: relative to the config file)
    urdf: str | None = None
    # inputs/outputs in the Dora dataflow
    state_input: str = ""
    command_output: str = ""

    def __post_init__(self) -> None:
        self.state_input = self.state_input or f"{self.name}_state"
        self.command_output = self.command_output or f"{self.name}_command"

    @property
    def names(self) -> list[str]:
        return [*self.joint_names, self.gripper_joint]


@dataclass
class CameraSpec:
    name: str                      # feature alias: observation.images.<name>
    input: str = ""                # Dora input id carrying forge_msgs.Image (rgb8 / bgr8)
    size: list[int] = field(default_factory=lambda: [320, 240])   # width, height

    def __post_init__(self) -> None:
        self.input = self.input or f"{self.name}_color"


@dataclass
class VlaSpec:
    arms: list[ArmSpec]
    cameras: list[CameraSpec]
    fps: int = 30

    @property
    def state_names(self) -> list[str]:
        return [f"{a.name}.{j}" for a in self.arms for j in a.names]

    @property
    def dim(self) -> int:
        return sum(len(a.names) for a in self.arms)

    def features(self, use_videos: bool = True) -> dict[str, dict[str, Any]]:
        names = self.state_names
        feats: dict[str, dict[str, Any]] = {
            "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
            "action": {"dtype": "float32", "shape": (len(names),), "names": names},
        }
        for c in self.cameras:
            w, h = c.size
            feats[f"observation.images.{c.name}"] = {
                "dtype": "video" if use_videos else "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            }
        return feats

    def pack(self, per_arm: dict[str, dict[str, float]]) -> np.ndarray:
        """Per-arm ``{joint: value}`` dicts -> flat float32 vector in spec order."""
        return np.array([float(per_arm[a.name][j]) for a in self.arms for j in a.names], dtype=np.float32)

    def unpack(self, vec) -> dict[str, list[float]]:
        """Flat vector -> per-arm values in ``ArmSpec.names`` order."""
        out, i = {}, 0
        for a in self.arms:
            n = len(a.names)
            out[a.name] = [float(v) for v in vec[i:i + n]]
            i += n
        return out


def prep_image(img: np.ndarray, size: list[int]) -> np.ndarray:
    """RGB uint8 HxWx3 -> resized RGB uint8 (height, width, 3); identity if already sized."""
    w, h = int(size[0]), int(size[1])
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[0] == h and img.shape[1] == w:
        return np.ascontiguousarray(img[:, :, :3])
    from PIL import Image

    return np.asarray(Image.fromarray(np.ascontiguousarray(img[:, :, :3])).resize((w, h), Image.BILINEAR))


def image_to_rgb(msg) -> np.ndarray:
    """forge_msgs.Image -> RGB uint8 HxWx3 (handles rgb8 / bgr8 / rgba8 / bgra8 / mono8)."""
    arr = msg.to_numpy()
    enc = str(getattr(msg, "encoding", "rgb8")).lower()
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8)
    if enc.startswith("bgr"):
        arr = arr[:, :, 2::-1]
    return np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8, copy=False)


def load_spec(raw: dict[str, Any], base_dir: Path | None = None) -> VlaSpec:
    arms = []
    for name, a in (raw.get("arms") or {}).items():
        a = dict(a)
        if a.get("urdf") and base_dir is not None and not Path(a["urdf"]).is_absolute():
            a["urdf"] = str((base_dir / a["urdf"]).resolve())
        arms.append(ArmSpec(name=name, **a))
    cams = [CameraSpec(**c) for c in (raw.get("cameras") or [])]
    if not arms or not cams:
        raise ValueError("vla config needs at least one arm and one camera")
    return VlaSpec(arms=arms, cameras=cams, fps=int(raw.get("fps", 30)))


def load_config(path: str | Path) -> tuple[dict[str, Any], VlaSpec]:
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text()) or {}
    return raw, load_spec(raw, p.parent)
