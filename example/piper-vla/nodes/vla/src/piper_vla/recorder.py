"""LeRobot dataset writer for teleoperated episodes (one dataset directory, appendable)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .spec import VlaSpec

logger = logging.getLogger("piper_vla.recorder")


class Recorder:
    def __init__(self, spec: VlaSpec, root: str | Path, repo_id: str, robot_type: str = "piper_dual",
                 use_videos: bool = True, image_writer_threads: int = 4, encoder_threads: int = 2) -> None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.spec = spec
        self.root = Path(root).expanduser()
        nthreads = image_writer_threads * len(spec.cameras)
        # streaming encoding writes video while recording, so saving an episode is quick
        common = dict(image_writer_threads=nthreads, streaming_encoding=use_videos, encoder_threads=encoder_threads)
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root, **common)
            want, have = spec.features(use_videos), self.ds.meta.features
            for k, v in want.items():
                if k not in have or tuple(have[k]["shape"]) != tuple(v["shape"]):
                    raise ValueError(f"existing dataset {self.root} has a different layout for {k!r}; "
                                     "use a new dataset name")
            if int(self.ds.meta.fps) != spec.fps:
                raise ValueError(f"existing dataset {self.root} is {self.ds.meta.fps} fps, config says {spec.fps}")
            logger.info("appending to %s (%d episodes)", self.root, self.ds.meta.total_episodes)
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.ds = LeRobotDataset.create(repo_id=repo_id, fps=spec.fps, root=self.root, robot_type=robot_type,
                                            features=spec.features(use_videos), use_videos=use_videos, **common)
            logger.info("created %s", self.root)
        self.frames = 0

    @property
    def episodes(self) -> int:
        return int(self.ds.meta.total_episodes)

    def add(self, state: np.ndarray, action: np.ndarray, images: dict[str, np.ndarray], task: str) -> None:
        frame: dict[str, Any] = {"observation.state": state.astype(np.float32),
                                 "action": action.astype(np.float32), "task": task}
        for name, img in images.items():
            frame[f"observation.images.{name}"] = img
        self.ds.add_frame(frame)
        self.frames += 1

    def save(self) -> int:
        n = self.frames
        if n:
            self.ds.save_episode()
        self.frames = 0
        return n

    def discard(self) -> None:
        if self.frames:
            self.ds.clear_episode_buffer()
        self.frames = 0

    def close(self) -> None:
        self.discard()
        self.ds.finalize()
