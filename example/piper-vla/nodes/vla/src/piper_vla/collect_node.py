"""Dora node: teleoperate the arms (keyboard / gamepad + IK) and record LeRobot episodes.

Inputs : ``tick`` (control/record clock, run it at the dataset fps), ``<arm>_state``
         (forge_msgs.JointState from the driver or simulator), one image input per camera.
Outputs: ``<arm>_command`` (forge_msgs.JointCommand: joint1..6 + gripper, position mode).

Every tick sends a command for every arm (idle arms hold their target) and, while recording,
appends one frame: measured state, commanded action, the latest image of each camera. A frame
is skipped (and the screen turns red) if any state or image is stale; discard such episodes.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .spec import VlaSpec, image_to_rgb, load_config, prep_image
from .teleop import TeleopArm, Twist
from .ui import Console, GamepadMap, Speeds

logger = logging.getLogger("piper_vla.collect")

STATE_STALE_S = 0.3
IMAGE_STALE_S = 0.5


class Autopilot:
    """Scripted operator for headless pipeline tests: ``VLA_AUTOPILOT="episodes=2,seconds=4"``.

    Homes, records an episode of smooth motion with one gripper toggle, saves, repeats, quits.
    """

    def __init__(self, text: str) -> None:
        kv = dict(item.split("=", 1) for item in text.split(",") if "=" in item)
        self.episodes = int(kv.get("episodes", 2))
        self.seconds = float(kv.get("seconds", 4))
        self.phase, self.ep, self.t, self.toggled = "wait", 0, 0.0, False

    def step(self, col: "Collector", dt: float):
        from .ui import Commands

        cmd, tw = Commands(), Twist()
        ready = all(a.ready for a in col.arms.values()) and all(c.name in col.images for c in col.spec.cameras)
        if self.phase == "wait" and ready:
            cmd.home, self.phase = True, "homing"
        elif self.phase == "homing" and not any(a.homing for a in col.arms.values()):
            cmd.toggle_record, self.phase, self.t, self.toggled = True, "rec", 0.0, False
        elif self.phase == "rec":
            self.t += dt
            w = 1.5
            tw = Twist(vx=0.05 * math.cos(w * self.t), vy=0.05 * math.sin(w * self.t), vz=-0.03 * math.sin(self.t))
            if self.t > self.seconds / 2 and not self.toggled:
                cmd.toggle_gripper, self.toggled = True, True
            if self.t >= self.seconds:
                cmd.toggle_record = True
                self.ep += 1
                self.phase = "wait" if self.ep < self.episodes else "done"
        elif self.phase == "done":
            cmd.quit = True
        return tw, cmd


class Collector:
    def __init__(self, node: Any, raw: dict[str, Any], spec: VlaSpec, dataset_name: str | None,
                 tasks: list[str]) -> None:
        from forge_msgs import Image, JointCommand, JointState

        self._Image, self._JointCommand, self._JointState = Image, JointCommand, JointState
        self.node = node
        self.spec = spec
        tcfg = raw.get("teleop") or {}
        yaw = tcfg.get("control_yaw_deg") or {}
        self.arms = {a.name: TeleopArm(a, float(tcfg.get("max_joint_speed", 2.0)), float(yaw.get(a.name, 0.0)))
                     for a in spec.arms}
        self.order = [a.name for a in spec.arms]
        self.home_speed = float(tcfg.get("home_joint_speed", 0.6))
        self.console = Console(speeds=Speeds(**(tcfg.get("speeds") or {})),
                               pad=GamepadMap(**(tcfg.get("gamepad") or {})))
        dcfg = raw.get("dataset") or {}
        self.dataset_name = dataset_name or dcfg.get("name") or "piper_vla"
        self.dataset_root = Path(dcfg.get("root", "~/.PhyAgentOS/vla/datasets")).expanduser() / self.dataset_name
        self.use_videos = bool(dcfg.get("use_videos", True))
        # hardware-dependent (CPU): laptop 4 / 2, workstation can use more
        self.writer_threads = int(dcfg.get("image_writer_threads", 4))
        self.encoder_threads = int(dcfg.get("encoder_threads", 2))
        self.tasks = tasks or list(raw.get("tasks") or []) or ["do the task"]
        self.task_i = 0
        self.sel = 0
        self.state_t = {a: 0.0 for a in self.order}
        self.images: dict[str, tuple[float, np.ndarray]] = {}
        self.cam_by_input = {c.input: c for c in spec.cameras}
        self.arm_by_input = {a.state_input: a.name for a in spec.arms}
        self.rec = None
        self.recording = False
        self.paused = False
        self.dropped = 0
        self.ep_frames = 0
        self.last_tick = None
        self.message = "ready: H = home, Enter = start recording"
        auto = os.environ.get("VLA_AUTOPILOT")
        self.autopilot = Autopilot(auto) if auto else None

    # ------------------------------------------------------------------ inputs

    def on_input(self, input_id: str, value: Any) -> bool:
        if input_id == "tick":
            return self.tick()
        arm = self.arm_by_input.get(input_id)
        if arm is not None:
            js = self._JointState.from_arrow(value)
            self.arms[arm].on_state(list(js.name), list(js.position))
            self.state_t[arm] = time.monotonic()
            return True
        cam = self.cam_by_input.get(input_id)
        if cam is not None:
            rgb = image_to_rgb(self._Image.from_arrow(value))
            self.images[cam.name] = (time.monotonic(), prep_image(rgb, cam.size))
        return True

    # ------------------------------------------------------------------ control tick

    def tick(self) -> bool:
        now = time.monotonic()
        dt = 1.0 / self.spec.fps if self.last_tick is None else min(0.1, max(0.0, now - self.last_tick))
        self.last_tick = now
        tw, cmd = self.console.poll()
        if self.autopilot is not None:
            tw, cmd = self.autopilot.step(self, dt)
        if cmd.quit:
            return False
        self._commands(cmd)
        sel_name = self.order[self.sel]
        for name, arm in self.arms.items():
            if self.paused:
                continue
            arm.step(tw if name == sel_name else Twist(), dt)
            if arm.ready:
                c = self._JointCommand(name=arm.spec.names, mode="position", position=arm.command_values())
                self.node.send_output(arm.spec.command_output, c.to_arrow())
        stale = self._stale(now)
        if self.recording:
            if stale:
                self.dropped += 1
            else:
                self._record_frame()
        self._draw(now, stale)
        return True

    def _commands(self, cmd) -> None:
        n = len(self.order)
        if cmd.select_arm is not None and cmd.select_arm < n:
            self.sel = cmd.select_arm
        if cmd.next_arm:
            self.sel = (self.sel + cmd.next_arm) % n
        if cmd.task is not None and cmd.task < len(self.tasks):
            self.task_i = cmd.task
            self.message = f"task: {self.tasks[self.task_i]}"
        if cmd.pause:
            self.paused = not self.paused
            if not self.paused:
                for arm in self.arms.values():
                    arm.sync()
            self.message = "paused" if self.paused else "resumed (targets synced to measured)"
        if cmd.home:
            dur = max([1.0] + [a.home_duration(self.home_speed) for a in self.arms.values()])
            for arm in self.arms.values():
                arm.start_home(dur)
            self.message = f"homing ({dur:.1f} s)"
        if cmd.toggle_gripper:
            arm = self.arms[self.order[self.sel]]
            if arm.ready:
                arm.g_cmd = 0.0 if (arm.g_cmd or 0.0) > arm.spec.max_opening_m / 2 else arm.spec.max_opening_m
        if cmd.discard and self.recording:
            self.rec.discard()
            self.recording = False
            self.message = "episode discarded"
        if cmd.toggle_record:
            if self.recording:
                self._save()
            else:
                self._start()

    def _start(self) -> None:
        if not all(a.ready for a in self.arms.values()):
            self.message = "cannot record: waiting for arm states"
            return
        missing = [c.name for c in self.spec.cameras if c.name not in self.images]
        if missing:
            self.message = f"cannot record: no image from {missing}"
            return
        if self.rec is None:
            from .recorder import Recorder

            self.rec = Recorder(self.spec, self.dataset_root, f"local/{self.dataset_name}", use_videos=self.use_videos,
                                image_writer_threads=self.writer_threads, encoder_threads=self.encoder_threads)
        far = [n for n, a in self.arms.items()
               if float(np.max(np.abs(np.asarray(a.spec.home_joints) - a.q_cmd))) > 0.08]
        self.recording, self.dropped, self.ep_frames = True, 0, 0
        self.message = "RECORDING" + (f" (warning: {far} not at home)" if far else "")

    def _save(self) -> None:
        self.console.draw([], ["saving episode (encoding video)..."])
        n = self.rec.save()
        self.recording = False
        self.message = f"saved episode {self.rec.episodes - 1}: {n} frames, {self.dropped} dropped" if n else "empty episode"

    def _stale(self, now: float) -> list[str]:
        out = [f"{a} state" for a in self.order if now - self.state_t[a] > STATE_STALE_S]
        out += [f"{c.name} image" for c in self.spec.cameras
                if c.name not in self.images or now - self.images[c.name][0] > IMAGE_STALE_S]
        return out

    def _record_frame(self) -> None:
        meas = {n: dict(zip(a.spec.names, [*map(float, a.q_meas), float(a.g_meas)])) for n, a in self.arms.items()}
        cmd = {n: dict(zip(a.spec.names, a.command_values())) for n, a in self.arms.items()}
        imgs = {c.name: self.images[c.name][1] for c in self.spec.cameras}
        self.rec.add(self.spec.pack(meas), self.spec.pack(cmd), imgs, self.tasks[self.task_i])
        self.ep_frames += 1

    def _draw(self, now: float, stale: list[str]) -> None:
        imgs = [(c.name, self.images[c.name][1] if c.name in self.images else None) for c in self.spec.cameras]
        lines = []
        for i, name in enumerate(self.order):
            a = self.arms[name]
            xyz = a.tcp_xyz()
            mark = ">" if i == self.sel else " "
            pos = "-" if xyz is None else "[" + ", ".join(f"{v:+.3f}" for v in xyz) + "]"
            grip = "-" if a.g_cmd is None else f"{a.g_cmd * 1000:.0f} mm"
            lines.append(f"{mark}{i + 1} {name:6s} tcp {pos} grip {grip}  {a.last_status}")
        eps = self.rec.episodes if self.rec is not None else "-"
        rec = f"REC {self.ep_frames / self.spec.fps:5.1f}s dropped {self.dropped}" if self.recording else "idle"
        lines.append(f"{rec} | dataset {self.dataset_name} episodes {eps} | task [F{self.task_i + 1}] {self.tasks[self.task_i]}")
        if stale:
            lines.append("STALE: " + ", ".join(stale))
        lines.append(self.message)
        self.console.draw(imgs, lines, alert=bool(stale) and self.recording)

    def close(self) -> None:
        if self.rec is not None:
            if self.recording:
                logger.warning("quitting while recording: episode discarded")
            self.rec.close()
            logger.info("dataset finalized at %s (%d episodes)", self.dataset_root, self.rec.episodes)
        self.console.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="piper-vla teleop + LeRobot recorder (Dora node)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", help="dataset name (directory under dataset.root)")
    ap.add_argument("--task", action="append", default=[], help="task text (repeatable; F1..F9 select)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw, spec = load_config(args.config)
    # collect.sh passes the dataset and task texts through the environment (texts have spaces)
    dataset = args.dataset or os.environ.get("VLA_DATASET") or None
    tasks = args.task + [t for t in os.environ.get("VLA_TASKS", "").split("\n") if t.strip()]
    from dora import Node

    node = Node()
    col = Collector(node, raw, spec, dataset, tasks)
    try:
        for event in node:
            if event.get("type") == "STOP":
                break
            if event.get("type") != "INPUT":
                continue
            try:
                if not col.on_input(str(event.get("id")), event["value"]):
                    break
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("ignored input %s: %s", event.get("id"), exc)
    finally:
        col.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
