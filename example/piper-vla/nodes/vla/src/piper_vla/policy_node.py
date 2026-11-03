"""Dora node: run a trained LeRobot policy (ACT or pi0.5) on the arm, and move it home.

Inputs : ``tick`` (control clock at the dataset fps), ``<arm>_state``, one image input per
         camera, ``policy_cmd`` (UTF-8 JSON from vla_server:
         ``{"op": "start", "run_id", "policy", "task"}``, ``{"op": "stop", "run_id"}``,
         ``{"op": "home", "run_id"}`` or ``{"op": "status"}``).
Outputs: ``<arm>_command`` (forge_msgs.JointCommand, straight to the driver) and
         ``policy_status`` (UTF-8 JSON).

The node never stops a run by itself: a run ends on ``stop`` from vla_server (the model
decides how long a policy acts) or when the dataflow stops. ``home`` moves the arm to the
demonstrations' start pose along a smooth joint path and opens the gripper. Models load in
the background at start-up; ``policy_status`` reports ``loading`` / ``ready`` / ``error`` per
policy, the current run, the homing progress and the latest joint state. Inference runs in a
worker thread so the Dora loop never blocks on the GPU; a slow chunk only delays the next
command (the driver holds the last target meanwhile).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from .spec import VlaSpec, image_to_rgb, load_config, prep_image

logger = logging.getLogger("piper_vla.policy")

STATE_STALE_S = 0.3
IMAGE_STALE_S = 0.5


@dataclass
class PolicyEntry:
    name: str
    path: str                     # LeRobot pretrained_model directory
    description: str = ""
    task: str = ""                # default instruction (pi0.5); ACT ignores text
    sha256: str | None = None     # of model.safetensors, checked at load
    device: str = "cuda"
    n_action_steps: int | None = None   # override chunk execution length
    max_joint_step_rad: float = 0.12    # per tick clamp on commanded joint change
    device_used: str = "cpu"
    state: str = "loading"
    error: str = ""
    policy: Any = None
    pre: Any = None
    post: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_policy(entry: PolicyEntry) -> None:
    """Load weights + processors (blocking; call from a background thread)."""
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    path = Path(entry.path).expanduser()
    weights = path / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing")
    if entry.sha256:
        got = _sha256(weights)
        if got != entry.sha256:
            raise ValueError(f"sha256 mismatch for {weights}: {got}")
    cfg = PreTrainedConfig.from_pretrained(str(path))
    cfg.device = entry.device if torch.cuda.is_available() or entry.device == "cpu" else "cpu"
    if entry.n_action_steps is not None and hasattr(cfg, "n_action_steps"):
        cfg.n_action_steps = int(entry.n_action_steps)
    cls = get_policy_class(cfg.type)
    # pi0/pi0.5 from_pretrained only *prints* when the weights fail to load and returns a randomly
    # initialised model; capture its output and refuse such a model
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        policy = cls.from_pretrained(str(path), config=cfg)
    out = buf.getvalue()
    if "without loading pretrained weights" in out or "Could not load state dict" in out:
        raise RuntimeError(f"weights did not load from {path}: {out.strip()[-300:]}")
    policy.to(cfg.device)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={"device_processor": {"device": cfg.device}})
    entry.policy, entry.pre, entry.post = policy, pre, post
    entry.device_used = cfg.device


class PolicyRunner:
    def __init__(self, node: Any, raw: dict[str, Any], spec: VlaSpec) -> None:
        from forge_msgs import Image, JointCommand, JointState

        self._Image, self._JointCommand, self._JointState = Image, JointCommand, JointState
        self.node = node
        self.spec = spec
        self.entries = {p["name"]: PolicyEntry(**p) for p in (raw.get("policies") or [])}
        self.arm_by_input = {a.state_input: a for a in spec.arms}
        self.cam_by_input = {c.input: c for c in spec.cameras}
        self.state: dict[str, tuple[float, dict[str, float]]] = {}
        self.images: dict[str, tuple[float, np.ndarray]] = {}
        self.run: dict[str, Any] | None = None       # {run_id, policy, task, steps, started}
        self.homing: dict[str, Any] | None = None    # {run_id, t0, start, target, duration, done, error}
        self.home_joint_speed = float(raw.get("home_joint_speed") or 0.5)   # rad/s along the home path
        self.last_cmd: np.ndarray | None = None
        self._busy = threading.Event()
        self._out_lock = threading.Lock()
        self._pending_cmd: np.ndarray | None = None
        self._status_dirty = True
        self._last_status_t = 0.0
        # one long-lived inference thread: CUDA/cuBLAS handles are per thread, a fresh thread per
        # step would rebuild them every time (tens of ms)
        self._jobs: queue.Queue = queue.Queue(maxsize=1)
        threading.Thread(target=self._worker, name="infer", daemon=True).start()
        for e in self.entries.values():
            threading.Thread(target=self._load, args=(e,), name=f"load-{e.name}", daemon=True).start()

    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            self._infer(*job)

    def _load(self, e: PolicyEntry) -> None:
        t0 = time.monotonic()
        try:
            load_policy(e)
            e.state = "ready"
            logger.info("policy %s ready (%.1f s)", e.name, time.monotonic() - t0)
        except Exception as exc:  # reported in policy_status; the node keeps serving others
            e.state, e.error = "error", f"{type(exc).__name__}: {exc}"[:400]
            logger.exception("policy %s failed to load", e.name)
        self._status_dirty = True

    # ------------------------------------------------------------------ status

    def status(self) -> dict[str, Any]:
        return {
            "policies": [{"name": e.name, "state": e.state, "error": e.error, "description": e.description,
                          "task": e.task} for e in self.entries.values()],
            "run": None if self.run is None else {k: self.run[k] for k in ("run_id", "policy", "task", "steps",
                                                                              "skipped", "busy", "infer_ms_mean",
                                                                              "infer_ms_max", "error")}
                     | {"elapsed_s": round(time.monotonic() - self.run["started"], 2)},
            "last_command": {} if self.last_cmd is None else self.spec.unpack(self.last_cmd),
            "home": None if self.homing is None else {k: self.homing[k] for k in ("run_id", "done", "error")}
                     | {"elapsed_s": round(time.monotonic() - self.homing["t0"], 2)},
            "state": {a.name: [round(float(self.state[a.name][1].get(j, 0.0)), 5) for j in a.names]
                      for a in self.spec.arms if a.name in self.state},
        }

    def _send_status(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or self._status_dirty or now - self._last_status_t > 0.5:
            self.node.send_output("policy_status", pa.array([json.dumps(self.status())]))
            self._status_dirty, self._last_status_t = False, now

    # ------------------------------------------------------------------ inputs

    def on_input(self, input_id: str, value: Any) -> None:
        if input_id == "tick":
            self.tick()
            return
        if input_id == "policy_cmd":
            self.on_cmd(json.loads(value[0].as_py()))
            return
        arm = self.arm_by_input.get(input_id)
        if arm is not None:
            js = self._JointState.from_arrow(value)
            self.state[arm.name] = (time.monotonic(), dict(zip(js.name, js.position)))
            return
        cam = self.cam_by_input.get(input_id)
        if cam is not None:
            rgb = image_to_rgb(self._Image.from_arrow(value))
            self.images[cam.name] = (time.monotonic(), prep_image(rgb, cam.size))

    def on_cmd(self, cmd: dict[str, Any]) -> None:
        op = cmd.get("op")
        if op == "start":
            self.homing = None
            e = self.entries.get(str(cmd.get("policy")))
            err = ("unknown policy" if e is None else "" if e.state == "ready" else f"policy {e.state}: {e.error}")
            self.run = {"run_id": cmd.get("run_id"), "policy": cmd.get("policy"),
                        "task": cmd.get("task") or (e.task if e else ""), "steps": 0, "skipped": 0,
                        "busy": 0, "infer_ms_mean": 0.0, "infer_ms_max": 0.0,
                        "started": time.monotonic(), "error": err}
            if not err:
                with e.lock:
                    e.policy.reset()
                self.last_cmd = None
                logger.info("run %s start policy=%s task=%r", self.run["run_id"], e.name, self.run["task"])
            self._send_status(force=True)
        elif op == "stop":
            if self.run is not None and (cmd.get("run_id") in (None, self.run["run_id"])):
                logger.info("run %s stop after %d steps", self.run["run_id"], self.run["steps"])
                self.run = None
            if self.homing is not None and (cmd.get("run_id") in (None, self.homing["run_id"])):
                self.homing = None                   # the driver holds the last commanded pose
            self._send_status(force=True)
        elif op == "home":
            self.run = None
            self.homing = self._home_plan(cmd.get("run_id"))
            self._send_status(force=True)
        elif op == "status":
            self._send_status(force=True)

    def _home_plan(self, run_id: Any) -> dict[str, Any]:
        """Smooth joint path from the current state to the home pose (gripper opened)."""
        plan = {"run_id": run_id, "t0": time.monotonic(), "done": False, "error": ""}
        now = time.monotonic()
        if any(a.name not in self.state or now - self.state[a.name][0] > STATE_STALE_S for a in self.spec.arms):
            plan.update(done=True, error="no fresh joint state")
            return plan
        start = self.spec.pack({a.name: self.state[a.name][1] for a in self.spec.arms})
        target = self.spec.pack({a.name: {**dict(zip(a.joint_names, a.home_joints)), a.gripper_joint: a.home_gripper_m}
                                 for a in self.spec.arms})
        joint_idx = [i for i, n in enumerate(self.spec.state_names) if not n.endswith(".gripper")]
        span = float(np.max(np.abs(target[joint_idx] - start[joint_idx]))) if joint_idx else 0.0
        plan.update(start=start, target=target, duration=max(1.0, span / max(1e-3, self.home_joint_speed)))
        logger.info("home %s over %.1f s", run_id, plan["duration"])
        return plan

    def _home_step(self) -> None:
        h = self.homing
        if h is None or h["done"]:
            return
        x = min(1.0, (time.monotonic() - h["t0"]) / h["duration"])
        a = x * x * (3.0 - 2.0 * x)                  # smoothstep: zero speed at both ends
        vec = h["start"] + (h["target"] - h["start"]) * a
        self.last_cmd = vec.astype(np.float64)
        self._send_command(vec)
        if x >= 1.0:
            h["done"] = True
            self._status_dirty = True

    # ------------------------------------------------------------------ control

    def _observation(self, now: float) -> dict[str, Any] | None:
        per_arm = {}
        for a in self.spec.arms:
            item = self.state.get(a.name)
            if item is None or now - item[0] > STATE_STALE_S:
                return None
            per_arm[a.name] = item[1]
        for c in self.spec.cameras:
            item = self.images.get(c.name)
            if item is None or now - item[0] > IMAGE_STALE_S:
                return None
        return {"state": self.spec.pack(per_arm), "images": {c.name: self.images[c.name][1] for c in self.spec.cameras}}

    def tick(self) -> None:
        self._home_step()
        with self._out_lock:
            cmd, self._pending_cmd = self._pending_cmd, None
        if cmd is not None and self.run is not None:
            self._send_command(cmd)
        if self.run is not None and not self.run["error"] and self._busy.is_set():
            self.run["busy"] += 1
        elif self.run is not None and not self.run["error"]:
            obs = self._observation(time.monotonic())
            if obs is None:
                self.run["skipped"] += 1
            else:
                self._busy.set()
                run_id = self.run["run_id"]
                e = self.entries[self.run["policy"]]
                self._jobs.put((e, obs, self.run["task"], run_id))
        self._send_status()

    def _infer(self, e: PolicyEntry, obs: dict[str, Any], task: str, run_id: Any) -> None:
        import torch

        t0 = time.perf_counter()
        try:
            dev = e.device_used
            batch: dict[str, Any] = {"observation.state": torch.from_numpy(obs["state"]).to(dev).unsqueeze(0),
                                     "task": [task]}
            for name, img in obs["images"].items():
                # uint8 to the GPU first, convert there (CPU float conversion is slow under load)
                t = torch.from_numpy(img).to(dev, non_blocking=True).permute(2, 0, 1).float().div_(255.0)
                batch[f"observation.images.{name}"] = t.unsqueeze(0)
            t1 = time.perf_counter()
            with e.lock, torch.inference_mode():
                b = e.pre(batch)
                t2 = time.perf_counter()
                act = e.policy.select_action(b)
                t3 = time.perf_counter()
                act = e.post(act)
            t4 = time.perf_counter()
            self._prof = getattr(self, "_prof", np.zeros(4)) + np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3])
            self._nprof = getattr(self, "_nprof", 0) + 1
            if self._nprof % 30 == 0:
                logger.info("infer ms (batch, pre, model, post) = %s", np.round(self._prof / self._nprof * 1000, 2).tolist())
            vec = act.squeeze(0).float().cpu().numpy() if hasattr(act, "cpu") else np.asarray(act).reshape(-1)
            vec = self._safe(vec, obs["state"], e)
            with self._out_lock:
                if self.run is not None and self.run["run_id"] == run_id:
                    self._pending_cmd = vec
                    r = self.run
                    r["steps"] += 1
                    ms = (time.perf_counter() - t0) * 1000.0
                    r["infer_ms_mean"] = round(r["infer_ms_mean"] + (ms - r["infer_ms_mean"]) / r["steps"], 2)
                    r["infer_ms_max"] = round(max(r["infer_ms_max"], ms), 1)
        except Exception as exc:
            logger.exception("inference failed")
            if self.run is not None and self.run["run_id"] == run_id:
                self.run["error"] = f"{type(exc).__name__}: {exc}"[:400]
                self._status_dirty = True
        finally:
            self._busy.clear()

    def _safe(self, vec: np.ndarray, state: np.ndarray, e: PolicyEntry) -> np.ndarray:
        """Clamp the per-tick joint change (against the last command) and the gripper range."""
        ref = self.last_cmd if self.last_cmd is not None else state
        out = vec.astype(np.float64).copy()
        i = 0
        for a in self.spec.arms:
            nj = len(a.joint_names)
            out[i:i + nj] = ref[i:i + nj] + np.clip(out[i:i + nj] - ref[i:i + nj], -e.max_joint_step_rad, e.max_joint_step_rad)
            out[i + nj] = float(np.clip(out[i + nj], 0.0, a.max_opening_m))
            i += nj + 1
        self.last_cmd = out
        return out

    def _send_command(self, vec: np.ndarray) -> None:
        for a, (name, values) in zip(self.spec.arms, self.spec.unpack(vec).items()):
            c = self._JointCommand(name=a.names, mode="position", position=values)
            self.node.send_output(a.command_output, c.to_arrow())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="piper-vla policy runner and homing (Dora node)")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw, spec = load_config(args.config)
    import os

    import torch

    # few CPU threads: the node shares the machine with drivers, cameras and perception
    # (hardware-dependent: 2 on the RTX 5060 laptop, raise on the workstation)
    torch.set_num_threads(int(raw.get("torch_threads") or os.environ.get("VLA_TORCH_THREADS") or 2))
    from dora import Node

    node = Node()
    runner = PolicyRunner(node, raw, spec)
    for event in node:
        if event.get("type") == "STOP":
            break
        if event.get("type") != "INPUT":
            continue
        try:
            runner.on_input(str(event.get("id")), event["value"])
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ignored input %s: %s", event.get("id"), exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
