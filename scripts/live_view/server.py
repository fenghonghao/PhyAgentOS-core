#!/usr/bin/env python3
"""Read-only live view of the two Orbbec cameras (RGB + aligned depth) and the PIPER pose.

Nothing here controls hardware and nothing is saved: frames live in memory only and are pushed to the
browser as JPEG; no video, image or log file is written.
  * cameras  - two sources, both read-only:
      dora (default): attaches to the running skill as a *dynamic Dora node* and receives the camera
                      outputs the dataflow routes to it. The cameras stay owned by the skill, so the
                      page works while the skill runs. The dataflow must declare the node (see below).
                      Start it before or after the skill; it waits for the dataflow and re-attaches
                      after a skill restart (by re-executing itself, see DoraFeed).
      usb           : opens the Orbbec devices directly (no device property is written). A device can be
                      opened by one process only, so the skill must be stopped first.
  * arm      - listens passively on a SocketCAN interface (the 0x2A1-0x2A8 feedback frames the arm
               broadcasts by itself). The socket never sends, so the arm is not enabled, not queried
               and not moved, and it can run next to the driver or Piper_sdk_ui.

Dataflow entry for the dora source (both packages name their camera nodes cam_head / cam_wrist):

    - id: live_view
      path: dynamic
      inputs:
        head_color:  {source: cam_head/image/color,  queue_size: 1}
        head_depth:  {source: cam_head/image/depth,  queue_size: 1}
        wrist_color: {source: cam_wrist/image/color, queue_size: 1}
        wrist_depth: {source: cam_wrist/image/depth, queue_size: 1}

Inputs are named <camera>_color (forge_msgs Image rgb8 / bgr8) and <camera>_depth (Image 32FC1, metres).
`paos skill start` of piper-dual-tabletop starts this page (the bundle carries this directory as
nodes/live_view, see nodes/run_live_view.sh), so open http://127.0.0.1:8780. By hand, from the repo (the
piper-vla venv has dora-rs, forge_msgs, opencv, aiohttp, pyorbbecsdk):

    ~/.PhyAgentOS/skills/.venvs/piper-vla/bin/python scripts/live_view/server.py
    # directly on the cameras, skill stopped:  ... server.py --source usb
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from aiohttp import WSCloseCode, web

HERE =Path(__file__).resolve().parent
# This file is <skill>/nodes/live_view/server.py when installed, scripts/live_view/server.py in the repo.
SKILL_ROOT = HERE.parents[1] if (HERE.parents[1] / "skill.yaml").is_file() else HERE.parents[1] / "example" / "piper-dual-tabletop"
ROBOT_DIR = SKILL_ROOT / "assets" / "robots" / "piper"
URDF = ROBOT_DIR / "piper_with_gripper_real.urdf"
MESH_DIR = ROBOT_DIR / "mjcf" / "assets"
SKILL_LOG_DIR = Path.home() / ".PhyAgentOS" / "logs" / "piper-dual-tabletop"  # written by the skill, only read here
LEDGER_TAIL = 300  # robot.run records sent when the page opens
SESSION_DIR = Path.home() / ".PhyAgentOS" / "workspace" / "sessions"  # PAOS conversations, only read here
SCENE_TAIL = 8  # most recent scene conversations shown

CAMERAS = {"head": "AY6Z36300RB", "wrist": "AY6Z363005R"}
COLOR_MODE = (640, 480, 30)
DEPTH_MODE = (640, 400, 30)
STREAM_HZ = 30

CTRL_MODE = {0: "STANDBY", 1: "CAN_CTRL", 2: "TEACHING", 3: "ETHERNET", 4: "WIFI", 5: "REMOTE",
             6: "LINKAGE_TEACHING", 7: "OFFLINE_TRAJECTORY"}
ARM_STATUS = {0: "NORMAL", 1: "EMERGENCY_STOP", 2: "NO_SOLUTION", 3: "SINGULARITY", 4: "TARGET_OVER_LIMIT",
              5: "JOINT_COMM_ERR", 6: "BRAKE_NOT_RELEASED", 7: "COLLISION", 8: "DRAG_OVERSPEED",
              9: "JOINT_STATUS_ERR", 10: "OTHER_ERR", 11: "TEACH_RECORD", 12: "TEACH_REPLAY",
              13: "TEACH_PAUSE", 14: "MAIN_NTC_OVERTEMP", 15: "RESISTOR_NTC_OVERTEMP"}
TEACH_STATUS = {0: "OFF", 1: "RECORDING", 2: "RECORD_STOPPED", 3: "REPLAYING", 4: "PAUSED",
                5: "RESUMED", 6: "TERMINATED", 7: "MOVE_TO_START"}


class FrameStore:
    """Latest colour and depth frame of one camera, plus the JPEG encoding cache."""

    def __init__(self, name: str, serial: str):
        self.name, self.serial = name, serial
        self.error: str | None = None
        self.lock = threading.Lock()
        self.color: np.ndarray | None = None     # BGR uint8
        self.depth_mm: np.ndarray | None = None  # uint16, millimetres, 0 = invalid
        self.seq = {"color": 0, "depth": 0}
        self.stamp = {"color": 0.0, "depth": 0.0}
        self._times: dict[str, list[float]] = {"color": [], "depth": []}
        self._cache: dict[str, tuple] = {}

    def start(self) -> None:
        """Frames are pushed in by a feed (DoraFeed), nothing to start."""

    def stop(self) -> None:
        """Nothing owned."""

    def publish(self, kind: str, frame: np.ndarray) -> None:
        now = time.monotonic()
        with self.lock:
            if kind == "color":
                self.color = frame
            else:
                self.depth_mm = frame
            self.seq[kind] += 1
            self.stamp[kind] = now
            times = self._times[kind]
            times.append(now)
            while times and now - times[0] > 2.0:
                times.pop(0)

    def fps(self, kind: str) -> float:
        with self.lock:
            times = list(self._times[kind])
        if len(times) < 2 or time.monotonic() - times[-1] > 1.0:
            return 0.0
        return (len(times) - 1) / (times[-1] - times[0])

    def jpeg(self, kind: str, near: int, far: int) -> tuple[int, bytes, dict] | None:
        """Encoded latest frame as (seq, jpeg, info); cached per sequence number."""
        with self.lock:
            seq = self.seq[kind]
            source = self.color if kind == "color" else self.depth_mm
        if source is None:
            return None
        key = (seq, near, far)
        cached = self._cache.get(kind)
        if cached and cached[0] == key:
            return seq, cached[1], cached[2]
        if kind == "color":
            data, info = cv2.imencode(".jpg", source, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), {}
        else:
            data = cv2.imencode(".jpg", depth_colormap(source, near, far), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            info = {"valid": round(float(np.count_nonzero(source)) / source.size, 3)}
        self._cache[kind] = (key, data, info)
        return seq, data, info

    def depth_at(self, x: int, y: int, radius: int = 2) -> int | None:
        """Median valid depth (mm) in a small window around the pixel, None if there is none."""
        with self.lock:
            depth = self.depth_mm
        if depth is None:
            return None
        h, w = depth.shape
        if not (0 <= x < w and 0 <= y < h):
            return None
        window = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
        valid = window[window > 0]
        return int(np.median(valid)) if valid.size else None


class UsbCamera(FrameStore):
    """Opens one Orbbec device directly: colour + depth software-aligned to the colour grid."""

    def __init__(self, name: str, serial: str, device):
        super().__init__(name, serial)
        self.device = device
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{name}")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Leave the read loop and wait for pipeline.stop(): SDK threads still running at interpreter
        exit abort the process and can leave the device needing a re-enumeration."""
        self._stop_requested.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        import pyorbbecsdk as ob

        try:
            pipeline = ob.Pipeline(self.device)
            config = ob.Config()
            cw, ch, cf = COLOR_MODE
            dw, dh, df = DEPTH_MODE
            colors = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
            depths = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            config.enable_stream(colors.get_video_stream_profile(cw, ch, ob.OBFormat.RGB, cf))
            config.enable_stream(depths.get_video_stream_profile(dw, dh, ob.OBFormat.Y16, df))
            config.set_align_mode(ob.OBAlignMode.SW_MODE)
            pipeline.start(config)
        except Exception as exc:  # most likely: the device is open in another process, or no udev rule
            self.error = f"{type(exc).__name__}: {exc}"
            return
        while not self._stop_requested.is_set():
            try:
                frames = pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                color = frames.get_color_frame()
                if color is not None:
                    w, h = color.get_width(), color.get_height()
                    rgb = np.frombuffer(color.get_data(), dtype=np.uint8).reshape(h, w, 3)
                    self.publish("color", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                depth = frames.get_depth_frame()
                if depth is not None:
                    w, h = depth.get_width(), depth.get_height()
                    raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(h, w)
                    self.publish("depth", (raw * float(depth.get_depth_scale())).astype(np.uint16))
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                break
        pipeline.stop()


def restart_process() -> None:
    """Replace this process by a fresh copy of itself (same arguments)."""
    sys.stdout.flush()
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def _graceful_exit() -> None:
    raise web.GracefulExit()


class DoraFeed(threading.Thread):
    """A dynamic Dora node that routes the camera outputs of the running dataflow into the stores.

    Inputs are named <camera>_color and <camera>_depth.

    Dora 0.4.1 allows one Node per process: a second Node() panics in the Rust runtime (tracing is set
    up once), leaks a thread pool, and the runtime replaces Python's SIGINT handler. So the feed never
    calls Node() speculatively. It polls `dora node list` until a running dataflow declares the dynamic
    node, attaches exactly once, and when that dataflow ends the whole process re-executes itself, which
    gives the next dataflow a fresh runtime and keeps the page alive across skill restarts."""

    def __init__(self, node_id: str, stores: dict[str, FrameStore]):
        super().__init__(daemon=True, name="dora-feed")
        if shutil.which("dora") is None:
            raise RuntimeError("the dora CLI is not on PATH")
        from dora import Node
        from forge_msgs import Image

        self._Node, self._Image = Node, Image
        self.node_id, self.stores = node_id, stores
        self.loop: asyncio.AbstractEventLoop | None = None
        self._stop_requested = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()
        if self.is_alive():
            self.join(timeout=5)

    def _declared(self) -> bool:
        """True when a running dataflow declares the dynamic node (an unattached one is listed as Unknown)."""
        try:
            out = subprocess.run(["dora", "node", "list", "--format", "json"],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
        return any(json.loads(line).get("node") == self.node_id for line in out.splitlines() if line.startswith("{"))

    def run(self) -> None:
        while not self._stop_requested.is_set() and not self._declared():
            self._set_error(f"waiting: no running dataflow declares the dynamic node '{self.node_id}'")
            self._stop_requested.wait(1.0)
        if self._stop_requested.is_set():
            return
        self._set_error(None)
        try:
            node = self._Node(self.node_id)
        except Exception as exc:  # e.g. another process is already attached as this node
            self._set_error(f"cannot attach as '{self.node_id}': {exc}")
            if not self._stop_requested.wait(3.0):
                restart_process()
            return
        # The Dora runtime takes over Ctrl-C (it would arrive as a STOP event and look like the end of the
        # dataflow). It does so a moment after Node() returns, so put the clean shutdown back later than that.
        self.loop.call_soon_threadsafe(
            self.loop.call_later, 2.0, lambda: self.loop.add_signal_handler(signal.SIGINT, _graceful_exit))
        while not self._stop_requested.is_set():
            event = node.next(timeout=0.5)  # a timeout comes back as an ERROR event, None means the stream closed
            if event is None or event["type"] in ("STOP", "ALL_INPUTS_CLOSED"):
                break
            if event["type"] == "INPUT":
                self._on_input(event["id"], event["value"])
        del node  # detach from the daemon
        if not self._stop_requested.is_set():
            restart_process()

    def _set_error(self, message: str | None) -> None:
        for store in self.stores.values():
            store.error = message

    def _on_input(self, input_id: str, value) -> None:
        camera, _, kind = input_id.partition("_")
        store = self.stores.get(camera)
        if store is None or kind not in ("color", "depth"):
            return
        try:
            image = self._Image.from_arrow(value)
            array = image.to_numpy()
            if kind == "color":
                if image.encoding not in ("rgb8", "bgr8"):
                    raise ValueError(f"unsupported colour encoding {image.encoding}")
                frame = array if image.encoding == "bgr8" else cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            else:
                if image.encoding != "32FC1":
                    raise ValueError(f"unsupported depth encoding {image.encoding} (expected 32FC1, metres)")
                frame = (np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0).clip(0, 65535).astype(np.uint16)
        except Exception as exc:
            store.error = f"{input_id}: {type(exc).__name__}: {exc}"
            return
        store.error = None
        store.publish(kind, frame)


def depth_colormap(depth_mm: np.ndarray, near: int, far: int) -> np.ndarray:
    """Turbo colormap, near = red, far = blue, invalid (0) = black."""
    span = max(far - near, 1)
    level = 255 - np.clip((depth_mm.astype(np.float32) - near) * (255.0 / span), 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
    image[depth_mm == 0] = 0
    return image


class ArmListener(threading.Thread):
    """Passive SocketCAN listener for the PIPER feedback frames. It never transmits."""

    def __init__(self, interface: str):
        super().__init__(daemon=True, name="arm-can-listener")
        self.interface = interface
        self.error: str | None = None
        self.lock = threading.Lock()
        self.joints: list[float | None] = [None] * 6   # degrees
        self.pose: list[float | None] = [None] * 6     # x y z (mm), rx ry rz (deg), firmware end pose
        self.gripper = {"mm": None, "effort": None, "status": None}
        self.status: dict | None = None
        self.stamp = {"joints": 0.0, "pose": 0.0, "gripper": 0.0, "status": 0.0}
        self._joint_times: list[float] = []

    def run(self) -> None:
        while True:
            try:
                sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                # Only the arm feedback IDs 0x2A0-0x2AF reach this process.
                sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, struct.pack("=II", 0x2A0, 0x7F0))
                sock.bind((self.interface,))
                sock.settimeout(0.5)
            except OSError as exc:
                self.error = f"{self.interface}: {exc}"
                time.sleep(1.0)
                continue
            self.error = None
            try:
                while True:
                    try:
                        frame = sock.recv(16)
                    except socket.timeout:
                        continue
                    can_id, _dlc, payload = struct.unpack("=IB3x8s", frame)
                    self._decode(can_id & 0x7FF, payload)
            except OSError as exc:  # the interface is down (after a reboot, or while it is re-activated): open again later
                self.error = f"{self.interface}: {exc}"
            finally:
                sock.close()
            time.sleep(1.0)

    def _decode(self, can_id: int, p: bytes) -> None:
        now = time.monotonic()
        with self.lock:
            if can_id == 0x2A1:
                self.status = {"ctrl": p[0], "arm": p[1], "mode": p[2], "teach": p[3], "motion": p[4],
                               "err": int.from_bytes(p[6:8], "big")}
                self.stamp["status"] = now
            elif can_id in (0x2A2, 0x2A3, 0x2A4):
                a, b = struct.unpack(">ii", p)
                first = (can_id - 0x2A2) * 2
                self.pose[first], self.pose[first + 1] = a * 1e-3, b * 1e-3  # mm for xyz, deg for rx ry rz
                self.stamp["pose"] = now
            elif can_id in (0x2A5, 0x2A6, 0x2A7):
                a, b = struct.unpack(">ii", p)
                first = (can_id - 0x2A5) * 2
                self.joints[first], self.joints[first + 1] = a * 1e-3, b * 1e-3
                self.stamp["joints"] = now
                if can_id == 0x2A5:
                    self._joint_times.append(now)
                    while self._joint_times and now - self._joint_times[0] > 2.0:
                        self._joint_times.pop(0)
            elif can_id == 0x2A8:
                angle, effort = struct.unpack(">ih", p[:6])
                self.gripper = {"mm": angle * 1e-3, "effort": effort * 1e-3, "status": p[6]}
                self.stamp["gripper"] = now

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self.lock:
            joints = list(self.joints)
            times = list(self._joint_times)
            age = {k: (round((now - v) * 1000) if v else None) for k, v in self.stamp.items()}
            has_joints = all(j is not None for j in joints)
            rate = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and now - times[-1] < 1 else 0.0
            status = None
            if self.status:
                s = self.status
                status = {"ctrl": CTRL_MODE.get(s["ctrl"], f"0x{s['ctrl']:02X}"),
                          "arm": ARM_STATUS.get(s["arm"], f"0x{s['arm']:02X}"),
                          "teach": TEACH_STATUS.get(s["teach"], f"0x{s['teach']:02X}"),
                          "motion_reached": s["motion"] == 0, "err": s["err"]}
            return {"joints": joints if has_joints else None, "pose": list(self.pose),
                    "gripper": dict(self.gripper), "status": status, "age_ms": age,
                    "joint_hz": round(rate, 1), "error": self.error}


async def websocket(request: web.Request) -> web.WebSocketResponse:
    app = request.app
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    app["websockets"].add(ws)
    loop = asyncio.get_running_loop()
    depth_range = [200, 1500]
    last_seq: dict[tuple[str, str], int] = {}

    async def listen() -> None:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if "depth_range" in data:
                    near, far = (int(v) for v in data["depth_range"])
                    if 0 <= near < far <= 20000:
                        depth_range[:] = [near, far]

    listener = asyncio.create_task(listen())
    try:
        while not ws.closed:
            started = time.monotonic()
            for name, camera in app["cameras"].items():
                for kind in ("color", "depth"):
                    if camera.seq[kind] == last_seq.get((name, kind)):
                        continue
                    packed = await loop.run_in_executor(None, camera.jpeg, kind, *depth_range)
                    if packed is None:
                        continue
                    seq, data, info = packed
                    last_seq[(name, kind)] = seq
                    header = json.dumps({"cam": name, "kind": kind, **info}).encode()
                    await ws.send_bytes(struct.pack(">I", len(header)) + header + data)
            await ws.send_json({"type": "arm", **app["arm"].snapshot()})
            await asyncio.sleep(max(0.0, 1 / STREAM_HZ - (time.monotonic() - started)))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        listener.cancel()
        app["websockets"].discard(ws)
    return ws


async def status(request: web.Request) -> web.Response:
    app = request.app
    cameras = {
        name: {"serial": cam.serial, "error": cam.error, "color_fps": round(cam.fps("color"), 1),
               "depth_fps": round(cam.fps("depth"), 1)}
        for name, cam in app["cameras"].items()
    }
    return web.json_response({"cameras": cameras, "arm": app["arm"].snapshot()})


async def depth_at(request: web.Request) -> web.Response:
    camera = request.app["cameras"].get(request.match_info["cam"])
    if camera is None:
        raise web.HTTPNotFound()
    mm = camera.depth_at(int(request.query["x"]), int(request.query["y"]))
    return web.json_response({"mm": mm})


async def colorbar(request: web.Request) -> web.Response:
    near, far = int(request.query.get("near", 200)), int(request.query.get("far", 1500))
    ramp = np.linspace(near, far, 256, dtype=np.float32).reshape(1, 256)
    image = cv2.resize(depth_colormap(ramp.astype(np.uint16), near, far), (256, 12), interpolation=cv2.INTER_NEAREST)
    return web.Response(body=cv2.imencode(".png", image)[1].tobytes(), content_type="image/png")


async def ledger(request: web.Request) -> web.Response:
    """New robot.run records since byte offset `since` (negative: the last LEDGER_TAIL), as raw JSON lines.

    The skill appends one record when a code block ends, so calls appear block by block."""
    path = SKILL_LOG_DIR / "ledger.jsonl"
    since = int(request.query.get("since", -1))
    size = path.stat().st_size if path.is_file() else 0
    start = since if 0 <= since <= size else 0
    data = b""
    if size:
        with path.open("rb") as fh:
            fh.seek(start)
            data = fh.read()
    end = data.rfind(b"\n") + 1  # a line still being written waits for the next poll
    lines = [line for line in data[:end].splitlines() if line.strip()]
    if since < 0:
        lines = lines[-LEDGER_TAIL:]
    # json.dumps writes NaN / Infinity, which JSON.parse in the browser rejects
    body = re.sub(rb"-?Infinity|NaN", b"null", b",".join(lines))
    return web.Response(body=b'{"offset":%d,"runs":[%s]}' % (start + end, body), content_type="application/json")


class LiveScene:
    """Replies of the scene in progress, from the agent log run_scene.sh tees while PAOS runs.

    PAOS writes the conversation file only when the scene ends; until then this log is the only record of
    what the model says. The log has no times, so each reply is stamped when first seen. Replies already in
    the log when it is first read are stamped with the scene's start. The CLI wraps lines at 80 columns."""

    LOG_NAME = re.compile(r"^(\d{8}-\d{6})_(.+)_agent\.log$")

    def __init__(self) -> None:
        self.path: Path | None = None
        self.offset = 0
        self.replies: list[dict] = []
        self.start = 0.0
        self.scene = ""
        self.stamp = ""

    def poll(self) -> None:
        logs = list((SKILL_LOG_DIR / "scenes").glob("*_agent.log"))
        if not logs:
            return
        newest = max(logs, key=lambda p: p.stat().st_mtime)
        match = self.LOG_NAME.match(newest.name)
        if match is None:
            return
        first_read = newest != self.path
        if first_read:
            self.path, self.offset, self.replies = newest, 0, []
            self.stamp, self.scene = match.groups()
            self.start = time.mktime(time.strptime(self.stamp, "%Y%m%d-%H%M%S"))
        with newest.open("rb") as fh:
            fh.seek(self.offset)
            data = fh.read()
        end = data.rfind(b"\n") + 1
        self.offset += end
        now = self.start if first_read else time.time()
        for line in data[:end].decode("utf-8", "replace").splitlines():
            if line.startswith("  \u21b3 "):
                self.replies.append({"t": now, "text": line[4:].strip()})
            elif line.strip() and self.replies:
                previous = self.replies[-1]["text"]
                piece = line.strip()
                gap = " " if previous[-1:].isascii() and piece[:1].isascii() else ""  # undo the wrapping
                self.replies[-1]["text"] = previous + gap + piece

    def snapshot(self) -> dict:
        task, ended = "", False
        meta = SKILL_LOG_DIR / "scenes" / f"{self.stamp}_{self.scene}.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                task, ended = data.get("task", ""), "end" in data
            except ValueError:  # being rewritten: next poll
                pass
        saved = (SESSION_DIR / f"scene_{self.scene}_{self.stamp}.jsonl").is_file()
        # An interrupted agent leaves neither an end time nor a conversation file.
        alive = subprocess.run(["pgrep", "-f", f"paos agent .*scene:{self.scene}:{self.stamp}"],
                               capture_output=True).returncode == 0
        ended = ended or not alive
        return {"scene": self.scene, "stamp": self.stamp, "start": self.start, "task": task, "ended": ended,
                "saved": saved, "replies": [] if saved else self.replies}


async def watch_live_scene(app: web.Application) -> None:
    while True:
        app["live_scene"].poll()
        await asyncio.sleep(0.5)


async def start_live_scene(app: web.Application) -> None:
    app["live_scene_task"] = asyncio.create_task(watch_live_scene(app))


async def stop_live_scene(app: web.Application) -> None:
    app["live_scene_task"].cancel()


async def live_scene(request: web.Request) -> web.Response:
    return web.json_response(request.app["live_scene"].snapshot())


def _message_text(content) -> str:
    """Text of a stored message; image parts are stored as the placeholder "[image]" and left out."""
    if isinstance(content, str):
        return content
    parts = [p.get("text", "") for p in content or [] if isinstance(p, dict)]
    return "\n".join(t for t in parts if t != "[image]")


def _load_scene(path: Path) -> dict:
    """One PAOS scene conversation, reduced to what the page shows. PAOS writes the file when the scene ends."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = rows[0] if rows and rows[0].get("_type") == "metadata" else {}
    messages = []
    for row in rows:
        role = row.get("role")
        if role is None:
            continue
        text = _message_text(row.get("content"))
        message = {"role": role, "text": text[:3000] if role == "tool" else text}
        if role == "assistant":
            message["reasoning"] = (row.get("reasoning_content") or "")[:8000]
            message["calls"] = [{"id": c.get("id"), "name": c.get("function", {}).get("name"),
                                 "args": (c.get("function", {}).get("arguments") or "")[:4000]}
                                for c in row.get("tool_calls") or []]
        elif role == "tool":
            found = re.search(r'"invocation_id":\s*"([^"]+)"', text)
            message.update(call_id=row.get("tool_call_id"), name=row.get("name"), invocation=found and found.group(1))
        messages.append(message)
    return {"name": path.name, "key": meta.get("key"), "created_at": meta.get("created_at"), "messages": messages}


async def scenes(request: web.Request) -> web.Response:
    """The most recent scene conversations: names and modification times, so the page fetches only changes."""
    files = sorted(SESSION_DIR.glob("scene_*.jsonl"), key=lambda p: p.stat().st_mtime)[-SCENE_TAIL:]
    return web.json_response([{"name": p.name, "mtime": p.stat().st_mtime} for p in files])


async def scene(request: web.Request) -> web.Response:
    name = request.query.get("name", "")
    path = SESSION_DIR / name
    if Path(name).name != name or not name.startswith("scene_") or not name.endswith(".jsonl") or not path.is_file():
        raise web.HTTPNotFound()
    return web.json_response(_load_scene(path))


async def shot(request: web.Request) -> web.FileResponse:
    """An image the skill saved for a show() call; nothing outside the skill's log folder is served."""
    path = Path(request.query.get("path", "")).resolve()
    if (not path.is_relative_to(SKILL_LOG_DIR.resolve()) or path.suffix.lower() not in (".jpg", ".jpeg", ".png")
            or not path.is_file()):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


async def mesh(request: web.Request) -> web.FileResponse:
    name = request.match_info["name"]
    path = MESH_DIR / f"{name}.stl"
    if not re.fullmatch(r"[a-z0-9_]+", name) or not path.is_file():
        raise web.HTTPNotFound()
    return web.FileResponse(path)


@web.middleware
async def same_origin_only(request: web.Request, handler):
    """Keep other websites out: the page streams the cameras, and browsers let any site open a
    WebSocket to 127.0.0.1. A foreign Host (DNS rebinding) or a foreign Origin on /ws is refused;
    requests without an Origin header (curl, scripts) are not browsers and pass."""
    if request.url.host not in request.app["allowed_hosts"]:
        raise web.HTTPForbidden(text="unexpected Host header")
    origin = request.headers.get("Origin")
    if request.path == "/ws" and origin is not None and urlparse(origin).netloc != request.host:
        raise web.HTTPForbidden(text="cross-origin WebSocket refused")
    return await handler(request)


async def close_websockets(app: web.Application) -> None:
    for ws in set(app["websockets"]):
        await ws.close(code=WSCloseCode.GOING_AWAY, message=b"server shutdown")


async def start_feed(app: web.Application) -> None:
    app["feed"].loop = asyncio.get_running_loop()
    app["feed"].start()


async def stop_sources(app: web.Application) -> None:
    """Detach / stop the frame sources cleanly (a Dora node must leave the daemon, Orbbec pipelines must stop)."""
    loop = asyncio.get_running_loop()
    if "feed" in app:
        await loop.run_in_executor(None, app["feed"].stop)
    for camera in app["cameras"].values():
        await loop.run_in_executor(None, camera.stop)


def make_app(args: argparse.Namespace) -> web.Application:
    app = web.Application(middlewares=[same_origin_only])
    app["allowed_hosts"] = {"127.0.0.1", "localhost", "::1", args.host}
    app["cameras"] = {}
    app["websockets"] = set()
    app.on_shutdown.append(close_websockets)
    if args.source == "dora":
        app["cameras"] = {name: FrameStore(name, "via Dora") for name in ("head", "wrist")}
        app["feed"] = DoraFeed(args.node_id, app["cameras"])
        app.on_startup.append(start_feed)
    else:
        import pyorbbecsdk as ob

        ob.Context.set_logger_to_console(ob.OBLogLevel.ERROR)  # the SDK would otherwise write ./Log/ to disk
        context = ob.Context()
        devices = context.query_devices()
        for name, serial in (("head", args.head), ("wrist", args.wrist)):
            try:
                device = devices.get_device_by_serial_number(serial)
            except Exception as exc:
                camera = UsbCamera(name, serial, None)
                camera.error = f"device {serial} not found: {exc}"
            else:
                camera = UsbCamera(name, serial, device)
                camera.start()
                time.sleep(1.0)  # stagger USB enumeration between the two cameras
            app["cameras"][name] = camera
        app["_orbbec_context"] = context  # keep the SDK context alive for the lifetime of the app
    app.on_cleanup.append(stop_sources)
    app["live_scene"] = LiveScene()
    app.on_startup.append(start_live_scene)
    app.on_cleanup.append(stop_live_scene)
    app["arm"] = ArmListener(args.can)
    app["arm"].start()

    app.router.add_get("/", lambda r: web.FileResponse(HERE / "index.html", headers={"Cache-Control": "no-store"}))
    app.router.add_get("/favicon.ico", lambda r: web.Response(status=204))
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/depth/{cam}", depth_at)
    app.router.add_get("/api/colorbar.png", colorbar)
    app.router.add_get("/api/ledger", ledger)
    app.router.add_get("/api/shot", shot)
    app.router.add_get("/api/scenes", scenes)
    app.router.add_get("/api/scene", scene)
    app.router.add_get("/api/live_scene", live_scene)
    app.router.add_get("/robot/piper.urdf", lambda r: web.FileResponse(URDF))
    app.router.add_get("/robot/mesh/{name}.stl", mesh)
    app.router.add_static("/vendor/", HERE / "vendor")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--can", default="can0", help="SocketCAN interface of the arm (listen only)")
    parser.add_argument("--source", choices=("dora", "usb"), default="dora",
                        help="dora: attach to the running skill as a dynamic node; usb: open the cameras directly")
    parser.add_argument("--node-id", default="live_view", help="id of the dynamic node in the dataflow (dora source)")
    parser.add_argument("--head", default=CAMERAS["head"], help="head camera serial number (usb source)")
    parser.add_argument("--wrist", default=CAMERAS["wrist"], help="wrist camera serial number (usb source)")
    args = parser.parse_args()

    web.run_app(make_app(args), host=args.host, port=args.port, shutdown_timeout=2.0, print=lambda *_: print(
        f"live view: http://{args.host}:{args.port}  (read-only; cameras: {args.source}; CAN {args.can})"))


if __name__ == "__main__":
    main()
