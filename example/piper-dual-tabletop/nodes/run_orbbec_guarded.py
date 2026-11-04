#!/usr/bin/env python3
"""Run the official Orbbec 1.1.0 adapter with a Dora 0.4.1 release guard."""
import os
from pathlib import Path
import sys

deps = Path(os.environ.get("CAMERA_NODES_DIR", str(Path.home() / ".PhyAgentOS/camera-nodes"))) / "orbbec-camera-1.1.0-shmfix"
if not (deps / "forge_devices_orbbec_camera").is_dir():
    raise SystemExit("Missing guarded camera dependencies; run scripts/setup_guarded_camera.sh")
sys.path.insert(0, str(deps))

# Importing the adapter sets its telemetry environment before importing Dora.
from forge_devices_orbbec_camera.main import main
import dora
from dora_shm_guard import ReclaimingNode

_Node = dora.Node


def guarded_node(*args, **kwargs):
    return ReclaimingNode(_Node(*args, **kwargs))


if __name__ == "__main__":
    dora.Node = guarded_node
    raise SystemExit(main())
