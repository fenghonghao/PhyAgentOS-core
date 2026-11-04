#!/usr/bin/env python3
"""Fix the v1.1.0 capture worker's multiprocessing queue shutdown deadlock."""
from pathlib import Path
import sys


def patch(root: Path) -> None:
    path = root / "forge_devices_orbbec_camera" / "isolated_backend.py"
    source = path.read_text()
    original = '    """Run pyorbbecsdk in a clean child process with no Dora imports."""\n'
    replacement = original + (
        "    # The parent stops draining frames at shutdown. Do not wait for the\n"
        "    # queue feeder to flush old images into a pipe nobody will read.\n"
        "    frame_queue.cancel_join_thread()\n"
    )
    if replacement in source:
        return
    if source.count(original) != 1:
        raise RuntimeError(f"Unexpected Orbbec adapter source: {path}")
    path.write_text(source.replace(original, replacement))


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
