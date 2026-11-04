"""Bound Dora 0.4.1 camera buffers until upstream drop-stream fix #1417 is deployed.

Small, unsubscribed outputs service Dora's pending release notifications without allocating
shared-memory frames. This is a compatibility workaround, not a replacement for the upstream fix.
"""
from pathlib import Path
import time

import pyarrow as pa

RECLAIM_OUTPUT = "_shm_reclaim"


def mapped_segments() -> int:
    """Count this process's Dora mappings, including unlinked-but-still-mapped regions."""
    return len({line.split('/dev/shm/', 1)[1].split()[0]
                for line in Path('/proc/self/maps').read_text().splitlines()
                if '/dev/shm/shmem_' in line})


class ReclaimingNode:
    """Keep sender memory bounded even when a subscriber stops releasing frames."""

    def __init__(self, node, *, limit: int = 24, timeout: float = 2.0):
        self._node = node
        self._limit = limit
        self._timeout = timeout
        self._pulse = pa.array([0], type=pa.uint8())  # below Dora's 4096-byte SHM threshold

    def __iter__(self):
        return iter(self._node)

    def __getattr__(self, name):
        return getattr(self._node, name)

    def send_output(self, output_id, data, metadata=None):
        self._node.send_output(output_id, data, metadata=metadata or {})
        # A zero-capacity Rust channel may deliver only one acknowledgement per send.
        # Multiple inline sends let the background thread advance without allocating frames.
        for _ in range(4):
            self._node.send_output(RECLAIM_OUTPUT, self._pulse)
        deadline = time.monotonic() + self._timeout
        while mapped_segments() > self._limit:
            if time.monotonic() >= deadline:
                raise RuntimeError("Dora shared-memory release stalled; stopping camera before /dev/shm fills")
            time.sleep(0.001)
            self._node.send_output(RECLAIM_OUTPUT, self._pulse)
