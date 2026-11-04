"""Remaining time of a timed scene, shown to the model on every request.

A scene runner (e.g. a competition launcher) exports ``PAOS_SCENE_DEADLINE`` as a Unix
timestamp in seconds. When it is set, every model request carries the seconds left so the
model can decide when to finish. This is information only: nothing here stops or limits
execution.
"""

from __future__ import annotations

import os
import time

from loguru import logger

SCENE_DEADLINE_ENV = "PAOS_SCENE_DEADLINE"


def scene_time_left(now: float | None = None) -> float | None:
    """Seconds until the scene deadline (negative once passed), or None when unset/invalid."""
    raw = os.environ.get(SCENE_DEADLINE_ENV, "").strip()
    if not raw:
        return None
    try:
        deadline = float(raw)
    except ValueError:
        logger.warning("Ignoring {}={!r}: not a Unix timestamp", SCENE_DEADLINE_ENV, raw)
        return None
    return round(deadline - (time.time() if now is None else now), 1)


def time_left_note(now: float | None = None) -> str | None:
    """``[time_left_s=…]`` for appending to the newest message, or None when no deadline is set."""
    left = scene_time_left(now)
    return None if left is None else f"[time_left_s={left:.1f}]"
