"""Thread-safe shared state between inference thread and display loop."""

from __future__ import annotations

import threading

from app.types import TrackView


class SharedState:
    """
    Published by the inference thread, consumed by the display loop.
    Copy-on-read under a lock so the display never blocks on inference.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_id: int = -1
        self._views: list[TrackView] = []
        self._status: dict = {}

    def publish(self, frame_id: int, views: list[TrackView], status: dict) -> None:
        with self._lock:
            self._frame_id = frame_id
            self._views = list(views)
            self._status = dict(status)

    def snapshot(self) -> tuple[int, list[TrackView], dict]:
        """Return a snapshot copy (frame_id, views, status). Never blocks display."""
        with self._lock:
            return self._frame_id, list(self._views), dict(self._status)

