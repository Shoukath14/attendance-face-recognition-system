"""Camera health monitoring: FPS estimation and frozen-frame detection."""

from __future__ import annotations

import time
from collections import deque


class CameraHealth:
    """Tracks camera FPS, last-frame age, and frozen-frame state."""

    def __init__(self, fps_window_s: float = 2.0, watchdog_timeout_s: float = 3.0) -> None:
        self._fps_window_s = fps_window_s
        self._watchdog_timeout_s = watchdog_timeout_s
        self._timestamps: deque[float] = deque()
        self._last_time: float = 0.0
        self._is_frozen: bool = False

    def record_frame(self, timestamp: float | None = None) -> None:
        """Call once per new frame received."""
        now = timestamp or time.time()
        self._last_time = now
        self._timestamps.append(now)
        cutoff = now - self._fps_window_s
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
        self._is_frozen = False

    def mark_frozen(self) -> None:
        """Call when the capture thread detects a frozen frame."""
        self._is_frozen = True

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / span if span > 0 else 0.0

    @property
    def last_frame_age_s(self) -> float:
        if self._last_time == 0.0:
            return float("inf")
        return time.time() - self._last_time

    @property
    def is_online(self) -> bool:
        return self.last_frame_age_s < self._watchdog_timeout_s and not self._is_frozen

    def status_dict(self) -> dict:
        return {
            "fps": round(self.fps, 1),
            "last_frame_age_s": round(self.last_frame_age_s, 2),
            "is_online": self.is_online,
            "is_frozen": self._is_frozen,
        }

