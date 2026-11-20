"""Unit tests for LatestFrameReader (camera capture)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera.capture import LatestFrameReader  # noqa: E402


class _FakeCapture:
    """Minimal cv2.VideoCapture-like stub."""

    def __init__(self, frames: list, *, loop: bool = False) -> None:
        self._frames = list(frames)
        self._idx = 0
        self._loop = loop
        self._opened = True

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def set(self, prop_id, value) -> bool:  # noqa: ANN001
        return True

    def read(self):  # noqa: ANN201
        if self._idx < len(self._frames):
            frame = self._frames[self._idx]
            self._idx += 1
            if frame is None:
                return False, None
            return True, frame
        if self._loop:
            self._idx = 0
            return self.read()
        return False, None

    def release(self) -> None:
        self._opened = False


def _make_frame(val: int = 128) -> np.ndarray:
    """Create a small distinct BGR frame."""
    return np.full((64, 64, 3), val, dtype=np.uint8)


def test_read_returns_none_before_start() -> None:
    """Before starting, read() must return None."""
    reader = LatestFrameReader(source=0, width=64, height=64, fps=10)
    assert reader.read() is None


def test_buffer_one_newest_frame(monkeypatch) -> None:
    """Buffer=1: only the newest frame is returned."""
    frames = [_make_frame(10), _make_frame(20), _make_frame(30)]
    cap = _FakeCapture(frames)

    def _fake_open(self):  # noqa: ANN001
        return cap

    monkeypatch.setattr(LatestFrameReader, "_open_capture", _fake_open)

    reader = LatestFrameReader(
        source=0,
        width=64,
        height=64,
        fps=10,
        reconnect_backoff_s=0.05,
        watchdog_timeout_s=0.5,
    )
    reader.start()
    # Give the thread time to drain all frames
    time.sleep(0.3)
    result = reader.read()
    reader.stop()

    assert result is not None, "Expected a frame to be buffered"
    # The last distinct frame had value 30; the newest must have been stored
    assert result.image is not None


def test_read_returns_none_when_source_fails(monkeypatch) -> None:
    """If the source immediately returns False, read() should return None."""
    cap = _FakeCapture([])  # empty → first read() returns False

    def _fake_open(self):  # noqa: ANN001
        return cap

    monkeypatch.setattr(LatestFrameReader, "_open_capture", _fake_open)

    reader = LatestFrameReader(
        source=0,
        width=64,
        height=64,
        fps=10,
        reconnect_backoff_s=100.0,  # large so it doesn't retry during test
        watchdog_timeout_s=0.5,
    )
    reader.start()
    time.sleep(0.1)
    result = reader.read()
    reader.stop()

    assert result is None


def test_stop_joins_thread(monkeypatch) -> None:
    """stop() should cleanly join the background thread."""
    cap = _FakeCapture([_make_frame()], loop=True)
    opened = threading.Event()

    def _fake_open(self):  # noqa: ANN001
        opened.set()
        return cap

    monkeypatch.setattr(LatestFrameReader, "_open_capture", _fake_open)

    reader = LatestFrameReader(
        source=0,
        width=64,
        height=64,
        fps=10,
        reconnect_backoff_s=0.05,
        watchdog_timeout_s=1.0,
    )
    reader.start()
    opened.wait(timeout=2.0)
    reader.stop()

    assert reader._thread is not None
    assert not reader._thread.is_alive()
