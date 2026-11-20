"""Camera capture thread: always keeps only the newest frame, reconnects on failure."""

from __future__ import annotations

import hashlib
import logging
import threading
import time

import cv2

from app.types import FrameData

logger = logging.getLogger("attendance.capture")


class LatestFrameReader:
    """Thread that continuously reads from a camera and exposes only the newest frame."""

    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
        fps: int,
        rotate_deg: int = 0,
        reconnect_backoff_s: float = 1.0,
        watchdog_timeout_s: float = 3.0,
    ) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._fps = fps
        self._rotate_deg = rotate_deg
        self._reconnect_backoff_s = reconnect_backoff_s
        self._watchdog_timeout_s = watchdog_timeout_s

        self._lock = threading.Lock()
        self._frame_data: FrameData | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Stats
        self._last_frame_time: float = 0.0
        self._reconnect_count: int = 0
        self._frame_id: int = 0
        self._fps_window: list[float] = []

    def start(self) -> None:
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        self._thread.start()
        logger.info("Capture thread started for source %s", self._source)

    def read(self) -> FrameData | None:
        """Return the newest frame (non-blocking). Returns None if no frame yet."""
        with self._lock:
            return self._frame_data

    def is_alive(self) -> bool:
        """Return True if a fresh frame arrived within the watchdog window."""
        if self._last_frame_time == 0.0:
            return False
        return (time.time() - self._last_frame_time) < self._watchdog_timeout_s

    def stats(self) -> dict:
        """Return camera health stats."""
        age = time.time() - self._last_frame_time if self._last_frame_time > 0 else float("inf")
        fps = self._compute_fps()
        return {
            "fps": fps,
            "last_frame_age_s": age,
            "reconnect_count": self._reconnect_count,
            "is_alive": self.is_alive(),
            "frame_id": self._frame_id,
        }

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Capture thread stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Open a VideoCapture and configure it. Returns None on failure."""
        try:
            if isinstance(self._source, int):
                # Try CAP_DSHOW first on Windows, fall back to default if it fails
                cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(self._source)
            else:
                cap = cv2.VideoCapture(self._source)

            if not cap.isOpened():
                return None

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_FPS, self._fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as exc:
            logger.error("Failed to open capture: %s", exc)
            return None

    def _capture_loop(self) -> None:
        """Main loop: open camera, read frames, reconnect on failure."""
        backoff = self._reconnect_backoff_s
        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                logger.warning("Could not open camera source %s, retrying in %.1fs", self._source, backoff)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 1.5, 10.0)
                self._reconnect_count += 1
                continue

            backoff = self._reconnect_backoff_s
            logger.info("Camera opened: source=%s width=%d height=%d", self._source, self._width, self._height)
            prev_hash: bytes | None = None

            while not self._stop_event.is_set():
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        logger.warning("Camera read failed, reconnecting…")
                        break

                    now = time.time()
                    # Apply rotation if configured (e.g. mobile held vertically)
                    if self._rotate_deg in (90, -270):
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif self._rotate_deg in (180, -180):
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif self._rotate_deg in (270, -90):
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    # Detect frozen frame by MD5 of a downsampled thumbnail
                    thumb = cv2.resize(frame, (32, 18))
                    h = hashlib.md5(thumb.tobytes(), usedforsecurity=False).digest()
                    if h == prev_hash:
                        if (now - self._last_frame_time) > self._watchdog_timeout_s:
                            logger.warning("Frozen frame detected, reconnecting…")
                            break
                    else:
                        prev_hash = h
                        self._frame_id += 1
                        self._fps_window.append(now)
                        # Keep a 2-second window for FPS estimate
                        cutoff = now - 2.0
                        self._fps_window = [t for t in self._fps_window if t > cutoff]

                    with self._lock:
                        self._frame_data = FrameData(image=frame, timestamp=now, frame_id=self._frame_id)
                    self._last_frame_time = now

                except Exception as exc:
                    logger.error("Exception in capture loop: %s", exc)
                    break

            cap.release()
            if not self._stop_event.is_set():
                self._reconnect_count += 1
                logger.info("Reconnecting (attempt %d)…", self._reconnect_count)
                self._stop_event.wait(timeout=backoff)
                backoff = min(backoff * 1.5, 10.0)

    def _compute_fps(self) -> float:
        if len(self._fps_window) < 2:
            return 0.0
        span = self._fps_window[-1] - self._fps_window[0]
        if span <= 0:
            return 0.0
        return (len(self._fps_window) - 1) / span

