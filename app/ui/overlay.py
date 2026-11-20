"""OpenCV overlay drawing: boxes, labels, status bar, enrollment guide."""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.types import TrackView

# State colours (BGR)
_COLORS = {
    "verifying": (0, 200, 255),     # yellow-orange
    "confirmed": (0, 220, 60),      # green
    "unknown": (0, 50, 230),        # red
    "spoof": (0, 0, 200),           # dark red
    "poor_quality": (140, 140, 140),# gray
}
_DEFAULT_COLOR = (200, 200, 200)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LINE = cv2.LINE_AA


def draw_views(frame: np.ndarray, views: list[TrackView], status: dict) -> np.ndarray:
    """Draw bounding boxes and state labels for each track."""
    out = frame.copy()
    for v in views:
        color = _COLORS.get(v.state, _DEFAULT_COLOR)
        x1, y1, x2, y2 = (int(c) for c in v.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = _make_label(v)
        # Background strip
        (tw, th), _ = cv2.getTextSize(label, _FONT, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4), _FONT, 0.55, (255, 255, 255), 1, _LINE)

        # Hint below box
        if v.hint:
            cv2.putText(out, v.hint, (x1, y2 + 18), _FONT, 0.45, color, 1, _LINE)

    return out


def draw_attendance_banner(frame: np.ndarray, name: str, roll_no: str, time_left_s: float) -> np.ndarray:
    """Draw a celebratory banner when a student is recognized and attendance is marked."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Banner in lower third
    bw = int(w * 0.85)
    bh = 110
    bx1 = (w - bw) // 2
    by1 = h - bh - 30
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Semi-transparent dark green background
    overlay = out.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 100, 25), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Bright green border
    cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 230, 80), 3)

    # Text
    title = "ATTENDANCE MARKED SUCCESSFULLY!"
    details = f"Name: {name}   |   Roll No: {roll_no}"

    cv2.putText(out, title, (bx1 + 25, by1 + 42), _FONT, 0.85, (0, 255, 120), 2, _LINE)
    cv2.putText(out, details, (bx1 + 25, by1 + 82), _FONT, 0.75, (255, 255, 255), 2, _LINE)
    return out


def _make_label(v: TrackView) -> str:
    if v.state == "confirmed":
        name = v.name or "?"
        roll = f" [{v.roll_no}]" if v.roll_no else ""
        score_str = f" {v.score:.2f}" if v.score is not None else ""
        marker = " ✓" if v.already_marked else ""
        return f"{name}{roll}{score_str}{marker}"
    elif v.state == "verifying":
        return "Verifying…"
    elif v.state == "unknown":
        return "Unknown"
    elif v.state == "spoof":
        return "SPOOF"
    elif v.state == "poor_quality":
        return v.hint or "Poor Quality"
    return v.state


def draw_status_bar(frame: np.ndarray, status: dict) -> np.ndarray:
    """Draw session info, FPS, camera health, and marked count at the top."""
    h, w = frame.shape[:2]
    bar_h = 32
    cv2.rectangle(frame, (0, 0), (w, bar_h), (30, 30, 30), -1)

    session_name = status.get("session_name", "No Active Session")
    fps = status.get("fps", 0.0)
    marked = status.get("marked_count", 0)
    cam_ok = status.get("camera_online", True)
    cam_str = "CAM OK" if cam_ok else "CAM OFFLINE"
    cam_color = (0, 220, 60) if cam_ok else (0, 50, 230)

    ts = time.strftime("%H:%M:%S")

    left = f"Session: {session_name}  |  {ts}  |  Marked: {marked}"
    right = f"FPS: {fps:.1f}  "
    cv2.putText(frame, left, (8, 22), _FONT, 0.5, (220, 220, 220), 1, _LINE)
    (rw, _), _ = cv2.getTextSize(right, _FONT, 0.5, 1)
    cv2.putText(frame, right, (w - rw - 60, 22), _FONT, 0.5, (220, 220, 220), 1, _LINE)
    cv2.putText(frame, cam_str, (w - 55, 22), _FONT, 0.45, cam_color, 1, _LINE)

    return frame


def draw_enroll_guide(frame: np.ndarray, hint: str | None, progress: float) -> np.ndarray:
    """Draw enrollment oval guide, hint text, and progress bar."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    # Oval guide
    color = (0, 220, 60) if progress > 0.5 else (0, 200, 255)
    cv2.ellipse(frame, (cx, cy), (110, 145), 0, 0, 360, color, 3)

    # Hint text
    if hint:
        (tw, _), _ = cv2.getTextSize(hint, _FONT, 0.7, 2)
        cv2.putText(frame, hint, ((w - tw) // 2, cy + 180), _FONT, 0.7, color, 2, _LINE)

    # Progress bar
    bar_w = int(w * 0.6)
    bar_x = (w - bar_w) // 2
    bar_y = h - 50
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18), (60, 60, 60), -1)
    filled = int(bar_w * min(1.0, max(0.0, progress)))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 18), (0, 220, 60), -1)
    cv2.putText(frame, f"{int(progress * 100)}%", (bar_x + bar_w + 8, bar_y + 14), _FONT, 0.5, (220, 220, 220), 1)

    return frame

