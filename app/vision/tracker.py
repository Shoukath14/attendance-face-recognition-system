"""IoU-based face tracker producing stable track IDs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.types import Detection

logger = logging.getLogger("attendance.tracker")


@dataclass
class Track:
    track_id: int
    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    age: int = 0
    last_seen_frame: int = 0
    history: list[tuple[float, float, float, float]] = field(default_factory=list)
    detection: Detection | None = None


class IoUTracker:
    """Simple IoU tracker: matches detections to tracks each frame."""

    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 15) -> None:
        self._iou_threshold = iou_threshold
        self._max_age_frames = max_age_frames
        self._tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame_id: int) -> list[Track]:
        """Match detections to tracks. Returns all currently live tracks."""
        if not detections:
            self._age_out(frame_id)
            return list(self._tracks)

        det_boxes = np.array([d.box for d in detections], dtype=np.float32)
        track_boxes = np.array([t.box for t in self._tracks], dtype=np.float32) if self._tracks else np.empty((0, 4))

        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        if len(self._tracks) > 0:
            iou_mat = _iou_matrix(det_boxes, track_boxes)
            # Greedy matching by descending IoU
            for _ in range(min(len(detections), len(self._tracks))):
                if iou_mat.size == 0:
                    break
                ri, ci = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[ri, ci] < self._iou_threshold:
                    break
                di, ti = int(ri), int(ci)
                self._update_track(self._tracks[ti], detections[di], frame_id)
                matched_det.add(di)
                matched_trk.add(ti)
                iou_mat[ri, :] = -1
                iou_mat[:, ci] = -1

        # New tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                t = Track(
                    track_id=self._next_id,
                    box=det.box,
                    landmarks=det.landmarks,
                    last_seen_frame=frame_id,
                    detection=det,
                )
                self._tracks.append(t)
                self._next_id += 1

        self._age_out(frame_id)
        return list(self._tracks)

    def predict(self, frame_id: int) -> list[Track]:
        """On non-detection frames: age tracks and return current list."""
        for t in self._tracks:
            t.age += 1
        self._age_out(frame_id)
        return list(self._tracks)

    def _update_track(self, track: Track, det: Detection, frame_id: int) -> None:
        track.box = det.box
        track.landmarks = det.landmarks
        track.last_seen_frame = frame_id
        track.age = 0
        track.detection = det
        track.history.append(det.box)
        if len(track.history) > 10:
            track.history.pop(0)

    def _age_out(self, frame_id: int) -> None:
        self._tracks = [
            t for t in self._tracks
            if (frame_id - t.last_seen_frame) <= self._max_age_frames
        ]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _iou_matrix(dets: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    mat = np.zeros((len(dets), len(tracks)), dtype=np.float32)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            mat[i, j] = _iou(d, t)
    return mat

