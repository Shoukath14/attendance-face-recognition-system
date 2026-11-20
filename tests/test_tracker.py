"""Unit tests for IoUTracker."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.types import Detection  # noqa: E402
from app.vision.tracker import IoUTracker  # noqa: E402


def _det(x1: float, y1: float, x2: float, y2: float, score: float = 0.9) -> Detection:
    lm = np.zeros((5, 2), dtype=np.float32)
    return Detection(box=(x1, y1, x2, y2), score=score, landmarks=lm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_same_box_same_id() -> None:
    """Two consecutive identical boxes → same track ID."""
    tracker = IoUTracker(iou_threshold=0.3, max_age_frames=15)
    det = _det(10, 10, 110, 110)

    tracks1 = tracker.update([det], frame_id=1)
    id1 = tracks1[0].track_id

    tracks2 = tracker.update([det], frame_id=2)
    id2 = tracks2[0].track_id

    assert id1 == id2, f"Expected same track ID but got {id1} vs {id2}"


def test_track_expires_after_max_age() -> None:
    """After max_age frames with no detection → track is removed."""
    max_age = 3
    tracker = IoUTracker(iou_threshold=0.3, max_age_frames=max_age)
    det = _det(10, 10, 110, 110)

    tracker.update([det], frame_id=1)

    # Feed frames without detections until beyond max_age
    for fid in range(2, 2 + max_age + 2):
        tracks = tracker.update([], frame_id=fid)

    assert len(tracks) == 0, "Expected track to expire but it's still alive"


def test_non_overlapping_boxes_two_ids() -> None:
    """Two non-overlapping boxes in the same frame → two different IDs."""
    tracker = IoUTracker(iou_threshold=0.3, max_age_frames=15)
    det_a = _det(0, 0, 50, 50)
    det_b = _det(200, 200, 300, 300)  # no overlap with det_a

    tracks = tracker.update([det_a, det_b], frame_id=1)

    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2, f"Expected 2 unique IDs, got: {ids}"


def test_moving_track_keeps_id() -> None:
    """A box that moves slightly frame-to-frame keeps its track ID."""
    tracker = IoUTracker(iou_threshold=0.3, max_age_frames=15)

    boxes = [
        (10, 10, 110, 110),
        (12, 10, 112, 110),
        (14, 10, 114, 110),
    ]
    first_id = None
    for fid, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        tracks = tracker.update([_det(x1, y1, x2, y2)], frame_id=fid)
        assert len(tracks) == 1
        if first_id is None:
            first_id = tracks[0].track_id
        else:
            assert tracks[0].track_id == first_id
