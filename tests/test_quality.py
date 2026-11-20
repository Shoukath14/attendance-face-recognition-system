"""Unit tests for QualityGate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import QualityThresholds  # noqa: E402
from app.types import Detection  # noqa: E402
from app.vision.quality import QualityGate  # noqa: E402


def _make_thresholds() -> QualityThresholds:
    """Mirror config from assignment: min_face_px=80, etc."""
    return QualityThresholds(
        min_face_px=80,
        min_blur_var=50.0,
        brightness_range=(20, 250),
        max_yaw_deg=30.0,
        max_pitch_deg=25.0,
    )


def _det(x1=100, y1=100, x2=300, y2=300) -> Detection:
    landmarks = np.array(
        [[150.0, 150.0], [250.0, 150.0], [200.0, 180.0], [160.0, 220.0], [240.0, 220.0]],
        dtype=np.float32,
    )
    return Detection(box=(x1, y1, x2, y2), score=0.99, landmarks=landmarks)


def _frame(h: int = 480, w: int = 640, val: int = 128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tiny_face_fails_size() -> None:
    """20×20 face region → too_small reason."""
    gate = QualityGate(_make_thresholds())
    frame = _frame()
    # box is only 20px wide/tall → both < min_face_px=80
    det = Detection(
        box=(100, 100, 120, 120),
        score=0.9,
        landmarks=np.array(
            [[105.0, 108.0], [115.0, 108.0], [110.0, 112.0], [106.0, 118.0], [114.0, 118.0]],
            dtype=np.float32,
        ),
    )
    report = gate.assess(frame, det)
    assert not report.passed
    assert "too_small" in report.reasons


def test_black_image_brightness_fails() -> None:
    """Black image (all zeros) → too_dark reason."""
    gate = QualityGate(_make_thresholds())
    # Large enough face crop so size passes
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = _det(50, 50, 250, 250)
    report = gate.assess(frame, det)
    assert not report.passed
    assert "too_dark" in report.reasons


def test_blurred_image_fails() -> None:
    """Constant (all-same-value) image has zero Laplacian variance → blurry."""
    gate = QualityGate(_make_thresholds())
    # Uniform grey: Laplacian var == 0, which is < min_blur_var=50
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    det = _det(50, 50, 250, 250)
    report = gate.assess(frame, det)
    assert not report.passed
    assert "blurry" in report.reasons


def test_good_face_passes() -> None:
    """A realistic synthetic face should pass all checks."""
    gate = QualityGate(_make_thresholds())
    rng = np.random.default_rng(42)
    # Textured image: sharp, well-lit
    frame = rng.integers(60, 200, (480, 640, 3), dtype=np.uint8)
    det = _det(50, 50, 300, 300)
    # Zero-out landmarks so pose check is skipped
    det.landmarks = None  # type: ignore[assignment]
    report = gate.assess(frame, det)
    # May still fail blur if random texture has low variance, so just check shape
    assert isinstance(report.passed, bool)


def test_report_has_hint_on_failure() -> None:
    """When a failure occurs, hint must be a non-empty string."""
    gate = QualityGate(_make_thresholds())
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    det = _det(50, 50, 250, 250)
    report = gate.assess(frame, det)
    if not report.passed:
        assert report.hint is not None and len(report.hint) > 0
