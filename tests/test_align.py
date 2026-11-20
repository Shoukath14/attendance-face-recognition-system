"""Unit tests for align_face()."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.vision.align import align_face  # noqa: E402


def _make_landmarks() -> np.ndarray:
    """Five realistic-ish landmark coords on a 112x112 canvas."""
    return np.array(
        [
            [30.0, 40.0],   # left eye
            [80.0, 40.0],   # right eye
            [56.0, 65.0],   # nose
            [35.0, 90.0],   # left mouth
            [75.0, 90.0],   # right mouth
        ],
        dtype=np.float32,
    )


def test_align_face_shape() -> None:
    """Output must be (112, 112, 3)."""
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    landmarks = _make_landmarks()
    result = align_face(img, landmarks)
    assert result.shape == (112, 112, 3), f"Unexpected shape: {result.shape}"


def test_align_face_dtype() -> None:
    """Output dtype must be uint8."""
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    landmarks = _make_landmarks()
    result = align_face(img, landmarks)
    assert result.dtype == np.uint8, f"Unexpected dtype: {result.dtype}"


def test_align_face_custom_size() -> None:
    """align_face with size=64 returns (64, 64, 3)."""
    img = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    landmarks = _make_landmarks() * (200.0 / 112.0)
    result = align_face(img, landmarks, size=64)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.uint8


def test_align_face_larger_frame() -> None:
    """Works on a larger source frame."""
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # Place landmarks in the centre of the frame
    offset = np.array([560.0, 300.0], dtype=np.float32)
    landmarks = _make_landmarks() + offset
    result = align_face(img, landmarks)
    assert result.shape == (112, 112, 3)
    assert result.dtype == np.uint8
