"""Pytest fixtures for attendance system tests."""

import numpy as np
import pytest


@pytest.fixture
def sample_config_dict():
    return {
        "camera": {
            "source": 0,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "reconnect_backoff_s": 1.0,
            "watchdog_timeout_s": 3.0,
        },
        "detector": {
            "type": "scrfd",
            "model_pack": "buffalo_s",
            "det_size": 640,
            "score_threshold": 0.5,
            "detect_every_n_frames": 3,
        },
        "tracker": {
            "iou_threshold": 0.3,
            "max_age_frames": 15,
        },
        "quality": {
            "enroll": {
                "min_face_px": 150,
                "max_yaw_deg": 15.0,
                "max_pitch_deg": 15.0,
                "min_blur_var": 80.0,
                "brightness_range": [70, 190],
            },
            "recognize": {
                "min_face_px": 80,
                "max_yaw_deg": 30.0,
                "max_pitch_deg": 25.0,
                "min_blur_var": 40.0,
                "brightness_range": [50, 210],
            },
        },
        "liveness": {
            "enabled": True,
            "model_path": "models/minifasnet.onnx",
            "real_threshold": 0.7,
        },
        "recognition": {
            "embedder": "arcface",
            "threshold": 0.45,
            "margin": 0.05,
            "duplicate_enroll_threshold": 0.6,
            "evaluate_interval_s": 0.3,
            "vote_window": 8,
            "confirm_votes": 5,
            "unknown_votes": 6,
        },
        "enrollment": {
            "burst_seconds": 1.5,
            "augmentations": ["flip", "bright", "dark", "blur"],
        },
        "attendance": {
            "default_late_after_min": 10,
        },
        "paths": {
            "db": "data/test_attendance.db",
            "thumbnails": "data/test_thumbnails",
            "review_snapshots": "data/test_review_snapshots",
            "models": "models",
        },
        "display": {
            "window_name": "Attendance Test",
            "show_fps": True,
            "mirror_preview": False,
        },
    }


@pytest.fixture
def synthetic_face_frame():
    """Create a 720x1280 BGR image."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)

