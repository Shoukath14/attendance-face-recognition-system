"""Quality gate: size, blur, pose, brightness checks on face crops."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from app.config import QualityThresholds
from app.types import Detection, QualityReport

logger = logging.getLogger("attendance.quality")


class QualityGate:
    """Assess whether a detected face meets quality requirements."""

    def __init__(self, thresholds: QualityThresholds) -> None:
        self._t = thresholds

    def assess(self, frame: np.ndarray, det: Detection) -> QualityReport:
        reasons: list[str] = []
        hint: str | None = None
        score = 1.0

        x1, y1, x2, y2 = det.box
        face_w = x2 - x1
        face_h = y2 - y1

        # 1. Size check
        if face_w < self._t.min_face_px or face_h < self._t.min_face_px:
            reasons.append("too_small")
            hint = "Move closer to the camera"
            score *= 0.1

        # 2. Crop for per-face checks
        h, w = frame.shape[:2]
        cx1 = max(0, int(x1))
        cy1 = max(0, int(y1))
        cx2 = min(w, int(x2))
        cy2 = min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            return QualityReport(passed=False, score=0.0, reasons=["invalid_crop"], hint="Face out of frame")

        crop = frame[cy1:cy2, cx1:cx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 3. Blur check (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self._t.min_blur_var:
            reasons.append("blurry")
            if hint is None:
                hint = "Hold still, image is blurry"
            score *= max(0.1, lap_var / self._t.min_blur_var)

        # 4. Brightness check
        mean_brightness = float(gray.mean())
        lo, hi = self._t.brightness_range
        if mean_brightness < lo:
            reasons.append("too_dark")
            if hint is None:
                hint = "Too dark – improve lighting"
            score *= max(0.1, mean_brightness / lo)
        elif mean_brightness > hi:
            reasons.append("too_bright")
            if hint is None:
                hint = "Too bright – reduce lighting or glare"
            score *= max(0.1, hi / mean_brightness)

        # 5. Pose check from landmarks (yaw and pitch approximation)
        if det.landmarks is not None and det.landmarks.shape == (5, 2):
            yaw, pitch = _estimate_pose(det.landmarks)
            if abs(yaw) > self._t.max_yaw_deg:
                reasons.append("pose_yaw")
                if hint is None:
                    hint = "Face the camera directly"
                score *= max(0.1, 1.0 - (abs(yaw) - self._t.max_yaw_deg) / 30.0)
            if abs(pitch) > self._t.max_pitch_deg:
                reasons.append("pose_pitch")
                if hint is None:
                    hint = "Keep your head level"
                score *= max(0.1, 1.0 - (abs(pitch) - self._t.max_pitch_deg) / 30.0)

        passed = len(reasons) == 0
        return QualityReport(passed=passed, score=float(np.clip(score, 0.0, 1.0)), reasons=reasons, hint=hint)


def _estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
    """Rough yaw/pitch from 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)."""
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    eye_center = (left_eye + right_eye) / 2.0
    eye_width = float(np.linalg.norm(right_eye - left_eye))
    if eye_width < 1e-6:
        return 0.0, 0.0

    # Yaw: asymmetry of eye distances to nose
    left_dist = float(np.linalg.norm(nose - left_eye))
    right_dist = float(np.linalg.norm(nose - right_eye))
    yaw_ratio = (right_dist - left_dist) / (right_dist + left_dist + 1e-6)
    yaw_deg = yaw_ratio * 90.0

    # Pitch: how much the nose is above/below the eye-center line, relative to face size
    nose_drop = nose[1] - eye_center[1]
    pitch_deg = math.degrees(math.atan2(nose_drop, eye_width))

    return yaw_deg, pitch_deg

