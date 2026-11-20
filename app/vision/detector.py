"""Face detectors: SCRFD (InsightFace), YuNet (OpenCV), MediaPipe."""

from __future__ import annotations

import logging
from typing import Protocol

import cv2
import numpy as np

from app.types import Detection

logger = logging.getLogger("attendance.detector")


class Detector(Protocol):
    def detect(self, image: np.ndarray) -> list[Detection]: ...


# ---------------------------------------------------------------------------
# SCRFD via InsightFace
# ---------------------------------------------------------------------------

class ScrfdDetector:
    """Uses InsightFace's SCRFD detector."""

    def __init__(self, model_pack: str, det_size: int, score_threshold: float) -> None:
        from insightface.app import FaceAnalysis

        self._det_size = det_size
        self._score_threshold = score_threshold
        self._app = FaceAnalysis(
            name=model_pack,
            root="models",
            allowed_modules=["detection"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_thresh=score_threshold, det_size=(det_size, det_size))
        logger.info("ScrfdDetector ready: pack=%s det_size=%d", model_pack, det_size)

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:
            faces = self._app.get(image)
            results = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(float)
                lmk = face.kps.astype(np.float32) if face.kps is not None else np.zeros((5, 2), np.float32)
                results.append(Detection(box=(x1, y1, x2, y2), score=float(face.det_score), landmarks=lmk))
            return results
        except Exception as exc:
            logger.error("SCRFD detection error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# YuNet (OpenCV)
# ---------------------------------------------------------------------------

class YuNetDetector:
    """OpenCV YuNet face detector (lighter, no landmarks needed for comparison)."""

    def __init__(self, model_path: str, det_size: int, score_threshold: float) -> None:
        self._det_size = det_size
        self._score_threshold = score_threshold
        self._detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (det_size, det_size),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )
        logger.info("YuNetDetector ready: det_size=%d", det_size)

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:
            h, w = image.shape[:2]
            scale = self._det_size / max(h, w)
            resized = cv2.resize(image, (int(w * scale), int(h * scale)))
            self._detector.setInputSize((resized.shape[1], resized.shape[0]))
            _, faces = self._detector.detect(resized)
            if faces is None:
                return []
            results = []
            for f in faces:
                x, y, fw, fh = f[:4]
                x1 = x / scale
                y1 = y / scale
                x2 = (x + fw) / scale
                y2 = (y + fh) / scale
                # Build 5 landmarks from YuNet columns 4–13 (x0y0 ... x4y4)
                lmk = np.array(f[4:14], dtype=np.float32).reshape(5, 2) / scale
                results.append(Detection(box=(float(x1), float(y1), float(x2), float(y2)), score=float(f[14]), landmarks=lmk))
            return results
        except Exception as exc:
            logger.error("YuNet detection error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# MediaPipe (lightweight fallback)
# ---------------------------------------------------------------------------

class MediaPipeDetector:
    """MediaPipe FaceDetection (short-range, no landmarks)."""

    def __init__(self, score_threshold: float = 0.5, det_size: int = 640) -> None:
        import mediapipe as mp

        self._det_size = det_size
        self._threshold = score_threshold
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=score_threshold
        )
        logger.info("MediaPipeDetector ready")

    def detect(self, image: np.ndarray) -> list[Detection]:
        try:

            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self._detector.process(rgb)
            if not result.detections:
                return []
            dets = []
            for d in result.detections:
                bb = d.location_data.relative_bounding_box
                x1 = bb.xmin * w
                y1 = bb.ymin * h
                x2 = x1 + bb.width * w
                y2 = y1 + bb.height * h
                dets.append(Detection(
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    score=float(d.score[0]),
                    landmarks=np.zeros((5, 2), np.float32),
                ))
            return dets
        except Exception as exc:
            logger.error("MediaPipe detection error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(cfg) -> Detector:
    dtype = cfg.detector.type
    if dtype == "scrfd":
        return ScrfdDetector(cfg.detector.model_pack, cfg.detector.det_size, cfg.detector.score_threshold)
    elif dtype == "yunet":
        model_path = str(cfg.paths.models) + "/face_detection_yunet_2023mar.onnx"
        return YuNetDetector(model_path, cfg.detector.det_size, cfg.detector.score_threshold)
    elif dtype == "mediapipe":
        return MediaPipeDetector(cfg.detector.score_threshold, cfg.detector.det_size)
    else:
        raise ValueError(f"Unknown detector type: {dtype}")

