"""Anti-spoofing liveness checker (MiniFASNet ONNX or degraded/disabled mode)."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.types import Detection, LivenessResult

logger = logging.getLogger("attendance.liveness")


class LivenessChecker:
    """
    Passive liveness detection using MiniFASNet (ONNX).

    If the model file is missing, logs a warning and runs in disabled mode
    (always returns real=True, score=1.0) so the demo never crashes.
    """

    _MODEL_INPUT_SIZE = (128, 128)
    _CONTEXT_SCALE = 2.7  # extra context crop around the face

    def __init__(self, cfg) -> None:
        self._enabled = cfg.liveness.enabled
        self._threshold = cfg.liveness.real_threshold
        self._session = None

        if not self._enabled:
            logger.info("Liveness disabled in config")
            return

        model_path = Path(cfg.liveness.model_path)
        if not model_path.exists():
            logger.warning(
                "Liveness model not found at '%s'. Running in DISABLED mode (no spoof detection).",
                model_path,
            )
            self._enabled = False
            return

        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            logger.info("LivenessChecker ready: model=%s", model_path)
        except Exception as exc:
            logger.warning("Could not load liveness model: %s. Running disabled.", exc)
            self._enabled = False

    def check(self, frame: np.ndarray, det: Detection) -> LivenessResult:
        if not self._enabled or self._session is None:
            return LivenessResult(is_real=True, score=1.0)

        try:
            crop = self._context_crop(frame, det)
            if crop is None:
                return LivenessResult(is_real=True, score=1.0)

            inp = cv2.resize(crop, self._MODEL_INPUT_SIZE).astype(np.float32)
            inp = inp.transpose(2, 0, 1)[np.newaxis]  # NCHW
            out = self._session.run(None, {self._input_name: inp})[0]
            # Assume output [spoof_score, real_score] or softmax output
            if out.shape[-1] == 2:
                _e = np.exp(out[0] - out[0].max())
                real_score = float(_e[1] / _e.sum())
            else:
                real_score = float(out[0][0])

            return LivenessResult(is_real=real_score >= self._threshold, score=real_score)
        except Exception as exc:
            logger.error("Liveness check error: %s", exc)
            return LivenessResult(is_real=True, score=1.0)

    def _context_crop(self, frame: np.ndarray, det: Detection) -> np.ndarray | None:
        """Crop a larger region around the face for context."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        fw = (x2 - x1) * self._CONTEXT_SCALE / 2.0
        fh = (y2 - y1) * self._CONTEXT_SCALE / 2.0
        cx1 = max(0, int(cx - fw))
        cy1 = max(0, int(cy - fh))
        cx2 = min(w, int(cx + fw))
        cy2 = min(h, int(cy + fh))
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2]

