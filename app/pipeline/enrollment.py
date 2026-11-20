"""Enrollment service: burst capture, best-frame selection, duplicate check, storage."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from app.recognition.augment import augment_face
from app.types import Detection, EnrollmentProgress, EnrollmentResult, QualityReport

logger = logging.getLogger("attendance.enrollment")


class EnrollmentService:
    """State machine for enrolling a student from a live burst or a single image."""

    def __init__(self, detector, quality, liveness, embedder, matcher, repos, cfg) -> None:
        self._detector = detector
        self._quality = quality  # QualityGate(enroll_thresholds)
        self._liveness = liveness
        self._embedder = embedder
        self._matcher = matcher
        self._repos = repos  # (student_repo, embedding_repo)
        self._cfg = cfg

        self._state = "waiting"
        self._burst_start: float | None = None
        self._best_frame: np.ndarray | None = None
        self._best_score: float = 0.0
        self._best_det: Detection | None = None

    # ------------------------------------------------------------------
    # Live burst flow
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._state = "waiting"
        self._burst_start = None
        self._best_frame = None
        self._best_score = 0.0
        self._best_det = None

    def process_frame(self, frame: np.ndarray) -> EnrollmentProgress:
        """
        Feed each live frame; returns progress status.
        Call enroll_best() after state == "ready".
        """
        detections = self._detector.detect(frame)
        if len(detections) != 1:
            hint = "Show exactly one face" if len(detections) != 1 else None
            if self._state == "capturing":
                # Reset if face count changed
                self.reset()
            return EnrollmentProgress(state="waiting", progress=0.0, hint=hint or "Look at the camera")

        det = detections[0]
        report: QualityReport = self._quality.assess(frame, det)

        if not report.passed:
            if self._state == "capturing":
                self.reset()
            return EnrollmentProgress(state="waiting", progress=0.0, hint=report.hint)

        # Start or continue burst
        now = time.time()
        if self._state == "waiting":
            self._state = "capturing"
            self._burst_start = now

        elapsed = now - (self._burst_start or now)
        burst_s = self._cfg.enrollment.burst_seconds
        progress = min(elapsed / burst_s, 1.0)

        if report.score > self._best_score:
            self._best_score = report.score
            self._best_frame = frame.copy()
            self._best_det = det

        if elapsed >= burst_s:
            self._state = "ready"
            return EnrollmentProgress(state="ready", progress=1.0, quality_score=self._best_score)

        return EnrollmentProgress(state="capturing", progress=progress, hint="Hold still…", quality_score=report.score)

    def enroll_best(self, name: str, roll_no: str) -> EnrollmentResult:
        """Enroll from the best frame captured during the burst."""
        if self._best_frame is None or self._best_det is None:
            return EnrollmentResult(success=False, message="No good frame captured")
        return self._do_enroll(name, roll_no, self._best_frame, self._best_det)

    # ------------------------------------------------------------------
    # Single-image flow
    # ------------------------------------------------------------------

    def enroll_from_image(self, name: str, roll_no: str, image: np.ndarray) -> EnrollmentResult:
        detections = self._detector.detect(image)
        if len(detections) != 1:
            return EnrollmentResult(success=False, message=f"Expected 1 face, found {len(detections)}")
        det = detections[0]
        report = self._quality.assess(image, det)
        if not report.passed:
            return EnrollmentResult(success=False, message=f"Quality check failed: {', '.join(report.reasons)}")
        return self._do_enroll(name, roll_no, image, det)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _do_enroll(self, name: str, roll_no: str, frame: np.ndarray, det: Detection) -> EnrollmentResult:
        from app.vision.align import align_face

        # Liveness
        live = self._liveness.check(frame, det)
        if not live.is_real:
            return EnrollmentResult(success=False, message="Liveness check failed (possible spoof)")

        aligned = align_face(frame, det.landmarks)
        embedding = self._embedder.embed(aligned)

        # Duplicate guard
        dup_id, dup_score = self._check_duplicate(embedding)
        if dup_id is not None:
            student = self._repos[0].get_by_id(dup_id)
            name_str = student["name"] if student else f"ID {dup_id}"
            return EnrollmentResult(success=False, message=f"Already enrolled as '{name_str}' (similarity {dup_score:.2f})")

        # Create student record
        thumbnail_path = self._save_thumbnail(aligned, roll_no)
        student_id = self._repos[0].add(roll_no, name, thumbnail_path)

        # Store original embedding
        emb_repo = self._repos[1]
        model_name = self._embedder.name
        emb_repo.add(student_id, model_name, "original", embedding)

        # Augmented variants
        augs = augment_face(aligned, self._cfg.enrollment.augmentations)
        for kind, aug_img in augs.items():
            aug_emb = self._embedder.embed(aug_img)
            emb_repo.add(student_id, model_name, kind, aug_emb)

        # Reload matcher
        self._matcher.reload_from_db(emb_repo, model_name)

        logger.info("Enrolled student '%s' (id=%d) with %d embeddings", name, student_id, 1 + len(augs))
        return EnrollmentResult(success=True, student_id=student_id, message=f"Enrolled '{name}' successfully")

    def _check_duplicate(self, embedding: np.ndarray) -> tuple[int | None, float]:
        from app.types import MatchResult

        threshold = self._cfg.recognition.duplicate_enroll_threshold
        result: MatchResult = self._matcher.match(embedding)
        if result.student_id is not None and result.score >= threshold:
            return result.student_id, result.score
        return None, 0.0

    def _save_thumbnail(self, aligned: np.ndarray, roll_no: str) -> str | None:
        try:
            thumb_dir = Path(self._cfg.paths.thumbnails)
            thumb_dir.mkdir(parents=True, exist_ok=True)
            path = thumb_dir / f"{roll_no}.jpg"
            small = cv2.resize(aligned, (64, 64))
            cv2.imwrite(str(path), small)
            return str(path)
        except Exception as exc:
            logger.warning("Could not save thumbnail: %s", exc)
            return None

