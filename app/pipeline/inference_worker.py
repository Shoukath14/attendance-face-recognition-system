"""Inference worker thread: runs the full per-frame recognition pipeline."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.shared_state import SharedState
from app.recognition.voting import TrackVoter
from app.types import TrackView

logger = logging.getLogger("attendance.inference")


class InferenceWorker(threading.Thread):
    """
    Consumes newest frames from LatestFrameReader, runs the full pipeline,
    publishes TrackViews to SharedState, writes attendance on confirmation.
    """

    def __init__(
        self,
        reader,
        shared_state: SharedState,
        detector,
        tracker,
        quality,
        liveness,
        embedder,
        matcher,
        session_manager,
        repos,  # (student_repo, embedding_repo, attendance_repo, review_repo)
        cfg,
    ) -> None:
        super().__init__(name="InferenceWorker", daemon=True)
        self._reader = reader
        self._state = shared_state
        self._detector = detector
        self._tracker = tracker
        self._quality = quality
        self._liveness = liveness
        self._embedder = embedder
        self._matcher = matcher
        self._session_manager = session_manager
        self._student_repo, self._emb_repo, self._att_repo, self._review_repo = repos
        self._cfg = cfg

        self._stop_event = threading.Event()
        self._voters: dict[int, TrackVoter] = {}  # track_id → voter
        self._last_eval: dict[int, float] = {}     # track_id → last eval timestamp
        self._last_frame_id: int = -1

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("InferenceWorker started")
        detect_every = self._cfg.detector.detect_every_n_frames
        eval_interval = self._cfg.recognition.evaluate_interval_s

        while not self._stop_event.is_set():
            try:
                fd = self._reader.read()
                if fd is None or fd.frame_id == self._last_frame_id:
                    time.sleep(0.005)
                    continue

                self._last_frame_id = fd.frame_id
                frame = fd.image
                frame_id = fd.frame_id

                # Detection or tracker predict
                if frame_id % detect_every == 0:
                    dets = self._detector.detect(frame)
                    tracks = self._tracker.update(dets, frame_id)
                else:
                    tracks = self._tracker.predict(frame_id)

                # Expire old voters
                live_ids = {t.track_id for t in tracks}
                for tid in list(self._voters.keys()):
                    if tid not in live_ids:
                        del self._voters[tid]
                        self._last_eval.pop(tid, None)

                views: list[TrackView] = []
                session = self._session_manager.active_session()

                for track in tracks:
                    tid = track.track_id
                    voter = self._voters.setdefault(
                        tid,
                        TrackVoter(
                            self._cfg.recognition.vote_window,
                            self._cfg.recognition.confirm_votes,
                            self._cfg.recognition.unknown_votes,
                        ),
                    )

                    decision, student_id = voter.decision()

                    if decision == "confirmed":
                        student_info = self._get_student_info(student_id)
                        name = student_info.get("name") if student_info else None
                        roll_no = student_info.get("roll_no") if student_info else None
                        already = session is not None and self._session_manager.already_marked(session["id"], student_id)
                        if session and not already:
                            self._mark_attendance(session, student_id, voter)
                        views.append(TrackView(
                            track_id=tid, box=track.box, state="confirmed",
                            student_id=student_id, name=name, roll_no=roll_no,
                            already_marked=session is not None and self._session_manager.already_marked(session["id"], student_id) if student_id else False,
                        ))
                        continue

                    if decision in ("unknown", "spoof"):
                        self._add_to_review(frame, track, decision, session)
                        views.append(TrackView(track_id=tid, box=track.box, state=decision))
                        continue

                    # Throttle per-track evaluation
                    now = time.time()
                    if now - self._last_eval.get(tid, 0) < eval_interval:
                        views.append(TrackView(track_id=tid, box=track.box, state="verifying"))
                        continue

                    # Quality gate
                    if track.detection is None:
                        views.append(TrackView(track_id=tid, box=track.box, state="verifying"))
                        continue

                    q = self._quality.assess(frame, track.detection)
                    if not q.passed:
                        views.append(TrackView(track_id=tid, box=track.box, state="poor_quality", hint=q.hint))
                        continue

                    self._last_eval[tid] = now

                    # Full pipeline
                    try:
                        from app.vision.align import align_face
                        live = self._liveness.check(frame, track.detection)
                        aligned = align_face(frame, track.detection.landmarks)
                        emb = self._embedder.embed(aligned)
                        match = self._matcher.match(emb)
                        voter.add(match, live.is_real)
                    except Exception as exc:
                        logger.error("Pipeline error track %d: %s", tid, exc)

                    views.append(TrackView(track_id=tid, box=track.box, state="verifying"))

                # Build status
                reader_stats = self._reader.stats()
                status = {
                    "fps": reader_stats.get("fps", 0),
                    "camera_online": self._reader.is_alive(),
                    "session_name": session["name"] if session else "None",
                    "marked_count": len(self._session_manager._marked_set) if session else 0,
                }
                self._state.publish(frame_id, views, status)

            except Exception as exc:
                logger.error("InferenceWorker frame error: %s", exc, exc_info=True)
                time.sleep(0.01)

        logger.info("InferenceWorker stopped")

    def _get_student_info(self, student_id: int | None) -> dict | None:
        if student_id is None:
            return None
        s = self._student_repo.get_by_id(student_id)
        if s:
            return {"name": s["name"], "roll_no": s["roll_no"]}
        return {"name": f"ID:{student_id}", "roll_no": ""}

    def _get_name(self, student_id: int | None) -> str | None:
        info = self._get_student_info(student_id)
        return info["name"] if info else None

    def _mark_attendance(self, session: dict, student_id: int, voter: TrackVoter) -> None:
        try:
            best_score = max(
                (m.score for m in voter._match_history if m.student_id == student_id),
                default=None,
            )
            status = self._session_manager.status_for(session)
            marked = self._att_repo.mark(session["id"], student_id, status, confidence=best_score)
            if marked:
                self._session_manager.record_marked(student_id)
                name = self._get_name(student_id)
                logger.info("Attendance marked: %s (student_id=%d) status=%s", name, student_id, status)
        except Exception as exc:
            logger.error("Error marking attendance: %s", exc)

    def _add_to_review(self, frame: np.ndarray, track, reason: str, session) -> None:
        try:
            snap_dir = Path(self._cfg.paths.review_snapshots)
            snap_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            snap_path = str(snap_dir / f"{reason}_{track.track_id}_{ts}.jpg")
            x1, y1, x2, y2 = (int(c) for c in track.box)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                cv2.imwrite(snap_path, crop)
            sid = session["id"] if session else None
            self._review_repo.add(sid, snap_path, reason)
        except Exception as exc:
            logger.error("Review queue add error: %s", exc)

