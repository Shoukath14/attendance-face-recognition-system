"""Main live recognition script: capture + inference + display threads."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera.capture import LatestFrameReader
from app.config import load_config
from app.logging_setup import setup_logging
from app.pipeline.inference_worker import InferenceWorker
from app.pipeline.session_manager import SessionManager
from app.pipeline.shared_state import SharedState
from app.recognition.embedder import build_embedder
from app.recognition.matcher import Matcher
from app.storage.db import init_db
from app.storage.repositories import (
    AttendanceRepo,
    EmbeddingRepo,
    ReviewRepo,
    SessionRepo,
    StudentRepo,
)
from app.ui.overlay import draw_attendance_banner, draw_status_bar, draw_views
from app.vision.detector import build_detector
from app.vision.enhance import enhance_if_dark
from app.vision.liveness import LivenessChecker
from app.vision.quality import QualityGate
from app.vision.tracker import IoUTracker


def build_quality_gate(cfg, mode: str) -> QualityGate:
    thresholds = cfg.quality.enroll if mode == "enroll" else cfg.quality.recognize
    return QualityGate(thresholds)


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("attendance.run_live")

    cfg = load_config(args.config)
    db_path = cfg.paths.db
    init_db(db_path)

    # Repos
    student_repo = StudentRepo(db_path)
    emb_repo = EmbeddingRepo(db_path)
    session_repo = SessionRepo(db_path)
    att_repo = AttendanceRepo(db_path)
    review_repo = ReviewRepo(db_path)

    # Vision pipeline components
    logger.info("Loading detector and embedder…")
    detector = build_detector(cfg)
    embedder = build_embedder(cfg)
    matcher = Matcher(cfg.recognition.threshold, cfg.recognition.margin)
    matcher.reload_from_db(emb_repo, embedder.name)
    quality = build_quality_gate(cfg, "recognize")
    liveness = LivenessChecker(cfg)
    tracker = IoUTracker(cfg.tracker.iou_threshold, cfg.tracker.max_age_frames)

    # Camera
    source = args.source if args.source is not None else cfg.camera.source
    rotate_deg = getattr(args, "rotate", None)
    if rotate_deg is None:
        rotate_deg = getattr(cfg.camera, "rotate_deg", 0)

    reader = LatestFrameReader(
        source, cfg.camera.width, cfg.camera.height, cfg.camera.fps,
        rotate_deg=int(rotate_deg),
        reconnect_backoff_s=cfg.camera.reconnect_backoff_s,
        watchdog_timeout_s=cfg.camera.watchdog_timeout_s,
    )

    shared_state = SharedState()
    session_manager = SessionManager(session_repo, att_repo)
    session_manager.start()

    worker = InferenceWorker(
        reader=reader,
        shared_state=shared_state,
        detector=detector,
        tracker=tracker,
        quality=quality,
        liveness=liveness,
        embedder=embedder,
        matcher=matcher,
        session_manager=session_manager,
        repos=(student_repo, emb_repo, att_repo, review_repo),
        cfg=cfg,
    )

    reader.start()
    worker.start()

    logger.info("Live recognition started. Press 'q' to quit.")
    window_name = cfg.display.window_name
    display_fps_times: list[float] = []

    # Feedback state on detection
    celebration_until: float = 0.0
    celebration_name: str = ""
    celebration_roll: str = ""
    seen_confirmed_ids: set[int] = set()

    def play_beep() -> None:
        try:
            import winsound
            winsound.Beep(1200, 800)  # 1200 Hz tone for ~0.8-1.0 sec
        except Exception:
            pass

    try:
        while True:
            fd = reader.read()
            if fd is None:
                # Camera offline — show placeholder
                placeholder = __import__("numpy").zeros((480, 640, 3), dtype=__import__("numpy").uint8)
                cv2.putText(placeholder, "Camera Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 50, 230), 2)
                cv2.imshow(window_name, placeholder)
            else:
                frame = fd.image.copy()
                if cfg.display.mirror_preview:
                    frame = cv2.flip(frame, 1)

                _, views, status = shared_state.snapshot()

                # Trigger beep and 2-second screen highlight upon newly confirmed detection
                for v in views:
                    if v.state == "confirmed" and v.student_id:
                        if not v.already_marked:
                            seen_confirmed_ids.discard(v.student_id)

                        if v.student_id not in seen_confirmed_ids:
                            seen_confirmed_ids.add(v.student_id)
                            celebration_until = time.time() + 2.0  # 2-second pause / highlight
                            celebration_name = v.name or "Student"
                            celebration_roll = v.roll_no or ""
                            # Beep sound in non-blocking thread for ~1 sec
                            threading.Thread(target=play_beep, daemon=True).start()

                # Enhance if dark
                frame = enhance_if_dark(frame)

                frame = draw_views(frame, views, status)
                frame = draw_status_bar(frame, status)

                # Show celebratory banner if within the 2-second pause window
                now = time.time()
                if now < celebration_until:
                    frame = draw_attendance_banner(frame, celebration_name, celebration_roll, celebration_until - now)

                # Display FPS (independent of inference)
                display_fps_times.append(now)
                display_fps_times = [t for t in display_fps_times if t > now - 2.0]
                if len(display_fps_times) > 1:
                    disp_fps = (len(display_fps_times) - 1) / (display_fps_times[-1] - display_fps_times[0])
                    if cfg.display.show_fps:
                        cv2.putText(frame, f"Display: {disp_fps:.1f} FPS", (8, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

                cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Shutting down…")
        worker.stop()
        worker.join(timeout=5)
        session_manager.stop()
        reader.stop()
        cv2.destroyAllWindows()
        logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attendance Face Recognition – Live")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--source", default=None, help="Camera source override (integer index or URL)")
    parser.add_argument("--rotate", type=int, default=None, help="Rotate camera feed (90, 180, 270)")
    args = parser.parse_args()
    if args.source is not None:
        try:
            args.source = int(args.source)
        except ValueError:
            pass
    main(args)

