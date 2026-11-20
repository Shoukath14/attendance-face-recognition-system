"""OpenCV enrollment window: live burst capture → enroll student."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.camera.capture import LatestFrameReader
from app.config import load_config
from app.logging_setup import setup_logging
from app.pipeline.enrollment import EnrollmentService
from app.recognition.embedder import build_embedder
from app.recognition.matcher import Matcher
from app.storage.db import init_db
from app.storage.repositories import EmbeddingRepo, StudentRepo
from app.ui.overlay import draw_enroll_guide
from app.vision.detector import build_detector
from app.vision.liveness import LivenessChecker
from app.vision.quality import QualityGate


def main(args: argparse.Namespace) -> None:
    setup_logging()
    logger = logging.getLogger("attendance.enroll")
    cfg = load_config(args.config)
    init_db(cfg.paths.db)

    student_repo = StudentRepo(cfg.paths.db)
    emb_repo = EmbeddingRepo(cfg.paths.db)

    detector = build_detector(cfg)
    embedder = build_embedder(cfg)
    matcher = Matcher(cfg.recognition.threshold, cfg.recognition.margin)
    matcher.reload_from_db(emb_repo, embedder.name)
    quality = QualityGate(cfg.quality.enroll)
    liveness = LivenessChecker(cfg)

    service = EnrollmentService(
        detector, quality, liveness, embedder, matcher,
        (student_repo, emb_repo), cfg,
    )

    # --- Single image mode ---
    if args.from_image:
        img = cv2.imread(args.from_image)
        if img is None:
            print(f"Error: could not read image '{args.from_image}'")
            sys.exit(1)
        result = service.enroll_from_image(args.name, args.roll_no, img)
        if result.success:
            print(f"[OK] {result.message}")
        else:
            print(f"[FAIL] {result.message}")
            sys.exit(1)
        return

    # --- Live camera mode ---
    source = args.source if args.source is not None else cfg.camera.source
    rotate_deg = getattr(args, "rotate", None)
    if rotate_deg is None:
        rotate_deg = getattr(cfg.camera, "rotate_deg", 0)

    reader = LatestFrameReader(
        source, cfg.camera.width, cfg.camera.height, cfg.camera.fps,
        rotate_deg=int(rotate_deg),
    )
    reader.start()

    logger.info("Enrollment window open for '%s' (%s). Press Esc or Q to cancel.", args.name, args.roll_no)
    service.reset()

    try:
        while True:
            # Check if user closed the window via 'X' button
            if cv2.getWindowProperty("Enrollment", cv2.WND_PROP_VISIBLE) < 1:
                # If window was created once and closed with X, exit cleanly
                try:
                    # In OpenCV, getWindowProperty returns -1 or 0 when closed
                    pass
                except Exception:
                    break

            fd = reader.read()
            if fd is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for Camera...", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(frame, "Press ESC or Q to Exit", (170, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow("Enrollment", frame)
                k = cv2.waitKey(30) & 0xFF
                if k in (27, ord("q"), ord("Q")):
                    print("Enrollment cancelled.")
                    break
                # Also check window close
                if cv2.getWindowProperty("Enrollment", cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user.")
                    break
                continue

            frame = fd.image.copy()
            progress = service.process_frame(frame)
            frame = draw_enroll_guide(frame, progress.hint, progress.progress)
            cv2.imshow("Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("Enrollment cancelled.")
                break

            if cv2.getWindowProperty("Enrollment", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break

            if progress.state == "ready":
                result = service.enroll_best(args.name, args.roll_no)
                if result.success:
                    print(f"[OK] {result.message}")
                    # Flash green
                    green = np.zeros_like(frame)
                    green[:, :] = (0, 220, 60)
                    cv2.addWeighted(frame, 0.4, green, 0.6, 0, frame)
                    cv2.putText(frame, "ENROLLED!", (frame.shape[1]//2 - 90, frame.shape[0]//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                    cv2.imshow("Enrollment", frame)
                    cv2.waitKey(2000)
                    break
                else:
                    print(f"[FAIL] {result.message}")
                    service.reset()
    finally:
        reader.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a student via live camera or image")
    parser.add_argument("--name", required=True, help="Student name")
    parser.add_argument("--roll-no", required=True, dest="roll_no", help="Roll number")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--rotate", type=int, default=None, help="Rotate camera frames (e.g. 90, 270)")
    parser.add_argument("--from-image", default=None, dest="from_image", help="Enroll from a photo file")
    args = parser.parse_args()
    if args.source is not None:
        try:
            args.source = int(args.source)
        except ValueError:
            pass
    main(args)

