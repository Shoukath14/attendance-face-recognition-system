"""Benchmark detection, embedding, and matching stages on the current system."""

import os
import time

import numpy as np

from app.config import load_config
from app.logging_setup import setup_logging


def generate_synthetic_face_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a synthetic frame containing a simple face-like pattern for benchmarking."""
    import cv2
    img = np.full((height, width, 3), 180, dtype=np.uint8)
    # Draw face oval
    center = (width // 2, height // 2)
    cv2.ellipse(img, center, (120, 160), 0, 0, 360, (210, 220, 240), -1)
    # Eyes
    cv2.circle(img, (center[0] - 40, center[1] - 40), 15, (50, 50, 50), -1)
    cv2.circle(img, (center[0] + 40, center[1] - 40), 15, (50, 50, 50), -1)
    # Nose
    cv2.line(img, (center[0], center[1] - 10), (center[0], center[1] + 30), (100, 100, 100), 4)
    # Mouth
    cv2.ellipse(img, (center[0], center[1] + 70), (45, 20), 0, 0, 180, (80, 80, 180), 5)
    return img


def run_benchmark(num_runs: int = 30):
    setup_logging()
    cfg = load_config()
    print("=" * 60)
    print("Attendance Face Recognition - Stage Latency Benchmark")
    print("=" * 60)

    # 1. Matching benchmark (pure NumPy)
    print("\n1. Matching Stage (NumPy Cosine Similarity, 100 enrolled students x 5 embeddings):")
    enrolled_embeddings = np.random.randn(500, 512).astype(np.float32)
    # L2 normalize
    enrolled_embeddings /= np.linalg.norm(enrolled_embeddings, axis=1, keepdims=True)
    query_vector = np.random.randn(512).astype(np.float32)
    query_vector /= np.linalg.norm(query_vector)

    match_times = []
    for _ in range(num_runs * 10):
        t0 = time.perf_counter()
        scores = np.dot(enrolled_embeddings, query_vector)
        _ = np.max(scores)
        match_times.append((time.perf_counter() - t0) * 1000)

    print(f"   Runs: {len(match_times)}")
    print(f"   Mean: {np.mean(match_times):.3f} ms | p95: {np.percentile(match_times, 95):.3f} ms | Max: {np.max(match_times):.3f} ms")

    # 2. InsightFace detection and embedding benchmark
    try:
        from insightface.app import FaceAnalysis
        print(f"\n2. InsightFace Model Pack: '{cfg.detector.model_pack}'")
        models_root = os.path.abspath(cfg.paths.models)
        app = FaceAnalysis(name=cfg.detector.model_pack, root=models_root, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(cfg.detector.det_size, cfg.detector.det_size))

        test_frame = generate_synthetic_face_image(cfg.camera.width, cfg.camera.height)

        # Warm-up
        for _ in range(3):
            _ = app.get(test_frame)

        detect_times = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = app.get(test_frame)
            detect_times.append((time.perf_counter() - t0) * 1000)

        print("   Full InsightFace get() (Detect + Landmark + Embed):")
        print(f"   Runs: {num_runs}")
        print(f"   Mean: {np.mean(detect_times):.2f} ms | p95: {np.percentile(detect_times, 95):.2f} ms | Max: {np.max(detect_times):.2f} ms")
        effective_fps = 1000.0 / np.mean(detect_times)
        print(f"   Estimated max inference FPS (without skipping): {effective_fps:.1f} FPS")
        print(f"   With detect_every_n_frames={cfg.detector.detect_every_n_frames}: effective load is ~{effective_fps * cfg.detector.detect_every_n_frames:.1f} video FPS equivalent")

    except ImportError:
        print("\nInsightFace not yet installed; skipping live model benchmark.")
    except Exception as e:
        print(f"\nCould not run InsightFace benchmark: {e}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    run_benchmark()

