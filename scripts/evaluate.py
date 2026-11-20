"""
Evaluate ArcFace embedder on eval dataset.

Expected folder structure:
  data/eval/enroll/<student_id>/front.jpg   -- enrollment reference
  data/eval/probe/<student_id>/<condition>/<img>.jpg  -- probe images

Usage:
  .venv\\Scripts\\python.exe scripts/evaluate.py [--model arcface|facenet] [--out reports/]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger("attendance.evaluate")


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return 1 - cosine_similarity (0 = identical, 2 = opposite)."""
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))


def _embed_image(embedder, img_path: Path) -> np.ndarray | None:
    """Load image with OpenCV and return embedding, or None on failure."""
    try:
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read image: %s", img_path)
            return None
        return embedder.embed(img)
    except Exception as exc:
        logger.error("Error embedding %s: %s", img_path, exc)
        return None


def _build_embedder(model_name: str):
    """Instantiate the requested embedder."""
    from app.config import load_config
    from app.recognition.embedder import ArcFaceEmbedder, FaceNetEmbedder

    cfg = load_config("config.yaml")
    if model_name == "arcface":
        return ArcFaceEmbedder(cfg.detector.model_pack)
    elif model_name == "facenet":
        return FaceNetEmbedder()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """Return (eer, eer_threshold) from ROC arrays (sklearn convention)."""
    fnr = 1.0 - tpr
    # EER is where FPR ~ FNR
    idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold


def _tar_at_far(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, target_far: float) -> tuple[float, float]:
    """Return (TAR, threshold) at the given FAR operating point."""
    # Find index where FPR is closest to target_far from below
    idx = np.searchsorted(fpr, target_far, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx]), float(thresholds[idx])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate face embedder on eval dataset")
    parser.add_argument("--model", choices=["arcface", "facenet"], default="arcface")
    parser.add_argument("--out", default="reports/", help="Output directory for reports")
    args = parser.parse_args()

    enroll_root = Path("data/eval/enroll")
    probe_root = Path("data/eval/probe")

    # Graceful empty-data handling
    if not enroll_root.exists() or not any(enroll_root.iterdir()):
        print("No eval data, skipping")
        sys.exit(0)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building embedder: {args.model} ...")
    embedder = _build_embedder(args.model)

    # --- Enroll phase ---
    print("Loading enrollment images ...")
    enroll_embeddings: dict[str, np.ndarray] = {}
    for student_dir in sorted(enroll_root.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        # Try front.jpg first, then any .jpg/.png
        candidates = [student_dir / "front.jpg"] + list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.png"))
        for img_path in candidates:
            if img_path.exists():
                emb = _embed_image(embedder, img_path)
                if emb is not None:
                    enroll_embeddings[student_id] = emb
                    logger.info("Enrolled %s from %s", student_id, img_path.name)
                    break

    if not enroll_embeddings:
        print("No valid enrollment images found in data/eval/enroll/. No eval data, skipping")
        sys.exit(0)

    # --- Probe phase ---
    print("Loading probe images ...")
    # genuine_scores: same ID, impostor_scores: different ID
    genuine_scores: list[float] = []
    impostor_scores: list[float] = []

    probe_embeddings: dict[str, list[np.ndarray]] = {}  # student_id -> [embeddings]
    if probe_root.exists():
        for student_dir in sorted(probe_root.iterdir()):
            if not student_dir.is_dir():
                continue
            student_id = student_dir.name
            imgs = list(student_dir.rglob("*.jpg")) + list(student_dir.rglob("*.png"))
            embs = []
            for img_path in imgs:
                emb = _embed_image(embedder, img_path)
                if emb is not None:
                    embs.append(emb)
            if embs:
                probe_embeddings[student_id] = embs

    if not probe_embeddings:
        print("No probe images found. No eval data, skipping")
        sys.exit(0)

    # Compute genuine pairs (probe vs enrolled same ID)
    for sid, probe_embs in probe_embeddings.items():
        if sid not in enroll_embeddings:
            logger.warning("Probe student %s has no enrollment reference, skipping genuine pairs", sid)
            continue
        ref = enroll_embeddings[sid]
        for pemb in probe_embs:
            genuine_scores.append(_cosine_distance(ref, pemb))

    # Compute impostor pairs (probe vs enrolled different ID)
    enrolled_ids = list(enroll_embeddings.keys())
    for sid, probe_embs in probe_embeddings.items():
        for other_id in enrolled_ids:
            if other_id == sid:
                continue
            ref = enroll_embeddings[other_id]
            for pemb in probe_embs:
                impostor_scores.append(_cosine_distance(ref, pemb))

    if not genuine_scores or not impostor_scores:
        print("Insufficient pairs for ROC evaluation. No eval data, skipping")
        sys.exit(0)

    genuine_arr = np.array(genuine_scores, dtype=np.float64)
    impostor_arr = np.array(impostor_scores, dtype=np.float64)

    print(f"  Genuine pairs : {len(genuine_arr)}")
    print(f"  Impostor pairs: {len(impostor_arr)}")

    # --- ROC curve ---
    # We treat lower distance = more similar = positive match.
    # sklearn roc_curve expects higher score = more positive, so negate distances.
    from sklearn.metrics import roc_curve

    y_true = np.concatenate([np.ones(len(genuine_arr)), np.zeros(len(impostor_arr))])
    y_score = np.concatenate([-genuine_arr, -impostor_arr])  # negate: higher = more similar

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Convert negated thresholds back to distance thresholds
    dist_thresholds = -thresholds

    eer, eer_thresh = _compute_eer(fpr, tpr, dist_thresholds)
    tar_at_01, thresh_at_01 = _tar_at_far(fpr, tpr, dist_thresholds, target_far=0.1)
    tar_at_001, thresh_at_001 = _tar_at_far(fpr, tpr, dist_thresholds, target_far=0.01)

    print(f"\n=== Evaluation Results ({args.model}) ===")
    print(f"  EER            : {eer:.4f}  (threshold={eer_thresh:.4f})")
    print(f"  TAR @ FAR=0.10 : {tar_at_01:.4f}  (threshold={thresh_at_01:.4f})")
    print(f"  TAR @ FAR=0.01 : {tar_at_001:.4f}  (threshold={thresh_at_001:.4f})")

    # --- Save ROC plot ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"{args.model} (EER={eer:.3f})", color="steelblue", lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.scatter([fpr[np.argmin(np.abs(fpr - (1 - tpr)))]], [tpr[np.argmin(np.abs(fpr - (1 - tpr)))]], color="red", zorder=5, label=f"EER={eer:.3f}")
        ax.set_xlabel("False Positive Rate (FAR)")
        ax.set_ylabel("True Positive Rate (TAR)")
        ax.set_title(f"ROC Curve — {args.model}")
        ax.legend()
        ax.grid(alpha=0.3)
        roc_path = out_dir / f"roc_{args.model}.png"
        fig.savefig(roc_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nROC plot saved to: {roc_path}")
    except ImportError:
        print("matplotlib not available; skipping ROC plot")

    # --- Save JSON results ---
    results = {
        "model": args.model,
        "genuine_pairs": int(len(genuine_arr)),
        "impostor_pairs": int(len(impostor_arr)),
        "eer": eer,
        "eer_threshold": eer_thresh,
        "tar_at_far_0_10": {"tar": tar_at_01, "threshold": thresh_at_01, "far": 0.10},
        "tar_at_far_0_01": {"tar": tar_at_001, "threshold": thresh_at_001, "far": 0.01},
        "genuine_mean_dist": float(genuine_arr.mean()),
        "genuine_std_dist": float(genuine_arr.std()),
        "impostor_mean_dist": float(impostor_arr.mean()),
        "impostor_std_dist": float(impostor_arr.std()),
        # ROC arrays as lists for calibrate_threshold.py
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": dist_thresholds.tolist(),
        },
    }

    json_path = out_dir / f"eval_results_{args.model}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to : {json_path}")


if __name__ == "__main__":
    main()
