"""Still-image test: detect → align → embed → match across two folders."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import load_config
from app.logging_setup import setup_logging
from app.recognition.embedder import build_embedder
from app.vision.align import align_face
from app.vision.detector import build_detector


def embed_folder(folder: Path, detector, embedder, cfg) -> dict[str, np.ndarray]:
    embeddings = {}
    for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = detector.detect(img)
        if not dets:
            print(f"  No face in {img_path.name}")
            continue
        det = max(dets, key=lambda d: d.score)
        aligned = align_face(img, det.landmarks)
        emb = embedder.embed(aligned)
        embeddings[img_path.stem] = emb
        print(f"  Embedded: {img_path.name}")
    return embeddings


def main(args: argparse.Namespace) -> None:
    setup_logging()
    cfg = load_config(args.config)
    detector = build_detector(cfg)
    embedder = build_embedder(cfg)

    folder_a = Path(args.folder_a)
    folder_b = Path(args.folder_b)

    print(f"\nEmbedding folder A: {folder_a}")
    embs_a = embed_folder(folder_a, detector, embedder, cfg)
    print(f"\nEmbedding folder B: {folder_b}")
    embs_b = embed_folder(folder_b, detector, embedder, cfg)

    if not embs_a or not embs_b:
        print("No embeddings in one or both folders.")
        return

    # Similarity matrix
    names_a = list(embs_a.keys())
    names_b = list(embs_b.keys())
    matrix = np.array([[float(embs_a[a] @ embs_b[b]) for b in names_b] for a in names_a])

    print("\nSimilarity matrix (rows=A, cols=B):")
    header = "        " + "  ".join(f"{n[:8]:>8}" for n in names_b)
    print(header)
    for i, na in enumerate(names_a):
        row = f"{na[:8]:>8} " + "  ".join(f"{matrix[i, j]:8.3f}" for j in range(len(names_b)))
        print(row)

    # Match results
    threshold = cfg.recognition.threshold
    print(f"\nMatches (threshold={threshold}):")
    for i, na in enumerate(names_a):
        best_j = int(np.argmax(matrix[i]))
        best_score = matrix[i, best_j]
        match = names_b[best_j] if best_score >= threshold else "Unknown"
        print(f"  {na} → {match}  (score={best_score:.3f})")

    # Save aligned crops if requested
    if args.save_aligned:
        out_dir = Path(args.save_aligned)
        out_dir.mkdir(parents=True, exist_ok=True)
        for folder, _embs_dict in [(folder_a, embs_a), (folder_b, embs_b)]:
            for img_path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                dets = detector.detect(img)
                if not dets:
                    continue
                det = max(dets, key=lambda d: d.score)
                aligned = align_face(img, det.landmarks)
                cv2.imwrite(str(out_dir / f"aligned_{img_path.name}"), aligned)
        print(f"\nAligned crops saved to {args.save_aligned}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Still-image similarity test")
    parser.add_argument("folder_a", help="Folder of person A images")
    parser.add_argument("folder_b", help="Folder of person B images")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--save-aligned", default=None, dest="save_aligned", help="Directory to save aligned crops")
    main(parser.parse_args())

