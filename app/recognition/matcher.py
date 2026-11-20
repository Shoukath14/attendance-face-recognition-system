"""Cosine-similarity matcher with top-1/top-2 margin check."""

from __future__ import annotations

import logging

import numpy as np

from app.types import MatchResult

logger = logging.getLogger("attendance.matcher")


class Matcher:
    """
    Cosine similarity matcher.

    Several embeddings per student (original + augmented) are supported.
    Per-student score = max over that student's embeddings.
    """

    def __init__(self, threshold: float, margin: float, top_k: int = 3) -> None:
        self._threshold = threshold
        self._margin = margin
        self._top_k = top_k
        self._ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._vectors: np.ndarray = np.empty((0, 512), dtype=np.float32)

    def load(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Replace in-memory enrolled set. ids[i] is the student_id for vectors[i]."""
        self._ids = np.array(ids, dtype=np.int64)
        self._vectors = np.array(vectors, dtype=np.float32)
        logger.info("Matcher loaded: %d embeddings for %d unique students", len(ids), len(np.unique(ids)))

    def match(self, query: np.ndarray) -> MatchResult:
        """Match query embedding against enrolled set."""
        if len(self._ids) == 0:
            return MatchResult(student_id=None, score=0.0, second_score=0.0, margin=0.0)

        q = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Cosine similarity to all enrolled embeddings
        sims = self._vectors @ q  # shape (N,)

        # Aggregate per student: take max similarity
        unique_ids = np.unique(self._ids)
        per_student: list[tuple[int, float]] = []
        for sid in unique_ids:
            mask = self._ids == sid
            best = float(np.max(sims[mask]))
            per_student.append((int(sid), best))

        per_student.sort(key=lambda x: x[1], reverse=True)

        top_id, top_score = per_student[0]
        second_score = per_student[1][1] if len(per_student) > 1 else 0.0
        margin = top_score - second_score
        candidates = per_student[: self._top_k]

        if top_score < self._threshold or margin < self._margin:
            return MatchResult(
                student_id=None,
                score=top_score,
                second_score=second_score,
                margin=margin,
                candidates=candidates,
            )

        return MatchResult(
            student_id=top_id,
            score=top_score,
            second_score=second_score,
            margin=margin,
            candidates=candidates,
        )

    def reload_from_db(self, repo, model_name: str) -> None:
        """Reload enrolled embeddings from the database."""
        rows = repo.get_all_embeddings(model_name)
        if not rows:
            self._ids = np.empty(0, dtype=np.int64)
            self._vectors = np.empty((0, 512), dtype=np.float32)
            return
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        vecs = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows], dtype=np.float32)
        self.load(ids, vecs)

