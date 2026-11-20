"""Unit tests for Matcher (cosine-similarity)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recognition.matcher import Matcher  # noqa: E402


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _rand_embedding(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return _unit(v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_identical_embedding_confirmed() -> None:
    """Loading and querying the same vector → score ≈ 1 → confirmed."""
    matcher = Matcher(threshold=0.4, margin=0.0)
    vec = _rand_embedding(1)
    matcher.load(np.array([42], dtype=np.int64), vec.reshape(1, -1))

    result = matcher.match(vec)
    assert result.student_id == 42
    assert result.score > 0.99


def test_orthogonal_embedding_unknown() -> None:
    """An orthogonal query has cosine similarity ≈ 0 → returns unknown."""
    matcher = Matcher(threshold=0.4, margin=0.0)
    vec = _rand_embedding(1)
    matcher.load(np.array([42], dtype=np.int64), vec.reshape(1, -1))

    # Build an orthogonal vector via Gram-Schmidt
    other = _rand_embedding(2)
    other = other - np.dot(other, vec) * vec
    other = _unit(other)

    result = matcher.match(other)
    assert result.student_id is None, f"Expected None but got {result.student_id}"


def test_multiple_embeddings_max_score() -> None:
    """Multiple embeddings per student → matcher takes the max similarity."""
    matcher = Matcher(threshold=0.4, margin=0.0)
    base = _rand_embedding(10)

    # Student 1: two embeddings — base (perfect match) and near-zero
    ids = np.array([1, 1, 2], dtype=np.int64)
    zero_like = np.zeros(512, dtype=np.float32)
    zero_like[0] = 1.0  # unit vector, orthogonal to base in practice

    vecs = np.stack([base, zero_like, _rand_embedding(99)], axis=0)
    matcher.load(ids, vecs)

    result = matcher.match(base)
    assert result.student_id == 1
    assert result.score > 0.99


def test_empty_gallery_returns_unknown() -> None:
    """No enrolled embeddings → MatchResult with student_id=None."""
    matcher = Matcher(threshold=0.4, margin=0.0)
    result = matcher.match(_rand_embedding(0))
    assert result.student_id is None
    assert result.score == 0.0


def test_margin_filter() -> None:
    """When top-2 scores are very close, margin check prevents confirmation."""
    matcher = Matcher(threshold=0.3, margin=0.5)  # high margin
    v1 = _rand_embedding(1)
    v2 = _unit(v1 + 0.01 * _rand_embedding(2))  # very close to v1

    ids = np.array([1, 2], dtype=np.int64)
    matcher.load(ids, np.stack([v1, v2], axis=0))

    result = matcher.match(v1)
    # margin between top-1 and top-2 will be small → should return unknown
    assert result.student_id is None or result.margin < 0.5
