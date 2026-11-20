"""Unit tests for all repository classes (StudentRepo, EmbeddingRepo, SessionRepo, AttendanceRepo, ReviewRepo)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.storage.db import init_db  # noqa: E402
from app.storage.repositories import (  # noqa: E402
    AttendanceRepo,
    EmbeddingRepo,
    ReviewRepo,
    SessionRepo,
    StudentRepo,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Return a fresh temporary SQLite DB path."""
    p = tmp_path / "test.db"
    init_db(p)
    return p


# ---------------------------------------------------------------------------
# StudentRepo
# ---------------------------------------------------------------------------

class TestStudentRepo:
    def test_add_and_get(self, db_path: Path) -> None:
        repo = StudentRepo(db_path)
        sid = repo.add("S001", "Alice")
        student = repo.get_by_id(sid)
        assert student is not None
        assert student["roll_no"] == "S001"
        assert student["name"] == "Alice"

    def test_list_active(self, db_path: Path) -> None:
        repo = StudentRepo(db_path)
        repo.add("S001", "Alice")
        repo.add("S002", "Bob")
        students = repo.get_all_active()
        assert len(students) == 2
        names = {s["name"] for s in students}
        assert names == {"Alice", "Bob"}

    def test_deactivate(self, db_path: Path) -> None:
        repo = StudentRepo(db_path)
        sid = repo.add("S003", "Charlie")
        repo.deactivate(sid)
        students = repo.get_all_active()
        assert all(s["id"] != sid for s in students)

    def test_get_nonexistent_returns_none(self, db_path: Path) -> None:
        repo = StudentRepo(db_path)
        assert repo.get_by_id(9999) is None


# ---------------------------------------------------------------------------
# EmbeddingRepo
# ---------------------------------------------------------------------------

class TestEmbeddingRepo:
    def test_save_and_load_roundtrip(self, db_path: Path) -> None:
        srep = StudentRepo(db_path)
        erep = EmbeddingRepo(db_path)
        sid = srep.add("E001", "Dave")

        vec = np.random.default_rng(0).standard_normal(512).astype(np.float32)
        erep.add(sid, "arcface", "original", vec)

        rows = erep.get_all_embeddings("arcface")
        assert len(rows) == 1
        loaded = np.frombuffer(rows[0][1], dtype=np.float32)
        np.testing.assert_allclose(loaded, vec, rtol=1e-5)

    def test_delete_for_student(self, db_path: Path) -> None:
        srep = StudentRepo(db_path)
        erep = EmbeddingRepo(db_path)
        sid = srep.add("E002", "Eve")

        vec = np.ones(512, dtype=np.float32)
        erep.add(sid, "arcface", "original", vec)
        erep.delete_for_student(sid)

        rows = erep.get_all_embeddings("arcface")
        assert len(rows) == 0


# ---------------------------------------------------------------------------
# SessionRepo
# ---------------------------------------------------------------------------

class TestSessionRepo:
    def test_create_and_get(self, db_path: Path) -> None:
        repo = SessionRepo(db_path)
        sid = repo.create("Morning Class", "2024-01-01T08:00:00", late_after_min=10)
        session = repo.get_by_id(sid)
        assert session is not None
        assert session["name"] == "Morning Class"
        assert session["status"] == "scheduled"

    def test_start_stop_transitions(self, db_path: Path) -> None:
        repo = SessionRepo(db_path)
        sid = repo.create("Afternoon", "2024-01-01T13:00:00")
        repo.set_status(sid, "active")
        assert repo.get_by_id(sid)["status"] == "active"

        repo.set_status(sid, "closed", end_time="2024-01-01T15:00:00")
        session = repo.get_by_id(sid)
        assert session["status"] == "closed"
        assert session["end_time"] == "2024-01-01T15:00:00"

    def test_get_active_session(self, db_path: Path) -> None:
        repo = SessionRepo(db_path)
        sid = repo.create("Active Session", "2024-01-01T09:00:00")
        repo.set_status(sid, "active")
        active = repo.get_active_session()
        assert active is not None
        assert active["id"] == sid


# ---------------------------------------------------------------------------
# AttendanceRepo
# ---------------------------------------------------------------------------

class TestAttendanceRepo:
    def _setup(self, db_path: Path):
        srep = StudentRepo(db_path)
        sess = SessionRepo(db_path)
        arep = AttendanceRepo(db_path)
        sid = srep.add("A001", "Frank")
        session_id = sess.create("Test Session", "2024-01-01T10:00:00")
        sess.set_status(session_id, "active")
        return arep, session_id, sid

    def test_mark_returns_true(self, db_path: Path) -> None:
        arep, session_id, sid = self._setup(db_path)
        result = arep.mark(session_id, sid, "present")
        assert result is True

    def test_mark_idempotent(self, db_path: Path) -> None:
        """Second mark for the same (session, student) must return False."""
        arep, session_id, sid = self._setup(db_path)
        arep.mark(session_id, sid, "present")
        result = arep.mark(session_id, sid, "present")
        assert result is False

    def test_get_by_session(self, db_path: Path) -> None:
        arep, session_id, sid = self._setup(db_path)
        arep.mark(session_id, sid, "late", confidence=0.85)
        rows = arep.get_session_attendance(session_id)
        assert len(rows) == 1
        assert rows[0]["student_id"] == sid

    def test_is_marked(self, db_path: Path) -> None:
        arep, session_id, sid = self._setup(db_path)
        assert not arep.is_marked(session_id, sid)
        arep.mark(session_id, sid, "present")
        assert arep.is_marked(session_id, sid)


# ---------------------------------------------------------------------------
# ReviewRepo
# ---------------------------------------------------------------------------

class TestReviewRepo:
    def test_enqueue_and_get_pending(self, db_path: Path) -> None:
        repo = ReviewRepo(db_path)
        rid = repo.add(None, "/tmp/snap.jpg", "low_confidence", [])
        pending = repo.get_unresolved()
        assert any(r["id"] == rid for r in pending)

    def test_resolve(self, db_path: Path) -> None:
        repo = ReviewRepo(db_path)
        rid = repo.add(None, "/tmp/snap2.jpg", "spoof", [])
        repo.resolve(rid, "confirmed_spoof")
        pending = repo.get_unresolved()
        assert all(r["id"] != rid for r in pending)
