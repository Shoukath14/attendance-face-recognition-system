"""Repository classes for SQLite: Students, Embeddings, Sessions, Attendance, Review."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.storage.db import get_connection

logger = logging.getLogger("attendance.repos")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudentRepo:
    def __init__(self, db_path: str | Path) -> None:
        self._db = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db)

    def add(self, roll_no: str, name: str, thumbnail_path: str | None = None) -> int:
        with self._conn() as conn:
            now = _utcnow()
            cur = conn.execute(
                "INSERT INTO students (roll_no, name, thumbnail_path, consent_at, created_at) VALUES (?, ?, ?, ?, ?)",
                (roll_no, name, thumbnail_path, now, now),
            )
            return cur.lastrowid

    def get_by_id(self, student_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
            return dict(row) if row else None

    def get_all_active(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM students WHERE active = 1 ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    def delete(self, student_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM students WHERE id = ?", (student_id,))

    def set_thumbnail(self, student_id: int, path: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE students SET thumbnail_path = ? WHERE id = ?", (path, student_id))

    def deactivate(self, student_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE students SET active = 0 WHERE id = ?", (student_id,))


class EmbeddingRepo:
    def __init__(self, db_path: str | Path) -> None:
        self._db = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db)

    def add(self, student_id: int, model_name: str, kind: str, vector: np.ndarray) -> int:
        blob = vector.astype(np.float32).tobytes()
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO embeddings (student_id, model_name, kind, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                (student_id, model_name, kind, blob, _utcnow()),
            )
            return cur.lastrowid

    def get_all_embeddings(self, model_name: str) -> list[tuple[int, bytes]]:
        """Return list of (student_id, vector_blob) for the given model."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT student_id, vector FROM embeddings WHERE model_name = ?", (model_name,)
            ).fetchall()
            return [(r[0], r[1]) for r in rows]

    def delete_for_student(self, student_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))


class SessionRepo:
    def __init__(self, db_path: str | Path) -> None:
        self._db = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db)

    def create(self, name: str, start_time: str, late_after_min: int = 10) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (name, start_time, late_after_min, status) VALUES (?, ?, ?, 'scheduled')",
                (name, start_time, late_after_min),
            )
            return cur.lastrowid

    def set_status(self, session_id: int, status: str, end_time: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET status = ?, end_time = ? WHERE id = ?",
                (status, end_time, session_id),
            )

    def get_active_session(self) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE status = 'active' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_all(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    def get_by_id(self, session_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None


class AttendanceRepo:
    def __init__(self, db_path: str | Path) -> None:
        self._db = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db)

    def mark(
        self,
        session_id: int,
        student_id: int,
        status: str,
        confidence: float | None = None,
        source: str = "auto",
        note: str | None = None,
    ) -> bool:
        """Insert attendance row. Returns True if inserted, False if duplicate."""
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO attendance (session_id, student_id, marked_at, status, confidence, source, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, student_id, _utcnow(), status, confidence, source, note),
                )
            return True
        except sqlite3.IntegrityError:
            return False  # Duplicate

    def is_marked(self, session_id: int, student_id: int) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM attendance WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            ).fetchone()
            return row is not None

    def get_marked_student_ids(self, session_id: int) -> list[int]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT student_id FROM attendance WHERE session_id = ?", (session_id,)
            ).fetchall()
            return [r[0] for r in rows]

    def get_session_attendance(self, session_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT a.*, s.name, s.roll_no FROM attendance a JOIN students s ON a.student_id = s.id "
                "WHERE a.session_id = ? ORDER BY a.marked_at",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_for_student(self, student_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))

    def delete_student_from_session(self, session_id: int, student_id: int) -> bool:
        """Reset attendance for a specific student in a session."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM attendance WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            )
            return cur.rowcount > 0

    def clear_session_attendance(self, session_id: int) -> int:
        """Clear all attendance records for a session to retake from scratch."""
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM attendance WHERE session_id = ?", (session_id,))
            return cur.rowcount


class ReviewRepo:
    def __init__(self, db_path: str | Path) -> None:
        self._db = str(db_path)

    def _conn(self) -> sqlite3.Connection:
        return get_connection(self._db)

    def add(
        self,
        session_id: int | None,
        snapshot_path: str,
        reason: str,
        top_candidates: list | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO review_queue (session_id, snapshot_path, reason, top_candidates, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, snapshot_path, reason, json.dumps(top_candidates or []), _utcnow()),
            )
            return cur.lastrowid

    def get_unresolved(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM review_queue WHERE resolved = 0 ORDER BY created_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def resolve(self, review_id: int, resolution: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE review_queue SET resolved = 1, resolution = ? WHERE id = ?",
                (resolution, review_id),
            )

