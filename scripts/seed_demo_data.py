"""
Seed the database with demo data for offline demonstration.
Creates 3 synthetic students with pre-computed random embeddings
and a closed session with attendance records.

Usage:
  .venv\\Scripts\\python.exe scripts/seed_demo_data.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.storage.db import init_db
from app.storage.repositories import AttendanceRepo, EmbeddingRepo, SessionRepo, StudentRepo


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _yesterday_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _random_roll() -> str:
    """Generate a random 8-char hex student ID."""
    return str(uuid.uuid4())[:8].upper()


def main() -> None:
    db_path = Path("data/attendance.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialise schema
    init_db(db_path)

    student_repo = StudentRepo(db_path)
    embedding_repo = EmbeddingRepo(db_path)
    session_repo = SessionRepo(db_path)
    attendance_repo = AttendanceRepo(db_path)

    # --- Insert students ---
    students = [
        ("Alice Smith", _random_roll()),
        ("Bob Jones", _random_roll()),
        ("Carol Lee", _random_roll()),
    ]
    student_ids: list[int] = []
    for name, roll_no in students:
        sid = student_repo.add(roll_no=roll_no, name=name)
        student_ids.append(sid)

    # --- Insert random 512-dim float32 embeddings ---
    rng = np.random.default_rng(seed=42)
    for sid in student_ids:
        vec = rng.standard_normal(512).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-9  # L2-normalise
        embedding_repo.add(
            student_id=sid,
            model_name="arcface",
            kind="enroll",
            vector=vec,
        )

    # --- Create a closed session (yesterday) ---
    yesterday = _yesterday_iso()
    session_id = session_repo.create(
        name="Demo Session",
        start_time=yesterday,
        late_after_min=10,
    )
    # Mark active then closed to exercise transitions
    session_repo.set_status(session_id, "active")
    session_repo.set_status(session_id, "closed", end_time=_utcnow_iso())

    # --- Attendance records ---
    # Alice = present, on_time
    alice_id, bob_id = student_ids[0], student_ids[1]
    # Carol (student_ids[2]) is absent — not inserted so auto-absent at report time
    attendance_repo.mark(
        session_id=session_id,
        student_id=alice_id,
        status="present",
        confidence=0.92,
        source="auto",
        note="on_time",
    )
    attendance_repo.mark(
        session_id=session_id,
        student_id=bob_id,
        status="present",
        confidence=0.78,
        source="auto",
        note="late",
    )

    print("Seeded demo data: 3 students, 1 closed session, 2 attendance records")
    print(f"  DB path  : {db_path.resolve()}")
    print(f"  Students : Alice Smith ({students[0][1]}), Bob Jones ({students[1][1]}), Carol Lee ({students[2][1]})")
    print(f"  Session  : 'Demo Session' id={session_id} (closed)")
    print("  Attendance: Alice=present(on_time), Bob=present(late), Carol=absent (no record)")


if __name__ == "__main__":
    main()
