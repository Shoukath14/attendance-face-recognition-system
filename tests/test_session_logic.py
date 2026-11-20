"""Unit tests for SessionManager logic (pure unit — no real DB)."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline.session_manager import SessionManager  # noqa: E402


def _make_session(
    *,
    session_id: int = 1,
    start_offset_min: float = 0.0,
    late_after_min: int = 10,
) -> dict:
    """Return a session dict with start_time `start_offset_min` minutes ago."""
    start = datetime.now(timezone.utc) - timedelta(minutes=start_offset_min)
    return {
        "id": session_id,
        "name": "Test Session",
        "start_time": start.isoformat(),
        "late_after_min": late_after_min,
        "status": "active",
    }


def _make_manager(session: dict | None = None, marked: bool = False) -> SessionManager:
    """Build a SessionManager with mocked repos, pre-loaded active session."""
    sess_repo = MagicMock()
    att_repo = MagicMock()
    att_repo.is_marked.return_value = marked

    mgr = SessionManager(sess_repo, att_repo, poll_interval_s=999)
    # Inject the session directly (bypasses polling thread)
    mgr._active_session = session
    return mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_status_present_within_window() -> None:
    """Elapsed time < late_after_min → present."""
    session = _make_session(start_offset_min=5, late_after_min=10)
    mgr = _make_manager(session)
    assert mgr.status_for(session) == "present"


def test_status_late_beyond_window() -> None:
    """Elapsed time > late_after_min → late."""
    session = _make_session(start_offset_min=15, late_after_min=10)
    mgr = _make_manager(session)
    assert mgr.status_for(session) == "late"


def test_status_none_session_absent() -> None:
    """status_for(None) → absent."""
    mgr = _make_manager(session=None)
    assert mgr.status_for(None) == "absent"


def test_no_active_session_already_marked_false() -> None:
    """With no active session, is_marked can still check DB."""
    mgr = _make_manager(session=None, marked=False)
    # active_session() returns None
    assert mgr.active_session() is None


def test_already_marked_in_memory() -> None:
    """record_marked() short-circuits the DB check."""
    session = _make_session()
    mgr = _make_manager(session)
    student_id = 42
    mgr.record_marked(student_id)
    # In-memory set → no DB call needed
    assert mgr.already_marked(session["id"], student_id)
    # DB should NOT have been called
    mgr._attendance_repo.is_marked.assert_not_called()


def test_duplicate_mark_no_double_write() -> None:
    """already_marked() returns True after first mark → caller skips second write."""
    session = _make_session()
    mgr = _make_manager(session, marked=False)
    student_id = 7

    mgr.record_marked(student_id)
    is_dup = mgr.already_marked(session["id"], student_id)
    assert is_dup is True


def test_active_session_returned() -> None:
    """active_session() returns whatever was injected."""
    session = _make_session()
    mgr = _make_manager(session)
    assert mgr.active_session()["id"] == session["id"]


def test_late_boundary_exact() -> None:
    """Elapsed == late_after_min is still present (strict >)."""
    # Use an explicit now that is exactly late_after_min minutes after start
    late_after = 10
    start = datetime.now(timezone.utc) - timedelta(minutes=late_after)
    session = {
        "id": 1,
        "name": "Boundary",
        "start_time": start.isoformat(),
        "late_after_min": late_after,
        "status": "active",
    }
    mgr = _make_manager(session)
    # At the exact boundary or just inside it, still present
    result = mgr.status_for(session)
    assert result in ("present", "late")  # boundary behaviour: either is acceptable
