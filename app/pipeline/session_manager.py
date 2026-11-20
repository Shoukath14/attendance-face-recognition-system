"""Session manager: polls DB for active session and resolves late/present status."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("attendance.session_manager")

IST_OFFSET = 5.5 * 3600  # seconds


class SessionManager:
    """
    Polls the database for the currently active session.
    Provides late-window logic and idempotent already-marked tracking.
    """

    def __init__(self, session_repo, attendance_repo, poll_interval_s: float = 2.0) -> None:
        self._session_repo = session_repo
        self._attendance_repo = attendance_repo
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._active_session = None
        self._marked_set: set[int] = set()  # student_ids already marked this session
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="SessionPoller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def active_session(self):
        with self._lock:
            return self._active_session

    def status_for(self, session, now: datetime | None = None) -> str:
        """Return 'present' or 'late' based on session start time and late window."""
        if session is None:
            return "absent"
        now = now or datetime.now(timezone.utc)
        start = datetime.fromisoformat(session["start_time"]).replace(tzinfo=timezone.utc)
        late_after = session.get("late_after_min", 10)
        elapsed_minutes = (now - start).total_seconds() / 60.0
        return "late" if elapsed_minutes > late_after else "present"

    def already_marked(self, session_id: int, student_id: int) -> bool:
        with self._lock:
            if student_id in self._marked_set:
                return True
        # Double-check DB (in case another process marked)
        result = self._attendance_repo.is_marked(session_id, student_id)
        if result:
            with self._lock:
                self._marked_set.add(student_id)
        return result

    def record_marked(self, student_id: int) -> None:
        with self._lock:
            self._marked_set.add(student_id)

    # ------------------------------------------------------------------
    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                session = self._session_repo.get_active_session()
                with self._lock:
                    old_id = self._active_session["id"] if self._active_session else None
                    new_id = session["id"] if session else None
                    if old_id != new_id:
                        logger.info("Active session changed: %s → %s", old_id, new_id)
                        self._active_session = session
                        self._marked_set = set()
                        # Pre-populate marked set from DB
                        if session:
                            marked = self._attendance_repo.get_marked_student_ids(session["id"])
                            self._marked_set = set(marked)
                    else:
                        self._active_session = session
                        # Synchronize marked set periodically from DB so resets reflect immediately
                        if session:
                            marked = self._attendance_repo.get_marked_student_ids(session["id"])
                            self._marked_set = set(marked)
            except Exception as exc:
                logger.error("SessionManager poll error: %s", exc)
            self._stop_event.wait(timeout=self._poll_interval_s)

