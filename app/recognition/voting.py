"""Per-track temporal voting for identity confirmation."""

from __future__ import annotations

from collections import deque

from app.types import MatchResult


class TrackVoter:
    """
    Accumulates per-frame match results for a track and returns a decision.

    States:
      - "pending"   : not enough evidence yet
      - "confirmed" : same identity wins confirm_votes of the last window
      - "unknown"   : too many below-threshold results
      - "spoof"     : liveness check failed majority
    """

    def __init__(self, window: int = 8, confirm_votes: int = 5, unknown_votes: int = 6) -> None:
        self._window = window
        self._confirm_votes = confirm_votes
        self._unknown_votes = unknown_votes
        self._match_history: deque[MatchResult] = deque(maxlen=window)
        self._liveness_history: deque[bool] = deque(maxlen=window)

    def add(self, match: MatchResult, liveness_ok: bool) -> None:
        self._match_history.append(match)
        self._liveness_history.append(liveness_ok)

    def decision(self) -> tuple[str, int | None]:
        """Return (state, student_id | None)."""
        if len(self._match_history) == 0:
            return "pending", None

        # Spoof check: majority of liveness failures
        spoof_count = sum(1 for v in self._liveness_history if not v)
        if spoof_count >= max(1, len(self._liveness_history) // 2 + 1):
            return "spoof", None

        # Count votes per student
        votes: dict[int, int] = {}
        unknown_count = 0
        for m in self._match_history:
            if m.student_id is None:
                unknown_count += 1
            else:
                votes[m.student_id] = votes.get(m.student_id, 0) + 1

        # Confirmed: a single student wins enough votes
        for sid, count in votes.items():
            if count >= self._confirm_votes:
                return "confirmed", sid

        # Unknown: too many below-threshold
        if unknown_count >= self._unknown_votes:
            return "unknown", None

        return "pending", None

    def reset(self) -> None:
        self._match_history.clear()
        self._liveness_history.clear()

