"""Unit tests for TrackVoter."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.recognition.voting import TrackVoter  # noqa: E402
from app.types import MatchResult  # noqa: E402


def _match(sid: int | None, score: float = 0.8) -> MatchResult:
    return MatchResult(student_id=sid, score=score, second_score=0.0, margin=score)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_five_of_eight_votes_confirmed() -> None:
    """5 of 8 votes for the same student_id → confirmed."""
    voter = TrackVoter(window=8, confirm_votes=5, unknown_votes=6)
    for _ in range(5):
        voter.add(_match(7), liveness_ok=True)
    for _ in range(3):
        voter.add(_match(None), liveness_ok=True)

    state, sid = voter.decision()
    assert state == "confirmed"
    assert sid == 7


def test_all_none_votes_unknown() -> None:
    """All 8 votes are None (below threshold) → unknown."""
    voter = TrackVoter(window=8, confirm_votes=5, unknown_votes=6)
    for _ in range(8):
        voter.add(_match(None), liveness_ok=True)

    state, sid = voter.decision()
    assert state == "unknown"
    assert sid is None


def test_majority_liveness_false_spoof() -> None:
    """Majority liveness=False → spoof (overrides identity votes)."""
    voter = TrackVoter(window=8, confirm_votes=5, unknown_votes=6)
    for _ in range(5):
        voter.add(_match(3), liveness_ok=False)
    for _ in range(3):
        voter.add(_match(3), liveness_ok=True)

    state, sid = voter.decision()
    assert state == "spoof"
    assert sid is None


def test_pending_not_enough_votes() -> None:
    """Fewer votes than confirm_votes → pending."""
    voter = TrackVoter(window=8, confirm_votes=5, unknown_votes=6)
    for _ in range(3):
        voter.add(_match(1), liveness_ok=True)

    state, _ = voter.decision()
    assert state == "pending"


def test_reset_clears_history() -> None:
    """reset() must clear all history → pending again."""
    voter = TrackVoter(window=8, confirm_votes=5, unknown_votes=6)
    for _ in range(8):
        voter.add(_match(None), liveness_ok=True)

    voter.reset()
    state, _ = voter.decision()
    assert state == "pending"
