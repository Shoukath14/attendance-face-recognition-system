"""CSV exporters and report helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_session_csv(attendance_rows: list[dict], output_path: str | None = None) -> str:
    """
    Export session attendance to CSV. Returns CSV string (or writes to output_path).
    """
    df = pd.DataFrame(attendance_rows)
    if df.empty:
        csv_str = "No attendance records found.\n"
    else:
        cols = [c for c in ["roll_no", "name", "marked_at", "status", "confidence", "source", "note"] if c in df.columns]
        csv_str = df[cols].to_csv(index=False)

    if output_path:
        Path(output_path).write_text(csv_str, encoding="utf-8")
    return csv_str


def compute_attendance_percentage(student_id: int, attendance_repo, session_repo) -> dict:
    """Compute attendance percentage across all closed sessions for a student."""
    sessions = [s for s in session_repo.get_all() if s.get("status") == "closed"]
    total = len(sessions)
    if total == 0:
        return {"student_id": student_id, "total_sessions": 0, "attended": 0, "percentage": 0.0}
    attended = sum(
        1 for s in sessions if attendance_repo.is_marked(s["id"], student_id)
    )
    return {
        "student_id": student_id,
        "total_sessions": total,
        "attended": attended,
        "percentage": round(attended / total * 100, 1),
    }


def session_summary_table(session_id: int, attendance_repo, student_repo) -> pd.DataFrame:
    """Return a DataFrame summarising a session's attendance including absent students."""
    rows = attendance_repo.get_session_attendance(session_id)
    present_ids = {r["student_id"] for r in rows}
    all_students = student_repo.get_all_active()
    records = []
    for s in all_students:
        if s["id"] in present_ids:
            rec = next(r for r in rows if r["student_id"] == s["id"])
            records.append({
                "roll_no": s["roll_no"],
                "name": s["name"],
                "status": rec["status"],
                "marked_at": rec["marked_at"],
                "source": rec["source"],
            })
        else:
            records.append({
                "roll_no": s["roll_no"],
                "name": s["name"],
                "status": "absent",
                "marked_at": "",
                "source": "",
            })
    return pd.DataFrame(records)

