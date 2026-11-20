"""Review Queue page: resolve unknown/uncertain faces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.config import load_config
from app.storage.db import init_db
from app.storage.repositories import AttendanceRepo, ReviewRepo, SessionRepo, StudentRepo

st.set_page_config(page_title="Review Queue", layout="wide")
st.title("🔍 Review Queue")

cfg = load_config("config.yaml")
init_db(cfg.paths.db)
review_repo = ReviewRepo(cfg.paths.db)
att_repo = AttendanceRepo(cfg.paths.db)
student_repo = StudentRepo(cfg.paths.db)
session_repo = SessionRepo(cfg.paths.db)

items = review_repo.get_unresolved()
if not items:
    st.success("No unresolved review items.")
else:
    st.info(f"{len(items)} item(s) pending review.")
    students = student_repo.get_all_active()
    student_map = {s["id"]: f"{s['name']} ({s['roll_no']})" for s in students}

    for item in items:
        with st.expander(f"[{item['reason'].upper()}] {item['created_at']}  –  {Path(item['snapshot_path']).name}"):
            col1, col2 = st.columns([1, 2])

            with col1:
                snap = Path(item["snapshot_path"])
                if snap.exists():
                    st.image(str(snap), width=150, caption="Unknown Face")
                else:
                    st.warning("Snapshot not found")

            with col2:
                try:
                    candidates = json.loads(item.get("top_candidates") or "[]")
                except Exception:
                    candidates = []
                if candidates:
                    st.write("**Top candidates:**")
                    for sid, score in candidates[:3]:
                        st.write(f"  • {student_map.get(sid, f'ID:{sid}')} — score {score:.3f}")

                st.write("**Assign identity:**")
                selected_student = st.selectbox(
                    "Student", options=["(Dismiss)"] + [student_map[s["id"]] for s in students],
                    key=f"sel_{item['id']}"
                )
                reason_note = st.text_input("Reason (required for override)", key=f"note_{item['id']}")
                col_a, col_b = st.columns(2)
                if col_a.button("✅ Assign", key=f"assign_{item['id']}"):
                    if selected_student == "(Dismiss)":
                        st.warning("Select a student or dismiss.")
                    elif not reason_note.strip():
                        st.warning("Please enter a reason for the manual override.")
                    else:
                        matched_student = next((s for s in students if student_map[s["id"]] == selected_student), None)
                        if matched_student:
                            session = session_repo.get_by_id(item["session_id"]) if item.get("session_id") else None
                            if session:
                                att_repo.mark(session["id"], matched_student["id"], "present",
                                              source="manual", note=reason_note.strip())
                            review_repo.resolve(item["id"], f"assigned:{matched_student['id']}:{reason_note}")
                            st.success("Assigned and resolved.")
                            st.rerun()

                if col_b.button("❌ Dismiss", key=f"dismiss_{item['id']}"):
                    review_repo.resolve(item["id"], "dismissed")
                    st.rerun()

