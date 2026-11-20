"""Sessions page: create, start, stop sessions."""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st

from app.config import load_config
from app.storage.db import init_db
from app.storage.repositories import SessionRepo

st.set_page_config(page_title="Sessions", layout="wide")
st.title("🗓️ Sessions")

cfg = load_config("config.yaml")
init_db(cfg.paths.db)
repo = SessionRepo(cfg.paths.db)

# --- Create new session ---
st.subheader("Create New Session")
with st.form("create_session"):
    name = st.text_input("Session Name", placeholder="e.g. CSE 471 – Lecture 1")
    late_min = st.number_input("Late after (minutes)", min_value=0, max_value=120, value=10)
    submitted = st.form_submit_button("Create Session")
    if submitted and name.strip():
        start_time = datetime.now(timezone.utc).isoformat()
        sid = repo.create(name.strip(), start_time, int(late_min))
        repo.set_status(sid, "active")
        st.success(f"Session '{name}' created and set to ACTIVE (id={sid})")
        st.rerun()

st.markdown("---")
st.subheader("All Sessions")

sessions = repo.get_all()
if not sessions:
    st.info("No sessions yet.")
else:
    for s in sessions:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 2])
        col1.write(f"**{s['name']}**")
        col2.write(s["start_time"])
        col3.write(s["status"].upper())
        with col4:
            if s["status"] == "scheduled":
                if st.button("▶ Start", key=f"start_{s['id']}"):
                    repo.set_status(s["id"], "active")
                    st.rerun()
            elif s["status"] == "active":
                if st.button("⏹ Close", key=f"close_{s['id']}"):
                    repo.set_status(s["id"], "closed", datetime.now(timezone.utc).isoformat())
                    st.rerun()
            else:
                st.write("Closed")

