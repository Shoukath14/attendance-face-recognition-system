"""Reports page: session summaries, charts, CSV download."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

from app.config import load_config
from app.reports.exporters import export_session_csv, session_summary_table
from app.storage.db import init_db
from app.storage.repositories import AttendanceRepo, SessionRepo, StudentRepo

st.set_page_config(page_title="Reports", layout="wide")
st.title("📊 Reports")

cfg = load_config("config.yaml")
init_db(cfg.paths.db)
session_repo = SessionRepo(cfg.paths.db)
att_repo = AttendanceRepo(cfg.paths.db)
student_repo = StudentRepo(cfg.paths.db)

sessions = session_repo.get_all()
if not sessions:
    st.info("No sessions found.")
else:
    session_names = {f"{s['name']} (#{s['id']})": s["id"] for s in sessions}
    selected = st.selectbox("Select Session", list(session_names.keys()))
    sid = session_names[selected]
    session = session_repo.get_by_id(sid)

    df = session_summary_table(sid, att_repo, student_repo)
    st.subheader(f"Session: {session['name']}")
    st.dataframe(df, use_container_width=True)

    # Chart
    if not df.empty:
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig = px.bar(status_counts, x="Status", y="Count", color="Status", title="Attendance Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    # CSV download
    rows = att_repo.get_session_attendance(sid)
    csv_str = export_session_csv(rows)
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_str,
        file_name=f"session_{sid}_attendance.csv",
        mime="text/csv",
    )

