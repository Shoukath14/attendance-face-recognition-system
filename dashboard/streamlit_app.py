"""Streamlit dashboard entry point and navigation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🎓 Attendance System")
st.sidebar.markdown("---")
st.sidebar.info("Navigate using the pages in the sidebar.")

st.title("Face Recognition Attendance System")
st.markdown("""
Welcome to the **Face Recognition Attendance System** dashboard.

Use the sidebar to:
- 📹 **Live Attendance** — view the current session's attendance in real time
- 🗓️ **Sessions** — create, start, and stop sessions
- 👤 **Students / Enroll** — manage students and enroll new ones
- 📊 **Reports** — download session reports and view charts
- 🔍 **Review Queue** — resolve unknown or uncertain face detections
""")

