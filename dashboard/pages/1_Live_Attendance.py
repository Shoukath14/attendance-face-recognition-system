"""Live Attendance page: auto-refreshing table + live recognition launch for the active session."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import load_config
from app.storage.db import init_db
from app.storage.repositories import AttendanceRepo, SessionRepo, StudentRepo

st.set_page_config(page_title="Live Attendance", layout="wide")
st.title("📹 Live Attendance")

cfg = load_config("config.yaml")
init_db(cfg.paths.db)
session_repo = SessionRepo(cfg.paths.db)
att_repo = AttendanceRepo(cfg.paths.db)
student_repo = StudentRepo(cfg.paths.db)

if "live_proc" not in st.session_state:
    st.session_state.live_proc = None

session = session_repo.get_active_session()

if session is None:
    st.warning("⚠️ No active session found. Go to the **Sessions** page to create and start a session.")
else:
    st.success(f"🟢 **Active Session:** {session['name']}  |  Started: {session['start_time']}")

    # --- Live Recognition Controller Bar ---
    st.markdown("##### 🚀 Live Recognition Process")
    c_src, c_rot, c_btn1, c_btn2 = st.columns([2.0, 1.2, 1.0, 1.0])

    saved_source = st.session_state.get("cam_source", str(cfg.camera.source))
    saved_rotate = st.session_state.get("cam_rotate", getattr(cfg.camera, "rotate_deg", 90))

    with c_src:
        run_source = st.text_input(
            "Camera / DroidCam Source",
            value=str(saved_source),
            help="DroidCam URL (e.g. http://192.168.1.XX:4747/video) or device index (0, 1)",
        )

    with c_rot:
        rot_options = {"None (0°)": 0, "Rotate 90° Clockwise": 90, "Rotate 270° (-90°)": 270, "Rotate 180°": 180}
        curr_idx = 1 if saved_rotate == 90 else (0 if saved_rotate == 0 else 2)
        sel_rot = st.selectbox(
            "Orientation",
            options=list(rot_options.keys()),
            index=curr_idx,
            help="Select Rotate 90° Clockwise for portrait phones",
        )
        active_rot = rot_options[sel_rot]
        st.session_state.cam_rotate = active_rot

    # Check if live recognition process is running
    is_running = False
    if st.session_state.live_proc is not None:
        if st.session_state.live_proc.poll() is None:
            is_running = True
        else:
            st.session_state.live_proc = None

    with c_btn1:
        st.write("")
        st.write("")
        if not is_running:
            if st.button("▶️ Start Live Recognition", type="primary", use_container_width=True):
                cmd = [
                    sys.executable,
                    "scripts/run_live.py",
                    "--config",
                    "config.yaml",
                    "--source",
                    run_source.strip(),
                    "--rotate",
                    str(active_rot),
                ]
                try:
                    proc = subprocess.Popen(cmd)
                    st.session_state.live_proc = proc
                    st.session_state.cam_source = run_source.strip()
                    st.session_state.cam_rotate = active_rot
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to start recognition: {exc}")
        else:
            st.button("🟢 Recognition Running...", disabled=True, use_container_width=True)

    with c_btn2:
        st.write("")
        st.write("")
        if is_running:
            if st.button("⏹️ Stop Recognition", type="secondary", use_container_width=True):
                st.session_state.live_proc.terminate()
                st.session_state.live_proc = None
                st.success("Recognition window closed.")
                st.rerun()

    if is_running:
        st.info("💡 Recognition camera window is active on your screen. Point the camera at faces to log attendance automatically. (Press **Q** on the camera window or click **Stop Recognition** above to exit).")

    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        rows = att_repo.get_session_attendance(session["id"])
        all_students = student_repo.get_all_active()
        present_ids = {r["student_id"] for r in rows}

        records = []
        for s in all_students:
            if s["id"] in present_ids:
                rec = next(r for r in rows if r["student_id"] == s["id"])
                records.append({
                    "Roll No": s["roll_no"],
                    "Name": s["name"],
                    "Status": rec["status"].upper(),
                    "Marked At": rec["marked_at"],
                    "Source": rec["source"],
                })
            else:
                records.append({
                    "Roll No": s["roll_no"],
                    "Name": s["name"],
                    "Status": "PENDING",
                    "Marked At": "",
                    "Source": "",
                })

        df = pd.DataFrame(records)
        st.dataframe(df, use_container_width=True, height=450)

    with col2:
        present_count = sum(1 for r in records if r["Status"] in ("PRESENT", "LATE"))
        st.metric("Present / Late", present_count)
        st.metric("Total Students", len(all_students))
        st.metric("Pending", len(all_students) - present_count)

        st.markdown("---")
        st.markdown("##### 🔄 Reset Options")

        if st.button("🔄 Refresh Table", use_container_width=True):
            st.rerun()

        # Reset specific student
        if present_ids:
            marked_students = [s for s in all_students if s["id"] in present_ids]
            selected_student = st.selectbox(
                "Select student to reset",
                options=marked_students,
                format_func=lambda s: f"{s['name']} ({s['roll_no']})",
            )
            if st.button("↩️ Reset Selected Student", use_container_width=True):
                # Import directly to ensure fresh module loaded by Streamlit server
                import importlib

                import app.storage.repositories as repos_mod
                importlib.reload(repos_mod)
                fresh_att_repo = repos_mod.AttendanceRepo(cfg.paths.db)
                fresh_att_repo.delete_student_from_session(session["id"], selected_student["id"])
                st.success(f"Reset attendance for {selected_student['name']}. You can now retake attendance!")
                st.rerun()

            if st.button("⚠️ Reset All Attendance", type="secondary", use_container_width=True):
                import importlib

                import app.storage.repositories as repos_mod
                importlib.reload(repos_mod)
                fresh_att_repo = repos_mod.AttendanceRepo(cfg.paths.db)
                count = fresh_att_repo.clear_session_attendance(session["id"])
                st.warning(f"Cleared {count} attendance record(s). Ready to take fresh attendance!")
                st.rerun()
