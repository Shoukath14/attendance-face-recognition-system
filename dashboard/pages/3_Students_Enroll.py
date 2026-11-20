"""Students / Enrollment page with direct DroidCam IP & Camera controls."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import load_config
from app.storage.db import init_db
from app.storage.repositories import EmbeddingRepo, StudentRepo

st.set_page_config(page_title="Students & Enroll", layout="wide")
st.title("👤 Students & Enrollment")

cfg = load_config("config.yaml")
init_db(cfg.paths.db)
student_repo = StudentRepo(cfg.paths.db)
emb_repo = EmbeddingRepo(cfg.paths.db)

# ---------------------------------------------------------------------------
# Camera / DroidCam IP Configuration Card
# ---------------------------------------------------------------------------
st.subheader("📹 Camera / DroidCam Connection")

if "cam_source" not in st.session_state:
    st.session_state.cam_source = str(cfg.camera.source)

if "cam_rotate" not in st.session_state:
    st.session_state.cam_rotate = getattr(cfg.camera, "rotate_deg", 90)

col_mode, col_input, col_rot, col_test = st.columns([1.3, 2.2, 1.2, 1.1])

with col_mode:
    cam_type = st.radio(
        "Camera Type",
        ["DroidCam / Mobile IP", "Webcam Index (0, 1...)"],
        index=0 if ("http" in str(st.session_state.cam_source)) else 1,
        horizontal=True,
    )

with col_input:
    if cam_type == "DroidCam / Mobile IP":
        default_val = (
            st.session_state.cam_source
            if "http" in str(st.session_state.cam_source)
            else "http://192.168.1.15:4747/video"
        )
        source_input = st.text_input(
            "DroidCam Stream URL",
            value=default_val,
            help="Check DroidCam app on your phone for Wi-Fi IP and Port (e.g. http://192.168.1.XX:4747/video)",
        )
    else:
        default_idx = (
            st.session_state.cam_source
            if "http" not in str(st.session_state.cam_source)
            else "0"
        )
        source_input = st.text_input(
            "Camera Device Index",
            value=default_idx,
            help="0 is default integrated webcam, 1 is secondary/virtual webcam",
        )

with col_rot:
    rot_options = {"None (Landscape 0°)": 0, "Rotate 90° Clockwise (Portrait)": 90, "Rotate 270° (-90°)": 270, "Rotate 180°": 180}
    current_rot_idx = 1 if st.session_state.cam_rotate == 90 else (0 if st.session_state.cam_rotate == 0 else 2)
    selected_rot = st.selectbox(
        "Orientation / Rotation",
        options=list(rot_options.keys()),
        index=current_rot_idx,
        help="Choose 90° Clockwise if your phone is held vertically in portrait mode",
    )
    st.session_state.cam_rotate = rot_options[selected_rot]

with col_test:
    st.write("")
    st.write("")
    test_cam = st.button("🔍 Test Camera Stream", use_container_width=True)

final_source: int | str
try:
    final_source = int(source_input.strip())
except ValueError:
    final_source = source_input.strip()

st.session_state.cam_source = final_source

if test_cam:
    with st.spinner(f"Connecting to '{final_source}'..."):
        test_cap = cv2.VideoCapture(final_source)
        opened = test_cap.isOpened()
        ret, frame = (False, None)
        if opened:
            # Try reading up to 5 frames
            for _ in range(5):
                ret, frame = test_cap.read()
                if ret:
                    break
                time.sleep(0.05)
        test_cap.release()

    if opened and ret and frame is not None:
        # Apply selected rotation for preview
        if st.session_state.cam_rotate in (90, -270):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif st.session_state.cam_rotate in (180, -180):
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif st.session_state.cam_rotate in (270, -90):
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        st.success(f" Connected successfully! Frame shape: {frame.shape[1]}x{frame.shape[0]}")
        # Show snapshot preview
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb_frame, caption=f"Live Feed Preview from {final_source} (Rotated {st.session_state.cam_rotate}°)", width=320)
    else:
        st.error(
            f"❌ Could not connect to '{final_source}'.\n\n"
            "- If using **DroidCam IP**: Ensure phone and laptop are on the same Wi-Fi, DroidCam app is OPEN on the phone, and URL ends with `/video` (e.g., `http://192.168.1.XX:4747/video`).\n"
            "- If using **Camera Index**: Verify index (0 or 1)."
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Enrollment Form
# ---------------------------------------------------------------------------
st.subheader("Enroll a New Student")
st.info(
    "ℹ️ **Consent notice:** By proceeding, you consent to your facial biometric being captured "
    "and stored as an embedding for attendance purposes."
)

if "enroll_proc" not in st.session_state:
    st.session_state.enroll_proc = None

# Cancel/close button if a camera enrollment is running
if st.session_state.enroll_proc is not None:
    poll = st.session_state.enroll_proc.poll()
    if poll is None:
        st.warning("📷 Camera enrollment window is active! Look at the camera window, or click below to close it.")
        if st.button("❌ Close Camera Enrollment Window", type="primary"):
            st.session_state.enroll_proc.terminate()
            st.session_state.enroll_proc = None
            st.success("Camera enrollment window closed.")
            st.rerun()
    else:
        st.session_state.enroll_proc = None

tab_cam, tab_photo = st.tabs(["📷 Camera Enrollment (DroidCam / Webcam)", "📁 Photo File Upload"])

with tab_cam:
    c_name, c_roll = st.columns(2)
    name_cam = c_name.text_input("Full Name", key="name_cam")
    roll_cam = c_roll.text_input("Roll Number", key="roll_cam")

    btn_launch = st.button("🚀 Start Live Camera Enrollment", type="primary")

    if btn_launch:
        if not name_cam.strip() or not roll_cam.strip():
            st.error("Please provide both Full Name and Roll Number.")
        else:
            cmd = [
                sys.executable,
                "scripts/enroll.py",
                "--name",
                name_cam.strip(),
                "--roll-no",
                roll_cam.strip(),
                "--config",
                "config.yaml",
                "--source",
                str(st.session_state.cam_source),
                "--rotate",
                str(st.session_state.cam_rotate),
            ]
            st.info(
                f"Launching camera window using source `{st.session_state.cam_source}`...\n"
                "Look at your camera until the oval guide turns green, or press **ESC**/**Q** to close."
            )
            try:
                proc = subprocess.Popen(cmd)
                st.session_state.enroll_proc = proc
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to launch enrollment: {exc}")

with tab_photo:
    p_name, p_roll = st.columns(2)
    name_photo = p_name.text_input("Full Name", key="name_photo")
    roll_photo = p_roll.text_input("Roll Number", key="roll_photo")
    uploaded = st.file_uploader("Select Face Photo", type=["jpg", "jpeg", "png"])
    btn_upload = st.button("📥 Process & Enroll Photo")

    if btn_upload:
        if not name_photo.strip() or not roll_photo.strip():
            st.error("Please enter both Name and Roll Number.")
        elif not uploaded:
            st.error("Please select a photo file.")
        else:
            from app.pipeline.enrollment import EnrollmentService
            from app.recognition.embedder import build_embedder
            from app.recognition.matcher import Matcher
            from app.vision.detector import build_detector
            from app.vision.liveness import LivenessChecker
            from app.vision.quality import QualityGate

            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Could not decode image.")
            else:
                with st.spinner("Detecting face and computing embeddings..."):
                    detector = build_detector(cfg)
                    embedder = build_embedder(cfg)
                    matcher = Matcher(cfg.recognition.threshold, cfg.recognition.margin)
                    matcher.reload_from_db(emb_repo, embedder.name)
                    quality = QualityGate(cfg.quality.enroll)
                    liveness = LivenessChecker(cfg)
                    service = EnrollmentService(
                        detector, quality, liveness, embedder, matcher, (student_repo, emb_repo), cfg
                    )
                    result = service.enroll_from_image(name_photo.strip(), roll_photo.strip(), img)
                    if result.success:
                        st.success(f"🎉 {result.message}")
                    else:
                        st.error(f"❌ {result.message}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Enrolled Students Table
# ---------------------------------------------------------------------------
st.subheader("Enrolled Students Directory")
students = student_repo.get_all_active()
if not students:
    st.info("No students enrolled yet.")
else:
    for s in students:
        c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 1.5, 1])
        c1.write(f"**{s['roll_no']}**")
        c2.write(s["name"])
        c3.write(s["created_at"])
        thumb_path = s.get("thumbnail_path")
        if thumb_path and Path(thumb_path).exists():
            c4.image(thumb_path, width=50)
        else:
            c4.write("No photo")

        if c5.button("🗑 Delete", key=f"del_{s['id']}"):
            if thumb_path:
                try:
                    Path(thumb_path).unlink(missing_ok=True)
                except Exception:
                    pass
            emb_repo.delete_for_student(s["id"])
            student_repo.delete(s["id"])
            st.rerun()
