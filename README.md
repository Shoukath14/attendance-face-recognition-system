# Face Recognition Attendance System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__s-green.svg)](https://github.com/deepinsight/insightface)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)]()

> **Academic Project** — CSE 471 (Deep Learning for Computer Vision), Semester 5  
> Real-time multi-person face recognition attendance system with passive liveness detection, track voting, and a Streamlit management dashboard.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Key Features](#key-features)
3. [Prerequisites & Requirements](#prerequisites--requirements)
4. [Setup & Installation](#setup--installation)
5. [Camera Configuration](#camera-configuration)
6. [Usage Guide](#usage-guide)
   - [Pre-flight Verification](#1-pre-flight-verification)
   - [Database Initialization & Seeding](#2-database-initialization--seeding)
   - [Student Enrollment](#3-student-enrollment)
   - [Running Live Recognition](#4-running-live-recognition)
   - [Interactive Dashboard](#5-interactive-dashboard)
   - [Evaluation & Calibration](#6-evaluation--calibration)
7. [Project Structure](#project-structure)
8. [Configuration (`config.yaml`)](#configuration)
9. [Known Limitations](#known-limitations)
10. [Data Privacy (DPDP Act)](#data-privacy-dpdp-act)

---

## System Architecture

The pipeline uses decoupled threaded execution to maintain high display refresh rates independently of model inference times:

```
[ Camera Source ] (DirectShow / RTSP / Phone Webcam)
       │
       ▼
[ LatestFrameReader ] (Thread 1: Zero-latency buffer=1 capture)
       │
       ▼
[ InferenceWorker ] (Thread 2: Batch vision pipeline)
   ├── 1. Face Detection (SCRFD 500M)
   ├── 2. Spatial Tracking (IoU Tracker with state prediction)
   ├── 3. Quality Gate (Blur, Resolution, Brightness, Pose check)
   ├── 4. Passive Liveness (MiniFASNet ONNX)
   ├── 5. Face Alignment (Umeyama 5-point similarity transform -> 112x112)
   ├── 6. Feature Embedding (ArcFace MobileFaceNet 512-d)
   ├── 7. Vector Cosine Matcher (Gallery distance + margin filter)
   └── 8. Sliding-Window Track Voter (Confirmed / Spoof / Unknown)
       │
       ├──► [ SessionManager / SQLite ] (Thread-safe idempotent attendance write)
       ▼
[ SharedState ] ◄── Thread-safe atomic snapshot
       │
       ▼
[ Display Thread / UI ] (Thread 3: 25+ FPS real-time rendering)
   └── OpenCV Window / Streamlit Live Dashboard
```

Detailed design and component descriptions can be found in [docs/architecture.md](file:///d:/EDU/sem%205/cse%20471(DLCV)/project/attendance%20face%20recognitio%20system/docs/architecture.md).

---

## Key Features

- **Decoupled Architecture:** 3-thread pipeline (Capture, Inference, Display) ensures a smooth 25+ FPS feed even when running deep learning models on standard CPU hardware.
- **Robust Quality Filtering:** Pre-filters blurry images (Laplacian variance), extreme head poses (3D yaw/pitch), under/over-exposure, and small crops before invoking recognition.
- **Anti-Spoofing Defense:** Passive MiniFASNet anti-spoofing checker blocks 2D paper and screen replays. Gracefully falls back if model weight is not present.
- **Track Voting Engine:** Temporal voting over recent trajectory frames prevents flickering detections and false positives.
- **Student Enrollment:** Burst mode with automatic pose extraction, image augmentations (flips, exposure, blur), and embedding gallery construction.
- **Comprehensive Web Dashboard:** Multi-page Streamlit portal for live attendance monitoring, session lifecycle control, student directory, review queue, and CSV exports.

---

## Prerequisites & Requirements

- **OS:** Windows 10/11 (PowerShell recommended)
- **Python:** Python 3.12.4 (installed in virtual environment `.venv`)
- **Camera:** Standard USB Webcam, Laptop integrated camera, or Smartphone (DroidCam / Iriun)
- **Dependencies:** All listed in `requirements.txt` (OpenCV, InsightFace, ONNX Runtime, NumPy, Streamlit, Plotly, Pydantic)

---

## Setup & Installation

### 1. Clone & Navigate to Project Directory
```powershell
Set-Location "d:\EDU\sem 5\cse 471(DLCV)\project\attendance face recognitio system"
```

### 2. Activate Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Verify Model Weights
The InsightFace `buffalo_s` pack is stored in `models/models/buffalo_s/`. To re-download or inspect models:
```powershell
.venv\Scripts\python.exe scripts/download_models.py
```

---

## Camera Configuration

To check connected cameras and indices on Windows (DirectShow backend):
```powershell
.venv\Scripts\python.exe scripts/list_cameras.py
```

### Using Android Phone as Webcam
1. Install **DroidCam** or **Iriun Webcam** on your phone and PC client.
2. Connect via Wi-Fi or USB Debugging.
3. Once connected, run `list_cameras.py` to identify the virtual camera index (e.g., `0` or `1`).
4. Update `config.yaml`:
   ```yaml
   camera:
     source: 0  # Replace with DroidCam index or RTSP URL: "http://192.168.1.X:4747/video"
   ```

---

## Usage Guide

### 1. Pre-flight Verification
Run the system health checker before demonstrations:
```powershell
.venv\Scripts\python.exe scripts/preflight.py
```

### 2. Database Initialization & Seeding
Initialize the SQLite schema:
```powershell
.venv\Scripts\python.exe scripts/init_db.py
```
To seed demo students and a past session for offline testing:
```powershell
.venv\Scripts\python.exe scripts/seed_demo_data.py
```

### 3. Student Enrollment

#### Via Live Camera:
```powershell
.venv\Scripts\python.exe scripts/enroll.py --name "John Doe" --roll "CS2026-001"
```
*Look at the camera, follow visual feedback on the screen, and press Spacebar to complete.*

#### Via Still Image:
```powershell
.venv\Scripts\python.exe scripts/enroll.py --name "Jane Doe" --roll "CS2026-002" --from-image path/to/photo.jpg
```

### 4. Running Live Recognition
Launch the live recognition window:
```powershell
.venv\Scripts\python.exe scripts/run_live.py
```
- Green box: Verified student (Attendance marked)
- Yellow box: Processing / Quality warning
- Red box: Unknown or Spoof detected
- Press `Q` to exit.

### 5. Interactive Dashboard
Launch the Streamlit web dashboard:
```powershell
.venv\Scripts\streamlit.exe run dashboard/streamlit_app.py
```
**Dashboard Pages:**
- **Live Attendance:** Real-time log of recognized students with photos and timestamps.
- **Sessions:** Create, open, and close attendance sessions.
- **Students & Enroll:** Register new students or review existing profiles.
- **Reports:** Attendance percentage summaries and downloadable CSV sheets.
- **Review Queue:** Review and assign unknown/flagged faces captured during sessions.

### 6. Evaluation & Calibration
To evaluate the model on the benchmark dataset and plot ROC curves:
```powershell
.venv\Scripts\python.exe scripts/evaluate.py --model arcface
.venv\Scripts\python.exe scripts/calibrate_threshold.py
```

---

## Project Structure

```
├── app/
│   ├── camera/          # Video capture & health watchdog
│   ├── vision/          # Detection, tracking, quality gate, liveness, alignment
│   ├── recognition/     # ArcFace/FaceNet embedders, cosine matcher, temporal voting
│   ├── pipeline/        # Threaded inference worker, session manager, shared state
│   ├── storage/         # SQLite schema, repositories, DB connection
│   ├── ui/              # OpenCV HUD overlay drawing
│   └── reports/         # CSV exporters & summary analytics
├── dashboard/           # Multi-page Streamlit dashboard
├── docs/                # Architecture docs, specifications, known limitations
├── models/              # ONNX weights (buffalo_s: det_500m, w600k_mbf, etc.)
├── reports/             # Generated ROC graphs and attendance spreadsheets
├── scripts/             # CLI utilities (preflight, benchmark, enroll, run_live, etc.)
├── tests/               # 53 passing automated unit & integration tests
├── config.yaml          # Central system configuration
└── requirements.txt     # Python packages
```

---

## Known Limitations

For complete details, see [docs/known_limitations.md](file:///d:/EDU/sem%205/cse%20471(DLCV)/project/attendance%20face%20recognitio%20system/docs/known_limitations.md).
- **Passive Liveness:** MiniFASNet targets basic 2D replay (screens/printouts); sophisticated 3D silicone masks require active challenge-response.
- **Occlusions:** Masks or extreme head angles exceeding 35 degrees route to Unknown.
- **Hardware Profile:** Optimized for CPU environments with a ~150-200ms recognition latency.

---

## Data Privacy (DPDP Act)

In accordance with data protection guidelines:
- **No Raw Face Images:** The system stores only mathematical embedding vectors (512 float values) and miniature 80x80 identification thumbnails.
- **Local Storage:** All vectors and logs are kept strictly in local SQLite storage without external cloud transmission.

