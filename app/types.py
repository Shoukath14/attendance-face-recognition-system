"""Shared data types and dataclasses across the application."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameData:
    image: np.ndarray  # BGR, original resolution
    timestamp: float  # time.time() at capture
    frame_id: int


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original-frame pixels
    score: float
    landmarks: np.ndarray  # shape (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth


@dataclass
class QualityReport:
    passed: bool
    score: float  # 0..1 composite score
    reasons: list[str] = field(default_factory=list)  # e.g. ["too_small", "blurry", "pose_yaw"]
    hint: str | None = None  # friendly message for overlay


@dataclass
class LivenessResult:
    is_real: bool
    score: float


@dataclass
class MatchResult:
    student_id: int | None  # None means Unknown
    score: float  # best cosine similarity
    second_score: float  # best score of a different student
    margin: float
    candidates: list[tuple[int, float]] = field(default_factory=list)  # top-k (student_id, score)


@dataclass
class TrackView:
    track_id: int
    box: tuple[float, float, float, float]
    state: str  # "verifying" | "confirmed" | "unknown" | "spoof" | "poor_quality"
    student_id: int | None = None
    name: str | None = None
    roll_no: str | None = None
    score: float | None = None
    hint: str | None = None
    already_marked: bool = False


@dataclass
class EnrollmentProgress:
    state: str  # "waiting" | "capturing" | "done" | "failed"
    progress: float  # 0.0 to 1.0
    hint: str | None = None
    quality_score: float = 0.0


@dataclass
class EnrollmentResult:
    success: bool
    student_id: int | None = None
    message: str = ""

