"""Typed configuration loader and validator using Pydantic."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class CameraConfig(BaseModel):
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    rotate_deg: int = 0  # 0, 90, 180, 270 (-90) for portrait / mobile cameras
    reconnect_backoff_s: float = 1.0
    watchdog_timeout_s: float = 3.0


class DetectorConfig(BaseModel):
    type: Literal["scrfd", "yunet", "mediapipe"] = "scrfd"
    model_pack: str = "buffalo_s"
    det_size: int = 640
    score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    detect_every_n_frames: int = Field(default=3, ge=1)


class TrackerConfig(BaseModel):
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_age_frames: int = Field(default=15, ge=1)


class QualityThresholds(BaseModel):
    min_face_px: int = Field(ge=10)
    max_yaw_deg: float = Field(ge=0.0, le=90.0)
    max_pitch_deg: float = Field(ge=0.0, le=90.0)
    min_blur_var: float = Field(ge=0.0)
    brightness_range: tuple[int, int]


class QualityConfig(BaseModel):
    enroll: QualityThresholds
    recognize: QualityThresholds


class LivenessConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/minifasnet.onnx"
    real_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class RecognitionConfig(BaseModel):
    embedder: Literal["arcface", "facenet"] = "arcface"
    threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    margin: float = Field(default=0.05, ge=0.0, le=1.0)
    duplicate_enroll_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    evaluate_interval_s: float = Field(default=0.3, ge=0.0)
    vote_window: int = Field(default=8, ge=1)
    confirm_votes: int = Field(default=5, ge=1)
    unknown_votes: int = Field(default=6, ge=1)


class EnrollmentConfig(BaseModel):
    burst_seconds: float = Field(default=1.5, ge=0.5)
    augmentations: list[str] = Field(default_factory=lambda: ["flip", "bright", "dark", "blur"])


class AttendanceConfig(BaseModel):
    default_late_after_min: int = Field(default=10, ge=0)


class PathsConfig(BaseModel):
    db: str = "data/attendance.db"
    thumbnails: str = "data/thumbnails"
    review_snapshots: str = "data/review_snapshots"
    models: str = "models"


class DisplayConfig(BaseModel):
    window_name: str = "Attendance"
    show_fps: bool = True
    mirror_preview: bool = False


class AppConfig(BaseModel):
    camera: CameraConfig
    detector: DetectorConfig
    tracker: TrackerConfig
    quality: QualityConfig
    liveness: LivenessConfig
    recognition: RecognitionConfig
    enrollment: EnrollmentConfig
    attendance: AttendanceConfig
    paths: PathsConfig
    display: DisplayConfig


def load_config(config_path: str | Path = "config.yaml") -> AppConfig:
    """Load and validate the YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax in {path}: {e}") from e

    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Configuration validation error in {path}:\n{e}") from e

