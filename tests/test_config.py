"""Test configuration validation and loading."""

import pytest

from app.config import AppConfig, load_config


def test_load_config_valid():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.camera.fps == 30
    assert cfg.detector.det_size == 640
    assert cfg.recognition.threshold == 0.45


def test_invalid_config_raises(tmp_path):
    bad_config = tmp_path / "bad_config.yaml"
    bad_config.write_text("camera:\n  width: -100\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_config)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config.yaml")

