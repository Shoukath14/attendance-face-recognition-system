"""Enrollment-time augmentation: flip, brightness, blur variants."""

from __future__ import annotations

import cv2
import numpy as np


def augment_face(aligned_face: np.ndarray, augmentations: list[str]) -> dict[str, np.ndarray]:
    """
    Generate augmented variants of a 112×112 aligned face.

    Returns a dict of {kind: image} for each requested augmentation.
    """
    variants: dict[str, np.ndarray] = {}

    if "flip" in augmentations:
        variants["flip"] = cv2.flip(aligned_face, 1)

    if "bright" in augmentations:
        variants["bright"] = _adjust_brightness(aligned_face, 1.3)

    if "dark" in augmentations:
        variants["dark"] = _adjust_brightness(aligned_face, 0.7)

    if "blur" in augmentations:
        variants["blur"] = cv2.GaussianBlur(aligned_face, (3, 3), 0.8)

    return variants


def _adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    out = image.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)

