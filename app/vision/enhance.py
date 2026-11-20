"""Low-light enhancement helpers (CLAHE / gamma correction)."""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE to the L channel of an LAB image."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_chan = clahe.apply(l_chan)
    enhanced = cv2.merge([l_chan, a_chan, b_chan])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def gamma_correct(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """Apply gamma correction to brighten a dark image (gamma > 1 brightens)."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def enhance_if_dark(image: np.ndarray, brightness_threshold: float = 70.0) -> np.ndarray:
    """Apply CLAHE + mild gamma correction when mean brightness is below threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < brightness_threshold:
        image = apply_clahe(image)
        image = gamma_correct(image, gamma=1.3)
    return image

