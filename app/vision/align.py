"""Face alignment: 5-point similarity transform to 112×112."""

from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace 5-point template (112×112)
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def align_face(frame: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """
    Compute a similarity transform from the 5 detected landmarks to the ArcFace template
    and warp the face to `size × size` BGR.

    Args:
        frame: Full BGR frame.
        landmarks: (5, 2) float32 array.
        size: Output square size (default 112).

    Returns:
        BGR uint8 array of shape (size, size, 3).
    """
    dst = ARCFACE_TEMPLATE.copy()
    if size != 112:
        dst = dst * (size / 112.0)

    src = landmarks.astype(np.float32)
    transform = _umeyama(src, dst)
    aligned = cv2.warpAffine(frame, transform[:2], (size, size), flags=cv2.INTER_LINEAR)
    return aligned


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate a similarity transform (scale + rotation + translation) from src→dst."""
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    sigma_src = (src_demean ** 2).sum() / num
    cov = dst_demean.T @ src_demean / num

    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1

    scale = np.sum(S * D) / (sigma_src + 1e-9)
    R = U @ D @ Vt
    t = dst_mean - scale * R @ src_mean

    transform = np.eye(dim + 1)
    transform[:dim, :dim] = scale * R
    transform[:dim, dim] = t
    return transform.astype(np.float32)

