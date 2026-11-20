"""Face embedders: ArcFace (InsightFace) and FaceNet (facenet-pytorch)."""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger("attendance.embedder")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, aligned_face: np.ndarray) -> np.ndarray: ...


class ArcFaceEmbedder:
    """InsightFace ArcFace embedding (MobileFaceNet or ResNet50)."""

    name = "arcface_mbf"
    dim = 512

    def __init__(self, model_pack: str) -> None:
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=model_pack,
            root="models",
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=0, det_size=(640, 640))
        # Grab recognition model directly
        self._rec = self._app.models.get("recognition")
        logger.info("ArcFaceEmbedder ready: pack=%s", model_pack)

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Embed a 112×112 BGR aligned face. Returns L2-normalised float32 (512-d).
        """
        try:
            # InsightFace recognition model expects (1, 3, 112, 112) float32
            import cv2

            img = cv2.resize(aligned_face, (112, 112))
            img.astype(np.float32).transpose(2, 0, 1)[np.newaxis]  # NCHW
            # InsightFace model .get() accepts an aligned face dict or raw array via forward
            feat = self._rec.get_feat(img)
            vec = np.array(feat, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            return vec / (norm + 1e-9)
        except Exception as exc:
            logger.error("ArcFaceEmbedder.embed error: %s", exc)
            return np.zeros(self.dim, dtype=np.float32)


class FaceNetEmbedder:
    """FaceNet baseline using facenet-pytorch InceptionResnetV1."""

    name = "facenet_vggface2"
    dim = 512

    def __init__(self, pretrained: str = "vggface2") -> None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1

            self._model = InceptionResnetV1(pretrained=pretrained).eval()
            self._torch = torch
            logger.info("FaceNetEmbedder ready: pretrained=%s", pretrained)
        except ImportError:
            logger.warning("facenet-pytorch not installed; FaceNetEmbedder will return zeros.")
            self._model = None
            self._torch = None

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(self.dim, dtype=np.float32)
        try:
            import cv2

            img = cv2.resize(aligned_face, (160, 160))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor = self._torch.tensor(img_rgb, dtype=self._torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
            # Normalize
            tensor = (tensor - 0.5) / 0.5
            with self._torch.no_grad():
                feat = self._model(tensor).squeeze().numpy()
            norm = np.linalg.norm(feat)
            return feat / (norm + 1e-9)
        except Exception as exc:
            logger.error("FaceNetEmbedder.embed error: %s", exc)
            return np.zeros(self.dim, dtype=np.float32)


def build_embedder(cfg) -> Embedder:
    etype = cfg.recognition.embedder
    if etype == "arcface":
        return ArcFaceEmbedder(cfg.detector.model_pack)
    elif etype == "facenet":
        return FaceNetEmbedder()
    else:
        raise ValueError(f"Unknown embedder type: {etype}")

