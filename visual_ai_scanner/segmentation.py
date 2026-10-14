"""Optional SAM segmentation. Heavy; disabled by default."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class SamSegmenter:
    """Uses ultralytics' SAM 2 / SAM weights. Produces a boolean mask per
    bounding box. Keep this disabled unless your GPU is comfortable."""

    def __init__(self, weights: str = "mobile_sam.pt", device: str = "auto"):
        from ultralytics import SAM
        self.model = SAM(weights)
        self.device = "cuda" if (device == "auto" and _has_cuda()) else "cpu"
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"

    def segment(self, image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
        if not boxes:
            return []
        masks: list[np.ndarray] = []
        try:
            results = self.model(image, bboxes=list(boxes), device=self.device, verbose=False)
            if not results or results[0].masks is None:
                return [np.zeros(image.shape[:2], bool) for _ in boxes]
            md = results[0].masks.data.cpu().numpy()
            for m in md:
                if m.shape != image.shape[:2]:
                    import cv2
                    m = cv2.resize(m.astype(np.uint8), image.shape[:2][::-1],
                                   interpolation=cv2.INTER_NEAREST)
                masks.append(m.astype(bool))
        except Exception as e:
            log.warning("Segmentation failed: %s", e)
            return [np.zeros(image.shape[:2], bool) for _ in boxes]
        return masks


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False