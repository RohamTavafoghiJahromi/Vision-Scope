"""Overlay rendering: boxes, labels, masks, trails. Label placement avoids
clipping at frame edges."""
from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection

# Stable, high-contrast colors per source. Used to distinguish backends.
_PALETTE = {
    "yolo":      (90, 200, 250),
    "dino":      (120, 250, 120),
    "reference": (250, 160, 90),
}


def _fit_label(text: str, font_scale: float, thickness: int) -> tuple[int, int, int]:
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    return tw, th + base, base


def draw_detections(image: np.ndarray, detections: list[Detection],
                    show_conf: bool = True, show_track_id: bool = True,
                    thickness: int = 2, font_scale: float = 0.5,
                    trails: dict[int, list[tuple[int, int]]] | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    out = image.copy()

    # --- Adaptive sizing ---
    # Reference: 1280 px wide is the "1.0" baseline.
    # At 640 wide → 0.72×; at 1920 wide → 1.25×; clamped to [0.45, 1.6].
    scale_factor = max(0.45, min(1.6, w / 1280.0))
    font_scale = float(font_scale) * scale_factor
    thickness = max(1, int(round(thickness * scale_factor)))

    # Motion trails
    if trails:
        for tid, pts in trails.items():
            if len(pts) < 2:
                continue
            for a, b in zip(pts[:-1], pts[1:]):
                cv2.line(out, a, b, (200, 200, 200), 1, cv2.LINE_AA)

    for d in detections:
        color = _PALETTE.get(d.source, (200, 200, 200))
        x1, y1, x2, y2 = d.xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Mask
        if d.mask is not None and d.mask.any():
            overlay = np.zeros_like(out)
            overlay[d.mask] = color
            out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        # Compose label
        parts = [d.label]
        if show_track_id and d.track_id is not None:
            parts.append(f"#{d.track_id}")
        if show_conf:
            parts.append(f"{d.score:.2f}")
        text = "  ".join(parts)

        tw, th, base = _fit_label(text, font_scale, 1)
        pad = 4
        lx1 = x1
        ly1 = y1 - th - 2 * pad
        if ly1 < 0:                     # flip below box if no room above
            ly1 = y1 + 2
        lx2 = min(w - 1, lx1 + tw + 2 * pad)
        ly2 = ly1 + th + 2 * pad

        cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
        # Choose text color for contrast against the box fill.
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(out, text, (lx1 + pad, ly2 - pad - base + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)
    return out