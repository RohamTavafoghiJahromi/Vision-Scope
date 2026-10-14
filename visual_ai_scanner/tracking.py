"""Thin wrapper around ultralytics' built-in ByteTrack / BoT-SORT trackers."""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from .detection import Detection

log = logging.getLogger(__name__)


class Tracker:
    """Uses ultralytics YOLO.track which internally runs ByteTrack/BoT-SORT.
    Because ultralytics does detection+tracking in one call, we run tracking
    through the same YoloDetector instance and let it assign IDs."""

    def __init__(self, yolo, tracker_cfg: str = "bytetrack.yaml"):
        self.yolo = yolo
        self.cfg = tracker_cfg
        self._names: dict[int, str] = {}

    def track(self, image: np.ndarray, conf: float) -> list[Detection]:
        try:
            results = self.yolo.model.track(
                image, conf=conf, persist=True, tracker=self.cfg,
                device=self.yolo.device, half=self.yolo.half, verbose=False,
            )
        except Exception as e:
            log.warning("Tracking failed (%s); falling back to plain detection.", e)
            return self.yolo.infer(image, conf=conf)
        out: list[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(cls)
        for (x1, y1, x2, y2), c, s, tid in zip(xyxy, cls, sc, ids):
            label = r.names.get(int(c), str(int(c)))
            if tid is not None:
                self._names[int(tid)] = label
            out.append(Detection(int(x1), int(y1), int(x2), int(y2), label,
                                 float(s), track_id=(int(tid) if tid is not None else None),
                                 source="yolo"))
        return out